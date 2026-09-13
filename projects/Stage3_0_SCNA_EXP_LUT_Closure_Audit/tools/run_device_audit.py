#!/usr/bin/env python3
"""Run the preregistered Stage 2.9 device matrices without changing Stage2.75."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from model_lineage import stable_case_id


PROJECT = Path(__file__).resolve().parents[1]
SPEC = json.loads((PROJECT / "experiment_spec.json").read_text())
KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
JSON_MARKER = "VALUE_AUDIT_JSON "


def convert(value: str):
    if value.startswith("0x"):
        return value
    if value in {"nan", "-nan", "inf", "-inf"}:
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parsed_records(text: str, context: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    # ``phase`` is emitted by the host for warmup/measure samples.  Keep the
    # controller phase under a distinct name so it cannot overwrite that
    # sample-level field during parsing.
    base_context = dict(context)
    if "phase" in base_context:
        base_context["audit_phase"] = base_context.pop("phase")
    prefixes = {
        "FIG8_ATTENTION_HOST_TIMING": "attention_timing",
        "FIG8_ATTENTION_COMPARE": "attention_correctness",
        "FIG8_ATTENTION_WORKERS": "attention_workers",
        "FIG8_ATTENTION_TIMERS": "attention_timers",
        "FIG8_NUMERIC ": "attention_numeric",
        "FIG8_AUDIT_RESOURCE": "resource",
        "AUDIT_STARTUP_TIMING": "startup",
    }
    for line in text.splitlines():
        if JSON_MARKER in line:
            payload = line.split(JSON_MARKER, 1)[1]
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            record = {**base_context, **record}
            record["record_type"] = "nonlinear"
            records.append(record)
            continue
        for prefix, kind in prefixes.items():
            if prefix in line:
                record = {key: convert(value) for key, value in KEY_VALUE.findall(line)}
                record = {**base_context, **record}
                record["record_type"] = kind
                records.append(record)
                break
    return records


class Runner:
    def __init__(self, out_dir: Path, dry_run: bool = False):
        self.out_dir = out_dir.resolve()
        self.logs = self.out_dir / "logs"
        self.raw = self.out_dir / "raw"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run

    def append(self, phase: str, records: Iterable[dict[str, object]]) -> None:
        path = self.raw / f"{phase}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            for record in records:
                if "case_id" not in record:
                    identity = {key: record.get(key) for key in
                                ("record_type", "mode", "flavor", "mask", "q", "kv", "head_dim",
                                 "seed", "session", "iteration", "kind", "process", "distribution")
                                if key in record}
                    record["case_id"] = stable_case_id(identity)
                record.setdefault("attempt", 1)
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    def command(self, argv: list[str], log: Path, context: dict[str, object], timeout: int = 180,
                expected_failure: bool = False) -> str:
        printable = shlex.join(argv)
        if self.dry_run:
            print(printable)
            return ""
        started = time.monotonic_ns()
        proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        elapsed_us = (time.monotonic_ns() - started) // 1000
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout, encoding="utf-8")
        context["controller_elapsed_us"] = elapsed_us
        context["command"] = printable
        context["returncode"] = proc.returncode
        if proc.returncode and not expected_failure:
            raise RuntimeError(f"command failed ({proc.returncode}); see {log}")
        if not proc.returncode and expected_failure:
            raise RuntimeError(f"command unexpectedly succeeded; see {log}")
        return proc.stdout

    def remote(self, flavor: str) -> str:
        return f"/data/local/tmp/stage2_9_value_audit_{flavor}"

    def deploy(self, flavor: str) -> None:
        remote = self.remote(flavor)
        log = self.logs / f"deploy_{flavor}.log"
        self.command(
            [str(PROJECT / "scripts/deploy.sh"), "--flavor", flavor, "--remote-dir", remote],
            log,
            {"phase": "deploy", "flavor": flavor},
            timeout=180,
        )

    def htp(self, flavor: str, args: list[str], log: Path, context: dict[str, object], timeout: int = 180,
            expected_failure: bool = False) -> str:
        remote = self.remote(flavor)
        command = [str(PROJECT / "scripts/device_exec.sh"), "--remote-dir", remote, *args]
        return self.command(command, log, context, timeout=timeout, expected_failure=expected_failure)

    def run_nonlinear(self, sessions: int | None = None) -> None:
        phase = "nonlinear"
        cfg = SPEC[phase]
        sessions = sessions or cfg["sessions"]
        self.deploy("fair_combined")
        for dist_index, distribution in enumerate(cfg["distributions"]):
            for session in range(sessions):
                modes = list(cfg["modes"])
                if (dist_index + session) % 2:
                    modes.reverse()
                for order, mode in enumerate(modes):
                    args = ["--value-audit", "--mode", mode, "--distribution", distribution,
                            "--warmup", str(cfg["warmup"]), "--iters", str(cfg["measure"])]
                    if distribution != "exhaustive-fp16":
                        args += ["--elements", str(cfg["default_elements"])]
                    context = {"phase": phase, "flavor": "fair_combined", "session": session,
                               "order": order, "mode": mode, "distribution": distribution}
                    log = self.logs / phase / f"{distribution}_s{session}_{order}_{mode}.log"
                    text = self.htp("fair_combined", args, log, context)
                    self.append(phase, parsed_records(text, context))

    @staticmethod
    def attention_args(mode: str, mask: str, q: int, kv: int, dim: int, seed: int,
                       warmup: int, iters: int, workers: str = "auto", compare: bool = False) -> list[str]:
        args = ["--figure8-attn", "--mode", mode, "--scna-variant", "pair_static_d8",
                "--workers", workers, "--q-task-rows", "auto", "--mask-mode", mask,
                "--qo-len", str(q), "--kv-len", str(kv), "--n-heads", "12", "--n-kv-heads", "2",
                "--head-dim", str(dim), "--seed", str(seed), "--warmup", str(warmup),
                "--iters", str(iters), "--no-events"]
        if compare:
            args.append("--compare-reference")
        return args

    def run_correctness(self) -> None:
        phase = "attention_correctness"
        cfg = SPEC[phase]
        self.deploy("fair_combined")
        cases = list(itertools.product(cfg["masks"], cfg["q"], cfg["kv"], cfg["head_dim"], cfg["seeds"]))
        sentinel = cfg["long_context_sentinels"]
        cases += list(itertools.product(sentinel["masks"], sentinel["q"], [sentinel["kv"]],
                                        [sentinel["head_dim"]], [sentinel["seed"]]))
        for case_index, (mask, q, kv, dim, seed) in enumerate(cases):
            modes = list(cfg["modes"])
            if case_index % 2:
                modes.reverse()
            for order, mode in enumerate(modes):
                context = {"phase": phase, "flavor": "fair_combined", "case": case_index, "order": order,
                           "mode": mode, "mask": mask, "q": q, "kv": kv, "head_dim": dim, "seed": seed}
                log = self.logs / phase / f"c{case_index:03d}_{mode}.log"
                text = self.htp("fair_combined", self.attention_args(mode, mask, q, kv, dim, seed, 1, 1,
                                                                      compare=True) + ["--numeric-debug"],
                                log, context, timeout=600)
                self.append(phase, parsed_records(text, context))

    def performance_cases(self) -> list[tuple[str, int, int]]:
        cfg = SPEC["attention_performance"]
        cases = [(cfg["mask"], q, kv) for q, kv in itertools.product(cfg["q"], cfg["kv"])]
        anchor = cfg["full_mask_anchor"]
        cases.append(("full", anchor["q"], anchor["kv"]))
        return cases

    def run_performance(self, sessions: int | None = None) -> None:
        phase = "attention_performance"
        cfg = SPEC[phase]
        sessions = sessions or cfg["sessions"]
        self.deploy("fair_combined")
        for case_index, (mask, q, kv) in enumerate(self.performance_cases()):
            for session in range(sessions):
                modes = list(cfg["modes"])
                if (case_index + session) % 2:
                    modes.reverse()
                for order, mode in enumerate(modes):
                    context = {"phase": phase, "flavor": "fair_combined", "case": case_index,
                               "session": session, "order": order, "mode": mode, "mask": mask,
                               "q": q, "kv": kv, "head_dim": 128, "seed": 12029}
                    log = self.logs / phase / f"c{case_index:02d}_s{session}_{order}_{mode}.log"
                    text = self.htp("fair_combined", self.attention_args(
                        mode, mask, q, kv, 128, 12029, cfg["warmup"], cfg["measure"]), log, context, timeout=600)
                    self.append(phase, parsed_records(text, context))

    def run_resource(self, processes: int | None = None) -> None:
        phase = "resource"
        cfg = SPEC[phase]
        processes = processes or cfg["fresh_processes"]
        pairs = [("lut_only", "lut-exp"), ("scna_only", "scna-fp16")]
        for flavor, _ in pairs:
            self.deploy(flavor)
        self.deploy("fair_combined")
        for process_index in range(processes):
            ordered = pairs if process_index % 2 == 0 else list(reversed(pairs))
            for order, (flavor, mode) in enumerate(ordered):
                context = {"phase": phase, "flavor": flavor, "mode": mode,
                           "process": process_index, "order": order, "kind": "fresh_process"}
                log = self.logs / phase / f"fresh_{process_index:02d}_{order}_{flavor}.log"
                text = self.htp(flavor, self.attention_args(
                    mode, "full", 32, 4096, 128, 12029, 1, 20), log, context, timeout=600)
                self.append(phase, parsed_records(text, context))
        for dedicated_flavor, mode in pairs:
            for flavor, kind in [("fair_combined", "combined_equivalence"),
                                 (dedicated_flavor, "dedicated_equivalence")]:
                for session in range(5):
                    context = {"phase": phase, "flavor": flavor, "mode": mode, "kind": kind,
                               "session": session, "q": 32, "kv": 4096}
                    log = self.logs / phase / f"equiv_{mode}_{flavor}_s{session}.log"
                    text = self.htp(flavor, self.attention_args(
                        mode, "full", 32, 4096, 128, 12029, 5, 20), log, context, timeout=600)
                    self.append(phase, parsed_records(text, context))
        stress = cfg["stress"]
        for flavor, mode in pairs:
            for worker in stress["workers"]:
                for session in range(5):
                    context = {"phase": phase, "flavor": flavor, "mode": mode, "kind": "worker_stress",
                               "worker": worker, "session": session, "q": stress["q"], "kv": stress["kv"]}
                    log = self.logs / phase / f"stress_{flavor}_w{worker}_s{session}.log"
                    text = self.htp(flavor, self.attention_args(
                        mode, "full", stress["q"], stress["kv"], 128, 12029, 5, 20,
                        workers=str(worker)), log, context, timeout=600)
                    self.append(phase, parsed_records(text, context))
        for mode in ("lut-exp", "scna-fp16"):
            for worker in stress["workers"]:
                for session in range(5):
                    context = {"phase": phase, "flavor": "fair_combined", "mode": mode,
                               "kind": "worker_stress_combined", "worker": worker, "session": session,
                               "q": stress["q"], "kv": stress["kv"]}
                    log = self.logs / phase / f"stress_fair_{mode}_w{worker}_s{session}.log"
                    text = self.htp("fair_combined", self.attention_args(
                        mode, "full", stress["q"], stress["kv"], 128, 12029, 5, 20,
                        workers=str(worker)), log, context, timeout=600)
                    self.append(phase, parsed_records(text, context))

    def run_smoke(self) -> None:
        self.deploy("fair_combined")
        for mode in ("lut-exp", "scna-fp16"):
            context = {"phase": "smoke", "flavor": "fair_combined", "mode": mode}
            log = self.logs / "smoke" / f"value_{mode}.log"
            text = self.htp("fair_combined", ["--value-audit", "--mode", mode, "--distribution", "dense",
                                               "--elements", "4096", "--warmup", "1", "--iters", "1"],
                            log, context)
            self.append("smoke", parsed_records(text, context))
            log = self.logs / "smoke" / f"attention_{mode}.log"
            text = self.htp("fair_combined", self.attention_args(
                mode, "full", 1, 512, 128, 12029, 1, 1, compare=True), log, context)
            self.append("smoke", parsed_records(text, context))
        for flavor, supported, rejected in (("lut_only", "lut-exp", "scna-fp16"),
                                             ("scna_only", "scna-fp16", "lut-exp")):
            self.deploy(flavor)
            base_args = ["--value-audit", "--distribution", "boundary", "--elements", "4096",
                         "--warmup", "1", "--iters", "1"]
            context = {"phase": "variant_validation", "flavor": flavor, "mode": supported,
                       "expectation": "supported"}
            log = self.logs / "variant_validation" / f"{flavor}_{supported}_supported.log"
            text = self.htp(flavor, ["--value-audit", "--mode", supported, *base_args[1:]], log, context)
            self.append("variant_validation", parsed_records(text, context))
            context = {"phase": "variant_validation", "flavor": flavor, "mode": rejected,
                       "expectation": "rejected"}
            log = self.logs / "variant_validation" / f"{flavor}_{rejected}_rejected.log"
            self.htp(flavor, ["--value-audit", "--mode", rejected, *base_args[1:]], log, context,
                     expected_failure=True)
            self.append("variant_validation", [{**context, "record_type": "mode_rejection",
                                                   "rejected": True}])


def main() -> int:
    global SPEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "nonlinear", "correctness", "performance", "resource", "all"],
                        required=True)
    parser.add_argument("--spec", type=Path, default=PROJECT / "experiment_spec.json")
    parser.add_argument("--out-dir", "--results-dir", dest="out_dir", type=Path,
                        default=PROJECT / "results/formal")
    parser.add_argument("--sessions", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    SPEC = json.loads(args.spec.resolve().read_text())
    runner = Runner(args.out_dir, args.dry_run)
    phases = [args.phase] if args.phase != "all" else ["nonlinear", "correctness", "performance", "resource"]
    try:
        for phase in phases:
            if phase == "smoke": runner.run_smoke()
            elif phase == "nonlinear": runner.run_nonlinear(args.sessions)
            elif phase == "correctness": runner.run_correctness()
            elif phase == "performance": runner.run_performance(args.sessions)
            elif phase == "resource": runner.run_resource(args.processes)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
