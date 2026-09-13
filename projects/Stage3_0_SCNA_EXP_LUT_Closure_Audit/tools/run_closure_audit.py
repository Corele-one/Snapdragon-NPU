#!/usr/bin/env python3
"""Run the staged Stage 3.0 operator audit; ranking phases require Gate 0 PASS."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import shlex
import struct
import subprocess
import time
from pathlib import Path

from model_lineage import stable_case_id


PROJECT = Path(__file__).resolve().parents[1]
SPEC = json.loads((PROJECT / "experiment_spec.json").read_text())
KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value: str):
    if value.startswith("0x"):
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def records(text: str, context: dict[str, object]) -> list[dict[str, object]]:
    prefixes = {
        "FIG8_ATTENTION_HOST_TIMING": "attention_timing",
        "FIG8_ATTENTION_COMPARE ": "attention_correctness",
        "FIG8_ATTENTION_CHECKSUM": "attention_checksum",
        "FIG8_ATTENTION_WORKERS": "attention_workers",
        "FIG8_ATTENTION_TIMERS": "attention_timers",
        "FIG8_NUMERIC_BLOCK": "attention_numeric_block",
        "FIG8_NUMERIC ": "attention_numeric",
        "FIG8_AUDIT_RESOURCE": "resource",
        "AUDIT_STARTUP_TIMING": "startup",
    }
    output = []
    for line in text.splitlines():
        if "VALUE_AUDIT_JSON " in line:
            record = json.loads(line.split("VALUE_AUDIT_JSON ", 1)[1])
            output.append({**context, **record, "record_type": "nonlinear"})
            continue
        for prefix, kind in prefixes.items():
            if prefix in line:
                output.append({**context, **{k: scalar(v) for k, v in KV.findall(line)}, "record_type": kind})
                break
    return output


class Runner:
    def __init__(self, output: Path, gate0_verdict: Path | None = None):
        self.output = output.resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise RuntimeError(f"refusing to mix evidence in non-empty directory: {self.output}")
        self.raw = self.output / "raw"
        self.logs = self.output / "logs"
        self.raw.mkdir(parents=True)
        self.logs.mkdir(parents=True)
        self.gate0_verdict = gate0_verdict.resolve() if gate0_verdict else None
        manifest_path = PROJECT / "manifests/build_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("build provenance is missing; run scripts/build_closure_artifacts.sh first")
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text())
        self.build_manifest_sha256 = sha256(manifest_path)
        self.device: dict[str, object] = {}

    def capture_device(self) -> None:
        if self.device:
            return
        def adb_value(*args: str) -> str:
            proc = subprocess.run(["adb", *args], text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=30)
            if proc.returncode:
                raise RuntimeError(f"failed to capture device provenance: {proc.stdout.strip()}")
            return proc.stdout.strip()
        self.device = {
            "serial": adb_value("get-serialno"),
            "soc_model": adb_value("shell", "getprop", "ro.soc.model"),
            "product": adb_value("shell", "getprop", "ro.product.device"),
            "build_fingerprint": adb_value("shell", "getprop", "ro.build.fingerprint"),
            "dsp_architecture": SPEC["target"]["architecture"],
            "sdk": SPEC["target"]["sdk"],
        }
        (self.output / "device.json").write_text(json.dumps(self.device, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def remote(artifact: str) -> str:
        return f"/data/local/tmp/stage3_0_closure_{artifact}"

    def command(self, argv: list[str], log: Path, timeout: int = 600) -> str:
        started = time.monotonic_ns()
        proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(f"command failed ({proc.returncode}): {shlex.join(argv)}; see {log}")
        return proc.stdout + f"\nCONTROLLER_ELAPSED_US={(time.monotonic_ns() - started) // 1000}\n"

    def append(self, phase: str, values: list[dict[str, object]]) -> None:
        with (self.raw / f"{phase}.jsonl").open("a") as stream:
            for value in values:
                artifact = str(value.get("artifact", ""))
                artifact_record = self.manifest.get("artifacts", {}).get(artifact, {})
                if artifact in self.manifest.get("system_sources", {}):
                    source_record = self.manifest["system_sources"][artifact]
                else:
                    source_record = self.manifest.get("source", {})
                value.setdefault("experiment_spec_sha256", self.manifest["experiment_spec"]["sha256"])
                value.setdefault("build_manifest_sha256", self.build_manifest_sha256)
                value.setdefault("artifact_sha256", artifact_record.get("sha256"))
                value.setdefault("source_tree_sha256", source_record.get("tree_sha256"))
                value.setdefault("device_serial", self.device.get("serial"))
                value.setdefault("device_soc", self.device.get("soc_model"))
                value.setdefault("session", 0)
                value.setdefault("iteration", 0)
                identity = {key: value.get(key) for key in (
                    "record_type", "track", "artifact", "mode", "mask", "q", "kv", "head_dim",
                    "seed", "session", "iteration", "kind", "process", "distribution") if key in value}
                value.setdefault("case_id", stable_case_id(identity))
                value.setdefault("attempt", 1)
                stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")

    def deploy(self, artifact: str) -> None:
        self.capture_device()
        self.command([str(PROJECT / "scripts/deploy.sh"), "--flavor", artifact,
                      "--remote-dir", self.remote(artifact)], self.logs / f"deploy_{artifact}.log", timeout=240)

    @staticmethod
    def attn_args(mode: str, mask: str, q: int, kv: int, dim: int, seed: int,
                  warmup: int, measure: int, workers: str = "auto", numeric: bool = False) -> list[str]:
        args = ["--figure8-attn", "--mode", mode, "--scna-variant", "pair_static_d8",
                "--workers", workers, "--q-task-rows", "auto", "--mask-mode", mask,
                "--qo-len", str(q), "--kv-len", str(kv), "--n-heads", "12", "--n-kv-heads", "2",
                "--head-dim", str(dim), "--seed", str(seed), "--warmup", str(warmup),
                "--iters", str(measure), "--no-events", "--compare-reference"]
        if numeric:
            args.append("--numeric-debug")
        return args

    def run_attn(self, phase: str, artifact: str, context: dict[str, object], args: list[str], timeout: int = 600) -> None:
        name = "_".join(str(context[key]) for key in ("mode", "mask", "q", "kv", "head_dim", "seed") if key in context)
        if "session" in context:
            name += f"_s{context['session']}"
        text = self.command([str(PROJECT / "scripts/device_exec.sh"), "--remote-dir", self.remote(artifact), *args],
                            self.logs / phase / f"{name}.log", timeout=timeout)
        self.append(phase, records(text, context))

    def run_attn_batches(self, phase: str, artifact: str,
                         cases: list[tuple[dict[str, object], list[str]]], batch_size: int = 24) -> None:
        """Amortize ADB startup while retaining one htp_ops_test process per case."""
        for batch_start in range(0, len(cases), batch_size):
            batch = cases[batch_start:batch_start + batch_size]
            commands = [f"cd {shlex.quote(self.remote(artifact))}",
                        "export LD_LIBRARY_PATH=.", "export DSP_LIBRARY_PATH='./cdsp;./dsp;.'"]
            contexts: dict[int, dict[str, object]] = {}
            for offset, (context, args) in enumerate(batch):
                case_index = batch_start + offset
                contexts[case_index] = context
                # DSP worker logging can finish after the host process returns.
                # Leading/trailing newlines keep the next marker parseable even
                # when that late output did not terminate its final line.
                commands.append(f"printf '\\nSTAGE30_CASE_BEGIN case_index={case_index}\\n'")
                commands.append("./htp_ops_test " + " ".join(shlex.quote(value) for value in args))
            text = self.command(["adb", "shell", "; ".join(commands)],
                                self.logs / phase / f"batch_{batch_start // batch_size:04d}.log", timeout=3600)
            parts = re.split(r"STAGE30_CASE_BEGIN case_index=(\d+)\r?\n", text)
            observed = set()
            for index in range(1, len(parts), 2):
                case_index = int(parts[index])
                observed.add(case_index)
                self.append(phase, records(parts[index + 1], contexts[case_index]))
            missing = set(contexts) - observed
            if missing:
                raise RuntimeError(f"batch output is missing case markers: {sorted(missing)}")

    def gate0_preflight(self, artifact: str) -> bool:
        """Freeze the Stage 2.9 reproducer and fail before the 2160-case matrix."""
        frozen = {"mask": "causal", "q": 32, "kv": 32, "head_dim": 128, "seed": 29101}
        for mode in SPEC["gate0"]["modes"]:
            context = {"track": "gate0", "artifact": artifact, "mode": mode, **frozen,
                       "kind": "frozen-reproducer"}
            self.run_attn("gate0", artifact, context,
                          self.attn_args(mode, frozen["mask"], frozen["q"], frozen["kv"],
                                         frozen["head_dim"], frozen["seed"], 1, 1, numeric=True))
        values = [json.loads(line) for line in (self.raw / "gate0.jsonl").read_text().splitlines()]
        comparisons = [value for value in values if value.get("record_type") == "attention_correctness"
                       and value.get("kind") == "frozen-reproducer"]
        gates = SPEC["gate0"]["gates"]
        numeric = [value for value in values if value.get("record_type") == "attention_numeric"
                   and value.get("kind") == "frozen-reproducer"]
        numeric_blocks = [value for value in values if value.get("record_type") == "attention_numeric_block"
                          and value.get("kind") == "frozen-reproducer"
                          and int(str(value.get("p_scalar_sum_bits", "0x0")), 16) != 0]
        rowsum_ulps = []
        for value in numeric_blocks:
            scalar_sum = struct.unpack("<f", struct.pack("<I", int(str(value["p_scalar_sum_bits"]), 16)))[0]
            expected = struct.unpack("<H", struct.pack("<e", scalar_sum))[0]
            rowsum_ulps.append(abs(int(str(value["rowsum0_bits"]), 16) - expected))
        invariant_failure = (not numeric or not rowsum_ulps
                             or max(rowsum_ulps) > gates["rowsum_fp16_ulp_max"]
                             or any(int(value.get("masked_p_nonzero", 1)) != 0
                                    or int(value.get("tail_p_nonzero", 1)) != 0 for value in numeric))
        failures = [value for value in comparisons
                    if float(value.get("rmse", float("inf"))) > gates["rmse_max"]
                    or float(value.get("max_abs_error", float("inf"))) > gates["max_abs_max"]
                    or int(value.get("candidate_nonfinite", 1)) > gates["nonfinite_max"]
                    or int(value.get("reference_nonfinite", 1)) > gates["nonfinite_max"]]
        failed = bool(failures or len(comparisons) != len(SPEC["gate0"]["modes"]) or invariant_failure)
        verdict = {
            "schema_version": 1,
            "verdict": "AUDIT_INVALID" if failed else "PASS",
            "stage": "gate0-frozen-reproducer",
            "fail_fast": failed,
            "matrix_executed": False,
            "matrix_expected_cases": SPEC["gate0"]["expected_cases"],
            "reason": ("frozen reproducer exceeds a preregistered correctness threshold"
                       if failures else ("frozen reproducer invariant failure" if invariant_failure
                                         else "frozen reproducer passed; proceed to full Gate 0")),
            "max_rowsum_ulp": max(rowsum_ulps) if rowsum_ulps else None,
            "mask_tail_nonzero_records": sum(int(value.get("masked_p_nonzero", 0)) != 0
                                             or int(value.get("tail_p_nonzero", 0)) != 0
                                             for value in numeric),
            "comparisons": [{key: value.get(key) for key in
                             ("mode", "mask", "q", "kv", "head_dim", "seed", "rmse",
                              "relative_l2", "max_abs_error", "candidate_nonfinite", "pass")}
                            for value in comparisons],
        }
        verdict_path = self.output / "gate0_preflight_verdict.json"
        verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        lines = "\n".join(
            f"| {v.get('mode')} | {float(v.get('rmse', float('nan'))):.9g} | "
            f"{float(v.get('max_abs_error', float('nan'))):.9g} | {v.get('pass')} |"
            for v in comparisons)
        (self.output / "gate0_preflight_verdict.md").write_text(
            "# Gate 0 冻结复现点\n\n"
            f"裁决：`{verdict['verdict']}`\n\n"
            "| mode | RMSE | max-abs | kernel pass |\n|---|---:|---:|---:|\n"
            f"{lines}\n\n"
            "任一模式超过 RMSE 0.002 或 max-abs 0.01 即按预注册规则停止完整矩阵及后续轨。\n")
        return verdict["verdict"] == "PASS"

    def gate0(self) -> bool:
        cfg = SPEC["gate0"]
        artifact = cfg["artifact"]
        self.deploy(artifact)
        if not self.gate0_preflight(artifact):
            return False
        jobs: list[tuple[dict[str, object], list[str]]] = []
        cases = itertools.product(cfg["masks"], cfg["q"], cfg["kv"], cfg["head_dim"], cfg["seeds"])
        for index, (mask, q, kv, dim, seed) in enumerate(cases):
            modes = list(cfg["modes"])
            if index % 2:
                modes.reverse()
            for order, mode in enumerate(modes):
                context = {"track": "gate0", "artifact": artifact, "mode": mode, "mask": mask,
                           "q": q, "kv": kv, "head_dim": dim, "seed": seed, "order": order, "kind": "matrix"}
                jobs.append((context, self.attn_args(mode, mask, q, kv, dim, seed, 1, 1, numeric=True)))
        sentinel = cfg["long_context_sentinels"]
        for mask, q, mode in itertools.product(sentinel["masks"], sentinel["q"], cfg["modes"]):
            context = {"track": "gate0", "artifact": artifact, "mode": mode, "mask": mask, "q": q,
                       "kv": sentinel["kv"], "head_dim": sentinel["head_dim"], "seed": sentinel["seed"],
                       "kind": "long-sentinel"}
            jobs.append((context, self.attn_args(mode, mask, q, sentinel["kv"], sentinel["head_dim"],
                                                 sentinel["seed"], 1, 1, numeric=True)))
        for case in cfg["determinism"]["cases"]:
            for mode in cfg["modes"]:
                for process in range(cfg["determinism"]["processes"]):
                    context = {"track": "gate0", "artifact": artifact, "mode": mode, **case,
                               "kind": "determinism", "process": process}
                    jobs.append((context, self.attn_args(mode, case["mask"], case["q"], case["kv"],
                                                         case["head_dim"], case["seed"], 0, 1)))
        self.run_attn_batches("gate0", artifact, jobs)
        return True

    def nonlinear(self) -> None:
        cfg = SPEC["nonlinear"]
        self.deploy("fair_combined")
        for dist_index, distribution in enumerate(cfg["distributions"]):
            for session in range(cfg["sessions"]):
                modes = list(cfg["modes"])
                if (dist_index + session) % 2:
                    modes.reverse()
                for order, mode in enumerate(modes):
                    args = ["--value-audit", "--mode", mode, "--distribution", distribution,
                            "--warmup", str(cfg["warmup"]), "--iters", str(cfg["measure"])]
                    if distribution != "exhaustive-fp16":
                        args += ["--elements", str(cfg["default_elements"])]
                    context = {"track": "nonlinear", "artifact": "fair_combined", "mode": mode,
                               "distribution": distribution, "session": session, "order": order}
                    text = self.command([str(PROJECT / "scripts/device_exec.sh"), "--remote-dir",
                                         self.remote("fair_combined"), *args],
                                        self.logs / "nonlinear" / f"{distribution}_s{session}_{mode}.log")
                    self.append("nonlinear", records(text, context))

    def performance(self, track: str) -> None:
        self.require_gate0()
        cfg = SPEC["performance"]
        candidates = cfg["tracks"][track]
        for artifact in sorted({value["artifact"] for value in candidates}):
            self.deploy(artifact)
        cases = [(cfg["mask"], q, kv) for q, kv in itertools.product(cfg["q"], cfg["kv"])]
        cases.append(("full", cfg["full_mask_anchor"]["q"], cfg["full_mask_anchor"]["kv"]))
        for index, (mask, q, kv) in enumerate(cases):
            for session in range(cfg["sessions"]):
                order = list(candidates)
                if (index + session) % 2:
                    order.reverse()
                for order_index, candidate in enumerate(order):
                    context = {"track": track, "artifact": candidate["artifact"], "mode": candidate["candidate"],
                               "mask": mask, "q": q, "kv": kv, "head_dim": 128, "seed": 12029,
                               "session": session, "order": order_index, "kind": "ranking"}
                    self.run_attn(track, candidate["artifact"], context,
                                  self.attn_args(candidate["mode"], mask, q, kv, 128, 12029,
                                                 cfg["warmup"], cfg["measure"]), timeout=1200)

    def require_gate0(self) -> None:
        verdict = self.gate0_verdict
        if verdict is None:
            raise RuntimeError("--gate0-verdict is required before ranking phases")
        if not verdict.is_file() or json.loads(verdict.read_text()).get("verdict") != "PASS":
            raise RuntimeError("Gate 0 PASS is required before ranking phases")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["gate0", "nonlinear", "track-a", "track-b", "model", "resource", "thermal", "all"])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--gate0-verdict", type=Path)
    args = parser.parse_args()
    runner = Runner(args.results_dir, args.gate0_verdict)
    if args.phase == "gate0":
        return 0 if runner.gate0() else 3
    elif args.phase == "nonlinear":
        runner.require_gate0(); runner.nonlinear()
    elif args.phase in {"track-a", "track-b"}:
        runner.performance(args.phase)
    elif args.phase == "model":
        runner.require_gate0()
        subprocess.run(["python3", str(PROJECT / "tools/run_model_audit.py"),
                        "--spec", str(PROJECT / "experiment_spec.json"),
                        "--model-manifest", str(PROJECT / SPEC["model"]["import_manifest"]),
                        "--out-dir", str(args.results_dir)], check=True)
    elif args.phase == "resource":
        runner.require_gate0()
        subprocess.run(["python3", str(PROJECT / "tools/run_device_audit.py"),
                        "--spec", str(PROJECT / "experiment_spec.json"), "--phase", "resource",
                        "--out-dir", str(args.results_dir)], check=True)
    elif args.phase == "thermal":
        runner.require_gate0()
        subprocess.run(["python3", str(PROJECT / "tools/run_thermal_audit.py"),
                        "--spec", str(PROJECT / "experiment_spec.json"),
                        "--out-dir", str(args.results_dir)], check=True)
    elif args.phase == "all":
        raise RuntimeError("use scripts/run_closure_audit.sh for the ordered all-phase audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
