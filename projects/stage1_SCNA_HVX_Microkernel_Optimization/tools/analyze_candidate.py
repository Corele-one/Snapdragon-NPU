#!/usr/bin/env python3
"""Analyze a paired Stage 1 candidate run into an auditable JSON summary."""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_baseline import (
    bootstrap_median,
    fields,
    paired_ratio,
    parse_correctness,
    parse_disassembly,
    percentile,
    sha256,
)


QOS = (1, 4, 8, 16, 32)
PAIRRET = "d7_pairret_noinline"
LUT = "exp_lut"


def bootstrap_unpaired_ratio(numerator: list[float], denominator: list[float],
                             draws: int = 10000, seed: int = 0xD1) -> dict:
    if not numerator or not denominator:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = random.Random(seed)
    ratios = []
    for _ in range(draws):
        top = statistics.median(rng.choices(numerator, k=len(numerator)))
        bottom = statistics.median(rng.choices(denominator, k=len(denominator)))
        ratios.append(top / bottom)
    return {
        "median": statistics.median(numerator) / statistics.median(denominator),
        "ci_low": percentile(ratios, 0.025),
        "ci_high": percentile(ratios, 0.975),
        "n": min(len(numerator), len(denominator)),
    }


def micro_values(run_dir: Path, label: str) -> list[float]:
    values = []
    for path in sorted((run_dir / "raw/micro").glob(f"{label}_sample*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if "SCNA_EXP_BENCH" in line:
                values.append(float(fields(line)["paired_ns_per_64"]))
    return values


def lut_micro_values(run_dir: Path) -> list[float]:
    values = []
    for path in sorted((run_dir / "raw/lut_micro").glob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            columns = line.split(",")
            if len(columns) == 20 and columns[0] == "2" and columns[1] == "lut_exp" \
                    and columns[17] == "Gelem/s" and int(columns[18]) == 1:
                throughput = float(columns[16])
                if throughput > 0:
                    values.append(64.0 / throughput)
    return values


def parse_micro(run_dir: Path, candidate: str) -> dict:
    raw = {
        PAIRRET: micro_values(run_dir, PAIRRET),
        candidate: micro_values(run_dir, candidate),
        LUT: lut_micro_values(run_dir),
    }
    return {
        "summary": {label: bootstrap_median(values, seed=0xD100 + index)
                    for index, (label, values) in enumerate(raw.items())},
        "ratios": {
            "candidate_over_pairret": bootstrap_unpaired_ratio(raw[candidate], raw[PAIRRET], seed=0xD110),
            "candidate_over_exp_lut": bootstrap_unpaired_ratio(raw[candidate], raw[LUT], seed=0xD111),
        },
    }


def parse_attention(run_dir: Path, candidate: str) -> tuple[dict, dict]:
    labels = (PAIRRET, candidate, LUT)
    raw = {label: {q: {} for q in QOS} for label in labels}
    for path in sorted((run_dir / "raw/attention").glob("*.log")):
        match = re.match(r"(.+)_q(\d+)_s(\d+)\.log", path.name)
        if not match or match.group(1) not in labels:
            continue
        label, q, session = match.group(1), int(match.group(2)), int(match.group(3))
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_HOST_TIMING" not in line:
                continue
            data = fields(line)
            if data.get("phase") == "measure" and int(data.get("ret", "1")) == 0:
                raw[label][q][(session, int(data["iteration"]))] = float(data["host_elapsed_us"])

    summary = {label: {} for label in labels}
    ratios = {"candidate_over_pairret": {}, "candidate_over_exp_lut": {}}
    for label in labels:
        for q in QOS:
            summary[label][q] = bootstrap_median(list(raw[label][q].values()), seed=0xD200 + q)
    for q in QOS:
        ratios["candidate_over_pairret"][q] = paired_ratio(
            raw[candidate][q], raw[PAIRRET][q], seed=0xD300 + q)
        ratios["candidate_over_exp_lut"][q] = paired_ratio(
            raw[candidate][q], raw[LUT][q], seed=0xD400 + q)
    return summary, ratios


def parse_diagnostics(run_dir: Path, candidate: str) -> dict:
    result = {}
    for label in (PAIRRET, candidate, LUT):
        path = run_dir / "raw/diagnostic" / f"{label}_q32_kv4096.log"
        totals = defaultdict(float)
        host = []
        for line in path.read_text(errors="replace").splitlines():
            data = fields(line)
            if "FIG8_ATTENTION_TIMERS" in line and data.get("phase") == "measure":
                for key in ("scna_exp", "profiled_total"):
                    totals[key] += float(data.get(key, 0))
            elif "FIG8_ATTENTION_HOST_TIMING" in line and data.get("phase") == "measure":
                host.append(float(data["host_elapsed_us"]))
        result[label] = {"host_us": statistics.median(host) if host else None, **totals}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    project = args.project.resolve()
    candidate = args.candidate

    manifest_text = (run_dir / "evidence/device_manifest.txt").read_text(errors="replace")
    manifest = dict(line.split("=", 1) for line in manifest_text.splitlines() if "=" in line)
    attention, attention_ratios = parse_attention(run_dir, candidate)
    summary = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "candidate": candidate,
        "quick": manifest.get("quick") == "1",
        "source_commit": manifest.get("source_git_commit", "UNKNOWN"),
        "artifact_sha256": {
            PAIRRET: sha256(project / f"artifacts/variants/{PAIRRET}/libhtp_ops_skel.so"),
            candidate: sha256(project / f"artifacts/variants/{candidate}/libhtp_ops_skel.so"),
        },
        "static": {
            label: parse_disassembly(run_dir / f"static/{label}.v79.disasm.txt",
                                     "hvx_scna_exp2_pair_hot_return_vhf")
            for label in (PAIRRET, candidate)
        },
        "micro": parse_micro(run_dir, candidate),
        "correctness": parse_correctness(run_dir),
        "attention": attention,
        "attention_ratios": attention_ratios,
        "diagnostic": parse_diagnostics(run_dir, candidate),
        "recovery_attempts": sorted(path.name for path in (run_dir / "raw/recovery").glob("*.log")),
    }
    output = run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
