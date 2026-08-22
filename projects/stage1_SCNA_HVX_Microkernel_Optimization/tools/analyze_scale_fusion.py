#!/usr/bin/env python3
"""Analyze Experiment I scale-fusion probe logs."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from analyze_baseline import bootstrap_median, fields, percentile


METRICS = (
    "scale_only_ns_per_64",
    "separate_scale_lut_ns_per_64",
    "separate_scale_scna_ns_per_64",
    "fused_scale_scna_ns_per_64",
)


def bootstrap_ratio(top: list[float], bottom: list[float], seed: int) -> dict:
    rng = random.Random(seed)
    values = []
    for _ in range(10000):
        values.append(statistics.median(rng.choices(top, k=len(top))) /
                      statistics.median(rng.choices(bottom, k=len(bottom))))
    return {
        "median": statistics.median(top) / statistics.median(bottom),
        "ci_low": percentile(values, 0.025),
        "ci_high": percentile(values, 0.975),
        "n": min(len(top), len(bottom)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    raw = {metric: [] for metric in METRICS}
    correctness = []
    for path in sorted((args.run_dir / "raw").glob("sample*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if "SCNA_EXP_BENCH" not in line:
                continue
            data = fields(line)
            for metric in METRICS:
                raw[metric].append(float(data[metric]))
            correctness.append({
                "max_abs_diff": float(data["fused_scale_max_abs_diff"]),
                "mismatches": int(data["fused_scale_mismatches"]),
                "samples": int(data["fused_scale_samples"]),
                "checksum": data["scale_checksum"],
                "scale_head_dim": int(data["scale_head_dim"]),
                "scale_fp16_bits": data["scale_fp16_bits"],
            })
    summary = {
        "schema_version": 1,
        "run_id": args.run_dir.name,
        "metrics_ns_per_64": {
            metric: bootstrap_median(values, seed=0x1A00 + index)
            for index, (metric, values) in enumerate(raw.items())
        },
        "ratios": {
            "fused_scna_over_separate_scna": bootstrap_ratio(
                raw["fused_scale_scna_ns_per_64"], raw["separate_scale_scna_ns_per_64"], 0x1A10),
            "fused_scna_over_separate_lut": bootstrap_ratio(
                raw["fused_scale_scna_ns_per_64"], raw["separate_scale_lut_ns_per_64"], 0x1A11),
        },
        "correctness": {
            "samples_per_run": sorted({row["samples"] for row in correctness}),
            "max_abs_diff": max(row["max_abs_diff"] for row in correctness),
            "mismatches_per_run": sorted({row["mismatches"] for row in correctness}),
            "checksums": sorted({row["checksum"] for row in correctness}),
            "scale_head_dim": sorted({row["scale_head_dim"] for row in correctness}),
            "scale_fp16_bits": sorted({row["scale_fp16_bits"] for row in correctness}),
        },
    }
    output = args.run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
