#!/usr/bin/env python3
"""Summarize Figure8 attention logs into latency and SCNA Pareto CSV/Markdown."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


PAIR_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def fields(line: str) -> dict[str, str]:
    return dict(PAIR_RE.findall(line))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    args = parse_args()
    dsp_by_iteration: dict[tuple[str, str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    host_by_iteration: dict[tuple[str, str, str, str], float] = {}
    micro_by_config: dict[tuple[str, str], dict[str, float]] = {}
    for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        if "SCNA_EXP2_BENCH" in line:
            data = fields(line)
            try:
                micro_by_config[(data["mode"], data["width"])] = {
                    "micro_elapsed_us": float(data["elapsed_us"]),
                    "micro_pair_elapsed_us": float(data.get("pair_elapsed_us", "0")),
                    "micro_rmse": float(data["rmse"]),
                    "micro_max_abs_error": float(data["max_abs_error"]),
                    "micro_dense_samples": float(data.get("dense_samples", "0")),
                    "micro_dense_rmse": float(data.get("dense_rmse", "0")),
                    "micro_dense_max_abs_error": float(data.get("dense_max_abs_error", "0")),
                    "micro_pair_max_abs_diff": float(data.get("pair_max_abs_diff", "0")),
                    "micro_monotonic_violations": float(data.get("monotonic_violations", "0")),
                    "micro_negative_count": float(data.get("negative_count", "0")),
                    "micro_nan_count": float(data.get("nan_count", "0")),
                }
            except (KeyError, ValueError):
                pass
            continue
        if "phase=measure" not in line:
            continue
        data = fields(line)
        mode = data.get("mode", "unknown")
        qo_len = data.get("qo_len", "0")
        width = data.get("scna_width", "0")
        iteration = data.get("iteration", "0")
        if "FIG8_ATTENTION_HOST_TIMING" in line:
            try:
                host_by_iteration[(mode, qo_len, width, iteration)] = float(data["host_elapsed_us"])
            except (KeyError, ValueError):
                pass
            continue
        if "FIG8_ATTENTION_TIMERS" not in line:
            continue
        key = (mode, qo_len, width, data.get("nonlinear_mode", "0"), iteration)
        try:
            dsp_by_iteration[key].append({name: float(data.get(name, "0")) for name in
                                         ("profiled_total", "safe_sm", "scna_exp", "q_load", "k_load", "v_load", "qk_dot", "core_acc")})
        except ValueError:
            continue

    samples: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    for (mode, qo_len, width, precision, iteration), records in dsp_by_iteration.items():
        sample = {name: sum(record[name] for record in records) for name in records[0]}
        sample["host_elapsed"] = host_by_iteration.get((mode, qo_len, width, iteration), 0.0)
        samples[(mode, qo_len, width, precision)].append(sample)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for (mode, qo_len, width, precision), values in sorted(samples.items()):
        total = [sample["profiled_total"] for sample in values]
        row: dict[str, object] = {
            "mode": mode,
            "qo_len": int(qo_len),
            "scna_width": int(width),
            "nonlinear_mode": int(precision),
            "samples": len(values),
            "total_median_us": statistics.median(total),
            "total_p50_us": percentile(total, 0.50),
            "total_p95_us": percentile(total, 0.95),
            "host_median_us": statistics.median(sample["host_elapsed"] for sample in values),
            "host_p50_us": percentile([sample["host_elapsed"] for sample in values], 0.50),
            "host_p95_us": percentile([sample["host_elapsed"] for sample in values], 0.95),
            "micro_elapsed_us": 0.0,
            "micro_pair_elapsed_us": 0.0,
            "micro_rmse": 0.0,
            "micro_max_abs_error": 0.0,
            "micro_dense_samples": 0.0,
            "micro_dense_rmse": 0.0,
            "micro_dense_max_abs_error": 0.0,
            "micro_pair_max_abs_diff": 0.0,
            "micro_monotonic_violations": 0.0,
            "micro_negative_count": 0.0,
            "micro_nan_count": 0.0,
        }
        row.update(micro_by_config.get((mode, width), {}))
        for name in ("safe_sm", "scna_exp", "q_load", "k_load", "v_load", "qk_dot", "core_acc"):
            row[f"{name}_median_us"] = statistics.median(sample[name] for sample in values)
        rows.append(row)

    fieldnames = list(rows[0]) if rows else ["mode", "qo_len", "scna_width", "nonlinear_mode", "samples", "total_median_us"]
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    baseline = {int(row["qo_len"]): float(row["total_median_us"]) for row in rows if row["mode"] == "baseline"}
    pareto = []
    for row in rows:
        if not str(row["mode"]).startswith("scna-"):
            continue
        speedup = baseline.get(int(row["qo_len"]), 0.0) / float(row["total_median_us"]) if row["total_median_us"] else 0.0
        pareto.append({**row, "baseline_speedup": speedup})
    with (args.out_dir / "pareto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pareto[0]) if pareto else fieldnames + ["baseline_speedup"])
        writer.writeheader()
        writer.writerows(pareto)

    report = ["# V81 SCNA FlashAttention Summary", "", "| Mode | Qo | Width | Precision | Host median us | DSP median us | DSP P95 us | Safe softmax us | SCNA us | Single micro us | Pair micro us | Dense RMSE | Dense max abs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append("| {mode} | {qo_len} | {scna_width} | {nonlinear_mode} | {host_median_us:.2f} | {total_median_us:.2f} | {total_p95_us:.2f} | {safe_sm_median_us:.2f} | {scna_exp_median_us:.2f} | {micro_elapsed_us:.2f} | {micro_pair_elapsed_us:.2f} | {micro_dense_rmse:.6g} | {micro_dense_max_abs_error:.6g} |".format(**row))
    (args.out_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
