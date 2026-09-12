#!/usr/bin/env python3
"""Strict parser and report generator for the fixed three-way v79 benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


LABEL_MODE = {"baseline": "baseline", "lut_exp": "lut-exp", "scna": "scna-fp16"}
Q_VALUES = (4, 8, 16, 32)
COMPONENTS = (
    "profiled_total", "q_load", "k_load", "v_load", "qk_dot", "safe_sm",
    "scna_exp", "state_update", "param_prepare", "core_acc", "o_scale", "o_store",
)
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def fields(line: str) -> dict[str, str]:
    return dict(KV_RE.findall(line))


def percentile(sorted_values: list[float], probability: float) -> float:
    index = (len(sorted_values) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_median(values: list[float], seed: int, draws: int = 10000) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return {
        "count": n,
        "median": statistics.median(values),
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
    }


def parse_host_log(path: Path, expected_mode: str, q: int, session: int) -> list[dict]:
    config_modes = []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if "FIG8_ATTENTION_CONFIG " in line:
            config_modes.append(fields(line).get("mode"))
        if "FIG8_ATTENTION_HOST_TIMING " not in line or "phase=measure" not in line:
            continue
        data = fields(line)
        if data.get("mode") != expected_mode:
            raise ValueError(f"mode mismatch in {path}: {data.get('mode')} != {expected_mode}")
        if int(data.get("qo_len", -1)) != q or data.get("ret") != "0":
            raise ValueError(f"invalid q or nonzero return in {path}: {data}")
        rows.append({
            "session": session,
            "iteration": int(data["iteration"]),
            "q": q,
            "host_elapsed_us": float(data["host_elapsed_us"]),
            "ret": int(data["ret"]),
        })
    if config_modes != [expected_mode]:
        raise ValueError(f"missing or mismatched config in {path}: {config_modes}")
    if len(rows) != 20 or sorted(row["iteration"] for row in rows) != list(range(20)):
        raise ValueError(f"incomplete session in {path}: {len(rows)} measured rows")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_summary(root: Path) -> dict:
    result = {}
    for label, expected_mode in LABEL_MODE.items():
        path = root / "diagnostic" / f"{label}_q32.log"
        if not path.is_file():
            raise ValueError(f"missing diagnostic log {path}")
        grouped: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        workers = []
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_TIMERS " in line and "phase=measure" in line:
                data = fields(line)
                if data.get("mode") != expected_mode:
                    raise ValueError(f"diagnostic mode mismatch in {path}")
                iteration = int(data["iteration"])
                for component in COMPONENTS:
                    grouped[iteration][component] += float(data.get(component, 0))
            elif "FIG8_ATTENTION_WORKERS " in line and "phase=measure" in line:
                data = fields(line)
                workers.append({key: int(data[key]) for key in (
                    "active_workers", "hvx_contexts", "vtcm_worker_cap", "tasks", "q_task_rows"
                )})
        if sorted(grouped) != [0, 1, 2]:
            raise ValueError(f"incomplete diagnostic iterations in {path}")
        result[label] = {
            "units": "task work-sum microseconds; do not compare directly with host wall latency",
            "measured_iterations": len(grouped),
            "median_task_work_sum_us": {
                component: statistics.median(row[component] for row in grouped.values())
                for component in COMPONENTS
            },
            "worker_records": workers,
        }
    return result


def analyze(root: Path) -> dict:
    all_rows: dict[str, list[dict]] = {}
    indexed: dict[str, dict[tuple[int, int, int], float]] = {}
    mode_summary = {}
    for label, expected_mode in LABEL_MODE.items():
        expected = {f"q{q}_s{s}.log" for q in Q_VALUES for s in range(1, 6)}
        actual = {path.name for path in (root / label).glob("*.log")}
        if expected != actual:
            raise ValueError(f"{label} file set mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        rows = []
        for q in Q_VALUES:
            for session in range(1, 6):
                rows.extend(parse_host_log(root / label / f"q{q}_s{session}.log", expected_mode, q, session))
        if len(rows) != 400:
            raise ValueError(f"{label}: expected 400 valid Host rows, found {len(rows)}")
        all_rows[label] = rows
        indexed[label] = {(row["q"], row["session"], row["iteration"]): row["host_elapsed_us"] for row in rows}
        write_csv(root / label / "iterations.csv", rows)
        mode_summary[label] = {
            str(q): bootstrap_median(
                [row["host_elapsed_us"] for row in rows if row["q"] == q],
                seed=0x7900 + q + list(LABEL_MODE).index(label) * 100,
            ) for q in Q_VALUES
        }
        (root / label / "summary.json").write_text(json.dumps(mode_summary[label], indent=2) + "\n")

    ratios = {"lut_exp_over_baseline": {}, "scna_over_baseline": {}}
    for candidate in ("lut_exp", "scna"):
        for q in Q_VALUES:
            keys = sorted(key for key in indexed["baseline"] if key[0] == q)
            values = [indexed[candidate][key] / indexed["baseline"][key] for key in keys]
            ratios[f"{candidate}_over_baseline"][str(q)] = bootstrap_median(
                values, seed=0x7A00 + q + (100 if candidate == "scna" else 0)
            )

    diagnostics = diagnostic_summary(root)
    total_samples = sum(len(rows) for rows in all_rows.values())
    result = {
        "schema_version": 1,
        "valid_host_samples": total_samples,
        "all_ret_zero": True,
        "host_latency_us": mode_summary,
        "paired_ratios": ratios,
        "diagnostics": diagnostics,
        "comparison_scope": "original FlashAttention systems versus the frozen final SCNA system",
    }
    if total_samples != 1200:
        raise ValueError(f"expected 1200 valid measured Host samples, found {total_samples}")
    (root / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    csv_rows = []
    for q in Q_VALUES:
        row = {"q": q}
        for label in LABEL_MODE:
            value = mode_summary[label][str(q)]
            row[f"{label}_median_us"] = value["median"]
            row[f"{label}_ci95_low_us"] = value["ci95_low"]
            row[f"{label}_ci95_high_us"] = value["ci95_high"]
        for key in ratios:
            value = ratios[key][str(q)]
            row[f"{key}_median"] = value["median"]
            row[f"{key}_ci95_low"] = value["ci95_low"]
            row[f"{key}_ci95_high"] = value["ci95_high"]
        csv_rows.append(row)
    write_csv(root / "comparison.csv", csv_rows)

    lines = [
        "# v79 三方实测结果", "",
        "下表是 Host wall latency；baseline 与 LUT-exp 来自原版 `flashattention`，SCNA 来自最终优化系统。",
        "因此它是系统级比较，不是只替换 exp evaluator 的单变量实验。", "",
        "| Q | baseline (us) | LUT-exp (us) | SCNA (us) | LUT/base | SCNA/base |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for q in Q_VALUES:
        lines.append(
            f"| {q} | {mode_summary['baseline'][str(q)]['median']:.1f} | "
            f"{mode_summary['lut_exp'][str(q)]['median']:.1f} | {mode_summary['scna'][str(q)]['median']:.1f} | "
            f"{ratios['lut_exp_over_baseline'][str(q)]['median']:.4f} | "
            f"{ratios['scna_over_baseline'][str(q)]['median']:.4f} |"
        )
    lines += ["", "比值为同 session、同 iteration 配对后取中位数；JSON/CSV 中包含 10,000 次 bootstrap 95% CI。", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    print(json.dumps({"valid_host_samples": result["valid_host_samples"], "all_ret_zero": True}, indent=2))


if __name__ == "__main__":
    main()
