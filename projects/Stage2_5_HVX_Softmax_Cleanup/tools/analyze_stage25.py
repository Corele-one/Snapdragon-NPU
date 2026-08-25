#!/usr/bin/env python3
"""Analyze Stage 2.5 local cleanup experiments without inventing missing data."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_baseline import bootstrap_median, fields, paired_ratio


QOS = (1, 4, 8, 16, 32)


def parse_attention(run_dir: Path) -> tuple[dict, dict]:
    raw: dict[str, dict[int, dict[tuple[int, int], float]]] = defaultdict(
        lambda: {q: {} for q in QOS})
    for path in sorted((run_dir / "raw/attention").glob("*.log")):
        match = re.match(r"(.+)_q(\d+)_s(\d+)\.log", path.name)
        if not match:
            continue
        label, q, session = match.group(1), int(match.group(2)), int(match.group(3))
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_HOST_TIMING" not in line:
                continue
            data = fields(line)
            if data.get("phase") == "measure" and data.get("ret") == "0":
                raw[label][q][(session, int(data["iteration"]))] = float(data["host_elapsed_us"])
    summary = {label: {q: bootstrap_median(list(by_q[q].values()), seed=0x2500 + q)
                       for q in QOS} for label, by_q in raw.items()}
    ratios = {}
    baseline = raw.get("stage2_final", {})
    for label, by_q in raw.items():
        if label == "stage2_final":
            continue
        ratios[label] = {q: paired_ratio(by_q[q], baseline.get(q, {}), seed=0x2600 + q) for q in QOS}
    return summary, ratios


def parse_correctness(run_dir: Path) -> dict:
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((run_dir / "raw/accuracy").glob("*.log")):
        label = path.name.split("_full_q", 1)[0] if "_full_q" in path.name else None
        if label is None:
            for marker in ("_causal_q", "_padding_q"):
                if marker in path.name:
                    label = path.name.split(marker, 1)[0]
                    break
        if label is None:
            continue
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_COMPARE " in line:
                rows[label].append(fields(line))
    result = {}
    for label, values in rows.items():
        result[label] = {
            "cases": len(values),
            "pass_cases": sum(row.get("pass") == "1" for row in values),
            "max_rmse": max((float(row["rmse"]) for row in values), default=None),
            "max_abs": max((float(row["max_abs_error"]) for row in values), default=None),
            "nonfinite": sum(int(row.get("candidate_nonfinite", "0")) for row in values),
            "pass": bool(values) and all(row.get("pass") == "1" for row in values),
        }
    return result


def parse_diagnostics(run_dir: Path) -> dict:
    result = {}
    for path in sorted((run_dir / "raw/diagnostic").glob("*_q32_kv4096.log")):
        label = path.name.removesuffix("_q32_kv4096.log")
        totals = defaultdict(float)
        host = []
        for line in path.read_text(errors="replace").splitlines():
            data = fields(line)
            if "FIG8_ATTENTION_TIMERS" in line and data.get("phase") == "measure":
                for key in ("q_load", "k_load", "v_load", "qk_dot", "safe_sm", "core_acc",
                            "o_scale", "o_store", "scna_exp", "state_update", "rowmax_mem",
                            "rowsum_reduce", "profiled_total"):
                    totals[key] += float(data.get(key, 0))
            elif "FIG8_ATTENTION_HOST_TIMING" in line and data.get("phase") == "measure":
                host.append(float(data["host_elapsed_us"]))
        result[label] = {"host_us": statistics.median(host) if host else None, **totals}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    attention, ratios = parse_attention(args.run_dir)
    payload = {
        "schema_version": 1,
        "run_id": args.run_dir.name,
        "attention": attention,
        "ratios_vs_stage2_final": ratios,
        "correctness": parse_correctness(args.run_dir),
        "diagnostic": parse_diagnostics(args.run_dir),
        "recovery_attempts": sorted(path.name for path in (args.run_dir / "raw/recovery").glob("*.log")),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

