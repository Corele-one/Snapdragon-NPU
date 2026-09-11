#!/usr/bin/env python3
"""Aggregate Stage 2.75 scheduler logs without conflating work sums and wall time."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

METRICS = (
    "profiled_total",
    "q_load",
    "k_load",
    "v_load",
    "qk_dot",
    "safe_sm",
    "scna_exp",
    "state_update",
    "core_acc",
    "o_scale",
    "o_store",
)

MATRIX_RE = re.compile(
    r"(?P<label>.+)_q(?P<q>\d+)_kv(?P<kv>\d+)_r(?P<rows>auto|\d+)_w(?P<workers>auto|\d+)_s(?P<session>\d+)\.log$"
)
ORIGINAL_RE = re.compile(
    r"(?P<label>original_(?:baseline|lut-exp))_q(?P<q>\d+)_kv(?P<kv>\d+)_s(?P<session>\d+)\.log$"
)
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")


def kv(line: str) -> dict[str, str]:
    return dict(KV_RE.findall(line))


def median_ci(values: list[float], draws: int, seed: int) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "median": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    med = statistics.median(values)
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    n = len(values)
    boots: list[float] = []
    # Chunking bounds memory for the 96-group generalization matrix while
    # preserving the preregistered 10,000 bootstrap draws.
    for offset in range(0, draws, 1000):
        count = min(1000, draws - offset)
        indices = rng.integers(0, n, size=(count, n))
        boots.extend(np.median(array[indices], axis=1).tolist())
    boots.sort()
    lo = boots[int(0.025 * (draws - 1))]
    hi = boots[int(0.975 * (draws - 1))]
    return {"n": n, "median": med, "ci_low": lo, "ci_high": hi}


def parse_log(path: Path) -> tuple[dict[str, object], list[dict[str, float]]]:
    match = MATRIX_RE.match(path.name)
    original = False
    if not match:
        match = ORIGINAL_RE.match(path.name)
        original = bool(match)
    if not match:
        raise ValueError(f"Unrecognized scheduler log name: {path}")

    gd = match.groupdict()
    meta: dict[str, object] = {
        "label": gd["label"],
        "q": int(gd["q"]),
        "kv": int(gd["kv"]),
        "session": int(gd["session"]),
        "q_task_rows": 32 if original else ("auto" if gd["rows"] == "auto" else int(gd["rows"])),
        "requested_workers": "auto" if original else gd["workers"],
        "mapping": "original_kv_head" if original else "query_chunk_x_kv_head",
    }

    hosts: dict[int, float] = {}
    worker_meta: dict[int, dict[str, str]] = {}
    timer_sums: dict[int, Counter[str]] = defaultdict(Counter)
    task_workers: dict[int, Counter[int]] = defaultdict(Counter)

    for line in path.read_text(errors="replace").splitlines():
        if "phase=measure" not in line:
            continue
        fields = kv(line)
        if "iteration" not in fields:
            continue
        iteration = int(fields["iteration"])
        if "FIG8_ATTENTION_HOST_TIMING" in line:
            if int(fields.get("ret", "-1")) != 0:
                raise ValueError(f"Nonzero return in {path}: {line}")
            hosts[iteration] = float(fields["host_elapsed_us"])
        elif "FIG8_ATTENTION_WORKERS" in line:
            worker_meta[iteration] = fields
        elif "FIG8_ATTENTION_TIMERS" in line:
            for metric in METRICS:
                timer_sums[iteration][metric] += float(fields.get(metric, 0))
            task_workers[iteration][int(fields.get("worker", -1))] += 1

    rows: list[dict[str, float]] = []
    for iteration in sorted(hosts):
        row: dict[str, float] = {"iteration": float(iteration), "host_us": hosts[iteration]}
        row.update({metric: timer_sums[iteration][metric] for metric in METRICS})
        wm = worker_meta.get(iteration, {})
        if original:
            # Original has one complete-query task per KV head. The timer records
            # identify the workers that actually did work even though old profile
            # headers did not carry scheduler metadata.
            active = len([w for w in task_workers[iteration] if w >= 0])
            tasks = sum(task_workers[iteration].values())
            q_rows = 32
            hvx_contexts = 6
            vtcm_cap = 6
        else:
            active = int(wm["active_workers"])
            tasks = int(wm["tasks"])
            q_rows = int(wm["q_task_rows"])
            hvx_contexts = int(wm["hvx_contexts"])
            vtcm_cap = int(wm["vtcm_worker_cap"])
        counts = [count for worker, count in task_workers[iteration].items() if worker >= 0]
        row.update(
            {
                "active_workers": float(active),
                "tasks": float(tasks),
                "q_task_rows_actual": float(q_rows),
                "hvx_contexts": float(hvx_contexts),
                "vtcm_worker_cap": float(vtcm_cap),
                "vtcm_total_bytes": float(active * 1024 * 1024),
                "task_records_max_worker": float(max(counts) if counts else 0),
                "task_records_min_worker": float(min(counts) if counts else 0),
                "task_record_imbalance": float((max(counts) - min(counts)) if counts else 0),
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError(f"No measured iterations in {path}")
    return meta, rows


def key_for(meta: dict[str, object]) -> str:
    return "|".join(
        str(meta[name])
        for name in ("label", "q", "kv", "q_task_rows", "requested_workers", "mapping")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    metadata: dict[str, dict[str, object]] = {}
    sources: dict[str, list[str]] = defaultdict(list)
    for run_dir in args.run_dir:
        for path in sorted((run_dir / "raw" / "attention").glob("*.log")):
            meta, rows = parse_log(path)
            key = key_for(meta)
            grouped[key].extend(rows)
            metadata[key] = {k: v for k, v in meta.items() if k != "session"}
            sources[key].append(str(path))

    summary: dict[str, object] = {
        "schema_version": 1,
        "aggregation": "component timers are summed across task records per iteration; host_us remains wall time",
        "bootstrap_draws": args.draws,
        "groups": {},
    }
    csv_rows: list[dict[str, object]] = []
    for index, key in enumerate(sorted(grouped)):
        rows = grouped[key]
        metric_names = ["host_us", *METRICS, "active_workers", "tasks", "q_task_rows_actual",
                        "vtcm_total_bytes", "task_records_max_worker", "task_records_min_worker",
                        "task_record_imbalance"]
        metrics = {
            name: median_ci([row[name] for row in rows], args.draws, 27500 + index * 101 + i)
            for i, name in enumerate(metric_names)
        }
        group = {"meta": metadata[key], "metrics": metrics, "sources": sources[key]}
        summary["groups"][key] = group
        flat: dict[str, object] = {"key": key, **metadata[key]}
        for name, stats in metrics.items():
            flat[f"{name}_median"] = stats["median"]
            flat[f"{name}_ci_low"] = stats["ci_low"]
            flat[f"{name}_ci_high"] = stats["ci_high"]
        csv_rows.append(flat)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    if csv_rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
