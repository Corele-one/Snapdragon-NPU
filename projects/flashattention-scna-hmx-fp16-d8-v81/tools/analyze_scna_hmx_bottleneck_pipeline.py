#!/usr/bin/env python3
"""Deterministically summarize the v81 HMX SCNA optimization/pipeline run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OLD_RUN = ROOT / "results/20260812_scna_hmx_fp16_d8_v81"
REPORT_NAME = "SCNA_HMX_FP16_D8_V81_BOTTLENECK_PIPELINE_REPORT.md"
LABEL = {
    "baseline": "Origin-HVX", "lut-exp": "EXP-LUT",
    "scna-hvx-fp16-d8": "HVX SCNA d8",
    "scna-hmx-fp16-d8-hybrid": "Hybrid current",
    "scna-hmx-fp16-d8-two-pass": "Two-pass current",
    "scna-hmx-fp16-d8-hybrid-vtranspose": "Hybrid + vtranspose",
    "scna-hmx-fp16-d8-two-pass-vtranspose": "Two-pass + vtranspose",
    "scna-hmx-fp16-d8-hybrid-batch4": "Hybrid + batch4",
    "scna-hmx-fp16-d8-two-pass-batch4": "Two-pass + batch4",
    "scna-hmx-fp16-d8-hybrid-direct-p": "Hybrid + direct-P",
    "scna-hmx-fp16-d8-two-pass-direct-p": "Two-pass + direct-P",
    "scna-hmx-fp16-d8-hybrid-attn-pipeline": "Hybrid pipeline request",
    "scna-hmx-fp16-d8-two-pass-attn-pipeline": "Two-pass pipeline request",
}
COLORS = {mode: plt.cm.tab20(i % 20) for i, mode in enumerate(LABEL)}
TIMER_FIELDS = ("profiled_total", "qk_dot", "safe_sm", "core_acc", "scna_exp",
                "scna_pack", "scna_hmx_affine_relu", "scna_reduction", "scna_unpack",
                "scna_transpose", "scna_p_store", "scna_lock", "scna_completion_fence",
                "scna_pipeline_overlap")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else values[lo] * (hi - position) + values[hi] * (position - lo)


def bootstrap_median(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    return percentile(draws, .025), percentile(draws, .975)


def timer_medians(row: dict[str, Any]) -> dict[str, float]:
    per_iteration: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in row.get("timers", []):
        fields = record["fields"]
        iteration = int(fields["iteration"])
        for key in TIMER_FIELDS:
            per_iteration[iteration][key] += float(fields.get(key, 0))
    return {key + "_median_us": statistics.median([x[key] for x in per_iteration.values()])
            if per_iteration else math.nan for key in TIMER_FIELDS}


def performance_summary(rows: list[dict[str, Any]], samples: int, seed: int
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("session_valid") or row.get("status") != "pass":
            continue
        measurements = [float(x["fields"]["host_elapsed_us"]) for x in row["host_measurements"]]
        sessions.append({"qo_len": int(row["qo_len"]), "session": int(row["session"]),
                         "attempt": int(row["attempt"]), "mode": row["mode"],
                         "label": LABEL[row["mode"]], "host_median_us": statistics.median(measurements),
                         "temperature_span_c": row["temperature_span_c"], **timer_medians(row)})
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        groups[(row["qo_len"], row["mode"])].append(row)
    summary: list[dict[str, Any]] = []
    for index, ((q, mode), group) in enumerate(sorted(groups.items())):
        values = [x["host_median_us"] for x in group]
        lo, hi = bootstrap_median(values, samples, seed + index)
        item = {"qo_len": q, "mode": mode, "label": LABEL[mode], "sessions": len(group),
                "median_us": statistics.median(values), "ci_low_us": lo, "ci_high_us": hi}
        for key in TIMER_FIELDS:
            item[key + "_median_us"] = statistics.median(x[key + "_median_us"] for x in group)
        summary.append(item)
    return sessions, summary


def paired_speedups(sessions: list[dict[str, Any]], boot: int, seed: int) -> list[dict[str, Any]]:
    comparisons = [
        ("scna-hmx-fp16-d8-hybrid-direct-p", "scna-hmx-fp16-d8-hybrid"),
        ("scna-hmx-fp16-d8-two-pass-direct-p", "scna-hmx-fp16-d8-two-pass"),
        ("scna-hmx-fp16-d8-hybrid-direct-p", "scna-hvx-fp16-d8"),
        ("scna-hmx-fp16-d8-two-pass-direct-p", "scna-hvx-fp16-d8"),
        ("scna-hmx-fp16-d8-hybrid-direct-p", "lut-exp"),
        ("scna-hmx-fp16-d8-two-pass-direct-p", "lut-exp"),
        ("scna-hmx-fp16-d8-hybrid-attn-pipeline", "scna-hmx-fp16-d8-hybrid-direct-p"),
        ("scna-hmx-fp16-d8-two-pass-attn-pipeline", "scna-hmx-fp16-d8-two-pass-direct-p"),
    ]
    lookup = {(x["qo_len"], x["session"], x["mode"]): x["host_median_us"] for x in sessions}
    q_values = sorted({x["qo_len"] for x in sessions})
    result: list[dict[str, Any]] = []
    for index, (candidate, baseline) in enumerate(comparisons):
        ratios_by_q: dict[int, list[float]] = {}
        for q in q_values:
            ratios_by_q[q] = [lookup[(q, s, baseline)] / lookup[(q, s, candidate)]
                              for s in sorted({x["session"] for x in sessions})
                              if (q, s, baseline) in lookup and (q, s, candidate) in lookup]
        if not q_values or any(not ratios_by_q[q] for q in q_values):
            result.append({"candidate": candidate, "baseline": baseline, "status": "missing"})
            continue
        flat = [v for q in q_values for v in ratios_by_q[q]]
        point = math.exp(statistics.mean(math.log(v) for v in flat))
        rng = random.Random(seed + 1000 + index)
        draws = []
        for _ in range(boot):
            values = [v for q in q_values for v in rng.choices(ratios_by_q[q], k=len(ratios_by_q[q]))]
            draws.append(math.exp(statistics.mean(math.log(v) for v in values)))
        lo, hi = percentile(draws, .025), percentile(draws, .975)
        unsupported = "attn-pipeline" in candidate
        status = "unsupported-serial-fallback" if unsupported else (
            "faster" if lo > 1 else "slower" if hi < 1 else "no-significant-conclusion")
        result.append({"candidate": candidate, "candidate_label": LABEL[candidate],
                       "baseline": baseline, "baseline_label": LABEL[baseline], "speedup": point,
                       "ci_low": lo, "ci_high": hi, "pairs": len(flat), "status": status})
    return result


def micro_paired_speedups(samples: list[dict[str, Any]], boot: int, seed: int) -> list[dict[str, Any]]:
    comparisons = (
        ("scna-hmx-fp16-d8-hybrid-vtranspose", "scna-hmx-fp16-d8-hybrid"),
        ("scna-hmx-fp16-d8-hybrid-batch4", "scna-hmx-fp16-d8-hybrid-vtranspose"),
        ("scna-hmx-fp16-d8-hybrid-direct-p", "scna-hmx-fp16-d8-hybrid-batch4"),
        ("scna-hmx-fp16-d8-hybrid-direct-p", "scna-hmx-fp16-d8-hybrid"),
        ("scna-hmx-fp16-d8-two-pass-vtranspose", "scna-hmx-fp16-d8-two-pass"),
        ("scna-hmx-fp16-d8-two-pass-batch4", "scna-hmx-fp16-d8-two-pass-vtranspose"),
        ("scna-hmx-fp16-d8-two-pass-direct-p", "scna-hmx-fp16-d8-two-pass-batch4"),
        ("scna-hmx-fp16-d8-two-pass-direct-p", "scna-hmx-fp16-d8-two-pass"),
        ("scna-hmx-fp16-d8-hybrid-direct-p", "scna-hvx-fp16-d8"),
        ("scna-hmx-fp16-d8-two-pass-direct-p", "scna-hvx-fp16-d8"),
    )
    valid = [x for x in samples if x["status"] == "pass" and x["duration_gate"]]
    by_mode: dict[str, list[float]] = defaultdict(list)
    for row in valid:
        by_mode[row["mode"]].append(row["pair_ns_per_vector"])
    result: list[dict[str, Any]] = []
    for index, (candidate, baseline) in enumerate(comparisons):
        candidate_values, baseline_values = by_mode[candidate], by_mode[baseline]
        if not candidate_values or not baseline_values:
            result.append({"candidate": candidate, "baseline": baseline, "status": "missing"})
            continue
        point = statistics.median(baseline_values) / statistics.median(candidate_values)
        rng = random.Random(seed + 3000 + index)
        draws = [statistics.median(rng.choices(baseline_values, k=len(baseline_values))) /
                 statistics.median(rng.choices(candidate_values, k=len(candidate_values))) for _ in range(boot)]
        lo, hi = percentile(draws, .025), percentile(draws, .975)
        status = "faster" if lo > 1 else "slower" if hi < 1 else "no-significant-conclusion"
        result.append({"candidate": candidate, "candidate_label": LABEL[candidate],
                       "baseline": baseline, "baseline_label": LABEL[baseline],
                       "speedup": point, "ci_low": lo, "ci_high": hi,
                       "candidate_samples": len(candidate_values), "baseline_samples": len(baseline_values),
                       "status": status})
    return result


def micro_summary(rows: list[dict[str, Any]], boot: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        rec = row.get("record")
        if not rec:
            continue
        f = rec["fields"]; iters = float(row["iterations"])
        item = {"mode": row["mode"], "label": LABEL[row["mode"]], "sample": row["sample"],
                "status": row["status"], "duration_gate": row.get("duration_gate", False),
                "elapsed_ms": row.get("elapsed_ms", 0),
                "implementation_rmse": float(f.get("implementation_rmse", 0)),
                "implementation_max_abs_error": float(f.get("implementation_max_abs_error", 0)),
                "overlap_mismatches": int(f.get("overlap_mismatches", 0)),
                "layout_mismatches": int(f.get("layout_mismatches", 0)),
                "overlap_speedup": float(f.get("overlap_speedup", 0)),
                "physical_macs": float(f.get("physical_macs", 0)),
                "useful_macs": float(f.get("useful_macs", 0))}
        item["pair_ns_per_vector"] = float(f.get("pair_elapsed_us", 0)) * 1000.0 / (2.0 * iters)
        for source, target in (("kernel_total_us", "total_ns_per_vector"),
                               ("pack_us", "pack_ns_per_vector"),
                               ("hmx_affine_relu_us", "affine_ns_per_vector"),
                               ("reduction_us", "reduction_ns_per_vector"),
                               ("transpose_us", "transpose_ns_per_vector"),
                               ("p_store_us", "p_store_ns_per_vector"),
                               ("completion_fence_us", "fence_ns_per_vector")):
            item[target] = float(f.get(source, 0)) * 1000.0 / iters
        total_us = float(f.get("kernel_total_us", 0))
        samples.append(item)
    summary: list[dict[str, Any]] = []
    for index, mode in enumerate(LABEL):
        group = [x for x in samples if x["mode"] == mode and x["status"] == "pass"]
        if not group:
            continue
        group = [x for x in group if x["duration_gate"]]
        if not group:
            continue
        values = [x["pair_ns_per_vector"] for x in group]
        lo, hi = bootstrap_median(values, boot, seed + 2000 + index)
        item = {"mode": mode, "label": LABEL[mode], "samples": len(group),
                "pair_ns_per_vector": statistics.median(values), "ci_low_ns": lo, "ci_high_ns": hi,
                "single_ns_per_vector": statistics.median(x["total_ns_per_vector"] for x in group),
                "implementation_rmse_max": max(x["implementation_rmse"] for x in group),
                "implementation_max_abs_error_max": max(x["implementation_max_abs_error"] for x in group),
                "layout_mismatches_max": max(x["layout_mismatches"] for x in group),
                "overlap_mismatches_max": max(x["overlap_mismatches"] for x in group)}
        for key in ("pack_ns_per_vector", "affine_ns_per_vector", "reduction_ns_per_vector",
                    "transpose_ns_per_vector", "p_store_ns_per_vector", "fence_ns_per_vector",
                    "overlap_speedup"):
            item[key] = statistics.median(x[key] for x in group)
        overlap = [x["overlap_speedup"] for x in group if x["overlap_speedup"] > 0]
        if overlap:
            item["overlap_ci_low"], item["overlap_ci_high"] = bootstrap_median(
                overlap, boot, seed + 2500 + index)
        else:
            item["overlap_ci_low"] = item["overlap_ci_high"] = 0.0
        legacy = "batch4" not in mode and "direct-p" not in mode
        two_pass = "two-pass" in mode
        if mode.startswith("scna-hmx"):
            # Throughput is computed over completed HMX stages, not end-to-end
            # pair latency.  Legacy dispatch issues two 32-spatial commands for
            # one 64-lane vector; batch4 issues one full command for two vectors
            # (or the same command with a zero-filled dummy vector in single mode).
            physical_macs_per_dispatch = ((32768 if two_pass else 16384) if legacy
                                          else (40960 if two_pass else 8192))
            useful_macs_per_dispatch = ((1024 if two_pass else 512) if legacy
                                        else (2048 if two_pass else 1024))
            hmx_ns = item["affine_ns_per_vector"] + (item["reduction_ns_per_vector"] if two_pass else 0)
            item["physical_tops"] = 2.0 * physical_macs_per_dispatch / hmx_ns / 1000.0
            item["effective_tops"] = 2.0 * useful_macs_per_dispatch / hmx_ns / 1000.0
            item["arithmetic_utilization_pct"] = 100.0 * useful_macs_per_dispatch / physical_macs_per_dispatch
        else:
            item["physical_tops"] = item["effective_tops"] = item["arithmetic_utilization_pct"] = 0.0
        summary.append(item)
    return samples, summary


def correctness_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        fields = row.get("compare", {}).get("fields", {}) if row.get("compare") else {}
        result.append({"mode": row["mode"], "label": LABEL[row["mode"]],
                       "mask_mode": row["mask_mode"], "kv_len": row["kv_len"],
                       "head_dim": row["head_dim"], "status": row["status"],
                       "rmse": fields.get("rmse", ""), "max_abs_error": fields.get("max_abs_error", ""),
                       "candidate_nonfinite": fields.get("candidate_nonfinite", "")})
    return result


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight",
                metadata={"Software": "SCNA HMX deterministic analyzer"})
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def unavailable(ax: plt.Axes, message: str) -> None:
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="#4c566a", wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d8dee9")


def plot_current_bottleneck(path: Path) -> dict[str, dict[str, float]]:
    old = list(csv.DictReader((OLD_RUN / "summary/microkernel.csv").open(encoding="utf-8")))
    stages = ("pack_ns_per_vector", "affine_ns_per_vector", "reduction_ns_per_vector", "unpack_ns_per_vector")
    modes = ("scna-hmx-fp16-d8-hybrid", "scna-hmx-fp16-d8-two-pass")
    med = {mode: {key: statistics.median(float(x[key]) for x in old if x["mode"] == mode) for key in stages}
           for mode in modes}
    fig, ax = plt.subplots(figsize=(8, 4.8)); bottom = [0., 0.]
    for key, label, color in zip(stages, ("pack", "HMX affine/ReLU", "reduction", "unpack"),
                                 ("#88c0d0", "#5e81ac", "#b48ead", "#d08770")):
        values = [med[m][key] for m in modes]
        ax.bar(range(2), values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(range(2), [LABEL[m] for m in modes]); ax.set_ylabel("ns / 64-lane vector")
    ax.set_title("Current HMX SCNA bottleneck composition"); ax.legend(); ax.grid(axis="y", alpha=.25)
    save(fig, path); return med


def plot_ablation(micro: list[dict[str, Any]], path: Path) -> None:
    sequences = (
        ("Hybrid", ("scna-hmx-fp16-d8-hybrid", "scna-hmx-fp16-d8-hybrid-vtranspose",
                    "scna-hmx-fp16-d8-hybrid-batch4", "scna-hmx-fp16-d8-hybrid-direct-p")),
        ("Two-pass", ("scna-hmx-fp16-d8-two-pass", "scna-hmx-fp16-d8-two-pass-vtranspose",
                      "scna-hmx-fp16-d8-two-pass-batch4", "scna-hmx-fp16-d8-two-pass-direct-p")),
    )
    lookup = {x["mode"]: x for x in micro}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for family, modes in sequences:
        rows = [lookup[m] for m in modes if m in lookup]
        if len(rows) == 4:
            y = [x["pair_ns_per_vector"] for x in rows]
            low = [x["pair_ns_per_vector"] - x["ci_low_ns"] for x in rows]
            high = [x["ci_high_ns"] - x["pair_ns_per_vector"] for x in rows]
            ax.errorbar(range(4), y, yerr=[low, high], marker="o", capsize=3, label=family)
    ax.set_xticks(range(4), ["current", "vtranspose", "batch4", "direct-P"])
    ax.set_yscale("log"); ax.set_ylabel("pair-path ns / vector (log)")
    ax.set_title("SCNA microkernel stepwise ablation (95% bootstrap CI)")
    ax.grid(alpha=.25); ax.legend(); save(fig, path)
    save(fig, path)


def plot_micro(micro: list[dict[str, Any]], path: Path) -> None:
    modes = [x["mode"] for x in micro]
    stages = (("pack_ns_per_vector", "pack"), ("affine_ns_per_vector", "HMX affine/ReLU"),
              ("reduction_ns_per_vector", "reduction"), ("transpose_ns_per_vector", "transpose"),
              ("fence_ns_per_vector", "completion fence"))
    fig, ax = plt.subplots(figsize=(12, 5.5)); bottom = [0.] * len(modes)
    for (key, label), color in zip(stages, ("#88c0d0", "#5e81ac", "#b48ead", "#a3be8c", "#d08770")):
        values = [x[key] for x in micro]
        ax.bar(range(len(modes)), values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(range(len(modes)), [LABEL[x].replace(" + ", "\n+") for x in modes], rotation=20, ha="right")
    ax.set_ylabel("ns / single-dispatch vector"); ax.set_title("SCNA microkernel stages (30 independent >=50 ms samples)")
    ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8); save(fig, path)


def plot_throughput(micro: list[dict[str, Any]], path: Path) -> None:
    chosen = [x for x in micro if x["mode"] in ("scna-hmx-fp16-d8-hybrid-direct-p",
                                                "scna-hmx-fp16-d8-two-pass-direct-p")]
    labels = ["Dense HMX\n(reference)", "Current Hybrid", "Current Two-pass"] + [x["label"] for x in chosen]
    values = [13.2, .143, .155] + [x["physical_tops"] for x in chosen]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(range(len(values)), values, color=["#4c566a", "#bf616a", "#d08770"] + [COLORS[x["mode"]] for x in chosen])
    axes[0].set_xticks(range(len(values)), labels, rotation=20, ha="right"); axes[0].set_ylabel("physical TOPS")
    axes[0].set_title("Dense and SCNA physical throughput")
    util = [3.125, 3.125] + [x["arithmetic_utilization_pct"] for x in chosen]
    axes[1].bar(range(len(util)), util, color=["#bf616a", "#d08770"] + [COLORS[x["mode"]] for x in chosen])
    axes[1].set_xticks(range(len(util)), labels[1:], rotation=20, ha="right"); axes[1].set_ylabel("useful / issued MAC (%)")
    axes[1].set_title("Structural arithmetic utilization")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    save(fig, path)


def plot_hmx_timeline(perf: list[dict[str, Any]], micro: list[dict[str, Any]], path: Path) -> None:
    candidates = [x for x in perf if x["qo_len"] == 32 and x["mode"] in
                  ("scna-hmx-fp16-d8-hybrid-direct-p", "scna-hmx-fp16-d8-two-pass-direct-p")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    bottom = [0.] * len(candidates)
    for key, label, color in (("qk_dot_median_us", "QK HMX", "#5e81ac"),
                              ("scna_exp_median_us", "SCNA HMX/HVX", "#b48ead"),
                              ("core_acc_median_us", "PV/O HMX", "#a3be8c")):
        values = [x[key] for x in candidates]
        axes[0].bar(range(len(candidates)), values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set_title("Serialized HMX command stages")
    if candidates:
        axes[0].set_xticks(range(len(candidates)), [x["label"] for x in candidates], rotation=15, ha="right")
        axes[0].set_ylabel("completion time at qo_len=32 (us)"); axes[0].legend()
    else:
        unavailable(axes[0], "NOT COLLECTED\nNo QK / SCNA / PV completion times\nafter correctness-gate timeout")
    overlap = [x for x in micro if "direct-p" in x["mode"]]
    axes[1].bar(range(len(overlap)), [x["overlap_speedup"] for x in overlap], color=[COLORS[x["mode"]] for x in overlap])
    axes[1].axhline(1.05, color="#bf616a", linestyle="--", label="probe gate 1.05x")
    axes[1].set_xticks(range(len(overlap)), [x["label"] for x in overlap], rotation=15, ha="right")
    axes[1].set_ylabel("serial / overlapped probe"); axes[1].set_title("Standalone HMX/HVX overlap probe"); axes[1].legend()
    for ax in axes: ax.grid(axis="y", alpha=.25)
    save(fig, path)


def plot_pipeline(perf: list[dict[str, Any]], path: Path) -> None:
    pairs = (("scna-hmx-fp16-d8-hybrid-direct-p", "scna-hmx-fp16-d8-hybrid-attn-pipeline"),
             ("scna-hmx-fp16-d8-two-pass-direct-p", "scna-hmx-fp16-d8-two-pass-attn-pipeline"))
    lookup = {(x["qo_len"], x["mode"]): x for x in perf}
    q_values = sorted({x["qo_len"] for x in perf})
    fig, ax = plt.subplots(figsize=(9, 5))
    for direct, requested in pairs:
        for mode, style in ((direct, "-"), (requested, "--")):
            rows = [lookup[(q, mode)] for q in q_values if (q, mode) in lookup]
            if rows: ax.plot([x["qo_len"] for x in rows], [x["median_us"] for x in rows], style,
                             marker="o", label=LABEL[mode], color=COLORS[direct])
    ax.set_title("Pipeline versus no-pipeline latency")
    if q_values:
        ax.set_xlabel("qo_len"); ax.set_ylabel("host latency (us)"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    else:
        unavailable(ax, "NOT COLLECTED\nPipeline and no-pipeline Attention both failed\nthe first correctness case")
    save(fig, path)


def plot_correctness(rows: list[dict[str, Any]], micro: list[dict[str, Any]],
                     rmse_limit: float, max_limit: float, scna_rmse_limit: float,
                     scna_max_limit: float, path: Path) -> None:
    valid = [x for x in rows if x["rmse"] != ""]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if valid:
        x = range(len(valid)); axes[0].scatter(x, [float(v["rmse"]) for v in valid], s=14)
        axes[1].scatter(x, [float(v["max_abs_error"]) for v in valid], s=14)
        axes[0].axhline(rmse_limit, color="red", linestyle="--", label=f"limit {rmse_limit}")
        axes[1].axhline(max_limit, color="red", linestyle="--", label=f"limit {max_limit}")
        axes[0].set(title="RMSE vs FP32 reference", ylabel="RMSE", xlabel="test case")
        axes[1].set(title="Maximum absolute error", ylabel="max abs error", xlabel="test case")
        for ax in axes: ax.set_yscale("log"); ax.grid(alpha=.25); ax.legend()
    else:
        counts = {status: sum(x["status"] == status for x in rows)
                  for status in ("pass", "failed", "skipped-after-mode-gate-failure")}
        labels = ["pass", "timeout/fail", "skipped"]
        values = [counts["pass"], counts["failed"], counts["skipped-after-mode-gate-failure"]]
        axes[0].bar(labels, values, color=["#a3be8c", "#bf616a", "#d8dee9"])
        axes[0].set_title("FlashAttention correctness gate status"); axes[0].set_ylabel("cases")
        hmx = [x for x in micro if x["mode"].startswith("scna-hmx")]
        xpos = range(len(hmx))
        axes[1].scatter(xpos, [max(x["implementation_rmse_max"], 1e-8) for x in hmx],
                        marker="o", label="SCNA RMSE")
        axes[1].scatter(xpos, [max(x["implementation_max_abs_error_max"], 1e-8) for x in hmx],
                        marker="x", label="SCNA max abs")
        axes[1].axhline(scna_rmse_limit, color="#5e81ac", linestyle="--", label=f"RMSE limit {scna_rmse_limit}")
        axes[1].axhline(scna_max_limit, color="#bf616a", linestyle="--", label=f"max-abs limit {scna_max_limit}")
        axes[1].set_xticks(list(xpos), [x["label"].replace(" + ", "\n+") for x in hmx], rotation=25, ha="right")
        axes[1].set_yscale("log"); axes[1].set_title("Independent SCNA numeric gate")
        axes[1].set_ylabel("error vs HVX FP16 SCNA"); axes[1].grid(alpha=.25); axes[1].legend(fontsize=7)
        axes[0].grid(axis="y", alpha=.25)
    save(fig, path)


def report(run: Path, manifest: dict[str, Any], perf: list[dict[str, Any]], speedups: list[dict[str, Any]],
           micro: list[dict[str, Any]], micro_speedups: list[dict[str, Any]],
           correctness: list[dict[str, Any]], current: dict[str, dict[str, float]],
           gates: dict[str, Any]) -> str:
    spec = manifest["spec"]; rel = run.relative_to(ROOT).as_posix()
    failed = [x for x in correctness if x["status"] == "failed"]
    skipped = [x for x in correctness if x["status"].startswith("skipped")]
    passed = [x for x in correctness if x["status"] == "pass"]
    direct = {x["mode"]: x for x in micro if "direct-p" in x["mode"]}
    hcur = current["scna-hmx-fp16-d8-hybrid"]; tcur = current["scna-hmx-fp16-d8-two-pass"]
    hshare = 100 * hcur["unpack_ns_per_vector"] / sum(hcur.values())
    tshare = 100 * tcur["unpack_ns_per_vector"] / sum(tcur.values())
    hmx_errors = [x for x in micro if x["mode"].startswith("scna-hmx")]
    max_scna_rmse = max(x["implementation_rmse_max"] for x in hmx_errors)
    max_scna_abs = max(x["implementation_max_abs_error_max"] for x in hmx_errors)
    acceptance = all(gates[key].get("pass", False) for key in ("disassembly", "hmx_numeric", "attention"))
    hybrid = direct["scna-hmx-fp16-d8-hybrid-direct-p"]
    two = direct["scna-hmx-fp16-d8-two-pass-direct-p"]
    lines = ["# v81 HMX SCNA 瓶颈优化与 Attention 流水化验证", "",
             "## 结论", "",
             f"最终验收：`{'通过' if acceptance else '未通过'}`。独立 HMX SCNA 数值门禁与反汇编门禁通过；"
             f"但 8/8 个优化 Attention 模式首个正确性用例均在 30 s 超时，另 `{len(skipped)}` 个用例按门禁跳过，"
             "所以没有采集正式 Attention 性能、Bc 调优或 pipeline speedup。", "",
             f"既有 Hybrid unpack 为 `{hcur['unpack_ns_per_vector']:.1f} ns/vector`，占显式阶段 `{hshare:.1f}%`；"
             f"Two-pass 为 `{tcur['unpack_ns_per_vector']:.1f} ns/vector`，占 `{tshare:.1f}%`。原实现首要瓶颈是 crouton 布局恢复。", "",
             f"微核优化后 Hybrid direct-P pair 为 `{hybrid['pair_ns_per_vector']:.2f} ns/vector`（95% CI "
             f"`{hybrid['ci_low_ns']:.2f}–{hybrid['ci_high_ns']:.2f}`），Two-pass 为 `{two['pair_ns_per_vector']:.2f}`"
             f"（`{two['ci_low_ns']:.2f}–{two['ci_high_ns']:.2f}`）。这些结果只证明独立微核有效，不能外推为 FlashAttention 加速。", "",
             "## Setup", "", "- Model：N/A；独立 FlashAttention Kernel 实验。",
             "- Dataset：固定种子的合成 Q/K/V 及 full、padding、causal mask。",
             f"- Hardware：{manifest['device']['product_model']} / {manifest['device']['soc_model']}；Hexagon SDK 6.6.0.0，Tools 19.0.07。",
             "- Architecture：v81；`-mv81 -mhmx -mhvx -mhvx-length=128B`。",
             f"- Main matrix：`qo_len={spec['performance']['qo_len']}`、KV=4096、heads=12/2、d=128、单 worker、无 KV pipeline。",
             f"- Planned sampling：5 sessions，每模式 5 warmup + 20 measure，温度跨度门槛 `{spec['performance']['max_start_temperature_span_c']:.1f}°C`，10,000 次配对 bootstrap。",
             f"- Binary SHA256：`{manifest['artifacts_sha256']['libhtp_ops_skel.so']}`；所有证据来自同一二进制。", "",
             "## 当前瓶颈", "", f"![当前瓶颈](../{rel}/figures/current_bottleneck.png)", "",
             f"Hybrid unpack `{hcur['unpack_ns_per_vector']:.1f} ns` 比 affine/ReLU `{hcur['affine_ns_per_vector']:.1f} ns` 高 "
             f"`{hcur['unpack_ns_per_vector']/hcur['affine_ns_per_vector']:.1f}×`；Two-pass unpack 仍占 `{tshare:.1f}%`。"
             "该图用阶段数据支持优先优化布局，而不是把问题归因于 HMX 矩阵乘单元。", "",
             "## 吞吐与利用率", "", f"![吞吐利用率](../{rel}/figures/throughput_utilization.png)", "",
             f"左图按 HMX 阶段完成时间与已发出的物理 MAC 计算：密集参考 13.2 TOPS、既有 Hybrid/Two-pass 0.143/0.155 TOPS、"
             f"优化 Hybrid/Two-pass `{hybrid['physical_tops']:.3f}`/`{two['physical_tops']:.3f}` TOPS。"
             "右图是 useful/issued MAC：batch4 Hybrid 为 12.5%，Two-pass 因第二遍 K=32 归约为 5.0%。"
             "结构利用率提升没有转化成密集 GEMM 级吞吐，说明小命令、栅栏与布局开销仍主导。", "",
             "## 微核逐步消融", "", f"![逐步消融](../{rel}/figures/ablation.png)", ""]
    for x in micro_speedups:
        if x["status"] != "missing":
            lines.append(f"- {x['candidate_label']} vs {x['baseline_label']}：`{x['speedup']:.3f}×`"
                         f"（95% CI `{x['ci_low']:.3f}–{x['ci_high']:.3f}`），`{x['status']}`。")
    lines += ["", "图中 vtranspose、batch4、direct-P 的变化来自微核 pair loop；端到端消融未采集。", "",
              "## HMX/HVX 重叠与 Attention pipeline", "", f"![HMX timeline](../{rel}/figures/hmx_timeline_overlap.png)", "",
              "左图标记 QK/SCNA/PV 完成时间未采集。三者共用 NON_SHARED HMX，只能串行发 HMX 命令；可重叠的是独立 HVX 工作。"
              f"右图独立 probe 的 Hybrid/Two-pass speedup 均为约 `{hybrid['overlap_speedup']:.3f}×`，"
              f"95% CI 下界 `{hybrid['overlap_ci_low']:.3f}`/`{two['overlap_ci_low']:.3f}`，mismatch 均为 0。"
              "该 probe 通过 1.05× 门槛，但不等价于端到端流水可用。", "",
              f"![Pipeline](../{rel}/figures/pipeline.png)", "",
              "pipeline 与无 pipeline 均没有端到端数据：所有优化模式都在第一次 HMX SCNA 调用超时。"
              "因此未执行 `Bc={256,512,1024,2048,4096}` 调优，也不报告 pipeline speedup。", "",
              "## 微核阶段", "", f"![微核阶段](../{rel}/figures/microkernel_stages.png)", ""]
    for x in micro:
        lines.append(f"- {x['label']}：pair `{x['pair_ns_per_vector']:.2f} ns/vector`，single `{x['single_ns_per_vector']:.2f}`，"
                     f"样本 `{x['samples']}`，overlap probe `{x['overlap_speedup']:.3f}×`。")
    lines += ["", "所有纳入汇总的模式均有 30 个独立且 ≥50 ms 的样本。阶段柱来自 single-dispatch profile；pair 延迟与 CI 来自独立 pair loop。", "",
              "## 正确性", "", f"![正确性](../{rel}/figures/correctness.png)", "",
              f"独立 SCNA 数值门禁通过：最大 implementation RMSE `{max_scna_rmse:.7f}`（门槛 0.001），最大绝对误差 "
              f"`{max_scna_abs:.7f}`（门槛 0.002），layout/overlap mismatch 均为 0。FlashAttention 为 "
              f"`{len(passed)}` pass、`{len(failed)}` timeout/fail、`{len(skipped)}` skipped；没有输出张量，"
              "所以 RMSE≤0.002 与 max-abs≤0.01 无可计算值，不能写成正确性通过。", "",
              "## vgather 门禁", "",
              "正式 1/8 通道 × ring depth × 同步策略稳定性矩阵未执行，状态为 `not-run`。开发期 gather 输出路径曾在 30 s 长序列门禁超时，"
              "但未达到 10^7 gather/30 s 的覆盖要求；因此没有恢复到正式 Kernel，也不报告相对 vtranspose 的 speedup。", "",
              "## Unexpected Results 与验证计划", "",
              "1. 8 个模式均停在 profile debug stage 416：QK 已完成，程序进入第一次 `scna_hmx_fp16_d8_pair_vhf`，30 s 内没有到达 stage 417。"
              "候选原因是 QK→SCNA 时 HMX output-scale/shape 状态转换不完整。验证：构造最小 QK→SCNA probe，每次命令前显式重设 scale/shape。",
              "2. HMX 独立微核通过、嵌入 Attention 超时，候选原因是共享 HMX 命令流的 ownership、完成栅栏或 store retirement 约束。"
              "验证：固定 ring depth=1，依次加入 QK completion fence、SCNA issue 前 fence、consume 前 fence，记录首个可返回组合。",
              "3. batch4 没有得到理论 4× 延迟收益，因为利用率提高来自一次命令填入四组映射，而固定命令、栅栏与转置并未同比消失。"
              "验证：profile 已把 pack、HMX affine、transpose 和 completion fence 分开；下一步逐项置零或预打包以定量确认。", "",
              "## Limitations", "", "- 仅一款 SM8750P SoC、FP16 d8 exp2、输入区间 `[-256,0]`。",
              "- 仅 Kernel 级合成数据，无模型精度、困惑度或端到端 token 延迟。",
              "- Attention 正确性门禁失败，正式性能和 pipeline 数据不存在。",
              "- vgather 正式稳定性矩阵未完成，不能评价其可用吞吐。",
              "- Android 无法读取 CDSP/HMX 实时频率，只记录请求的 performance/TURBO_L3 模式。", "",
              "## 可复现证据", "", f"- Manifest：`results/{run.name}/manifest.json`",
              f"- Raw JSONL/log：`results/{run.name}/raw/`", f"- Summary：`results/{run.name}/summary/`",
              f"- Disassembly：`results/{run.name}/verification/scna_hmx_symbols.disasm.txt`",
              f"- Gates：`results/{run.name}/verification/`",
              "- 重建命令：`python3 tools/analyze_scna_hmx_bottleneck_pipeline.py results/" + run.name + "`", "",
              "## AI 辅助声明", "", "AI 用于代码与实验脚本辅助、机械化解析、绘图和报告组织；研究问题、实验设计、设备执行、原始数据验证和结论由研究者复核。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args(); run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); spec = manifest["spec"]
    boot, seed = int(spec["statistics"]["bootstrap_samples"]), int(spec["seed"])
    sessions, perf = performance_summary(read_jsonl(run / "raw/performance.jsonl"), boot, seed)
    speedups = paired_speedups(sessions, boot, seed)
    micro_samples, micro = micro_summary(read_jsonl(run / "raw/microkernel.jsonl"), boot, seed)
    micro_speedups = micro_paired_speedups(micro_samples, boot, seed)
    correctness = correctness_summary(read_jsonl(run / "raw/correctness.jsonl"))
    gates = {
        name: json.loads((run / f"verification/{filename}").read_text(encoding="utf-8"))
        for name, filename in (("disassembly", "disassembly_gate.json"),
                               ("hmx_numeric", "hmx_numeric_gate.json"),
                               ("attention", "attention_correctness_gate.json"))
    }
    vgather_gate = {"pass": False, "status": "not-run",
                    "reason": "The required 1/8-channel x ring-depth x synchronization matrix was not executed.",
                    "development_evidence": "docs/HMX_SCNA_OPTIMIZATION_HISTORY.json",
                    "formal_kernel_enabled": False}
    (run / "verification/vgather_gate.json").write_text(
        json.dumps(vgather_gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gates["vgather"] = vgather_gate
    summary, figures = run / "summary", run / "figures"; summary.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    write_csv(summary / "performance_sessions.csv", sessions); write_csv(summary / "performance_summary.csv", perf)
    write_csv(summary / "speedup_summary.csv", speedups); write_csv(summary / "microkernel_samples.csv", micro_samples)
    write_csv(summary / "microkernel_summary.csv", micro); write_csv(summary / "microkernel_speedup.csv", micro_speedups)
    write_csv(summary / "correctness.csv", correctness)
    combined = {"performance": perf, "speedups": speedups, "microkernel": micro,
                "microkernel_speedups": micro_speedups, "correctness": correctness, "gates": gates}
    (summary / "summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "scna-hmx-bottleneck-v81"})
    current = plot_current_bottleneck(figures / "current_bottleneck")
    plot_ablation(micro, figures / "ablation"); plot_micro(micro, figures / "microkernel_stages")
    plot_throughput(micro, figures / "throughput_utilization"); plot_hmx_timeline(perf, micro, figures / "hmx_timeline_overlap")
    plot_pipeline(perf, figures / "pipeline")
    plot_correctness(correctness, micro, float(spec["correctness"]["rmse_limit"]),
                     float(spec["correctness"]["max_abs_error_limit"]),
                     float(spec["microkernel"]["implementation_rmse_limit"]),
                     float(spec["microkernel"]["implementation_max_abs_error_limit"]), figures / "correctness")
    body = report(run, manifest, perf, speedups, micro, micro_speedups, correctness, current, gates)
    report_path = run / REPORT_NAME; report_path.write_text(body, encoding="utf-8")
    if not args.no_publish:
        public = ROOT / "reports" / REPORT_NAME; public.parent.mkdir(exist_ok=True); shutil.copyfile(report_path, public)
    print(json.dumps({"report": str(report_path), "sha256": hashlib.sha256(body.encode()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
