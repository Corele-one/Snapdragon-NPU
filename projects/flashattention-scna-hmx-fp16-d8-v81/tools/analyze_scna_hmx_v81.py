#!/usr/bin/env python3
"""Analyze one SCNA HMX v81 run and deterministically rebuild CSV, figures and report."""

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
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_HISTORY_PATH = ROOT / "docs/HMX_SCNA_OPTIMIZATION_HISTORY.json"
MODE_LABEL = {
    "baseline": "Origin-HVX",
    "lut-exp": "EXP-LUT",
    "scna-hvx-fp16-d8": "HVX SCNA d8",
    "scna-hmx-fp16-d8-hybrid": "HMX Hybrid",
    "scna-hmx-fp16-d8-two-pass": "HMX Two-pass",
}
COLORS = {
    "baseline": "#4C566A", "lut-exp": "#5E81AC", "scna-hvx-fp16-d8": "#A3BE8C",
    "scna-hmx-fp16-d8-hybrid": "#BF616A", "scna-hmx-fp16-d8-two-pass": "#D08770",
}
STAGES = ("scna_pack", "scna_hmx_affine_relu", "scna_reduction", "scna_unpack")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    x = (len(ordered) - 1) * p
    lo = int(math.floor(x)); hi = int(math.ceil(x))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - x) + ordered[hi] * (x - lo)


def bootstrap_median(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    draws = [statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)]
    return percentile(draws, .025), percentile(draws, .975)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def timer_session_medians(row: dict[str, Any]) -> dict[str, float]:
    by_iteration: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for timer in row["timers"]:
        fields = timer["fields"]
        iteration = int(fields["iteration"])
        for key in ("profiled_total", "safe_sm", "scna_exp", *STAGES):
            by_iteration[iteration][key] += float(fields.get(key, 0))
    result: dict[str, float] = {}
    for key in ("profiled_total", "safe_sm", "scna_exp", *STAGES):
        values = [entry[key] for entry in by_iteration.values()]
        result[key + "_median_us"] = statistics.median(values) if values else math.nan
    return result


def summarize_performance(rows: list[dict[str, Any]], boot: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("session_valid") or row.get("status") != "pass":
            continue
        values = [float(item["fields"]["host_elapsed_us"]) for item in row["host_measurements"]]
        session_rows.append({
            "qo_len": int(row["qo_len"]), "session": int(row["session"]), "attempt": int(row["attempt"]),
            "mode": row["mode"], "host_median_us": statistics.median(values),
            "host_min_us": min(values), "host_max_us": max(values),
            "temperature_span_c": row["temperature_span_c"], **timer_session_medians(row),
        })
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[(row["qo_len"], row["mode"])].append(row)
    summary: list[dict[str, Any]] = []
    for index, ((qo_len, mode), group) in enumerate(sorted(grouped.items())):
        values = [float(row["host_median_us"]) for row in group]
        lo, hi = bootstrap_median(values, boot, seed + index)
        item: dict[str, Any] = {"qo_len": qo_len, "mode": mode, "label": MODE_LABEL[mode],
                                "sessions": len(group), "median_us": statistics.median(values),
                                "ci_low_us": lo, "ci_high_us": hi}
        for key in ("profiled_total", "safe_sm", "scna_exp", *STAGES):
            item[key + "_median_us"] = statistics.median(float(row[key + "_median_us"]) for row in group)
        summary.append(item)
    return session_rows, summary


def paired_speedups(session_rows: list[dict[str, Any]], boot: int, seed: int) -> list[dict[str, Any]]:
    lookup = {(row["qo_len"], row["session"], row["mode"]): float(row["host_median_us"])
              for row in session_rows}
    q_values = sorted({row["qo_len"] for row in session_rows})
    sessions = sorted({row["session"] for row in session_rows})
    output: list[dict[str, Any]] = []
    baselines = ("baseline", "lut-exp", "scna-hvx-fp16-d8")
    hmx_modes = ("scna-hmx-fp16-d8-hybrid", "scna-hmx-fp16-d8-two-pass")
    for pair_index, hmx in enumerate(hmx_modes):
        for baseline in baselines:
            ratios: dict[int, list[float]] = {}
            for q in q_values:
                ratios[q] = [lookup[(q, session, baseline)] / lookup[(q, session, hmx)]
                             for session in sessions
                             if (q, session, baseline) in lookup and (q, session, hmx) in lookup]
            flat = [value for values in ratios.values() for value in values]
            if not flat or any(not values for values in ratios.values()):
                output.append({"hmx_mode": hmx, "baseline_mode": baseline, "status": "missing"})
                continue
            point = math.exp(statistics.mean(math.log(value) for value in flat))
            rng = random.Random(seed + 100 + pair_index * 10 + baselines.index(baseline))
            draws: list[float] = []
            for _ in range(boot):
                sampled: list[float] = []
                for q in q_values:
                    sampled.extend(rng.choices(ratios[q], k=len(ratios[q])))
                draws.append(math.exp(statistics.mean(math.log(value) for value in sampled)))
            lo, hi = percentile(draws, .025), percentile(draws, .975)
            output.append({"hmx_mode": hmx, "hmx_label": MODE_LABEL[hmx], "baseline_mode": baseline,
                           "baseline_label": MODE_LABEL[baseline], "speedup": point,
                           "ci_low": lo, "ci_high": hi, "pairs": len(flat), "status": "faster" if lo > 1 else
                           ("slower" if hi < 1 else "no-significant-conclusion")})
    return output


def summarize_correctness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        compare = row.get("compare")
        fields = compare["fields"] if compare else {}
        numeric = row.get("numeric", [])
        output.append({"mode": row["mode"], "label": MODE_LABEL[row["mode"]],
                       "mask_mode": row["mask_mode"], "kv_len": row["kv_len"],
                       "head_dim": row["head_dim"], "status": row["status"],
                       "rmse": fields.get("rmse", ""), "max_abs_error": fields.get("max_abs_error", ""),
                       "candidate_nonfinite": fields.get("candidate_nonfinite", ""),
                       "tail_zero_required": row.get("tail_zero_required", False),
                       "tail_zero": bool(numeric) and all(int(x["fields"].get("p0_last_bits", "-1"), 16) == 0
                                                          for x in numeric)})
    return output


def summarize_micro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        record = row.get("record")
        fields = record["fields"] if record else {}
        iters = float(row.get("iterations", 1))
        output.append({"mode": row["mode"], "label": MODE_LABEL[row["mode"]], "sample": row["sample"],
                       "status": row["status"], "duration_gate": row.get("duration_gate", False),
                       "elapsed_ms": row.get("elapsed_ms", ""),
                       "total_ns_per_vector": float(fields.get("kernel_total_us", 0)) * 1000 / iters,
                       "pack_ns_per_vector": float(fields.get("pack_us", 0)) * 1000 / iters,
                       "affine_ns_per_vector": float(fields.get("hmx_affine_relu_us", 0)) * 1000 / iters,
                       "reduction_ns_per_vector": float(fields.get("reduction_us", 0)) * 1000 / iters,
                       "unpack_ns_per_vector": float(fields.get("unpack_us", 0)) * 1000 / iters,
                       "implementation_rmse": fields.get("implementation_rmse", ""),
                       "implementation_max_abs_error": fields.get("implementation_max_abs_error", ""),
                       "random_samples": fields.get("random_samples", ""),
                       "random_implementation_rmse": fields.get("random_implementation_rmse", ""),
                       "random_implementation_max_abs_error": fields.get("random_implementation_max_abs_error", ""),
                       "tail_implementation_max_abs_error": fields.get("tail_implementation_max_abs_error", ""),
                       "dense_rmse_vs_exp2": fields.get("dense_rmse", ""),
                       "monotonic_violations": fields.get("monotonic_violations", ""),
                       "negative_count": fields.get("negative_count", ""), "nan_count": fields.get("nan_count", "")})
    return output


def summarize_optimization(history: dict[str, Any], micro: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in history["checkpoints"]:
        vectors = float(row["vectors"])
        def optional_ns(key: str) -> float | str:
            return float(row[key]) * 1_000_000 / vectors if key in row else ""
        output.append({
            "checkpoint": row["checkpoint"], "acceptance_status": row["acceptance_status"],
            "mode": row["mode"], "label": MODE_LABEL[row["mode"]], "samples": 1,
            "total_ns_per_vector": float(row["total_ms"]) * 1_000_000 / vectors,
            "pack_ns_per_vector": optional_ns("pack_ms"),
            "affine_ns_per_vector": optional_ns("hmx_affine_relu_ms"),
            "reduction_ns_per_vector": optional_ns("reduction_ms"),
            "unpack_ns_per_vector": optional_ns("unpack_ms"),
            "provenance": "development checkpoint; one 1000-vector batch; no CI",
        })
    for mode in ("scna-hmx-fp16-d8-hybrid", "scna-hmx-fp16-d8-two-pass"):
        group = [row for row in micro if row["mode"] == mode and row["status"] == "pass"]
        output.append({
            "checkpoint": "final-stable", "acceptance_status": "accepted", "mode": mode,
            "label": MODE_LABEL[mode], "samples": len(group),
            "total_ns_per_vector": statistics.median(float(row["total_ns_per_vector"]) for row in group),
            "pack_ns_per_vector": statistics.median(float(row["pack_ns_per_vector"]) for row in group),
            "affine_ns_per_vector": statistics.median(float(row["affine_ns_per_vector"]) for row in group),
            "reduction_ns_per_vector": statistics.median(float(row["reduction_ns_per_vector"]) for row in group),
            "unpack_ns_per_vector": statistics.median(float(row["unpack_ns_per_vector"]) for row in group),
            "provenance": "formal run; median of 30 independent >=50 ms samples",
        })
    return output


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=160, bbox_inches="tight", metadata={"Software": "SCNA v81 report"})
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def plot_latency(summary: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for mode in MODE_LABEL:
        rows = sorted((row for row in summary if row["mode"] == mode), key=lambda row: row["qo_len"])
        if not rows:
            continue
        x = [row["qo_len"] for row in rows]; y = [row["median_us"] for row in rows]
        low = [row["median_us"] - row["ci_low_us"] for row in rows]
        high = [row["ci_high_us"] - row["median_us"] for row in rows]
        ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, label=MODE_LABEL[mode], color=COLORS[mode])
    ax.set_yscale("log")
    ax.set(xlabel="qo_len", ylabel="Host latency (µs, log scale)", title="Five-mode v81 latency (session medians, paired experiment)")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); save_figure(fig, path)


def plot_speedup(rows: list[dict[str, Any]], path: Path) -> None:
    valid = [row for row in rows if row["status"] != "missing"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = [f"{row['hmx_label']}\nvs {row['baseline_label']}" for row in valid]
    values = [row["speedup"] for row in valid]
    errors = [[row["speedup"] - row["ci_low"] for row in valid],
              [row["ci_high"] - row["speedup"] for row in valid]]
    ax.bar(range(len(valid)), values, color=[COLORS[row["hmx_mode"]] for row in valid], yerr=errors, capsize=3)
    ax.axhline(1, color="black", linewidth=1); ax.set_xticks(range(len(valid)), labels, rotation=20, ha="right")
    ax.set(ylabel="Speedup (baseline / HMX)", title="HMX geometric-mean speedup across qo_len (paired bootstrap 95% CI)")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); save_figure(fig, path)


def plot_stages(summary: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in summary if row["mode"] in ("scna-hmx-fp16-d8-hybrid", "scna-hmx-fp16-d8-two-pass")]
    rows.sort(key=lambda row: (row["qo_len"], row["mode"]))
    fig, ax = plt.subplots(figsize=(10, 5.4)); bottoms = [0.0] * len(rows)
    stage_labels = {"scna_pack": "pack", "scna_hmx_affine_relu": "HMX affine/ReLU",
                    "scna_reduction": "reduction", "scna_unpack": "unpack"}
    colors = ("#88C0D0", "#5E81AC", "#B48EAD", "#D08770")
    for stage, color in zip(STAGES, colors):
        values = [row[stage + "_median_us"] for row in rows]
        ax.bar(range(len(rows)), values, bottom=bottoms, label=stage_labels[stage], color=color)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_xticks(range(len(rows)), [f"q{r['qo_len']}\n{r['label'].replace('HMX ', '')}" for r in rows])
    ax.set(ylabel="SCNA stage time per attention call (µs)", title="Hybrid vs two-pass stage ablation")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout(); save_figure(fig, path)


def plot_correctness(rows: list[dict[str, Any]], path: Path, rmse_limit: float, max_limit: float) -> None:
    valid = [row for row in rows if row["rmse"] != ""]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = list(range(len(valid)))
    axes[0].scatter(x, [float(row["rmse"]) for row in valid], c=[COLORS[row["mode"]] for row in valid], s=20)
    axes[0].axhline(rmse_limit, color="#BF616A", linestyle="--", label=f"limit {rmse_limit:g}")
    axes[0].set_yscale("log")
    axes[0].set(title="FP32 reference RMSE", xlabel="correctness case", ylabel="RMSE"); axes[0].legend()
    axes[1].scatter(x, [float(row["max_abs_error"]) for row in valid], c=[COLORS[row["mode"]] for row in valid], s=20)
    axes[1].axhline(max_limit, color="#BF616A", linestyle="--", label=f"limit {max_limit:g}")
    axes[1].set_yscale("log")
    axes[1].set(title="FP32 reference maximum absolute error", xlabel="correctness case", ylabel="Max abs error")
    axes[1].legend()
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle("FlashAttention correctness gates: full / padding / causal, kv 4093 / 4096, d64 / d128")
    fig.tight_layout(); save_figure(fig, path)


def plot_micro(rows: list[dict[str, Any]], path: Path) -> None:
    modes = list(MODE_LABEL)[2:]
    medians: dict[str, dict[str, float]] = {}
    for mode in modes:
        group = [row for row in rows if row["mode"] == mode and row["status"] == "pass"]
        medians[mode] = {stage: statistics.median(float(row[stage]) for row in group) for stage in
                         ("pack_ns_per_vector", "affine_ns_per_vector", "reduction_ns_per_vector", "unpack_ns_per_vector")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_total, ax_stages = axes
    total_values = [statistics.median(float(row["total_ns_per_vector"])
                    for row in rows if row["mode"] == mode and row["status"] == "pass") for mode in modes]
    ax_total.bar(range(len(modes)), total_values, color=[COLORS[mode] for mode in modes])
    ax_total.set_yscale("log")
    ax_total.set_xticks(range(len(modes)), [MODE_LABEL[mode].replace(" SCNA d8", "\nSCNA d8").replace("HMX ", "HMX\n") for mode in modes])
    ax_total.set(ylabel="Time per 64-lane vector (ns, log scale)", title="Total microkernel cost")
    ax_total.grid(axis="y", alpha=.25)
    hmx_modes = modes[1:]
    bottoms = [0.0] * len(hmx_modes)
    stages = (("pack_ns_per_vector", "pack"), ("affine_ns_per_vector", "HMX affine/ReLU"),
              ("reduction_ns_per_vector", "reduction"), ("unpack_ns_per_vector", "unpack"))
    for (key, label), color in zip(stages, ("#88C0D0", "#5E81AC", "#B48EAD", "#D08770")):
        values = [100 * medians[mode][key] / sum(medians[mode].values()) for mode in hmx_modes]
        ax_stages.bar(range(len(hmx_modes)), values, bottom=bottoms, label=label, color=color)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax_stages.set_xticks(range(len(hmx_modes)), [MODE_LABEL[mode] for mode in hmx_modes])
    ax_stages.set(ylabel="Share of explicit stages (%)", title="HMX stage composition")
    ax_stages.grid(axis="y", alpha=.25); ax_stages.legend(fontsize=8)
    fig.suptitle("SCNA microkernel: median of 30 independent ≥50 ms samples")
    fig.tight_layout(); save_figure(fig, path)


def plot_optimization(rows: list[dict[str, Any]], path: Path) -> None:
    checkpoints = ("naive-scalar-layout", "vector-scatter-gather", "final-stable")
    modes = ("scna-hmx-fp16-d8-hybrid", "scna-hmx-fp16-d8-two-pass")
    lookup = {(row["checkpoint"], row["mode"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = .34
    for mode_index, mode in enumerate(modes):
        x = [index + (mode_index - .5) * width for index in range(len(checkpoints))]
        values = [float(lookup[(checkpoint, mode)]["total_ns_per_vector"]) for checkpoint in checkpoints]
        bars = ax.bar(x, values, width, label=MODE_LABEL[mode], color=COLORS[mode])
        bars[1].set_hatch("//")
        for xpos, value in zip(x, values):
            ax.text(xpos, value * 1.08, f"{value / 1000:.3g} µs", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(checkpoints)), ("Naive scalar\n(micro only)",
                                            "Scatter + gather\n(rejected: timeout)",
                                            "Final stable\n(30 samples)"))
    ax.set(ylabel="Time per 64-lane vector (ns, log scale)",
           title="HMX SCNA optimization evolution: local speed versus end-to-end acceptance")
    ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout(); save_figure(fig, path)


def fnum(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_report(run_dir: Path, manifest: dict[str, Any], perf: list[dict[str, Any]], speedups: list[dict[str, Any]],
                 correctness: list[dict[str, Any]], micro: list[dict[str, Any]],
                 optimization: list[dict[str, Any]], optimization_history: dict[str, Any]) -> str:
    spec = manifest["spec"]; p_cfg = spec["performance"]; c_cfg = spec["correctness"]
    failures = [row for row in correctness if row["status"] != "pass"]
    valid_micro = [row for row in micro if row["status"] == "pass"]
    duration_failures = [row for row in valid_micro if not row["duration_gate"]]
    max_rmse = max((float(row["rmse"]) for row in correctness if row["rmse"] != ""), default=math.nan)
    max_abs = max((float(row["max_abs_error"]) for row in correctness if row["max_abs_error"] != ""), default=math.nan)
    two = next((row for row in speedups if row.get("hmx_mode", "").endswith("two-pass") and
                row.get("baseline_mode") == "scna-hvx-fp16-d8"), None)
    hybrid = next((row for row in speedups if row.get("hmx_mode", "").endswith("hybrid") and
                   row.get("baseline_mode") == "scna-hvx-fp16-d8"), None)
    hmx_perf = [row for row in perf if row["mode"].startswith("scna-hmx")]
    worst_unpack = max(hmx_perf, key=lambda row: row["scna_unpack_median_us"]) if hmx_perf else None
    correctness_gate = not failures and len(correctness) == 36
    micro_gate = len(valid_micro) == 90 and not duration_failures
    disasm = json.loads((run_dir / "verification/disassembly_gate.json").read_text())
    hmx_micro = {mode: [row for row in valid_micro if row["mode"] == mode]
                 for mode in ("scna-hmx-fp16-d8-hybrid", "scna-hmx-fp16-d8-two-pass")}
    optimization_lookup = {(row["checkpoint"], row["mode"]): row for row in optimization}
    hybrid_mode = "scna-hmx-fp16-d8-hybrid"
    two_pass_mode = "scna-hmx-fp16-d8-two-pass"
    naive_hybrid = optimization_lookup[("naive-scalar-layout", hybrid_mode)]
    naive_two = optimization_lookup[("naive-scalar-layout", two_pass_mode)]
    fast_hybrid = optimization_lookup[("vector-scatter-gather", hybrid_mode)]
    fast_two = optimization_lookup[("vector-scatter-gather", two_pass_mode)]
    final_hybrid = optimization_lookup[("final-stable", hybrid_mode)]
    final_two = optimization_lookup[("final-stable", two_pass_mode)]
    lines = [
        "# Hexagon v81 HMX 上的 FP16 d8 SCNA 实验报告", "",
        f"> Run ID：`{run_dir.name}`。正式结论只使用该 run 的原始数据；优化过程另引用版本化开发记录 `docs/HMX_SCNA_OPTIMIZATION_HISTORY.json`。缺失项不会插值。", "",
        "## 摘要", "",
        f"本次实验在同一个 v81 DSP 二进制中完成五模式比较。HMX 反汇编门禁为 **{'通过' if disasm['pass'] else '失败'}**；"
        f"FlashAttention 正确性矩阵通过 {len(correctness)-len(failures)}/{len(correctness)} 项，观测最大 RMSE 为 `{max_rmse:.6g}`、"
        f"最大绝对误差为 `{max_abs:.6g}`。微核 ≥50 ms 门禁通过 {len(valid_micro)-len(duration_failures)}/{len(valid_micro)} 个有效样本。", "",
    ]
    if two and hybrid:
        for item in (hybrid, two):
            verdict = "更快" if item["status"] == "faster" else ("更慢" if item["status"] == "slower" else "无显著结论")
            lines.append(f"相对 HVX SCNA，{item['hmx_label']} 的跨形状几何平均 speedup 为 "
                         f"`{item['speedup']:.3f}×`（95% CI `{item['ci_low']:.3f}–{item['ci_high']:.3f}`），结论为 **{verdict}**。")
        lines.append("")
    lines += [
        "## Setup", "",
        "- Model：N/A；这是独立 FlashAttention Kernel 实验。",
        "- Dataset：host 端固定种子生成的合成 Q/K/V 与 full、padding、causal mask 张量。",
        f"- Hardware：{manifest['device']['product_model']}，SoC `{manifest['device']['soc_model']}`，Android {manifest['device']['android']}。",
        f"- Toolchain：Hexagon SDK 6.6.0.0、Tools 19.0.07、`-mv81 -mhmx -mhvx -mhvx-length=128B`。",
        f"- 主矩阵：`qo_len={p_cfg['qo_len']}`、`kv_len={p_cfg['kv_len']}`、heads/KV-heads=`{p_cfg['n_heads']}/{p_cfg['n_kv_heads']}`、"
        f"`head_dim={p_cfg['head_dim']}`、full mask、单 worker、KV pipeline off。",
        "- Baseline：Origin-HVX、EXP-LUT、HVX SCNA FP16 direct d8；候选为 HMX Hybrid 与 HMX Two-pass。",
        f"- 采样：每 shape 5 个 session，每模式 5 次 warmup + 20 次测量；五模式使用循环 Latin-square 顺序；"
        f"配对 session 起始最高 CPU 温度跨度上限 {p_cfg['max_start_temperature_span_c']:.1f}°C。",
        "- DSP power：代码请求 `HAP_DCVS_V2_PERFORMANCE_MODE` 和 `TURBO_L3`；Android sysfs 不暴露 CDSP/HMX 实时频率，因此 manifest 明确记录为不可观测。",
        f"- DSP binary SHA256：`{manifest['artifacts_sha256']['libhtp_ops_skel.so']}`。",
        f"- SCNA 参数 SHA256：`{manifest['scna_params_sha256']}`，与来源文件 SHA256 "
        f"`{manifest['scna_params_source_sha256']}` {'一致' if manifest['scna_params_sha256'] == manifest['scna_params_source_sha256'] else '不一致'}。", "",
        "## HMX 优化过程：Motivation–Solution–Result", "",
        "下述前两个 checkpoint 来自开发阶段单批 1,000 次 vector 测量，没有独立样本和置信区间，只用于解释工程决策；"
        "最终 checkpoint 来自正式 30 个、每个不少于 50 ms 的独立样本。前期数据不会混入最终配对性能统计。", "",
        "### 阶段 0：Naive HMX，先建立正确布局", "",
        "**Motivation。** SCNA 数学形式是 8 个 `ReLU(w_k x+b_k)` 的求和，但 HMX 接收 crouton tile，而 FlashAttention 提供 64-lane HVX vector。"
        "首先需要验证 bias 编码、ReLU shape selector、32 spatial × 8 output-channel 布局和第二遍直接重载，性能不是这一阶段的首要目标。", "",
        "**Solution。** 每次处理 32 个标量，逐元素把输入写入 spatial position 的 channel 0，其余 channel 清零。"
        "第一遍 HMX 写出 channel 0–7。Hybrid 逐元素拆出八个 channel 后用七次 HVX `vadd`；two-pass 把第一遍 crouton 直接作为第二遍 activation，"
        "用 channel 0–7 的单位权重归约到 output channel 0，再逐元素读回。", "",
        f"**Result。** Hybrid 总耗时 `{naive_hybrid['total_ns_per_vector']/1000:.3f} µs/vector`，其中 pack "
        f"`{naive_hybrid['pack_ns_per_vector']/1000:.3f} µs`、unpack `{naive_hybrid['unpack_ns_per_vector']/1000:.3f} µs`；"
        f"二者占总耗时 `{100*(naive_hybrid['pack_ns_per_vector']+naive_hybrid['unpack_ns_per_vector'])/naive_hybrid['total_ns_per_vector']:.1f}%`。"
        f"Two-pass 总耗时 `{naive_two['total_ns_per_vector']/1000:.3f} µs/vector`，pack+unpack 占 "
        f"`{100*(naive_two['pack_ns_per_vector']+naive_two['unpack_ns_per_vector'])/naive_two['total_ns_per_vector']:.1f}%`。"
        "数据直接指出，naive 版本的瓶颈是布局搬运，不是 HMX affine/ReLU。", "",
        "### 阶段 1：向量化 pack/unpack，获得微核局部最优", "",
        "**Motivation。** Naive Hybrid 和 two-pass 分别有 98.7% 与 94.5% 的时间落在 pack/unpack，因此应先消除逐元素访存。", "",
        "**Solution。** pack 改用 `Q6_vscatter_QRMVhV`，以预计算 offset 一次写入 32 个 spatial 的 channel 0；unpack 改用 "
        "`Q6_vgather_ARMVh` 从 crouton 抽取目标 channel；Hybrid 的 8→1 归约保持七次真实 HVX FP16 `vadd`。", "",
        f"**Result。** Hybrid 从 `{naive_hybrid['total_ns_per_vector']/1000:.3f}` 降到 "
        f"`{fast_hybrid['total_ns_per_vector']/1000:.3f} µs/vector`，即 `{naive_hybrid['total_ns_per_vector']/fast_hybrid['total_ns_per_vector']:.2f}×`；"
        f"two-pass 从 `{naive_two['total_ns_per_vector']/1000:.3f}` 降到 `{fast_two['total_ns_per_vector']/1000:.3f} µs/vector`，"
        f"即 `{naive_two['total_ns_per_vector']/fast_two['total_ns_per_vector']:.2f}×`。Hybrid pack 从 5.504 降至 0.068 µs，"
        "unpack 从 14.270 降至 0.546 µs；two-pass pack 从 5.463 降至 0.052 µs，unpack 从 1.890 降至 0.023 µs。", "",
        "### 阶段 2：端到端门禁否决异步 gather 版本", "",
        "**Motivation。** 微核计时只证明短循环局部吞吐，不能证明它能在 FlashAttention 的 HMX QK/PV、VTCM 和在线 softmax 环境中稳定运行。"
        "因此必须先通过 `qo_len=4, kv_len=4096` 的长序列 warmup，再允许进入正式性能实验。", "",
        "**Solution。** 把主矩阵长序列 warmup 设为强制门禁，并保留 30 s host timeout、返回码和当时的实现配置；"
        "只要门禁失败，就停止该版本的正式性能结论，即使微核数字更低。", "",
        f"**Result。** 两种 HMX 模式均未通过该门禁。Two-pass 在 `{optimization_history['long_sequence_failure']['two_pass_observed_us']}` µs 后返回 "
        f"`ret={optimization_history['long_sequence_failure']['two_pass_ret']}`；Hybrid 同样达到 30 s timeout。"
        "开发实验没有单独隔离出某一条指令级根因，因此这里只能得出“异步 gather 版未通过端到端稳定性门禁”，不能声称发现了 HMX 硬件缺陷。", "",
        "### 阶段 3：稳定性优先的最终实现", "",
        "**Motivation。** 验收要求是长序列正确、可重复和可配对测量。一个 0.635 µs 但会在主矩阵超时的微核没有可用性能。", "",
        "**Solution。** 保留稳定且收益明确的 HVX scatter pack；移除 SCNA 输出侧异步 gather，改为按 crouton 索引确定性解包；"
        "在 HMX store 后通过 volatile load 建立完成点，并在 HMX manager setup/reset 显式清零进程内自旋锁。"
        "Hybrid 仍做 HVX 七次加法；two-pass 仍让第一遍 crouton 直接重载到第二遍。", "",
        f"**Result。** `kv_len=4096` 初步复测恢复为 `ret=0`：Hybrid `{optimization_history['stable_preliminary'][0]['host_elapsed_us']} µs`，"
        f"two-pass `{optimization_history['stable_preliminary'][1]['host_elapsed_us']} µs`。正式微核中 Hybrid 为 "
        f"`{final_hybrid['total_ns_per_vector']/1000:.3f} µs/vector`，two-pass 为 `{final_two['total_ns_per_vector']/1000:.3f} µs/vector`。"
        f"相对 naive，最终 Hybrid 为 `{naive_hybrid['total_ns_per_vector']/final_hybrid['total_ns_per_vector']:.2f}×`，two-pass 为 "
        f"`{naive_two['total_ns_per_vector']/final_two['total_ns_per_vector']:.2f}×`；相对未通过门禁的微核局部最优，最终版本分别慢 "
        f"`{final_hybrid['total_ns_per_vector']/fast_hybrid['total_ns_per_vector']:.2f}×` 和 `{final_two['total_ns_per_vector']/fast_two['total_ns_per_vector']:.2f}×`。"
        "这是为稳定性支付的可量化代价。正式配对主矩阵进一步表明，Hybrid 与 two-pass 相对 HVX SCNA 的跨形状 speedup 分别为 "
        f"`{hybrid['speedup']:.4f}×`（95% CI `{hybrid['ci_low']:.4f}–{hybrid['ci_high']:.4f}`）和 "
        f"`{two['speedup']:.4f}×`（95% CI `{two['ci_low']:.4f}–{two['ci_high']:.4f}`）。"
        "因此最终成果是“完成真实 HMX 映射并通过稳定性/正确性验收”，不是“取得性能提升”。", "",
        "![HMX SCNA 优化演进](../" + run_dir.relative_to(ROOT).as_posix() + "/figures/optimization_evolution.png)", "",
        f"图中阴影柱表示曾达到微核局部最优、但被长序列门禁否决的 scatter+gather 版本。最终 two-pass 比最终 Hybrid 快 "
        f"`{final_hybrid['total_ns_per_vector']/final_two['total_ns_per_vector']:.2f}×`。两者 affine/ReLU 中位数接近，分别为 "
        f"`{final_hybrid['affine_ns_per_vector']:.2f}` 与 `{final_two['affine_ns_per_vector']:.2f} ns/vector`；差异主要来自 unpack："
        f"Hybrid `{final_hybrid['unpack_ns_per_vector']:.2f} ns`，two-pass `{final_two['unpack_ns_per_vector']:.2f} ns`，后者减少 "
        f"`{100*(1-final_two['unpack_ns_per_vector']/final_hybrid['unpack_ns_per_vector']):.1f}%`。这说明第二遍 HMX 虽增加 reduction，"
        "但显著降低了需要恢复的输出通道数。", "",
        "## 数值门禁", "",
        "![正确性误差与门槛](../" + run_dir.relative_to(ROOT).as_posix() + "/figures/correctness.png)", "",
        f"图中包含 full、padding、causal，`kv_len=4093/4096` 与 `head_dim=64/128`。"
        f"最大 RMSE `{max_rmse:.6g}` 低于 `{c_cfg['rmse_limit']}`，最大绝对误差 `{max_abs:.6g}` 低于 `{c_cfg['max_abs_error_limit']}`；"
        f"失败 {len(failures)} 项。对需要尾部清零的 case，原始 `FIG8_NUMERIC` 记录用于检查最后 padding/mask lane。", "",
        "微测试把“SCNA 对真实 exp2 的逼近误差”和“HMX 对 HVX SCNA 的迁移误差”分开记录；前者不作为迁移失败。"
        f"Hybrid 对 HVX 的最大迁移 RMSE/最大绝对误差为 `"
        f"{max(float(row['implementation_rmse']) for row in hmx_micro['scna-hmx-fp16-d8-hybrid']):.6g}`/`"
        f"{max(float(row['implementation_max_abs_error']) for row in hmx_micro['scna-hmx-fp16-d8-hybrid']):.6g}`；"
        f"two-pass 为 `{max(float(row['implementation_rmse']) for row in hmx_micro['scna-hmx-fp16-d8-two-pass']):.6g}`/`"
        f"{max(float(row['implementation_max_abs_error']) for row in hmx_micro['scna-hmx-fp16-d8-two-pass']):.6g}`，"
        "均按 0.001/0.002 门槛判定。", "",
        "## 主性能结果", "",
        "![五模式延迟折线图](../" + run_dir.relative_to(ROOT).as_posix() + "/figures/latency.png)", "",
    ]
    for q in sorted({row["qo_len"] for row in perf}):
        group = {row["mode"]: row for row in perf if row["qo_len"] == q}
        if len(group) == 5:
            fastest = min(group.values(), key=lambda row: row["median_us"])
            slowest = max(group.values(), key=lambda row: row["median_us"])
            lines.append(f"`qo_len={q}` 时，最低中位延迟为 {fastest['label']} `{fastest['median_us']:.1f} µs`，"
                         f"最高为 {slowest['label']} `{slowest['median_us']:.1f} µs`；误差棒是五个 session 中位数的 bootstrap 95% CI。")
    lines += ["", "![HMX 相对 baseline 的 speedup](../" + run_dir.relative_to(ROOT).as_posix() + "/figures/speedup.png)", ""]
    for row in speedups:
        if row["status"] == "missing":
            lines.append(f"{MODE_LABEL[row['hmx_mode']]} vs {MODE_LABEL[row['baseline_mode']]}：数据缺失。")
        else:
            verdict = "更快" if row["ci_low"] > 1 else ("更慢" if row["ci_high"] < 1 else "无显著结论")
            lines.append(f"{row['hmx_label']} vs {row['baseline_label']}：`{row['speedup']:.3f}×`，95% CI "
                         f"`{row['ci_low']:.3f}–{row['ci_high']:.3f}`，因此标记为“{verdict}”。")
    lines += ["", "## 阶段消融", "",
              "![Hybrid 与 two-pass 阶段消融](../" + run_dir.relative_to(ROOT).as_posix() + "/figures/stages.png)", ""]
    if worst_unpack:
        total = sum(worst_unpack[stage + "_median_us"] for stage in STAGES)
        share = 100 * worst_unpack["scna_unpack_median_us"] / total if total else math.nan
        lines.append(f"最高解包开销出现在 `{worst_unpack['label']}, qo_len={worst_unpack['qo_len']}`："
                     f"unpack `{worst_unpack['scna_unpack_median_us']:.1f} µs`，占四个显式 SCNA 阶段 `{share:.1f}%`。"
                     "该比例直接量化 crouton→HVX 布局恢复成本。")
    lines += ["", "![SCNA 微核阶段开销](../" + run_dir.relative_to(ROOT).as_posix() + "/figures/microkernel.png)", ""]
    for mode in MODE_LABEL:
        group = [row for row in valid_micro if row["mode"] == mode]
        if group:
            median_total = statistics.median(float(row["total_ns_per_vector"]) for row in group)
            lines.append(f"{MODE_LABEL[mode]}：30 个独立样本的每 64-lane vector 中位总耗时 `{median_total:.2f} ns`。")
    for mode in ("scna-hmx-fp16-d8-hybrid", "scna-hmx-fp16-d8-two-pass"):
        group = hmx_micro[mode]
        medians = {key: statistics.median(float(row[key]) for row in group) for key in
                   ("pack_ns_per_vector", "affine_ns_per_vector", "reduction_ns_per_vector", "unpack_ns_per_vector")}
        explicit = sum(medians.values())
        lines.append(f"{MODE_LABEL[mode]} 的阶段组成中，unpack 为 `{medians['unpack_ns_per_vector']:.2f} ns`/vector，"
                     f"占四个显式阶段 `{100 * medians['unpack_ns_per_vector'] / explicit:.1f}%`；"
                     f"HMX affine/ReLU 占 `{100 * medians['affine_ns_per_vector'] / explicit:.1f}%`。")
    lines += ["", "## Unexpected Results 与验证计划", ""]
    if worst_unpack:
        lines += [
            f"1. 布局恢复是首要异常：最差 case 的 unpack 占 `{share:.1f}%`，而非 HMX affine/ReLU。验证计划：实现 `vdeal/vshuff` 原生解包并保持同一参数、同一二进制矩阵复测。",
            "2. HMX 低占用：每次只使用 32×1 输入映射到 8 个输出通道，矩阵阵列利用率受 d8 与单输入通道限制。验证计划：批量融合多个 softmax vector 后复测 HMX 阶段与总延迟。",
            "3. Two-pass 增加 intermediate→VTCM→activation 的第二遍往返。验证计划：对比保持第一遍 crouton 驻留、融合第二遍权重加载的版本，并分别报告 reduction 与 unpack。",
        ]
    if failures:
        lines.append(f"4. 有 {len(failures)} 个正确性 case 失败；应先按 mask/tail、bias/selector、head_dim 分层复现，性能数据不得用于成功结论。")
    lines += ["", "## Limitations", "",
              "- 仅验证一款 SM8750P SoC；结论不能外推到其他 HMX 实现或固件。",
              "- 仅覆盖 FP16、d8、exp2 与输入 `[-256, 0]`。",
              "- 仅是 Kernel 级合成数据，无真实模型与数据集。",
              "- 固定无 KV DMA/VTCM pipeline；不能代表 pipeline 打开后的相互作用。",
              "- 未验证模型级精度、困惑度和端到端 token 延迟。",
              "- Android sysfs 无法读取 CDSP/HMX 实时频率；实验只能证实已请求 performance/TURBO_L3，不能证实每个样本的实际 DSP 频率。", "",
              "## 验收状态", "",
              f"- HMX named-symbol 反汇编证据：{'通过' if disasm['pass'] else '失败'}。",
              f"- 全部正确性门禁：{'通过' if correctness_gate else '失败或不完整'}（{len(correctness)-len(failures)}/{len(correctness)}）。",
              f"- 微核 30×3 且每样本 ≥50 ms：{'通过' if micro_gate else '失败或不完整'}。",
              "- 五模式同一二进制：由每个 session 前远端 SHA256 校验，发现不一致时采集器立即拒绝。",
              "- 确定性重建：`tools/analyze_scna_hmx_v81.py` 以固定 seed=8108、10,000 次 bootstrap 从 JSONL 重建 CSV、图和本文。", "",
              "## 原始证据", "",
              f"- Manifest：`results/{run_dir.name}/manifest.json`",
              f"- 原始 JSONL/log：`results/{run_dir.name}/raw/`",
              f"- 汇总 CSV/JSON：`results/{run_dir.name}/summary/`",
              "- 优化 checkpoint 数据与证据边界：`docs/HMX_SCNA_OPTIMIZATION_HISTORY.json`",
              f"- HMX 反汇编：`results/{run_dir.name}/verification/scna_hmx_symbols.disasm.txt`", "",
              "## AI 辅助声明", "",
              "AI 用于代码与实验脚本辅助、机械化数据解析、绘图和报告组织；研究问题、实验设计、设备执行、原始数据验证与全部结论必须由研究者复核。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(); manifest = json.loads((run_dir / "manifest.json").read_text())
    spec = manifest["spec"]; boot = int(spec["statistics"]["bootstrap_samples"]); seed = int(spec["seed"])
    perf_raw = read_jsonl(run_dir / "raw/performance.jsonl")
    correctness_raw = read_jsonl(run_dir / "raw/correctness.jsonl")
    micro_raw = read_jsonl(run_dir / "raw/microkernel.jsonl")
    optimization_history = json.loads(OPTIMIZATION_HISTORY_PATH.read_text(encoding="utf-8"))
    session_rows, perf = summarize_performance(perf_raw, boot, seed)
    speedups = paired_speedups(session_rows, boot, seed)
    correctness = summarize_correctness(correctness_raw); micro = summarize_micro(micro_raw)
    optimization = summarize_optimization(optimization_history, micro)
    summary = run_dir / "summary"; figures = run_dir / "figures"
    write_csv(summary / "performance_sessions.csv", session_rows)
    write_csv(summary / "performance_summary.csv", perf)
    write_csv(summary / "speedup_summary.csv", speedups)
    write_csv(summary / "correctness.csv", correctness)
    write_csv(summary / "microkernel.csv", micro)
    write_csv(summary / "optimization_history.csv", optimization)
    combined = {"performance": perf, "speedups": speedups, "correctness": correctness,
                "microkernel": micro, "optimization": optimization}
    (summary / "summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "scna-hmx-v81"})
    plot_latency(perf, figures / "latency"); plot_speedup(speedups, figures / "speedup")
    plot_stages(perf, figures / "stages")
    plot_correctness(correctness, figures / "correctness", float(spec["correctness"]["rmse_limit"]),
                     float(spec["correctness"]["max_abs_error_limit"]))
    plot_micro(micro, figures / "microkernel")
    plot_optimization(optimization, figures / "optimization_evolution")
    report = build_report(run_dir, manifest, perf, speedups, correctness, micro,
                          optimization, optimization_history)
    report_path = run_dir / "SCNA_HMX_FP16_D8_V81_REPORT.md"; report_path.write_text(report, encoding="utf-8")
    if not args.no_publish:
        public = ROOT / "reports/SCNA_HMX_FP16_D8_V81_REPORT.md"; public.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_path, public)
    digest = hashlib.sha256(report.encode()).hexdigest()
    print(json.dumps({"report": str(report_path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
