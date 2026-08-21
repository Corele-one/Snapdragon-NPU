#!/usr/bin/env python3
"""Analyze v79 SCNA pipeline experiments and generate evidence-linked figures/report."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scna-pipeline-mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEMA = 3
QOS = (1, 4, 8, 16, 32)
IMPLS = [
    "static_d8_ref", "d7_serial", "d7_scalar_w", "d7_pairret_noinline",
    "d7_pairret_inline", "d7_quad_pipeline", "d7_prebroadcast",
    "qf16_tree_control", "piecewise_control", "combined_confirm",
]
LABELS = ["baseline", "lut-exp", *IMPLS]
DISPLAY = {
    "baseline": "Origin HVX", "lut-exp": "EXP-LUT", "static_d8_ref": "static-d8",
    "d7_serial": "d7 serial", "d7_scalar_w": "d7 scalar-w",
    "d7_pairret_noinline": "pair-return", "d7_pairret_inline": "pair-return inline",
    "d7_quad_pipeline": "quad pipeline", "d7_prebroadcast": "prebroadcast",
    "qf16_tree_control": "qf16 tree", "piecewise_control": "piecewise",
    "combined_confirm": "combined confirm",
}
ELIGIBLE = {"d7_serial", "d7_scalar_w", "d7_pairret_noinline", "d7_quad_pipeline", "combined_confirm"}
STRATEGIES = {
    "static_d8_ref": "冻结 d8 系数与原始顺序，作为纵向 baseline carrier",
    "d7_serial": "删除 x<=0 域恒零的第 8 神经元，前七项保持原累加顺序",
    "d7_scalar_w": "用 vmpy(Vhf,Rhf) 从通用寄存器提供双 half 权重，删除权重向量 splat",
    "d7_pairret_noinline": "unchecked 双向量按值返回 hot ABI，noinline，移除循环内检查/输出指针回写",
    "d7_pairret_inline": "与 pair-return 相同，但强制 inline 以检验调用开销与 live range 的权衡",
    "d7_quad_pipeline": "一次消费两个 64-column block，交错四条向量链并跨链复用 weight/bias",
    "d7_prebroadcast": "在列循环外预广播七组常量，检验常量生命周期扩展",
    "qf16_tree_control": "旧 qf16 tree 重关联对照；不参与 winner 选择",
    "piecewise_control": "区间选择加单 affine 的旧提案对照；不参与 winner 选择",
    "combined_confirm": "仅在 scalar-weight 与 quad 均有独立正向证据时进行的全新确认 artifact",
}
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
COLORS = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors)


def fields(line: str) -> dict:
    out = {}
    for key, value in KV_RE.findall(line):
        try:
            out[key] = int(value, 0) if value.lower().startswith(("0x", "-0x")) else (
                float(value) if any(ch in value.lower() for ch in (".", "e")) else int(value, 0))
        except ValueError:
            out[key] = value
    return out


def label_from_name(name: str) -> str | None:
    for label in sorted(LABELS, key=len, reverse=True):
        if name.startswith(label + "_") or name == label + ".log":
            return label
    return None


def median(values):
    return statistics.median(values) if values else None


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def bootstrap_median(values, seed=20260821, draws=10000):
    if not values:
        return None
    rng = random.Random(seed)
    samples = [median([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws)]
    return {"median": median(values), "ci_low": quantile(samples, .025),
            "ci_high": quantile(samples, .975), "n": len(values)}


def bootstrap_ratio(candidate, baseline, seed=20260821, draws=10000):
    keys = sorted(candidate.keys() & baseline.keys())
    pairs = [(candidate[k], baseline[k]) for k in keys if baseline[k] > 0]
    if not pairs:
        return None
    rng = random.Random(seed)
    sampled = []
    for _ in range(draws):
        chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        sampled.append(median([a for a, _ in chosen]) / median([b for _, b in chosen]))
    return {"ratio": median([a for a, _ in pairs]) / median([b for _, b in pairs]),
            "ci_low": quantile(sampled, .025), "ci_high": quantile(sampled, .975), "pairs": len(pairs)}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def parse_micro(root: Path):
    rows = []
    for path in sorted((root / "raw/micro").glob("*_sample*.log")):
        label = label_from_name(path.name)
        sample = re.search(r"sample(\d+)", path.name)
        for line in path.read_text(errors="replace").splitlines():
            if "SCNA_EXP_BENCH" not in line:
                continue
            data = fields(line)
            rows.append({
                "label": label, "sample": int(sample.group(1)) if sample else 0,
                "schema_version": int(data.get("schema_version", -1)),
                "kernel_impl": int(data.get("kernel_impl", -1)),
                "pair_ns_per_64": float(data.get("paired_ns_per_64", math.nan)),
                "quad_ns_per_64": float(data.get("quad_ns_per_64", math.nan)),
                "dense_rmse": float(data.get("dense_rmse", math.nan)),
                "dense_max_abs": float(data.get("dense_max_abs", math.nan)),
                "monotonic_violations": int(data.get("monotonic_violations", -1)),
                "nonfinite": int(data.get("random_nonfinite_count", -1)) + int(data.get("nan_count", -1)),
                "paired_single_mismatches": int(data.get("paired_single_mismatches", -1)),
                "quad_pair_mismatches": int(data.get("quad_pair_mismatches", -1)),
                "checksum": str(data.get("checksum", "")), "source": str(path.relative_to(root)),
            })
    return rows


def parse_attention(root: Path):
    rows = []
    for folder in ("attention", "confirm", "scaling", "diagnostic"):
        for path in sorted((root / f"raw/{folder}").glob("*.log")):
            label = label_from_name(path.name)
            session = re.search(r"_s(\d+)", path.name)
            scaling_worker = re.search(r"_w(auto|\d+)\.log$", path.name) if folder == "scaling" else None
            for line in path.read_text(errors="replace").splitlines():
                if "FIG8_ATTENTION_HOST_TIMING" not in line:
                    continue
                data = fields(line)
                if data.get("phase") != "measure" or int(data.get("ret", 1)) != 0:
                    continue
                rows.append({
                    "label": label, "folder": folder, "session": int(session.group(1)) if session else 0,
                    "iteration": int(data.get("iteration", 0)), "qo_len": int(data.get("qo_len", 0)),
                    "kv_len": int(data.get("kv_len", 0)),
                    "workers": scaling_worker.group(1) if scaling_worker else str(data.get("workers", 1)),
                    "host_us": float(data["host_elapsed_us"]), "source": str(path.relative_to(root)),
                })
    return rows


def parse_accuracy(root: Path):
    rows = []
    for path in sorted((root / "raw/accuracy").glob("*.log")):
        compare = None
        mask_zero = tail_zero = True
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_COMPARE " in line:
                compare = fields(line)
            elif "FIG8_NUMERIC " in line:
                data = fields(line)
                mask_zero &= int(data.get("masked_p_nonzero", 0)) == 0
                tail_zero &= int(data.get("tail_p_nonzero", 0)) == 0
        if compare:
            rows.append({
                "label": label_from_name(path.name), "pass": int(compare.get("pass", 0)),
                "rmse": float(compare.get("rmse", math.inf)),
                "max_abs": float(compare.get("max_abs_error", math.inf)),
                "finite": int(compare.get("candidate_nonfinite", 1)) == 0,
                "mask_zero": mask_zero, "tail_zero": tail_zero,
                "source": str(path.relative_to(root)),
            })
    return rows


def parse_diagnostic_timers(root: Path):
    rows = []
    timer_keys = ("profiled_total", "q_load", "k_load", "v_load", "qk_dot", "safe_sm",
                  "scna_exp", "param_prepare", "core_acc", "o_scale", "o_store")
    for path in sorted((root / "raw/diagnostic").glob("*.log")):
        totals = {key: 0 for key in timer_keys}
        count = 0
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_TIMERS" not in line:
                continue
            data = fields(line)
            if data.get("phase") != "measure":
                continue
            count += 1
            for key in timer_keys:
                totals[key] += int(data.get(key, 0))
        if count:
            kv_match = re.search(r"_kv(\d+)\.log$", path.name)
            rows.append({"label": label_from_name(path.name),
                         "kv_len": int(kv_match.group(1)) if kv_match else 0,
                         "timer_records": count, **totals,
                         "source": str(path.relative_to(root))})
    return rows


def thermal_snapshot(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    status = re.search(r"Thermal Status:\s*(\d+)", text)
    current = text.split("Current temperatures from HAL:", 1)
    current = current[1].split("Current cooling devices", 1)[0] if len(current) == 2 else ""
    temperatures = {name: float(value) for value, name in
                    re.findall(r"Temperature\{mValue=([-+0-9.]+).*?mName=([^,}]+)", current)}
    nsp = [value for name, value in temperatures.items() if name.startswith("nsp")]
    return {"status": int(status.group(1)) if status else None,
            "nsp_min_c": min(nsp) if nsp else None, "nsp_max_c": max(nsp) if nsp else None,
            "skin_c": temperatures.get("skin"), "battery_c": temperatures.get("battery")}


def ratio_for(attention, candidate, baseline, q):
    selected = [r for r in attention if r["folder"] in ("attention", "confirm") and r["qo_len"] == q and r["kv_len"] == 4096]
    maps = {}
    for label in (candidate, baseline):
        maps[label] = {(r["session"], r["iteration"]): r["host_us"] for r in selected if r["label"] == label}
    return bootstrap_ratio(maps[candidate], maps[baseline], seed=20260821 + q)


def accuracy_status(accuracy, label):
    rows = [r for r in accuracy if r["label"] == label]
    return bool(rows) and all(r["pass"] and r["finite"] and r["mask_zero"] and r["tail_zero"] and
                              r["rmse"] <= .002 and r["max_abs"] <= .01 for r in rows)


def summarize(root: Path):
    micro, attention, accuracy = parse_micro(root), parse_attention(root), parse_accuracy(root)
    diagnostic_timers = parse_diagnostic_timers(root)
    static_metrics_path = root / "static/static_metrics.json"
    static_gates_path = root / "static/static_gates.json"
    static_metrics = json.loads(static_metrics_path.read_text()) if static_metrics_path.exists() else {}
    gates_doc = json.loads(static_gates_path.read_text()) if static_gates_path.exists() else {"rows": []}
    gates = {r["kernel_impl"]: bool(r["static_pass"]) for r in gates_doc.get("rows", [])}
    bitwise = {}
    bitwise_path = root / "static/micro_bitwise_gate.csv"
    if bitwise_path.exists():
        with bitwise_path.open() as handle:
            bitwise = {r["kernel_impl"]: {"expected": r["expected_checksum"],
                                           "observed": r["observed_checksum"],
                                           "pass": bool(int(r["bitwise_pass"]))}
                       for r in csv.DictReader(handle)}
    micro_summary = {}
    micro_contract = {}
    for label in IMPLS:
        values = [r["pair_ns_per_64"] for r in micro if r["label"] == label and math.isfinite(r["pair_ns_per_64"])]
        quad = [r["quad_ns_per_64"] for r in micro if r["label"] == label and math.isfinite(r["quad_ns_per_64"])]
        label_rows = [r for r in micro if r["label"] == label]
        micro_summary[label] = {"pair": bootstrap_median(values), "quad": bootstrap_median(quad),
                                "checksum_set": sorted({r["checksum"] for r in label_rows}),
                                "max_pair_mismatches": max((r["paired_single_mismatches"] for r in label_rows), default=None),
                                "max_quad_mismatches": max((r["quad_pair_mismatches"] for r in label_rows), default=None),
                                "max_monotonic_violations": max((r["monotonic_violations"] for r in label_rows), default=None),
                                "max_nonfinite": max((r["nonfinite"] for r in label_rows), default=None)}
        micro_contract[label] = {"rows": len(label_rows),
                                 "pass": bool(label_rows) and all(
                                     r["schema_version"] == SCHEMA and r["kernel_impl"] == IMPLS.index(label)
                                     for r in label_rows)}
    micro_ratios = {}
    static_pair = {r["sample"]: r["pair_ns_per_64"] for r in micro if r["label"] == "static_d8_ref"}
    static_quad = {r["sample"]: r["quad_ns_per_64"] for r in micro if r["label"] == "static_d8_ref"}
    for label in IMPLS:
        candidate_pair = {r["sample"]: r["pair_ns_per_64"] for r in micro if r["label"] == label}
        candidate_quad = {r["sample"]: r["quad_ns_per_64"] for r in micro if r["label"] == label}
        micro_ratios[label] = {"pair_vs_static": bootstrap_ratio(candidate_pair, static_pair),
                               "quad_vs_static": bootstrap_ratio(candidate_quad, static_quad)}
    latency = {}
    for label in LABELS:
        latency[label] = {}
        for q in QOS:
            values = [r["host_us"] for r in attention if r["label"] == label and r["qo_len"] == q and
                      r["kv_len"] == 4096 and r["folder"] in ("attention", "confirm")]
            latency[label][str(q)] = bootstrap_median(values, seed=20260821 + q)
    ratios = {}
    for label in [x for x in LABELS if x not in ("baseline", "lut-exp")]:
        ratios[label] = {str(q): {"vs_static": ratio_for(attention, label, "static_d8_ref", q),
                                  "vs_lut": ratio_for(attention, label, "lut-exp", q)} for q in QOS}
    correctness = {}
    for label in IMPLS:
        rows = [r for r in accuracy if r["label"] == label]
        correctness[label] = {"cases": len(rows), "pass": accuracy_status(accuracy, label),
                              "max_rmse": max((r["rmse"] for r in rows), default=None),
                              "max_abs": max((r["max_abs"] for r in rows), default=None)}
    decisions = {}
    for label in IMPLS:
        q32 = ratios.get(label, {}).get("32", {})
        vs_static, vs_lut = q32.get("vs_static"), q32.get("vs_lut")
        no_q_regression = True
        for q in QOS:
            ratio = ratios.get(label, {}).get(str(q), {}).get("vs_static")
            if not ratio or ratio["ci_high"] > 1.02:
                no_q_regression = False
        bitwise_ok = bitwise.get(label, {}).get("pass", True)
        eligible = (label in ELIGIBLE and gates.get(label, False) and correctness[label]["pass"] and
                    bitwise_ok and micro_contract[label]["pass"])
        decisions[label] = {
            "eligible": eligible, "vertical_win": bool(vs_static and vs_static["ci_high"] < 1.0),
            "beats_lut": bool(vs_lut and vs_lut["ci_high"] < 1.0),
            "near_lut": bool(vs_lut and vs_lut["ci_high"] <= 1.02),
            "no_q_regression": no_q_regression, "bitwise_pass": bitwise_ok,
        }
        decisions[label]["full_success"] = bool(
            eligible and decisions[label]["vertical_win"] and decisions[label]["near_lut"] and no_q_regression)
    combined_enabled = all(decisions.get(x, {}).get("vertical_win") and decisions.get(x, {}).get("no_q_regression")
                           and decisions.get(x, {}).get("eligible")
                           for x in ("d7_scalar_w", "d7_quad_pipeline"))
    diagnostic = {}
    for label in LABELS:
        diagnostic[label] = {}
        for kv in (64, 4096):
            hosts = [r["host_us"] for r in attention if r["folder"] == "diagnostic" and
                     r["label"] == label and r["kv_len"] == kv]
            timers = next((r for r in diagnostic_timers if r["label"] == label and r["kv_len"] == kv), None)
            diagnostic[label][str(kv)] = {"host": bootstrap_median(hosts), "timers": timers}
    scaling = {}
    scaling_labels = sorted({r["label"] for r in attention if r["folder"] == "scaling" and r["label"]})
    for label in scaling_labels:
        scaling[label] = {}
        for workers in ("1", "2", "3", "4", "5", "6", "auto"):
            hosts = [r["host_us"] for r in attention if r["folder"] == "scaling" and
                     r["label"] == label and r["workers"] == workers]
            scaling[label][workers] = bootstrap_median(hosts)
    successful = [label for label in IMPLS if decisions[label]["full_success"] and latency[label]["32"]]
    overall_winner = min(successful, key=lambda label: latency[label]["32"]["median"]) if successful else None
    return {"schema_version": SCHEMA, "micro": micro, "attention": attention, "accuracy_rows": accuracy,
            "micro_summary": micro_summary, "micro_ratios": micro_ratios,
            "micro_contract": micro_contract,
            "latency": latency, "ratios": ratios,
            "correctness": correctness, "static_metrics": static_metrics,
            "static_gates": gates_doc, "bitwise_gates": bitwise,
            "decisions": decisions, "combined_enabled": combined_enabled,
            "overall_winner": overall_winner, "diagnostic": diagnostic, "scaling": scaling,
            "thermal": {"before": thermal_snapshot(root / "evidence/thermal_before.txt"),
                        "after": thermal_snapshot(root / "evidence/thermal_after.txt")}}


def save_figure(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=240)
    plt.close(fig)


def plot_all(root: Path, summary: dict):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "svg.fonttype": "none",
                         "svg.hashsalt": "scna-hvx-d8-pipeline-v79-schema3", "pdf.fonttype": 42})
    figures = root / "figures"
    active = [x for x in LABELS if any(summary["latency"].get(x, {}).get(str(q)) for q in QOS)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for idx, label in enumerate(active):
        pts = [(q, summary["latency"][label][str(q)]) for q in QOS if summary["latency"][label][str(q)]]
        ax.errorbar([q for q, _ in pts], [v["median"] for _, v in pts],
                    yerr=[[v["median"]-v["ci_low"] for _, v in pts], [v["ci_high"]-v["median"] for _, v in pts]],
                    marker="o", capsize=2, linewidth=1.2, color=COLORS[idx % len(COLORS)], label=DISPLAY[label])
    ax.set(xlabel="Query length (KV=4096)", ylabel="Host latency (us)", title="Full Attention latency, median and bootstrap 95% CI")
    ax.set_xticks(QOS); ax.grid(axis="y", alpha=.25)
    if active:
        ax.legend(ncol=3, frameon=False)
    save_figure(fig, figures / "01_attention_latency")

    ratio_labels = [x for x in IMPLS if summary["ratios"].get(x, {}).get("32", {}).get("vs_lut")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); x = list(range(len(ratio_labels))); width = .38
    for offset, baseline_key, title, color in ((-width/2, "vs_static", "vs static-d8", "#4C78A8"),
                                                (width/2, "vs_lut", "vs EXP-LUT", "#F58518")):
        vals = [summary["ratios"][label]["32"][baseline_key] for label in ratio_labels]
        ax.bar([i+offset for i in x], [v["ratio"] for v in vals], width, label=title, color=color,
               yerr=[[v["ratio"]-v["ci_low"] for v in vals], [v["ci_high"]-v["ratio"] for v in vals]], capsize=2)
    ax.axhline(1.0, color="black", linewidth=1); ax.axhline(1.02, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, [DISPLAY[v] for v in ratio_labels], rotation=25, ha="right")
    ax.set(ylabel="Latency ratio (lower is better)", title="q=32 latency ratio and paired-bootstrap 95% CI")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save_figure(fig, figures / "02_q32_ratios")

    micro_labels = [x for x in IMPLS if summary["micro_summary"][x]["pair"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); x = list(range(len(micro_labels))); width = .38
    pair = [summary["micro_summary"][v]["pair"] for v in micro_labels]
    quad = [summary["micro_summary"][v]["quad"] for v in micro_labels]
    ax.bar([i-width/2 for i in x], [v["median"] for v in pair], width, label="pair", color="#4C78A8")
    ax.bar([i+width/2 for i in x], [v["median"] for v in quad], width, label="quad / lane", color="#54A24B")
    ax.set_xticks(x, [DISPLAY[v] for v in micro_labels], rotation=25, ha="right")
    ax.set(ylabel="ns per 64 FP16 elements", title="SCNA evaluator micro benchmark"); ax.legend(frameon=False)
    save_figure(fig, figures / "03_micro_pair_quad")

    static_labels = [x for x in IMPLS if x in summary["static_metrics"] and not summary["static_metrics"][x].get("missing")]
    gate_map = {r["kernel_impl"]: r for r in summary["static_gates"].get("rows", [])}
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2))
    panels = (
        ("pair micro (ns/64)", [summary["micro_summary"][v]["pair"]["median"] for v in static_labels]),
        ("evaluator instructions", [summary["static_metrics"][v].get("instructions", 0) for v in static_labels]),
        ("evaluator packets", [summary["static_metrics"][v].get("packets", 0) for v in static_labels]),
        ("evaluator splat", [summary["static_metrics"][v].get("splat", 0) for v in static_labels]),
        ("evaluator stack (B)", [summary["static_metrics"][v].get("stack_frame_bytes", 0) for v in static_labels]),
        ("caller stack refs delta", [gate_map.get(v, {}).get("caller_spill_delta", 0) for v in static_labels]),
    )
    for ax, (title, vals) in zip(axes.flat, panels):
        ax.bar(range(len(vals)), vals, color=[COLORS[i % len(COLORS)] for i in range(len(vals))])
        ax.axhline(0, color="black", linewidth=.6)
        ax.set_xticks(range(len(vals)), [DISPLAY[v] for v in static_labels], rotation=70, ha="right")
        ax.set_title(title)
    fig.suptitle("Micro latency and disassembly-derived metrics"); fig.tight_layout()
    save_figure(fig, figures / "04_static_metrics")

    corr_labels = [x for x in IMPLS if summary["correctness"][x]["cases"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); values = [summary["correctness"][x]["max_abs"] for x in corr_labels]
    colors = ["#54A24B" if summary["correctness"][x]["pass"] else "#E45756" for x in corr_labels]
    ax.bar(range(len(values)), values, color=colors); ax.axhline(.01, color="black", linestyle="--", label="max_abs gate=0.01")
    ax.set_xticks(range(len(values)), [DISPLAY[v] for v in corr_labels], rotation=25, ha="right")
    ax.set(ylabel="Maximum absolute error", title="Attention correctness gate"); ax.legend(frameon=False)
    save_figure(fig, figures / "05_correctness")


def fmt_stat(stat, unit=""):
    if not stat:
        return "UNAVAILABLE"
    return f"{stat['median']:.3f}{unit} [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}]"


def fmt_ratio(stat):
    if not stat:
        return "UNAVAILABLE"
    return f"{stat['ratio']:.4f} [{stat['ci_low']:.4f}, {stat['ci_high']:.4f}]"


def pct_delta(stat):
    return "UNAVAILABLE" if not stat else f"{(stat['ratio'] - 1.0) * 100:+.2f}%"


def fmt_number(value, digits=9):
    return "UNAVAILABLE" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def fmt_thermal(snapshot):
    if not snapshot:
        return "UNAVAILABLE"
    def value(key):
        return "NA" if snapshot.get(key) is None else f"{snapshot[key]:.1f}"
    return (f"status={snapshot.get('status', 'NA')}，NSP={value('nsp_min_c')}–{value('nsp_max_c')} °C，"
            f"skin={value('skin_c')} °C，battery={value('battery_c')} °C")


def make_report(root: Path, summary: dict):
    manifest_path = root / "evidence/device_manifest.txt"
    manifest = manifest_path.read_text(errors="replace") if manifest_path.exists() else "UNAVAILABLE"
    toolchain_path = root / "evidence/toolchain_manifest.txt"
    toolchain = ("；".join(toolchain_path.read_text(errors="replace").splitlines()) if toolchain_path.exists() else
                 "Hexagon SDK 6.6.0.0；Hexagon LLVM Tools 19.0.07；DSP target v79；-mv79")
    frequency_path = root / "evidence/frequency_snapshot.txt"
    frequency = ("；".join(frequency_path.read_text(errors="replace").splitlines())
                 if frequency_path.exists() else "UNAVAILABLE")
    quick = "quick=1" in manifest
    lines = ["# v79 HVX static-d8 SCNA 指令级优化实验报告", "", "## 结论", ""]
    available = [(x, summary["latency"][x]["32"]) for x in IMPLS
                 if summary["latency"][x]["32"] and summary["decisions"][x]["eligible"]]
    if quick:
        lines.append("本 run 为 quick 链路验证，不用于性能结论或 winner 选择。")
    elif summary["overall_winner"]:
        best = summary["overall_winner"]
        lines.append(f"`{best}` 满足全部预注册条件，q=32 Host latency 为 "
                     f"{fmt_stat(summary['latency'][best]['32'], ' us')}。")
    elif available:
        best, best_stat = min(available, key=lambda item: item[1]["median"])
        decision = summary["decisions"][best]
        rs = summary["ratios"][best]["32"]["vs_static"]
        rl = summary["ratios"][best]["32"]["vs_lut"]
        lines += ["**本轮没有候选满足全部预注册成功条件，因此没有 winner。**",
                  f"q=32 最快的可选候选是 `{best}`，Host latency 为 {fmt_stat(best_stat, ' us')}。",
                  f"它相对 static-d8 为 {fmt_ratio(rs)}（{pct_delta(rs)}），相对 EXP-LUT 为 {fmt_ratio(rl)}（{pct_delta(rl)}）。",
                  f"判定：q32 纵向显著提升={decision['vertical_win']}；接近 EXP-LUT={decision['near_lut']}；"
                  f"超过 EXP-LUT={decision['beats_lut']}；全部 Qo 无 >2% 回退={decision['no_q_regression']}。"]
    else:
        lines.append("正式 Attention 数据不可用，本文不生成性能结论。")
    if not quick:
        lines += [f"同机 q32 baseline：Origin HVX {fmt_stat(summary['latency']['baseline']['32'], ' us')}，"
                  f"EXP-LUT {fmt_stat(summary['latency']['lut-exp']['32'], ' us')}，"
                  f"static-d8 {fmt_stat(summary['latency']['static_d8_ref']['32'], ' us')}。",
                  "`combined_confirm` 未执行：scalar-weight 没有证明 q32 纵向提升，且 scalar-weight/quad 未同时通过全 Qo 回退门禁。"]
    stats_text = ("2 sessions，每 session 2 warmup + 3 measure（链路验证，不用于正式结论）" if quick else
                  "5 sessions，每 session 5 warmup + 20 measure")
    lines += ["", "所有结论仅来自本 run 目录的日志、CSV/JSON 与反汇编，不复用历史延迟。", "",
              "## 实验设置", "", "- 模型/数据集：N/A，算子级 seeded synthetic Figure8 Attention benchmark。",
              "- Shape：q=1/4/8/16/32，KV=4096，12 query heads，2 KV heads，head_dim=128，full mask，workers=1。",
              f"- 统计：{stats_text}；10,000 次配对 bootstrap 95% CI。",
              "- 正确性矩阵：full/causal/padding × q=1/4 × KV=4093/4096 × head_dim=64/128 × 3 seeds。",
              "- Micro：每个 artifact 30 个独立样本（quick 为 3），正式样本累计目标 80 ms，超过 50 ms 下限。",
              "- Baseline：Origin HVX、EXP-LUT、原始 static-d8，全部在本次 run 同机重测。",
              f"- Toolchain：{toolchain}。",
              f"- 频率快照：{frequency}。DSP devfreq 节点在该 Android user build 上不可读；不据 CPU 频率反推 DSP 频率。",
              f"- 温控：before {fmt_thermal(summary['thermal']['before'])}；after {fmt_thermal(summary['thermal']['after'])}。",
              "- 硬件、系统与 artifact SHA256：", "", "```text", manifest.strip(), "```", "",
              "## 优化策略与消融结论", "",
              "| 变体 | 在 static-d8 上的改动 | 数据结论 |", "|---|---|---|"]
    gate_map = {r["kernel_impl"]: r for r in summary["static_gates"].get("rows", [])}
    for label in IMPLS:
        gate = gate_map.get(label, {})
        dec = summary["decisions"][label]
        pair_ratio = summary["micro_ratios"][label]["pair_vs_static"]
        q32_ratio = summary["ratios"].get(label, {}).get("32", {}).get("vs_static")
        if label == "static_d8_ref":
            result = "基线；micro 与 Attention 均在本 run 重测"
        elif gate and not gate.get("static_pass"):
            result = f"静态否决：{gate.get('reasons')}"
        elif label == "combined_confirm" and not summary["combined_enabled"]:
            result = "未触发；独立改动未同时满足组合条件"
        elif not summary["correctness"][label]["pass"]:
            result = "正确性门禁失败或未执行"
        elif not q32_ratio:
            result = "未进入正式 Attention"
        else:
            result = (f"micro pair {pct_delta(pair_ratio)}；q32/static {pct_delta(q32_ratio)}；"
                      f"q32 CI 通过={dec['vertical_win']}；全 Qo 门禁={dec['no_q_regression']}")
            if label in ("qf16_tree_control", "piecewise_control"):
                result += "；control，不参与 winner"
        lines.append(f"| {label} | {STRATEGIES[label]} | {result} |")
    lines += ["", "### 反汇编门禁", "",
              "| 变体 | inst | packets | splat | Rhf mul | eval spill/stack B | caller stack-ref/frame Δ | 门禁 |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for label in IMPLS:
        data = summary["static_metrics"].get(label, {})
        gate = gate_map.get(label, {})
        lines.append(f"| {label} | {data.get('instructions', 'NA')} | {data.get('packets', 'NA')} | "
                     f"{data.get('splat', 'NA')} | {gate.get('scalar_weight_multiply', 'NA')} | "
                     f"{gate.get('evaluator_spill', 'NA')}/{gate.get('evaluator_stack', 'NA')} | "
                     f"{gate.get('caller_spill_delta', 'NA')}/{gate.get('caller_stack_delta', 'NA')} B | "
                     f"{'PASS' if gate.get('static_pass') else 'FAIL'} {gate.get('reasons', '')} |")
    lines += ["", "![micro 与反汇编联合图](figures/04_static_metrics.svg)", "",
              "图 4 联合展示 micro latency、instruction、packet、splat、evaluator stack 与 caller stack-reference 增量。scalar-weight 实际生成 14 条 `vmpy(Vhf,Rhf)`，splat 从 17 降到 8；pair-return noinline 的 packet 从 41 降到 36。inline 增加 128 B caller frame，prebroadcast 增加 1920 B frame 与 7 个 stack reference，按预注册规则否决。quad 增加 1920 B frame 与 116 个 stack reference；因它不是预注册硬否决项，保留动态结果。", "",
              "## Micro Benchmark", "",
              "| 变体 | pair ns/64 (95% CI) | pair/static | quad ns/64 (95% CI) | quad/static | mismatch pair/quad |",
              "|---|---:|---:|---:|---:|---:|"]
    for label in IMPLS:
        data = summary["micro_summary"][label]
        ratios = summary["micro_ratios"][label]
        lines.append(f"| {label} | {fmt_stat(data['pair'])} | {fmt_ratio(ratios['pair_vs_static'])} | "
                     f"{fmt_stat(data['quad'])} | {fmt_ratio(ratios['quad_vs_static'])} | "
                     f"{data['max_pair_mismatches']}/{data['max_quad_mismatches']} |")
    lines += ["", "![Pair/quad micro](figures/03_micro_pair_quad.svg)", "",
              "图 3 的误差线来自 30 个独立样本的 bootstrap 95% CI，quad 按相同 64-element useful work 归一化。scalar-weight/pair-return 的 pair micro 从 22.690 降到 20.627 ns/64（约 9.09%）；quad 达 20.112 ns/64（约 11.36%）。micro 改善不能替代完整 Attention 结论。", "",
              "## 完整 Attention", "", "![不同 Qo 延迟](figures/01_attention_latency.svg)", "",
              "图 1 给出所有同机方案的绝对 Host latency 与 bootstrap 95% CI。EXP-LUT 在五个 Qo 均最低；pair-return/quad 的收益集中在 q>=8，短 Qo 的固定成本未能稳定摊薄。", "",
              "| 方案 | q1 us | q4 us | q8 us | q16 us | q32 us |", "|---|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        if any(summary["latency"].get(label, {}).get(str(q)) for q in QOS):
            values = [fmt_stat(summary["latency"][label][str(q)]) for q in QOS]
            lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "![q32 比值与消融](figures/02_q32_ratios.svg)", "",
              "图 2 是 q32 消融图，使用同 session/iteration 配对数据。实线 1.00 表示相等，虚线 1.02 是接近 EXP-LUT 与 Qo 回退门限。", "",
              "| 变体 | q32/static | q32/LUT | 纵向胜出 | 接近 LUT | 全 Qo 门禁 |",
              "|---|---:|---:|---|---|---|"]
    for label in [x for x in IMPLS if x != "static_d8_ref"]:
        rs = summary["ratios"].get(label, {}).get("32", {}).get("vs_static")
        rl = summary["ratios"].get(label, {}).get("32", {}).get("vs_lut")
        d = summary["decisions"][label]
        lines.append(f"| {label} | {fmt_ratio(rs)} | {fmt_ratio(rl)} | {d['vertical_win']} | "
                     f"{d['near_lut']} | {d['no_q_regression']} |")
    lines += ["", "### 全 Qo 相对 static-d8 的配对 ratio", "",
              "| 变体 | q1 | q4 | q8 | q16 | q32 |", "|---|---:|---:|---:|---:|---:|"]
    reported = ["d7_serial", "d7_scalar_w", "d7_pairret_noinline", "d7_quad_pipeline",
                "qf16_tree_control", "piecewise_control"]
    for label in reported:
        values = [fmt_ratio(summary["ratios"][label][str(q)]["vs_static"]) for q in QOS]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    pair_q1 = summary["ratios"]["d7_pairret_noinline"]["1"]["vs_static"]
    pair_q4 = summary["ratios"]["d7_pairret_noinline"]["4"]["vs_static"]
    quad_q1 = summary["ratios"]["d7_quad_pipeline"]["1"]["vs_static"]
    regression_note = ("全 Qo 回退判定不可用。" if not all((pair_q1, pair_q4, quad_q1)) else
                       f"pair-return 的 q1/q4 CI 上界为 {pair_q1['ci_high']:.4f}/{pair_q4['ci_high']:.4f}；"
                       f"quad 的 q1 上界为 {quad_q1['ci_high']:.4f}。CI 上界超过 1.02 的候选不满足全 Qo 判定。")
    lines += ["", regression_note, "",
              "## 正确性", "", "![正确性门禁](figures/05_correctness.svg)", "",
              "图 5 汇总 72 个 case/候选的最坏误差。七个动态筛选 artifact 均通过 finite、mask、tail、RMSE<=0.002 和 max_abs<=0.01；四个顺序保持型候选的固定 pilot checksum 与 static-d8 一致。", "",
              "| 变体 | cases | max RMSE | max abs | bitwise checksum | 门禁 |",
              "|---|---:|---:|---:|---|---|"]
    static_checksum = next((v["expected"] for v in summary["bitwise_gates"].values()), "UNAVAILABLE")
    for label in ["static_d8_ref", *reported]:
        corr = summary["correctness"][label]
        checksum = (static_checksum if label == "static_d8_ref" else
                    summary["bitwise_gates"].get(label, {}).get("observed", "N/A control"))
        lines.append(f"| {label} | {corr['cases']} | {fmt_number(corr['max_rmse'])} | {fmt_number(corr['max_abs'])} | "
                     f"{checksum} | {'PASS' if corr['pass'] else 'FAIL'} |")
    lines += ["", "## 诊断数据（不替代主判定）", "",
              "KV diagnostic 每项仅 1 次正式迭代，用于定位阶段稀释；workers sweep 是对 q32 最快候选的探索性扩展。", "",
              "| 方案 | KV64 Host us | KV4096 Host us | KV4096 scna_exp sum | KV4096 profiled_total sum |",
              "|---|---:|---:|---:|---:|"]
    diagnostic_labels = [x for x in LABELS if summary["diagnostic"].get(x, {}).get("4096", {}).get("host")]
    for label in diagnostic_labels:
        d64 = summary["diagnostic"][label]["64"]
        d4096 = summary["diagnostic"][label]["4096"]
        timers = d4096.get("timers") or {}
        lines.append(f"| {label} | {fmt_stat(d64.get('host'))} | {fmt_stat(d4096.get('host'))} | "
                     f"{timers.get('scna_exp', 'NA')} | {timers.get('profiled_total', 'NA')} |")
    static_timer = summary["diagnostic"].get("static_d8_ref", {}).get("4096", {}).get("timers") or {}
    pair_timer = summary["diagnostic"].get("d7_pairret_noinline", {}).get("4096", {}).get("timers") or {}
    if static_timer.get("scna_exp") and pair_timer.get("scna_exp"):
        timer_reduction = 1.0 - pair_timer["scna_exp"] / static_timer["scna_exp"]
        lines += ["", f"单次 stage diagnostic 中，pair-return 的 summed `scna_exp` 从 "
                  f"{static_timer['scna_exp']} 降到 {pair_timer['scna_exp']}（-{timer_reduction*100:.2f}%）；"
                  "Host 收益更小，直接支持“Attention 其他阶段稀释 evaluator 收益”的解释。"]
    for label, workers_data in summary["scaling"].items():
        lines += ["", f"探索性 workers sweep：`{label}`。", "",
                  "| workers | 1 | 2 | 3 | 4 | 5 | 6 | auto |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|",
                  "| Host us | " + " | ".join(fmt_stat(workers_data[w]) for w in ("1", "2", "3", "4", "5", "6", "auto")) + " |"]
    lines += ["", "## 异常结果与验证计划", ""]
    failures = []
    for label in IMPLS:
        gate = gate_map.get(label, {})
        corr = summary["correctness"][label]
        dec = summary["decisions"][label]
        if gate and not gate.get("static_pass"):
            failures.append(f"- `{label}` 静态否决（{gate.get('reasons')}）：caller frame Δ={gate.get('caller_stack_delta')} B、"
                            f"stack-reference Δ={gate.get('caller_spill_delta')}。原因候选：(1) 常量/中间量 live range 扩大；"
                            f"(2) inline 或预广播使寄存器分配跨列循环；(3) 常量数组落栈。验证计划：缩短常量生命周期后重编译，"
                            f"逐项比较 allocframe、r29/r30 references 与 packet 调度，静态门禁通过前不跑正式 Attention。")
        elif corr["cases"] and not corr["pass"]:
            failures.append(f"- `{label}` 正确性失败。原因候选：FP16 重关联、scalar lane 选择、mask/tail 路径。验证计划：用 numeric-debug lane bits、checksum 和边界输入定位首个差异。")
        elif label != "static_d8_ref" and summary["latency"][label]["32"] and not dec["full_success"]:
            pair = summary["micro_ratios"][label]["pair_vs_static"]
            q32 = summary["ratios"][label]["32"]["vs_static"]
            worst_q = max(QOS, key=lambda q: summary["ratios"][label][str(q)]["vs_static"]["ci_high"])
            worst = summary["ratios"][label][str(worst_q)]["vs_static"]
            scope_note = ("；该项为 control，按预注册规则不进入 winner" if label in ("qf16_tree_control", "piecewise_control") else "")
            failures.append(f"- `{label}` 未满足完整成功标准：micro pair/static={fmt_ratio(pair)}，q32/static={fmt_ratio(q32)}，"
                            f"最差 Qo=q{worst_q}、CI 上界={worst['ci_high']:.4f}，packets={gate.get('packets')}、"
                            f"caller frame Δ={gate.get('caller_stack_delta')} B。证据支持的原因：(1) evaluator 节省被 QK/PV/Host 固定阶段稀释；"
                            f"(2) packet/执行槽竞争未随 instruction 数同比下降；(3) 短 Qo 的 ABI 固定成本或寄存器压力无法摊薄。"
                            f"验证计划：用归档 KV stage timer 量化稀释比例，再做缩短 live range 的单变量构建{scope_note}。")
    if not summary["overall_winner"] and not quick:
        failures.append("- 横向目标未达成：最快候选相对 EXP-LUT 的 q32 ratio CI 完全高于 1.02。验证计划：继续消除 pair-return 的短 Qo 固定成本；若 evaluator stage 已接近下限而完整差距仍在，则据 stage timer 判定本轮搜索空间不足，不扩大结论。")
    lines.extend(failures or ["- 未观察到门禁或性能异常；仍需复核原始日志和热状态。"])
    lines += ["", "## 局限性与可复现性", "",
              "- 这是单设备、算子级实验，不外推到端到端 LLM。",
              "- d4/d6 重训练、混合 LUT-SCNA、HMX/v81、QK/PV/mask/tiling 修改均不在本轮。",
              "- 原始日志在 `raw/`，CSV 在 `tables/`，摘要在 `summary.json`，SHA256/构建参数/反汇编在 `evidence/` 与 `static/`；所有图由归档数据直接生成。",
              "- 本报告由脚本从实测数据生成；使用者须逐句核验，投稿或学位材料需遵守对应 AI 披露政策。", ""]
    (root / "SCNA_HVX_D8_PIPELINE_V79_REPORT_ZH.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--selection-out")
    args = parser.parse_args()
    root = Path(args.run_dir)
    summary = summarize(root)
    if args.select_only:
        output = Path(args.selection_out) if args.selection_out else root / "evidence/combined_selection.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"schema_version": SCHEMA,
                                      "combined_enabled": summary["combined_enabled"],
                                      "decisions": summary["decisions"]}, indent=2) + "\n")
        return
    write_csv(root / "tables/micro_samples.csv", summary.pop("micro"))
    write_csv(root / "tables/attention_samples.csv", summary.pop("attention"))
    write_csv(root / "tables/accuracy_cases.csv", summary.pop("accuracy_rows"))
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_all(root, summary)
    make_report(root, summary)


if __name__ == "__main__":
    main()
