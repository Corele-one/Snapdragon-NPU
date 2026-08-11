#!/usr/bin/env python3
"""Analyze the v79 accuracy-first gate and emit reproducible CSV/JSON/SVG artifacts."""

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
COMPARE_RE = re.compile(r"FIG8_ATTENTION_COMPARE\s+(.*)")
LAYOUT_RE = re.compile(r"FIG8_ATTENTION_LAYOUT_COMPARE\s+(.*)")
MICRO_RE = re.compile(r"SCNA_EXP_BENCH\s+(.*)")


def kv(text):
    out = {}
    for key, value in KV_RE.findall(text):
        try:
            out[key] = float(value) if any(c in value for c in ".eE") else int(value, 0)
        except ValueError:
            out[key] = value
    return out


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_median(values, seed, draws=10000):
    rng = random.Random(seed)
    samples = [statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def bootstrap_ratio(numerator, denominator, seed, draws=10000):
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        a = statistics.median(rng.choices(numerator, k=len(numerator)))
        b = statistics.median(rng.choices(denominator, k=len(denominator)))
        samples.append(a / b)
    return percentile(samples, 0.025), percentile(samples, 0.975)


def svg_grouped_bars(path, title, groups, series, ylabel, reference=None, value_format=".3f"):
    width, height = 1080, 520
    left, right, top, bottom = 90, 30, 55, 100
    chart_w, chart_h = width - left - right, height - top - bottom
    values = [item["value"] for group in groups for item in group["items"]]
    ci_values = [item.get("ci_high", item["value"]) for group in groups for item in group["items"]]
    ymax = max(values + ci_values + ([reference] if reference is not None else [0])) * 1.18
    if ymax == 0:
        ymax = 1
    colors = ["#3366cc", "#dc3912", "#109618", "#ff9900", "#990099"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="27" text-anchor="middle" font-family="sans-serif" font-size="19">{title}</text>',
           f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#333"/>',
           f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>',
           f'<text x="20" y="{top+chart_h/2}" transform="rotate(-90 20 {top+chart_h/2})" text-anchor="middle" font-family="sans-serif" font-size="13">{ylabel}</text>']
    for tick in range(6):
        val = ymax * tick / 5
        y = top + chart_h * (1 - val / ymax)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        out.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{val:{value_format}}</text>')
    if reference is not None:
        y = top + chart_h * (1 - reference / ymax)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#111" stroke-dasharray="7,5"/>')
        out.append(f'<text x="{left+chart_w-4}" y="{y-5:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">gate={reference:{value_format}}</text>')
    group_w = chart_w / len(groups)
    for gi, group in enumerate(groups):
        item_w = group_w * 0.72 / max(len(group["items"]), 1)
        base_x = left + gi * group_w + group_w * 0.14
        for si, item in enumerate(group["items"]):
            x = base_x + si * item_w
            y = top + chart_h * (1 - item["value"] / ymax)
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{item_w*0.82:.1f}" height="{top+chart_h-y:.1f}" fill="{colors[si]}"/>')
            if "ci_low" in item:
                yl = top + chart_h * (1 - item["ci_low"] / ymax)
                yh = top + chart_h * (1 - item["ci_high"] / ymax)
                mid = x + item_w * 0.41
                out.extend([f'<line x1="{mid:.1f}" y1="{yl:.1f}" x2="{mid:.1f}" y2="{yh:.1f}" stroke="#111"/>',
                            f'<line x1="{mid-5:.1f}" y1="{yl:.1f}" x2="{mid+5:.1f}" y2="{yl:.1f}" stroke="#111"/>',
                            f'<line x1="{mid-5:.1f}" y1="{yh:.1f}" x2="{mid+5:.1f}" y2="{yh:.1f}" stroke="#111"/>'])
            out.append(f'<text x="{x+item_w*0.41:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{item["value"]:{value_format}}</text>')
        out.append(f'<text x="{left+(gi+0.5)*group_w:.1f}" y="{top+chart_h+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{group["label"]}</text>')
    lx = left
    for index, label in enumerate(series):
        out.append(f'<rect x="{lx}" y="{height-40}" width="13" height="13" fill="{colors[index]}"/>')
        out.append(f'<text x="{lx+18}" y="{height-29}" font-family="sans-serif" font-size="12">{label}</text>')
        lx += max(150, len(label) * 8 + 35)
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_accuracy(result_dir, analysis):
    rows = []
    layout_rows = []
    with (result_dir / "accuracy/case_status.tsv").open(encoding="utf-8") as handle:
        for status in csv.DictReader(handle, delimiter="\t"):
            text = Path(status["log"]).read_text(errors="replace")
            compares = [kv(m.group(1)) for m in COMPARE_RE.finditer(text)]
            layouts = [kv(m.group(1)) for m in LAYOUT_RE.finditer(text)]
            candidate = compares[-1] if compares else {}
            row = {k: status[k] for k in ("stage", "mode", "layout", "seed", "mask", "qo_len", "kv_len", "head_dim")}
            row.update({
                "rmse": candidate.get("rmse", math.nan),
                "max_abs_error": candidate.get("max_abs_error", math.nan),
                "candidate_nonfinite": candidate.get("candidate_nonfinite", -1),
                "reference_nonfinite": candidate.get("reference_nonfinite", -1),
                "compare_pass": int(status["compare_pass"]),
                "layout_pass": int(status["layout_pass"]),
                "mask_tail_pass": int(status["mask_tail_pass"]),
                "checksum_unique_count": int(status["checksum_unique_count"]),
                "log": status["log"],
            })
            rows.append(row)
            if layouts:
                item = layouts[-1]
                layout_rows.append({k: item[k] for k in ("seed", "mask_mode", "qo_len", "kv_len", "head_dim", "rmse", "max_abs_error", "serial_nonfinite", "lane8_nonfinite", "pass")})
    write_csv(analysis / "accuracy_matrix.csv", rows)
    write_csv(analysis / "layout_compare_matrix.csv", layout_rows)
    (analysis / "accuracy_matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (analysis / "layout_compare_matrix.json").write_text(json.dumps(layout_rows, indent=2) + "\n", encoding="utf-8")
    stages = {}
    for stage in ("baseline", "scna_serial", "scna_lane8"):
        subset = [row for row in rows if row["stage"] == stage]
        worst_rmse = max(subset, key=lambda row: row["rmse"])
        worst_abs = max(subset, key=lambda row: row["max_abs_error"])
        stages[stage] = {
            "cases": len(subset),
            "pass": sum(row["compare_pass"] and row["layout_pass"] and row["mask_tail_pass"] for row in subset),
            "worst_rmse": worst_rmse["rmse"], "worst_rmse_case": {k: worst_rmse[k] for k in ("seed", "mask", "qo_len", "kv_len", "head_dim")},
            "worst_max_abs": worst_abs["max_abs_error"], "worst_max_abs_case": {k: worst_abs[k] for k in ("seed", "mask", "qo_len", "kv_len", "head_dim")},
            "nonfinite_total": sum(row["candidate_nonfinite"] + row["reference_nonfinite"] for row in subset),
            "mask_tail_failures": sum(not row["mask_tail_pass"] for row in subset),
            "checksum_unique_min": min(row["checksum_unique_count"] for row in subset),
        }
    layout_summary = {
        "cases": len(layout_rows), "pass": sum(item["pass"] for item in layout_rows),
        "worst_rmse": max(item["rmse"] for item in layout_rows),
        "worst_max_abs": max(item["max_abs_error"] for item in layout_rows),
        "nonfinite_total": sum(item["serial_nonfinite"] + item["lane8_nonfinite"] for item in layout_rows),
    }
    summary = {"total_cases": len(rows), "all_pass": all(v["pass"] == v["cases"] for v in stages.values()), "stages": stages, "layout_compare": layout_summary}
    (analysis / "accuracy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    groups = []
    for stage, label in (("baseline", "Origin-HVX"), ("scna_serial", "SCNA serial"), ("scna_lane8", "SCNA lane8")):
        item = stages[stage]
        groups.append({"label": label, "items": [
            {"value": 100 * item["worst_rmse"] / 0.002},
            {"value": 100 * item["worst_max_abs"] / 0.01},
        ]})
    groups.append({"label": "serial vs lane8", "items": [
        {"value": 100 * layout_summary["worst_rmse"] / 1e-4},
        {"value": 100 * layout_summary["worst_max_abs"] / 1e-3},
    ]})
    svg_grouped_bars(analysis / "attention_accuracy_gate.svg", "Attention error as percentage of each gate", groups, ["RMSE / gate", "max-abs / gate"], "% of threshold", reference=100, value_format=".1f")
    return summary


def analyze_micro(result_dir, analysis):
    records = []
    for filename in ("30samples_final.log", "30samples_lane8_final.log"):
        for line in (result_dir / "performance/micro" / filename).read_text(errors="replace").splitlines():
            match = MICRO_RE.search(line)
            if match:
                records.append(kv(match.group(1)))
    by_variant = defaultdict(list)
    for record in records:
        label = "serial" if record["layout"] == "serial" else record["variant"]
        by_variant[label].append(record["paired_ns_per_64"])
    required = ("serial", "current", "sequential-pair", "split4-pair", "pack-once")
    if any(len(by_variant[name]) != 30 for name in required):
        raise SystemExit(f"incomplete micro samples: {dict((k, len(v)) for k, v in by_variant.items())}")
    sample_rows, summary = [], {}
    for record in records:
        label = "serial" if record["layout"] == "serial" else record["variant"]
        sample_rows.append({"variant": label, "sample": record["sample"], "paired_ns_per_64": record["paired_ns_per_64"], "checksum": hex(record["checksum"])})
    write_csv(analysis / "micro_samples.csv", sample_rows)
    serial = by_variant["serial"]
    for index, name in enumerate(required):
        values = by_variant[name]
        lo, hi = bootstrap_median(values, 20260810 + index)
        item = {"n": len(values), "median_ns_per_64": statistics.median(values), "p95_ns_per_64": percentile(values, .95), "median_ci_low": lo, "median_ci_high": hi, "checksum_unique": len({r["checksum"] for r in records if ("serial" if r["layout"] == "serial" else r["variant"]) == name})}
        if name != "serial":
            speed_lo, speed_hi = bootstrap_ratio(serial, values, 20260900 + index)
            item.update({"serial_speedup": statistics.median(serial) / statistics.median(values), "speedup_ci_low": speed_lo, "speedup_ci_high": speed_hi})
        summary[name] = item
    candidates = required[1:]
    fastest = min(candidates, key=lambda name: summary[name]["median_ns_per_64"])
    gate = summary[fastest]["speedup_ci_low"] > 1
    result = {"variants": summary, "fastest_lane8": fastest, "micro_gate_pass": gate, "attention_performance_gate_executed": False if not gate else None}
    (analysis / "micro_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    latency_groups = [{"label": name, "items": [{"value": summary[name]["median_ns_per_64"], "ci_low": summary[name]["median_ci_low"], "ci_high": summary[name]["median_ci_high"]}]} for name in required]
    svg_grouped_bars(analysis / "micro_latency.svg", "Paired SCNA evaluator latency (30 DSP qtimer samples)", latency_groups, ["median and bootstrap 95% CI"], "ns / 64 scores", value_format=".1f")
    speed_groups = [{"label": name, "items": [{"value": summary[name]["serial_speedup"], "ci_low": summary[name]["speedup_ci_low"], "ci_high": summary[name]["speedup_ci_high"]}]} for name in candidates]
    svg_grouped_bars(analysis / "micro_speedup_ci.svg", "Serial / lane8 paired evaluator speedup", speed_groups, ["speedup and bootstrap 95% CI"], "speedup (higher is better)", reference=1, value_format=".3f")
    first = records[0]
    evaluator = {key: first[key] for key in ("dense_samples", "dense_rmse", "dense_max_abs", "monotonic_violations", "negative_count", "nan_count", "canonical_oracle_mismatches", "paired_single_mismatches", "reciprocal_max_relative_error", "reciprocal_nonfinite_count", "reciprocal_pass")}
    (analysis / "evaluator_summary.json").write_text(json.dumps(evaluator, indent=2) + "\n", encoding="utf-8")
    eval_groups = [
        {"label": "dense SCNA RMSE", "items": [{"value": evaluator["dense_rmse"]}]},
        {"label": "dense SCNA max-abs", "items": [{"value": evaluator["dense_max_abs"]}]},
        {"label": "reciprocal max-rel", "items": [{"value": evaluator["reciprocal_max_relative_error"]}]},
    ]
    svg_grouped_bars(analysis / "evaluator_error.svg", "Evaluator-level approximation and reciprocal error", eval_groups, ["measured error"], "absolute or relative error", value_format=".4f")
    return result, evaluator


def analyze_static(result_dir, analysis):
    source = result_dir / "performance/static/scna_disassembly_summary.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    shutil_rows = []
    for row in rows:
        shutil_rows.append({"variant": row["variant"], "instructions": row["instructions"], "shuffle_reduction": row["shuffle_reduction"], "spill_memory_ops": row["spill_memory_ops"], "vector_load_store": row["vector_load_store"]})
    write_csv(analysis / "static_instruction_counts.csv", shutil_rows)
    groups = [{"label": row["variant"], "items": [{"value": row["shuffle_reduction"]}, {"value": row["spill_memory_ops"]}]} for row in shutil_rows]
    svg_grouped_bars(analysis / "static_shuffle_spill.svg", "Packetized disassembly pressure by function body", groups, ["shuffle/reduction", "r29 stack vector-memory ops"], "static instruction count", value_format=".0f")
    source_cost = [
        {"label": "expand", "items": [{"value": 16}]},
        {"label": "affine+ReLU", "items": [{"value": 32}]},
        {"label": "reduce", "items": [{"value": 96}]},
        {"label": "pack", "items": [{"value": 48}]},
    ]
    svg_grouped_bars(analysis / "lane8_stage_source_ops.svg", "Current paired lane8 source-level HVX operations", source_cost, ["intrinsic operations per two vectors"], "source intrinsic count", value_format=".0f")
    return shutil_rows


def timeline_svg(trace_path, output):
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    slices = [event for event in trace["traceEvents"] if event.get("ph") == "X" and event.get("args", {}).get("qo_len") == 4 and event.get("args", {}).get("iteration") == 0]
    pids = sorted({event["pid"] for event in slices})
    selected = []
    for pid in pids:
        selected.extend(sorted([event for event in slices if event["pid"] == pid], key=lambda e: (e["ts"], e["tid"]))[:180])
    lanes = sorted({(event["pid"], event["tid"]) for event in selected})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    min_ts, max_ts = min(e["ts"] for e in selected), max(e["ts"] + e["dur"] for e in selected)
    width, left, chart_w = 1200, 135, 1020
    height = 65 + 17 * len(lanes)
    scale = chart_w / max(max_ts - min_ts, 1)
    colors = {"q_load": "#8dd3c7", "k_load": "#80b1d3", "qk_dot": "#377eb8", "safe_sm": "#ff7f00", "scna_exp": "#e41a1c", "v_load": "#b3de69", "core_acc": "#4daf4a", "o_scale": "#984ea3", "o_store": "#a65628"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="600" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">DSP qtimer Perfetto replay excerpt: serial and sequential-pair, Qo=4</text>']
    for event in selected:
        y = 42 + lane_index[(event["pid"], event["tid"])] * 17
        x = left + (event["ts"] - min_ts) * scale
        w = max(event["dur"] * scale, 0.8)
        comp = event.get("args", {}).get("component", "kernel")
        out.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="12" fill="{colors.get(comp, "#bbb")}"/>')
    out.append('</svg>')
    output.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.result_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    accuracy = analyze_accuracy(args.result_dir, analysis)
    micro, evaluator = analyze_micro(args.result_dir, analysis)
    static = analyze_static(args.result_dir, analysis)
    timeline_svg(args.result_dir / "performance/trace/scna_serial_vs_sequential_pair_all_q.perfetto.json", analysis / "perfetto_timeline_export.svg")
    combined = {"accuracy": accuracy, "micro": micro, "evaluator": evaluator, "static": static}
    (analysis / "summary.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
