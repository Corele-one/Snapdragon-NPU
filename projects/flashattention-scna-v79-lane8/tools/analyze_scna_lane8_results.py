#!/usr/bin/env python3
"""Data analysis and SVG generation for the three-session SCNA lane8 run."""

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


KV = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
HOST = re.compile(r"FIG8_ATTENTION_HOST_TIMING\s+(.*)")
TIMERS = re.compile(r"FIG8_ATTENTION_TIMERS\s+(.*)")
COMPARE = re.compile(r"FIG8_ATTENTION_COMPARE\s+(.*)")
MICRO = re.compile(r"SCNA_EXP_BENCH\s+(.*)")


def values(text):
    result = {}
    for key, value in KV.findall(text):
        try:
            result[key] = float(value) if any(char in value for char in ".eE") else int(value, 0)
        except ValueError:
            result[key] = value
    return result


def percentile(items, p):
    ordered = sorted(items)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def bootstrap_speedup(serial, lane8, seed=20260808, count=10000):
    rng = random.Random(seed)
    sessions = sorted(set(serial) & set(lane8))
    draws = []
    for _ in range(count):
        s_values, l_values = [], []
        for sampled_session in rng.choices(sessions, k=len(sessions)):
            s = serial[sampled_session]
            l = lane8[sampled_session]
            s_values.extend(rng.choices(s, k=len(s)))
            l_values.extend(rng.choices(l, k=len(l)))
        draws.append(statistics.median(s_values) / statistics.median(l_values))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def bootstrap_median(by_session, seed=20260808, count=10000):
    """Hierarchical bootstrap: resample sessions, then observations in each session."""
    rng = random.Random(seed)
    sessions = sorted(by_session)
    draws = []
    for _ in range(count):
        sampled = []
        for sampled_session in rng.choices(sessions, k=len(sessions)):
            observations = by_session[sampled_session]
            sampled.extend(rng.choices(observations, k=len(observations)))
        draws.append(statistics.median(sampled))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def svg_latency(summary, path):
    modes = ["Origin-HVX", "LUT-EXP", "SCNA serial d8", "SCNA lane8 d8"]
    colors = ["#3366cc", "#ff9900", "#dc3912", "#109618"]
    qos = [4, 8, 16, 32]
    vals = [summary[mode] for mode in modes]
    ymax = max(item[str(q)]["median_us"] for item in vals for q in qos) * 1.12
    w, h, left, top, cw, ch = 900, 500, 75, 45, 760, 350
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="white"/>',
           '<text x="450" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Attention latency vs Qo (median host elapsed)</text>']
    out += [f'<line x1="{left}" y1="{top+ch}" x2="{left+cw}" y2="{top+ch}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ch}" stroke="black"/>']
    for mode, color in zip(modes, colors):
        pts = []
        for i, qo in enumerate(qos):
            x = left + i * cw / 3
            y = top + ch * (1 - summary[mode][str(qo)]["median_us"] / ymax)
            pts.append(f"{x:.1f},{y:.1f}")
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
    for i, qo in enumerate(qos):
        x = left + i * cw / 3
        out.append(f'<text x="{x:.1f}" y="{top+ch+22}" text-anchor="middle" font-family="sans-serif" font-size="13">{qo}</text>')
    for i, (mode, color) in enumerate(zip(modes, colors)):
        x = 80 + i * 195
        out.append(f'<line x1="{x}" y1="455" x2="{x+28}" y2="455" stroke="{color}" stroke-width="3"/><text x="{x+34}" y="460" font-family="sans-serif" font-size="12">{mode}</text>')
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def svg_bars(items, title, path, value_key="median", ci=True):
    w, h, left, top, cw, ch = 900, 470, 75, 45, 760, 330
    ymax = max(item[value_key] if not ci else item["ci_high"] for item in items) * 1.15
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">', '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="450" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
           f'<line x1="{left}" y1="{top+ch}" x2="{left+cw}" y2="{top+ch}" stroke="black"/>']
    bw = cw / (len(items) * 1.7)
    for i, item in enumerate(items):
        x = left + (i + 0.35) * cw / len(items)
        val = item[value_key]
        y = top + ch * (1 - val / ymax)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{top+ch-y:.1f}" fill="#109618"/>')
        if ci:
            yl = top + ch * (1 - item["ci_low"] / ymax); yh = top + ch * (1 - item["ci_high"] / ymax)
            mid = x + bw / 2
            out += [f'<line x1="{mid:.1f}" y1="{yl:.1f}" x2="{mid:.1f}" y2="{yh:.1f}" stroke="black"/>', f'<line x1="{mid-6:.1f}" y1="{yl:.1f}" x2="{mid+6:.1f}" y2="{yl:.1f}" stroke="black"/>', f'<line x1="{mid-6:.1f}" y1="{yh:.1f}" x2="{mid+6:.1f}" y2="{yh:.1f}" stroke="black"/>']
        out.append(f'<text x="{x+bw/2:.1f}" y="{top+ch+20}" text-anchor="middle" font-family="sans-serif" font-size="12">{item["label"]}</text>')
        out.append(f'<text x="{x+bw/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{val:.3f}</text>')
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def timeline_svg(trace_path, output):
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    slices = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("args", {}).get("iteration") == 0]
    slices = sorted(slices, key=lambda e: (e.get("pid", 0), e.get("tid", 0), e.get("ts", 0)))[:300]
    tids = sorted({(e.get("pid", 0), e.get("tid", 0)) for e in slices})
    row = {key: i for i, key in enumerate(tids)}
    min_ts = min(e["ts"] for e in slices); max_ts = max(e["ts"] + e["dur"] for e in slices)
    scale = 1050 / max(max_ts - min_ts, 1)
    h = 70 + len(tids) * 18
    colors = {"scna_exp": "#dc3912", "safe_sm": "#ff9900", "qk_dot": "#3366cc", "core_acc": "#109618"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{h}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="600" y="22" text-anchor="middle" font-family="sans-serif" font-size="18">Perfetto event replay export (iteration 0)</text>']
    for event in slices:
        y = 42 + row[(event.get("pid", 0), event.get("tid", 0))] * 18
        x = 120 + (event["ts"] - min_ts) * scale
        width = max(event["dur"] * scale, 1)
        comp = event.get("args", {}).get("component", "kernel")
        out.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="13" fill="{colors.get(comp, "#999")}" title="{event.get("name", "")}"/>')
    out.append('</svg>')
    output.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.result_dir / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    samples = defaultdict(lambda: defaultdict(list))
    timer_shares = defaultdict(lambda: defaultdict(list))
    rows = []
    labels = {"baseline": "Origin-HVX", "lut_exp": "LUT-EXP", "scna_serial": "SCNA serial d8", "scna_lane8": "SCNA lane8 d8"}
    for session_dir in sorted((args.result_dir / "raw").glob("session_*")):
        session = int(session_dir.name.split("_")[-1])
        for path in session_dir.glob("*.log"):
            stem = path.stem
            key = next((name for name in labels if stem.startswith(name + "_q")), None)
            if not key: continue
            qo = int(re.search(r"_q(\d+)", stem).group(1))
            timer_by_iteration = defaultdict(lambda: {"scna_exp": 0, "safe_sm": 0, "profiled_total": 0})
            for line in path.read_text(errors="replace").splitlines():
                match = HOST.search(line)
                if match:
                    item = values(match.group(1))
                    if item.get("phase") == "measure" and item.get("ret") == 0:
                        value = item["host_elapsed_us"]
                        samples[(labels[key], qo)][session].append(value)
                        rows.append({"session": session, "evaluator": labels[key], "qo_len": qo, "host_elapsed_us": value, "source": path.name})
                timer_match = TIMERS.search(line)
                if timer_match:
                    timer_item = values(timer_match.group(1))
                    if timer_item.get("phase") == "measure":
                        group = timer_by_iteration[timer_item["iteration"]]
                        for field in group:
                            group[field] += timer_item.get(field, 0)
            for group in timer_by_iteration.values():
                timer_shares[(labels[key], qo)][session].append({
                    "profiled_percent": 100.0 * group["scna_exp"] / group["profiled_total"] if group["profiled_total"] else 0.0,
                    "safe_sm_percent": 100.0 * group["scna_exp"] / group["safe_sm"] if group["safe_sm"] else 0.0,
                })
    if not rows:
        raise SystemExit("no measured headline rows")
    with (analysis / "attention_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary = defaultdict(dict)
    for (mode, qo), by_session in samples.items():
        flat = [v for values_ in by_session.values() for v in values_]
        ci_low, ci_high = bootstrap_median(by_session, seed=20260808 + qo + sum(ord(c) for c in mode))
        share_rows = [v for values_ in timer_shares[(mode, qo)].values() for v in values_]
        summary[mode][str(qo)] = {
            "n": len(flat),
            "sessions": len(by_session),
            "median_us": statistics.median(flat),
            "median_ci_low_us": ci_low,
            "median_ci_high_us": ci_high,
            "p95_us": percentile(flat, .95),
            "scna_profiled_percent_median": statistics.median(v["profiled_percent"] for v in share_rows),
            "scna_safe_sm_percent_median": statistics.median(v["safe_sm_percent"] for v in share_rows),
        }
    speedups = []
    for qo in (4, 8, 16, 32):
        lo, hi = bootstrap_speedup(samples[("SCNA serial d8", qo)], samples[("SCNA lane8 d8", qo)], seed=20260808 + qo)
        med = summary["SCNA serial d8"][str(qo)]["median_us"] / summary["SCNA lane8 d8"][str(qo)]["median_us"]
        speedups.append({"label": f"Qo={qo}", "qo_len": qo, "median": med, "ci_low": lo, "ci_high": hi})

    micro = defaultdict(list)
    micro_checks = []
    for path in (args.result_dir / "micro").glob("*_sample_*.log"):
        layout = path.name.split("_")[0]
        for line in path.read_text(errors="replace").splitlines():
            match = MICRO.search(line)
            if match:
                item = values(match.group(1)); micro[layout].append(item["paired_ns_per_64"]); micro_checks.append(item)
    micro_items = []
    for layout in ("serial", "lane8"):
        micro_items.append({"label": layout, "median": statistics.median(micro[layout]), "ci_low": min(micro[layout]), "ci_high": max(micro[layout])})
    micro_lo, micro_hi = bootstrap_speedup({0: micro["serial"]}, {0: micro["lane8"]}, count=10000)
    micro_speedup = statistics.median(micro["serial"]) / statistics.median(micro["lane8"])

    correctness = []
    for path in (args.result_dir / "correctness").glob("*.log"):
        for line in path.read_text(errors="replace").splitlines():
            match = COMPARE.search(line)
            if match: correctness.append({"source": path.name, **values(match.group(1))})
    correctness_pass = len(correctness) == 24 and all(item.get("pass") == 1 for item in correctness)
    dense_pass = all(item.get("monotonic_violations") == 0 and item.get("negative_count") == 0 and item.get("nan_count") == 0 and item.get("lane_oracle_mismatches") == 0 for item in micro_checks)
    gate = correctness_pass and dense_pass and micro_lo > 1 and all(item["ci_low"] > 1 for item in speedups)
    result = {"attention": summary, "speedups": speedups, "micro": {"items": micro_items, "speedup": micro_speedup, "ci_low": micro_lo, "ci_high": micro_hi}, "correctness": {"cases": correctness, "pass": correctness_pass, "dense_pass": dense_pass}, "d16_d32_expansion_gate": gate}
    (analysis / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    svg_latency(summary, analysis / "attention_latency_vs_qo.svg")
    svg_bars(speedups, "SCNA serial/lane8 speedup with hierarchical bootstrap 95% CI", analysis / "lane8_speedup_ci.svg")
    svg_bars(micro_items, "Paired evaluator latency (ns / 64 scores; min-max over 30 samples)", analysis / "microkernel_latency.svg")
    correctness_items = []
    for layout in ("serial", "lane8"):
        layout_rows = [item for item in correctness if item.get("layout") == layout]
        correctness_items.extend([
            {"label": f"{layout} max RMSE", "median": max(item["rmse"] for item in layout_rows)},
            {"label": f"{layout} max abs", "median": max(item["max_abs_error"] for item in layout_rows)},
        ])
    svg_bars(correctness_items, "Attention correctness worst case over mask/tail shapes", analysis / "attention_correctness.svg", ci=False)
    timeline_svg(args.result_dir / "trace/scna_serial_vs_lane8_all_q.perfetto.json", analysis / "perfetto_timeline_export.svg")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
