#!/usr/bin/env python3
"""Analyze the v81 pre/post numeric-fix ABBA performance matrix."""

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


FILE_RE = re.compile(r"s(?P<session>\d+)_q(?P<qo>\d+)_(?P<mode>baseline|lut-exp|scna-d8)_leg(?P<leg>\d+)_(?P<revision>pre|post)\.log$")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
STAGES = ("q_load", "k_load", "qk_dot", "safe_sm", "scna_exp", "v_load", "core_acc", "o_scale", "o_store")


def fields(line):
    return dict(KV_RE.findall(line))


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def hierarchical_bootstrap(pre, post, seed, draws=10000):
    rng = random.Random(seed)
    sessions = sorted(set(pre) & set(post))
    speedups = []
    deltas = []
    for _ in range(draws):
        pre_sample = []
        post_sample = []
        for _ in sessions:
            session = rng.choice(sessions)
            p0 = pre[session]
            p1 = post[session]
            pre_sample.extend(rng.choice(p0) for _ in p0)
            post_sample.extend(rng.choice(p1) for _ in p1)
        pre_med = statistics.median(pre_sample)
        post_med = statistics.median(post_sample)
        speedups.append(pre_med / post_med)
        deltas.append((post_med / pre_med - 1.0) * 100.0)
    return {
        "speedup_ci_low": percentile(speedups, 0.025),
        "speedup_ci_high": percentile(speedups, 0.975),
        "delta_pct_ci_low": percentile(deltas, 0.025),
        "delta_pct_ci_high": percentile(deltas, 0.975),
    }


def svg_escape(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_document(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.small{font-size:12px}.label{font-size:13px}.title{font-size:16px;font-weight:600}</style>\n'
        + "\n".join(body) + "\n</svg>\n"
    )


def write_latency_svg(rows, output):
    width, height = 1000, 430
    body = ['<text class="title" x="500" y="25" text-anchor="middle">Numeric fix: pre/post host latency</text>']
    colors = {"pre": "#4c78a8", "post": "#e45756"}
    labels = {"baseline": "Origin-HVX", "scna-d8": "SCNA direct d8"}
    ymax = math.ceil(max(max(r["pre_median_us"], r["post_median_us"]) for r in rows) / 500) * 500
    for panel, mode in enumerate(("baseline", "scna-d8")):
        left, top, pw, ph = 70 + panel * 490, 55, 410, 300
        body.append(f'<text class="label" x="{left + pw/2}" y="45" text-anchor="middle">{labels[mode]}</text>')
        for tick in range(0, int(ymax) + 1, 500):
            y = top + ph - ph * tick / ymax
            body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
            if panel == 0:
                body.append(f'<text class="small" x="{left-8}" y="{y+4:.1f}" text-anchor="end">{tick}</text>')
        body.extend([
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>',
            f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>',
        ])
        subset = [r for r in rows if r["mode"] == mode]
        for rev in ("pre", "post"):
            points = []
            for index, row in enumerate(subset):
                x = left + 35 + index * (pw - 70) / 3
                value = row[f"{rev}_median_us"]
                y = top + ph - ph * value / ymax
                points.append(f"{x:.1f},{y:.1f}")
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[rev]}"/>')
                body.append(f'<text class="small" x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle">{value:.0f}</text>')
                if rev == "post":
                    body.append(f'<text class="small" x="{x:.1f}" y="{top+ph+20}" text-anchor="middle">{row["qo_len"]}</text>')
            body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[rev]}" stroke-width="2.5"/>')
    body.extend([
        '<text class="label" x="500" y="405" text-anchor="middle">Qo length</text>',
        '<text class="label" x="15" y="210" text-anchor="middle" transform="rotate(-90 15 210)">Median host latency (us, lower is better)</text>',
        '<line x1="405" y1="385" x2="430" y2="385" stroke="#4c78a8" stroke-width="3"/><text class="small" x="436" y="389">Pre-fix</text>',
        '<line x1="505" y1="385" x2="530" y2="385" stroke="#e45756" stroke-width="3"/><text class="small" x="536" y="389">Post-fix</text>',
    ])
    output.write_text(svg_document(width, height, body), encoding="utf-8")


def write_delta_svg(rows, output):
    width, height = 1000, 470
    left, top, pw, ph = 75, 45, 870, 340
    ymin, ymax = -10.0, 55.0
    ycoord = lambda value: top + ph - ph * (value - ymin) / (ymax - ymin)
    body = ['<text class="title" x="500" y="25" text-anchor="middle">Post-fix host latency change with hierarchical bootstrap 95% CI</text>']
    for tick in range(-10, 56, 10):
        y = ycoord(tick)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
        body.append(f'<text class="small" x="{left-8}" y="{y+4:.1f}" text-anchor="end">{tick}%</text>')
    body.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>',
        f'<line class="axis" x1="{left}" y1="{ycoord(0):.1f}" x2="{left+pw}" y2="{ycoord(0):.1f}"/>',
    ])
    colors = {"baseline": "#4c78a8", "scna-d8": "#f2a541"}
    labels = {"baseline": "Origin-HVX", "scna-d8": "SCNA direct d8"}
    q_values = (4, 8, 16, 32)
    for qi, qo in enumerate(q_values):
        center = left + (qi + 0.5) * pw / len(q_values)
        for mi, mode in enumerate(("baseline", "scna-d8")):
            row = next(r for r in rows if r["mode"] == mode and r["qo_len"] == qo)
            value = row["post_minus_pre_pct"]
            x = center + (-28 if mi == 0 else 28)
            zero = ycoord(0)
            y = ycoord(value)
            body.append(f'<rect x="{x-20}" y="{min(y,zero):.1f}" width="40" height="{abs(zero-y):.1f}" fill="{colors[mode]}"/>')
            low_y, high_y = ycoord(row["delta_pct_ci_low"]), ycoord(row["delta_pct_ci_high"])
            body.extend([
                f'<line x1="{x}" y1="{high_y:.1f}" x2="{x}" y2="{low_y:.1f}" stroke="#222"/>',
                f'<line x1="{x-6}" y1="{high_y:.1f}" x2="{x+6}" y2="{high_y:.1f}" stroke="#222"/>',
                f'<line x1="{x-6}" y1="{low_y:.1f}" x2="{x+6}" y2="{low_y:.1f}" stroke="#222"/>',
                f'<text class="small" x="{x}" y="{y-7:.1f}" text-anchor="middle">{value:+.1f}%</text>',
            ])
        body.append(f'<text class="small" x="{center}" y="{top+ph+20}" text-anchor="middle">{qo}</text>')
    body.extend([
        '<text class="label" x="510" y="430" text-anchor="middle">Qo length</text>',
        '<rect x="360" y="444" width="14" height="14" fill="#4c78a8"/><text class="small" x="380" y="456">Origin-HVX</text>',
        '<rect x="505" y="444" width="14" height="14" fill="#f2a541"/><text class="small" x="525" y="456">SCNA direct d8</text>',
    ])
    output.write_text(svg_document(width, height, body), encoding="utf-8")


def write_accuracy_svg(rows, output):
    width, height = 820, 420
    left, top, pw, ph = 90, 45, 670, 290
    log_min, log_max = -5.0, -2.0
    ycoord = lambda value: top + ph - ph * (math.log10(value) - log_min) / (log_max - log_min)
    body = ['<text class="title" x="410" y="25" text-anchor="middle">Worst FP32-reference RMSE across Qo values</text>']
    for exponent in (-5, -4, -3, -2):
        y = ycoord(10 ** exponent)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
        body.append(f'<text class="small" x="{left-8}" y="{y+4:.1f}" text-anchor="end">1e{exponent}</text>')
    gate_y = ycoord(0.002)
    body.append(f'<line x1="{left}" y1="{gate_y:.1f}" x2="{left+pw}" y2="{gate_y:.1f}" stroke="#c44" stroke-dasharray="6 4"/>')
    body.append(f'<text class="small" x="{left+pw-3}" y="{gate_y-5:.1f}" text-anchor="end">RMSE gate 0.002</text>')
    labels = {"baseline": "Origin-HVX", "scna-d8": "SCNA direct d8"}
    for mi, mode in enumerate(("baseline", "scna-d8")):
        center = left + (mi + 0.5) * pw / 2
        values = {
            "pre": max(r["pre_worst_rmse"] for r in rows if r["mode"] == mode),
            "post": max(r["post_worst_rmse"] for r in rows if r["mode"] == mode),
        }
        for ri, rev in enumerate(("pre", "post")):
            x = center + (-45 if ri == 0 else 45)
            y = ycoord(values[rev])
            base = ycoord(1e-5)
            color = "#4c78a8" if rev == "pre" else "#e45756"
            body.append(f'<rect x="{x-28}" y="{y:.1f}" width="56" height="{base-y:.1f}" fill="{color}"/>')
            body.append(f'<text class="small" x="{x}" y="{y-7:.1f}" text-anchor="middle">{values[rev]:.2e}</text>')
        body.append(f'<text class="label" x="{center}" y="{top+ph+24}" text-anchor="middle">{labels[mode]}</text>')
    body.extend([
        '<rect x="315" y="390" width="14" height="14" fill="#4c78a8"/><text class="small" x="335" y="402">Pre-fix</text>',
        '<rect x="430" y="390" width="14" height="14" fill="#e45756"/><text class="small" x="450" y="402">Post-fix</text>',
    ])
    output.write_text(svg_document(width, height, body), encoding="utf-8")


def write_stage_svg(stage_rows, output):
    width, height = 1120, 450
    body = ['<text class="title" x="560" y="25" text-anchor="middle">Qo=32 DSP stage medians (sum across KV heads)</text>']
    labels = {"baseline": "Origin-HVX", "scna-d8": "SCNA direct d8"}
    colors = {"pre": "#4c78a8", "post": "#e45756"}
    ymax = math.ceil(max(max(r["pre_median_us"], r["post_median_us"]) for r in stage_rows) / 100) * 100
    for panel, mode in enumerate(("baseline", "scna-d8")):
        left, top, pw, ph = 65 + panel * 550, 55, 490, 300
        body.append(f'<text class="label" x="{left+pw/2}" y="45" text-anchor="middle">{labels[mode]}</text>')
        for tick in range(0, int(ymax) + 1, 200):
            y = top + ph - ph * tick / ymax
            body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>')
            if panel == 0:
                body.append(f'<text class="small" x="{left-7}" y="{y+4:.1f}" text-anchor="end">{tick}</text>')
        subset = [r for r in stage_rows if r["mode"] == mode]
        step = pw / len(subset)
        for index, row in enumerate(subset):
            center = left + (index + 0.5) * step
            for ri, rev in enumerate(("pre", "post")):
                value = row[f"{rev}_median_us"]
                x = center + (-10 if ri == 0 else 10)
                y = top + ph - ph * value / ymax
                body.append(f'<rect x="{x-9}" y="{y:.1f}" width="18" height="{top+ph-y:.1f}" fill="{colors[rev]}"/>')
            body.append(f'<text class="small" x="{center}" y="{top+ph+15}" text-anchor="middle" transform="rotate(35 {center} {top+ph+15})">{svg_escape(row["stage"])}</text>')
    body.extend([
        '<text class="label" x="18" y="210" text-anchor="middle" transform="rotate(-90 18 210)">DSP qtimer median (us)</text>',
        '<rect x="470" y="420" width="14" height="14" fill="#4c78a8"/><text class="small" x="490" y="432">Pre-fix</text>',
        '<rect x="570" y="420" width="14" height="14" fill="#e45756"/><text class="small" x="590" y="432">Post-fix</text>',
    ])
    output.write_text(svg_document(width, height, body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    host = defaultdict(lambda: defaultdict(list))
    dsp_total = defaultdict(lambda: defaultdict(list))
    stage_samples = defaultdict(list)
    accuracy = defaultdict(list)
    temperatures = []
    file_audit = []

    for path in sorted(args.input_dir.glob("*.log")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        meta = match.groupdict()
        session = int(meta["session"])
        qo = int(meta["qo"])
        mode = meta["mode"]
        revision = meta["revision"]
        key = (mode, qo, revision)
        iteration_totals = defaultdict(float)
        iteration_stages = defaultdict(lambda: defaultdict(float))
        measured = 0
        compare_count = 0
        for line in path.read_text(errors="replace").splitlines():
            data = fields(line)
            if line.startswith("NUMERIC_FIX_RUN "):
                temperatures.append(float(data["temp_before_tenths_c"]) / 10.0)
            elif line.startswith("NUMERIC_FIX_RUN_END "):
                temperatures.append(float(data["temp_after_tenths_c"]) / 10.0)
            elif line.startswith("FIG8_ATTENTION_HOST_TIMING ") and data.get("phase") == "measure":
                host[key][session].append(float(data["host_elapsed_us"]))
                measured += 1
            elif line.startswith("FIG8_ATTENTION_TIMERS ") and data.get("phase") == "measure":
                iteration = int(data["iteration"])
                iteration_totals[iteration] += float(data["profiled_total"])
                for stage in STAGES:
                    iteration_stages[iteration][stage] += float(data.get(stage, 0.0))
            elif line.startswith("FIG8_ATTENTION_COMPARE "):
                accuracy[key].append({
                    "rmse": float(data["rmse"]),
                    "max_abs": float(data["max_abs_error"]),
                    "nonfinite": int(data["candidate_nonfinite"]),
                })
                compare_count += 1
        dsp_total[key][session].extend(iteration_totals.values())
        for stage_values in iteration_stages.values():
            for stage in STAGES:
                stage_samples[(mode, qo, revision, stage)].append(stage_values[stage])
        file_audit.append({"file": path.name, "measured": measured, "compare_count": compare_count})

    if len(file_audit) != 96:
        raise SystemExit(f"expected 96 ABBA logs, found {len(file_audit)}")
    if any(row["measured"] != 15 or row["compare_count"] != 1 for row in file_audit):
        raise SystemExit("one or more ABBA logs failed completeness audit")

    rows = []
    for mode in ("baseline", "scna-d8"):
        for qo in (4, 8, 16, 32):
            pre = host[(mode, qo, "pre")]
            post = host[(mode, qo, "post")]
            pre_values = [x for values in pre.values() for x in values]
            post_values = [x for values in post.values() for x in values]
            if len(pre_values) != 90 or len(post_values) != 90:
                raise SystemExit(f"expected 90 samples/revision for {mode} q{qo}")
            ci = hierarchical_bootstrap(pre, post, seed=810000 + qo + len(mode))
            pre_med = statistics.median(pre_values)
            post_med = statistics.median(post_values)
            pre_dsp = [x for values in dsp_total[(mode, qo, "pre")].values() for x in values]
            post_dsp = [x for values in dsp_total[(mode, qo, "post")].values() for x in values]
            pre_acc = accuracy[(mode, qo, "pre")]
            post_acc = accuracy[(mode, qo, "post")]
            rows.append({
                "mode": mode,
                "qo_len": qo,
                "sessions": 3,
                "samples_per_revision": len(pre_values),
                "pre_median_us": pre_med,
                "pre_p95_us": percentile(pre_values, 0.95),
                "post_median_us": post_med,
                "post_p95_us": percentile(post_values, 0.95),
                "post_minus_pre_pct": (post_med / pre_med - 1.0) * 100.0,
                "pre_over_post_speedup": pre_med / post_med,
                **ci,
                "pre_dsp_total_median_us": statistics.median(pre_dsp),
                "post_dsp_total_median_us": statistics.median(post_dsp),
                "pre_worst_rmse": max(x["rmse"] for x in pre_acc),
                "post_worst_rmse": max(x["rmse"] for x in post_acc),
                "pre_worst_max_abs": max(x["max_abs"] for x in pre_acc),
                "post_worst_max_abs": max(x["max_abs"] for x in post_acc),
                "pre_nonfinite": sum(x["nonfinite"] for x in pre_acc),
                "post_nonfinite": sum(x["nonfinite"] for x in post_acc),
            })

    csv_path = args.output_dir / "numeric_fix_performance_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "contract": {
            "hardware": "SM8750P 25091RP04C",
            "isa": "v81",
            "shape": "heads/kv-heads=12/2, kv=4096, head-dim=128",
            "qo": [4, 8, 16, 32],
            "modes": ["baseline", "scna-d8"],
            "sessions": 3,
            "warmup_per_leg": 5,
            "measured_per_leg": 15,
            "order": "pre-post-post-pre",
            "events": False,
            "bootstrap_draws": 10000,
        },
        "temperature_c": {
            "min": min(temperatures),
            "median": statistics.median(temperatures),
            "max": max(temperatures),
        },
        "file_audit": file_audit,
        "rows": rows,
    }
    (args.output_dir / "numeric_fix_performance_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    stage_rows = []
    for mode in ("baseline", "scna-d8"):
        for stage in STAGES:
            pre_values = stage_samples[(mode, 32, "pre", stage)]
            post_values = stage_samples[(mode, 32, "post", stage)]
            pre_median = statistics.median(pre_values)
            post_median = statistics.median(post_values)
            stage_rows.append({
                "mode": mode,
                "qo_len": 32,
                "stage": stage,
                "samples_per_revision": len(pre_values),
                "pre_median_us": pre_median,
                "post_median_us": post_median,
                "post_minus_pre_us": post_median - pre_median,
                "post_minus_pre_pct": ((post_median / pre_median - 1.0) * 100.0) if pre_median else None,
            })
    with (args.output_dir / "numeric_fix_q32_stage_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    (args.output_dir / "numeric_fix_q32_stage_summary.json").write_text(
        json.dumps(stage_rows, indent=2) + "\n", encoding="utf-8"
    )

    write_latency_svg(rows, args.output_dir / "numeric_fix_latency_pre_post.svg")
    write_delta_svg(rows, args.output_dir / "numeric_fix_latency_delta_ci.svg")
    write_accuracy_svg(rows, args.output_dir / "numeric_fix_accuracy_pre_post.svg")
    write_stage_svg(stage_rows, args.output_dir / "numeric_fix_q32_stage_pre_post.svg")


if __name__ == "__main__":
    main()
