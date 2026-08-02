#!/usr/bin/env python3
"""Analyze paired SCNA KV-pipeline runs and generate SVG figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


PAIR_RE = re.compile(r"(\w+)=([^\s]+)")
COMPONENTS = (
    "profiled_total", "safe_sm", "scna_exp", "q_load", "k_load", "v_load",
    "qk_dot", "core_acc", "o_scale", "o_store", "kv_dma_issue", "kv_dma_wait", "kv_transform",
)


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


def median_ci(values: list[float], seed_key: str) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else 0.0
        return value, value
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def median_ratio_ci(numerators: list[float], denominators: list[float], seed_key: str) -> tuple[float, float]:
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    ratios = []
    for _ in range(3000):
        numerator = statistics.median(rng.choices(numerators, k=len(numerators)))
        denominator = statistics.median(rng.choices(denominators, k=len(denominators)))
        ratios.append(numerator / denominator)
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def normalize(data: dict[str, str]) -> tuple[str, str, str, str, int, int, int]:
    mode = data.get("mode", "unknown")
    if mode.startswith("scna-"):
        function = data.get("scna_function", "exp2")
        if function in ("0", "1"):
            function = "exp" if function == "1" else "exp2"
        kernel = data.get("scna_kernel", "direct")
        if kernel in ("0", "1"):
            kernel = "tree" if kernel == "1" else "direct"
        pipeline = data.get("scna_pipeline", "off")
        if pipeline in ("0", "1"):
            pipeline = "on" if pipeline == "1" else "off"
        width = int(data.get("scna_width", "16"))
    else:
        function, kernel, pipeline, width = "none", "none", "off", 0
    return mode, function, kernel, pipeline, width, int(data.get("qo_len", "0")), int(data.get("iteration", "0"))


def svg_begin(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0}.axis{stroke:#30343b;stroke-width:1.4}.grid{stroke:#d9dde3;stroke-width:1}.tick{font-size:12px;fill:#3e4651}.label{font-size:14px;fill:#20242a}.title{font-size:19px;font-weight:700;fill:#15181d}.legend{font-size:12px;fill:#20242a}</style>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def latency_svg(rows: list[dict[str, object]], path: Path) -> tuple[str, str, str, int]:
    on_q32 = [row for row in rows if row["pipeline"] == "on" and int(row["qo_len"]) == 32]
    selected = min(on_q32, key=lambda row: float(row["total_median_us"]))
    config = (str(selected["mode"]), str(selected["function"]), str(selected["kernel"]), int(selected["scna_width"]))
    baseline = sorted((row for row in rows if row["mode"] == "baseline"), key=lambda row: int(row["qo_len"]))
    off = sorted((row for row in rows if (row["mode"], row["function"], row["kernel"], int(row["scna_width"])) == config
                  and row["pipeline"] == "off"), key=lambda row: int(row["qo_len"]))
    on = sorted((row for row in rows if (row["mode"], row["function"], row["kernel"], int(row["scna_width"])) == config
                 and row["pipeline"] == "on"), key=lambda row: int(row["qo_len"]))
    series = [("Baseline", "#252a31", baseline),
              (f"{config[0]} {config[1]} {config[2]} d{config[3]} off", "#d4553d", off),
              (f"{config[0]} {config[1]} {config[2]} d{config[3]} on", "#178f86", on)]
    width, height = 900, 510
    left, top, plot_w, plot_h = 82, 58, 620, 365
    ymax = max(float(row["total_ci_high_us"]) for _, _, values in series for row in values) * 1.12
    qs = [4, 8, 16, 32]
    sx = lambda q: left + qs.index(q) * plot_w / 3
    sy = lambda value: top + plot_h - value / ymax * plot_h
    out = svg_begin(width, height, "V81 SCNA KV Pipeline DSP Latency (95% Median CI)")
    for i in range(6):
        value = ymax * i / 5
        y = sy(value)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>']
    out += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>']
    for q in qs:
        out.append(f'<text x="{sx(q):.1f}" y="{top + plot_h + 23}" text-anchor="middle" class="tick">{q}</text>')
    out += [f'<text x="{left + plot_w / 2}" y="{height - 23}" text-anchor="middle" class="label">Query length (tokens)</text>',
            f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2})" class="label">DSP latency (us)</text>']
    for index, (label, color, values) in enumerate(series):
        points = " ".join(f'{sx(int(row["qo_len"])):.1f},{sy(float(row["total_median_us"])):.1f}' for row in values)
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in values:
            x, y = sx(int(row["qo_len"])), sy(float(row["total_median_us"]))
            hi, lo = sy(float(row["total_ci_high_us"])), sy(float(row["total_ci_low_us"]))
            out += [f'<line x1="{x:.1f}" y1="{hi:.1f}" x2="{x:.1f}" y2="{lo:.1f}" stroke="{color}"/>',
                    f'<line x1="{x - 4:.1f}" y1="{hi:.1f}" x2="{x + 4:.1f}" y2="{hi:.1f}" stroke="{color}"/>',
                    f'<line x1="{x - 4:.1f}" y1="{lo:.1f}" x2="{x + 4:.1f}" y2="{lo:.1f}" stroke="{color}"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>']
        ly = 82 + index * 31
        out += [f'<line x1="724" y1="{ly}" x2="752" y2="{ly}" stroke="{color}" stroke-width="2.5"/>',
                f'<circle cx="738" cy="{ly}" r="4" fill="{color}"/>',
                f'<text x="760" y="{ly + 4}" class="legend">{escape(label)}</text>']
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return config


def speedup_svg(rows: list[dict[str, object]], path: Path) -> None:
    q32_on = [row for row in rows if row["pipeline"] == "on" and row["function"] == "exp2" and int(row["qo_len"]) == 32]
    styles = {
        ("scna-fp16", "direct"): "#d4553d", ("scna-fp16", "tree"): "#8f2f6d",
        ("scna-int8", "direct"): "#167fbc", ("scna-int8", "tree"): "#178f86",
    }
    width, height = 820, 470
    left, top, plot_w, plot_h = 78, 55, 550, 325
    widths = (8, 16, 32)
    series = []
    for key, color in styles.items():
        values = [next(float(row["pipeline_speedup"]) for row in q32_on
                       if (row["mode"], row["kernel"], int(row["scna_width"])) == (key[0], key[1], model_width))
                  for model_width in widths]
        series.append((f"{key[0]} {key[1]}", color, values))
    interval_rows = {(str(row["mode"]), str(row["kernel"]), int(row["scna_width"])): row for row in q32_on}
    ymin = min(0.9, min(float(row["pipeline_speedup_ci_low"]) for row in q32_on) * 0.95)
    ymax = max(float(row["pipeline_speedup_ci_high"]) for row in q32_on) * 1.08
    sx = lambda index: left + index * plot_w / 2
    sy = lambda value: top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
    out = svg_begin(width, height, "Exp2 Pipeline Speedup at qo_len=32")
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = sy(value)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.2f}x</text>']
    out += [f'<line x1="{left}" y1="{sy(1.0):.1f}" x2="{left + plot_w}" y2="{sy(1.0):.1f}" stroke="#30343b" stroke-dasharray="5 4"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>']
    for index, model_width in enumerate(widths):
        out.append(f'<text x="{sx(index):.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">d{model_width}</text>')
    out += [f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" class="label">SCNA width</text>',
            f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2})" class="label">Pipeline off / on speedup</text>']
    for index, (label, color, values) in enumerate(series):
        points = " ".join(f"{sx(i):.1f},{sy(value):.1f}" for i, value in enumerate(values))
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for i, value in enumerate(values):
            row = interval_rows[(label.split()[0], label.split()[1], widths[i])]
            hi = sy(float(row["pipeline_speedup_ci_high"]))
            lo = sy(float(row["pipeline_speedup_ci_low"]))
            out += [f'<line x1="{sx(i):.1f}" y1="{hi:.1f}" x2="{sx(i):.1f}" y2="{lo:.1f}" stroke="{color}"/>',
                    f'<line x1="{sx(i) - 4:.1f}" y1="{hi:.1f}" x2="{sx(i) + 4:.1f}" y2="{hi:.1f}" stroke="{color}"/>',
                    f'<line x1="{sx(i) - 4:.1f}" y1="{lo:.1f}" x2="{sx(i) + 4:.1f}" y2="{lo:.1f}" stroke="{color}"/>',
                    f'<circle cx="{sx(i):.1f}" cy="{sy(value):.1f}" r="4" fill="{color}"/>',
                    f'<text x="{sx(i):.1f}" y="{sy(value) - 8:.1f}" text-anchor="middle" class="tick">{value:.2f}x</text>']
        ly = 83 + index * 29
        out += [f'<line x1="650" y1="{ly}" x2="677" y2="{ly}" stroke="{color}" stroke-width="2.3"/>',
                f'<circle cx="663.5" cy="{ly}" r="4" fill="{color}"/>',
                f'<text x="684" y="{ly + 4}" class="legend">{escape(label)}</text>']
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def breakdown_svg(rows: list[dict[str, object]], config: tuple[str, str, str, int], path: Path) -> None:
    values = {str(row["pipeline"]): row for row in rows
              if (row["mode"], row["function"], row["kernel"], int(row["scna_width"]), int(row["qo_len"]))
              == (config[0], config[1], config[2], config[3], 32)}
    metrics = [("DSP total", "total_median_us"), ("K+V load", "kv_load_median_us"),
               ("DMA wait", "kv_dma_wait_median_us"), ("KV transform", "kv_transform_median_us")]
    width, height = 820, 480
    left, top, plot_w, plot_h = 78, 58, 560, 330
    ymax = max(float(values[p][field]) for _, field in metrics for p in ("off", "on")) * 1.12
    sy = lambda value: top + plot_h - value / ymax * plot_h
    group_w = plot_w / len(metrics)
    bar_w = 38
    out = svg_begin(width, height, "qo_len=32 Pipeline Timing Breakdown")
    for i in range(6):
        value = ymax * i / 5
        y = sy(value)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>']
    out += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
            f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2})" class="label">Median latency (us)</text>']
    colors = {"off": "#d4553d", "on": "#178f86"}
    for index, (label, field) in enumerate(metrics):
        center = left + (index + 0.5) * group_w
        for offset, pipeline in ((-bar_w / 2, "off"), (bar_w / 2, "on")):
            value = float(values[pipeline][field])
            x = center + offset - bar_w / 2
            y = sy(value)
            out += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + plot_h - y:.1f}" fill="{colors[pipeline]}"/>',
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="tick">{value:.0f}</text>']
        out.append(f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{escape(label)}</text>')
    for index, pipeline in enumerate(("off", "on")):
        y = 90 + index * 31
        out += [f'<rect x="670" y="{y - 10}" width="16" height="16" fill="{colors[pipeline]}"/>',
                f'<text x="695" y="{y + 3}" class="legend">pipeline {pipeline}</text>']
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    host: dict[tuple[str, str, str, str, int, int, int], float] = {}
    dsp: dict[tuple[str, str, str, str, int, int, int], list[dict[str, float]]] = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        if "phase=measure" not in line:
            continue
        data = fields(line)
        if "FIG8_ATTENTION_HOST_TIMING" in line:
            host[normalize(data)] = float(data["host_elapsed_us"])
        elif "FIG8_ATTENTION_TIMERS" in line:
            dsp[normalize(data)].append({name: float(data.get(name, "0")) for name in COMPONENTS})

    samples: dict[tuple[str, str, str, str, int, int], list[dict[str, float]]] = defaultdict(list)
    raw_rows: list[dict[str, object]] = []
    for (mode, function, kernel, pipeline, model_width, qo_len, iteration), records in sorted(dsp.items()):
        sample = {name: sum(record[name] for record in records) for name in COMPONENTS}
        sample["host_elapsed"] = host[(mode, function, kernel, pipeline, model_width, qo_len, iteration)]
        key = (mode, function, kernel, pipeline, model_width, qo_len)
        samples[key].append(sample)
        raw_rows.append({"mode": mode, "function": function, "kernel": kernel, "pipeline": pipeline,
                         "scna_width": model_width, "qo_len": qo_len, "iteration": iteration, **sample})

    rows: list[dict[str, object]] = []
    for key, values in sorted(samples.items()):
        mode, function, kernel, pipeline, model_width, qo_len = key
        total = [value["profiled_total"] for value in values]
        host_values = [value["host_elapsed"] for value in values]
        total_ci = median_ci(total, "-".join(map(str, key)) + "-dsp")
        host_ci = median_ci(host_values, "-".join(map(str, key)) + "-host")
        row: dict[str, object] = {
            "mode": mode, "function": function, "kernel": kernel, "pipeline": pipeline,
            "scna_width": model_width, "qo_len": qo_len, "samples": len(values),
            "total_median_us": statistics.median(total), "total_p50_us": percentile(total, 0.5),
            "total_p95_us": percentile(total, 0.95), "total_ci_low_us": total_ci[0],
            "total_ci_high_us": total_ci[1], "host_median_us": statistics.median(host_values),
            "host_p95_us": percentile(host_values, 0.95), "host_ci_low_us": host_ci[0],
            "host_ci_high_us": host_ci[1],
        }
        for name in COMPONENTS[1:]:
            row[f"{name}_median_us"] = statistics.median(value[name] for value in values)
        row["kv_load_median_us"] = float(row["k_load_median_us"]) + float(row["v_load_median_us"])
        row["host_control_median_us"] = float(row["host_median_us"]) - float(row["total_median_us"])
        row["scna_share_pct"] = 100.0 * float(row["scna_exp_median_us"]) / float(row["total_median_us"])
        rows.append(row)

    baseline = {int(row["qo_len"]): float(row["total_median_us"]) for row in rows if row["mode"] == "baseline"}
    off = {(row["mode"], row["function"], row["kernel"], int(row["scna_width"]), int(row["qo_len"])): row
           for row in rows if row["pipeline"] == "off"}
    for row in rows:
        row["baseline_speedup"] = baseline[int(row["qo_len"])] / float(row["total_median_us"])
        if row["mode"] == "baseline":
            row["pipeline_speedup"] = 1.0
            row["pipeline_speedup_ci_low"] = 1.0
            row["pipeline_speedup_ci_high"] = 1.0
            row["host_pipeline_speedup"] = 1.0
            row["host_pipeline_speedup_ci_low"] = 1.0
            row["host_pipeline_speedup_ci_high"] = 1.0
        else:
            reference = off[(row["mode"], row["function"], row["kernel"], int(row["scna_width"]), int(row["qo_len"]))]
            row["pipeline_speedup"] = float(reference["total_median_us"]) / float(row["total_median_us"])
            row["host_pipeline_speedup"] = float(reference["host_median_us"]) / float(row["host_median_us"])
            if row["pipeline"] == "on":
                off_key = (str(row["mode"]), str(row["function"]), str(row["kernel"]), "off",
                           int(row["scna_width"]), int(row["qo_len"]))
                on_key = (str(row["mode"]), str(row["function"]), str(row["kernel"]), "on",
                          int(row["scna_width"]), int(row["qo_len"]))
                speedup_ci = median_ratio_ci(
                    [value["profiled_total"] for value in samples[off_key]],
                    [value["profiled_total"] for value in samples[on_key]], "-".join(map(str, on_key)) + "-ratio")
                host_speedup_ci = median_ratio_ci(
                    [value["host_elapsed"] for value in samples[off_key]],
                    [value["host_elapsed"] for value in samples[on_key]], "-".join(map(str, on_key)) + "-host-ratio")
                row["pipeline_speedup_ci_low"], row["pipeline_speedup_ci_high"] = speedup_ci
                row["host_pipeline_speedup_ci_low"], row["host_pipeline_speedup_ci_high"] = host_speedup_ci
            else:
                row["pipeline_speedup_ci_low"] = 1.0
                row["pipeline_speedup_ci_high"] = 1.0
                row["host_pipeline_speedup_ci_low"] = 1.0
                row["host_pipeline_speedup_ci_high"] = 1.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, output_rows in (("iteration_samples.csv", raw_rows), ("summary.csv", rows)):
        with (args.out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(output_rows)

    selected = latency_svg(rows, args.out_dir / "pipeline_latency.svg")
    speedup_svg(rows, args.out_dir / "pipeline_speedup_q32.svg")
    breakdown_svg(rows, selected, args.out_dir / "pipeline_breakdown_q32.svg")

    selected_on = next(row for row in rows if row["pipeline"] == "on" and int(row["qo_len"]) == 32 and
                       (row["mode"], row["function"], row["kernel"], int(row["scna_width"])) == selected)
    selected_off = off[(selected[0], selected[1], selected[2], selected[3], 32)]
    report = [
        "# V81 SCNA KV Pipeline Summary", "",
        "Generated from adjacent pipeline-off/on runs. Confidence intervals are deterministic bootstrap 95% CIs of the median.", "",
        "![Pipeline latency](pipeline_latency.svg)", "",
        "![Pipeline speedup](pipeline_speedup_q32.svg)", "",
        "![Pipeline breakdown](pipeline_breakdown_q32.svg)", "",
        "## Selected q32 configuration", "",
        f"- Configuration: {selected[0]} {selected[1]} {selected[2]} d{selected[3]}",
        f"- DSP median: {float(selected_off['total_median_us']):.1f} us off -> {float(selected_on['total_median_us']):.1f} us on ({float(selected_on['pipeline_speedup']):.3f}x)",
        f"- DSP speedup 95% CI: [{float(selected_on['pipeline_speedup_ci_low']):.3f}x, {float(selected_on['pipeline_speedup_ci_high']):.3f}x]",
        f"- Host speedup: {float(selected_on['host_pipeline_speedup']):.3f}x, 95% CI [{float(selected_on['host_pipeline_speedup_ci_low']):.3f}x, {float(selected_on['host_pipeline_speedup_ci_high']):.3f}x]",
        f"- K+V median: {float(selected_off['kv_load_median_us']):.1f} us off -> {float(selected_on['kv_load_median_us']):.1f} us on",
        f"- Pipeline DMA issue/wait/transform: {float(selected_on['kv_dma_issue_median_us']):.1f} / {float(selected_on['kv_dma_wait_median_us']):.1f} / {float(selected_on['kv_transform_median_us']):.1f} us", "",
        "## q32 matrix", "",
        "| Mode | Function | Kernel | Width | Off DSP us | On DSP us | Pipeline speedup [95% CI] | Baseline speedup | On K+V us | DMA wait us |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["pipeline"] != "on" or int(row["qo_len"]) != 32:
            continue
        reference = off[(row["mode"], row["function"], row["kernel"], int(row["scna_width"]), 32)]
        report.append(f"| {row['mode']} | {row['function']} | {row['kernel']} | {row['scna_width']} | "
                      f"{float(reference['total_median_us']):.1f} | {float(row['total_median_us']):.1f} | "
                      f"{float(row['pipeline_speedup']):.3f}x [{float(row['pipeline_speedup_ci_low']):.3f}, {float(row['pipeline_speedup_ci_high']):.3f}] | "
                      f"{float(row['baseline_speedup']):.3f}x | "
                      f"{float(row['kv_load_median_us']):.1f} | {float(row['kv_dma_wait_median_us']):.1f} |")
    (args.out_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
