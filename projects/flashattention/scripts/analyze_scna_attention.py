#!/usr/bin/env python3
"""Summarize v81 SCNA attention logs and generate reproducible SVG figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


PAIR_RE = re.compile(r"(\w+)=([^\s]+)")
COMPONENTS = ("profiled_total", "safe_sm", "scna_exp", "q_load", "k_load", "v_load", "qk_dot", "core_acc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


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
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    boot = [statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)]
    return percentile(boot, 0.025), percentile(boot, 0.975)


def normalize_config(data: dict[str, str]) -> tuple[str, str, str, int, int, int]:
    mode = data.get("mode", "unknown")
    if mode.startswith("scna-"):
        function = data.get("scna_function", data.get("function", "exp2"))
        if function in ("0", "1"):
            function = "exp" if function == "1" else "exp2"
        kernel = data.get("scna_kernel", data.get("kernel", "direct"))
        if kernel in ("0", "1"):
            kernel = "tree" if kernel == "1" else "direct"
        width = int(data.get("scna_width", data.get("width", "16")))
    else:
        function, kernel, width = "none", "none", 0
    return mode, function, kernel, width, int(data.get("qo_len", "0")), int(data.get("iteration", "0"))


def svg_begin(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0}.axis{stroke:#30343b;stroke-width:1.4}.grid{stroke:#d9dde3;stroke-width:1}.tick{font-size:12px;fill:#3e4651}.label{font-size:14px;fill:#20242a}.title{font-size:19px;font-weight:700;fill:#15181d}.legend{font-size:12px;fill:#20242a}</style>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def render_latency_svg(rows: list[dict[str, object]], path: Path) -> list[str]:
    baseline = {int(row["qo_len"]): row for row in rows if row["mode"] == "baseline"}
    selected: dict[str, tuple[str, str, int]] = {}
    for mode in ("scna-fp16", "scna-int8"):
        candidates = [row for row in rows if row["mode"] == mode and int(row["qo_len"]) == 32]
        if candidates:
            best = min(candidates, key=lambda row: float(row["total_median_us"]))
            selected[mode] = (str(best["function"]), str(best["kernel"]), int(best["scna_width"]))
    series: list[tuple[str, str, list[dict[str, object]]]] = []
    if baseline:
        series.append(("Baseline", "#252a31", [baseline[q] for q in sorted(baseline)]))
    colors = {"scna-fp16": "#d4553d", "scna-int8": "#178f86"}
    for mode, config in selected.items():
        values = [row for row in rows if row["mode"] == mode and
                  (row["function"], row["kernel"], int(row["scna_width"])) == config]
        values.sort(key=lambda row: int(row["qo_len"]))
        label = f"{mode} {config[0]} {config[1]} d{config[2]}"
        series.append((label, colors[mode], values))
    if not series:
        return []
    width, height = 860, 500
    left, top, plot_w, plot_h = 82, 58, 610, 360
    all_values = [float(row["total_ci_high_us"]) for _, _, values in series for row in values]
    ymax = max(all_values) * 1.12 if all_values else 1.0
    q_values = [4, 8, 16, 32]
    sx = lambda q: left + q_values.index(q) * plot_w / (len(q_values) - 1)
    sy = lambda value: top + plot_h - value / ymax * plot_h
    out = svg_begin(width, height, "V81 FlashAttention DSP Latency (95% Median CI)")
    for i in range(6):
        value = ymax * i / 5
        y = sy(value)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>']
    out += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>']
    for q in q_values:
        x = sx(q)
        out.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{q}</text>')
    out += [f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" class="label">Query length (tokens)</text>',
            f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2})" class="label">DSP latency (us)</text>']
    for index, (label, color, values) in enumerate(series):
        points = " ".join(f'{sx(int(row["qo_len"])):.1f},{sy(float(row["total_median_us"])):.1f}' for row in values)
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in values:
            x = sx(int(row["qo_len"]))
            y = sy(float(row["total_median_us"]))
            lo, hi = sy(float(row["total_ci_low_us"])), sy(float(row["total_ci_high_us"]))
            out += [f'<line x1="{x:.1f}" y1="{hi:.1f}" x2="{x:.1f}" y2="{lo:.1f}" stroke="{color}"/>',
                    f'<line x1="{x - 4:.1f}" y1="{hi:.1f}" x2="{x + 4:.1f}" y2="{hi:.1f}" stroke="{color}"/>',
                    f'<line x1="{x - 4:.1f}" y1="{lo:.1f}" x2="{x + 4:.1f}" y2="{lo:.1f}" stroke="{color}"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>']
        ly = 82 + index * 30
        out += [f'<line x1="715" y1="{ly}" x2="745" y2="{ly}" stroke="{color}" stroke-width="2.5"/>',
                f'<circle cx="730" cy="{ly}" r="4" fill="{color}"/>',
                f'<text x="752" y="{ly + 4}" class="legend">{escape(label)}</text>']
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return [label for label, _, _ in series]


def render_pareto_svg(rows: list[dict[str, object]], path: Path) -> None:
    points = [row for row in rows if str(row["mode"]).startswith("scna-") and int(row["qo_len"]) == 32]
    if not points:
        return
    width, height = 900, 540
    left, top, plot_w, plot_h = 90, 58, 620, 390
    xmax = max(float(row["micro_dense_rmse"]) for row in points) * 1.08
    ymin = min(float(row["total_ci_low_us"]) for row in points) * 0.92
    ymax = max(float(row["total_ci_high_us"]) for row in points) * 1.08
    sx = lambda value: left + value / xmax * plot_w
    sy = lambda value: top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
    styles = {
        ("scna-fp16", "direct"): ("#d4553d", "circle"),
        ("scna-fp16", "tree"): ("#8f2f6d", "square"),
        ("scna-int8", "direct"): ("#167fbc", "circle"),
        ("scna-int8", "tree"): ("#178f86", "square"),
    }
    out = svg_begin(width, height, "SCNA Accuracy-Latency Pareto at qo_len=32 (95% Median CI)")
    for i in range(6):
        xvalue = xmax * i / 5
        x = sx(xvalue)
        out += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>',
                f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{xvalue:.3f}</text>']
        yvalue = ymin + (ymax - ymin) * i / 5
        y = sy(yvalue)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{yvalue:.0f}</text>']
    out += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" class="label">Dense exp approximation RMSE</text>',
            f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2})" class="label">DSP latency (us)</text>']
    for row in points:
        color, shape = styles[(str(row["mode"]), str(row["kernel"]))]
        x = sx(float(row["micro_dense_rmse"]))
        y = sy(float(row["total_median_us"]))
        lo, hi = sy(float(row["total_ci_low_us"])), sy(float(row["total_ci_high_us"]))
        out += [f'<line x1="{x:.1f}" y1="{hi:.1f}" x2="{x:.1f}" y2="{lo:.1f}" stroke="{color}"/>']
        if shape == "circle":
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        else:
            out.append(f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" fill="{color}"/>')
        short_fn = "e2" if row["function"] == "exp2" else "e"
        out.append(f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" class="tick">{short_fn}/d{row["scna_width"]}</text>')
    for index, ((mode, kernel), (color, shape)) in enumerate(styles.items()):
        ly = 90 + index * 30
        if shape == "circle":
            out.append(f'<circle cx="742" cy="{ly}" r="4.5" fill="{color}"/>')
        else:
            out.append(f'<rect x="737.5" y="{ly - 4.5}" width="9" height="9" fill="{color}"/>')
        out.append(f'<text x="756" y="{ly + 4}" class="legend">{mode} {kernel}</text>')
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def render_micro_speedup_svg(micro: dict[tuple[str, str, str, int], list[dict[str, float]]], path: Path) -> None:
    width, height = 760, 450
    left, top, plot_w, plot_h = 78, 55, 500, 310
    widths = (8, 16, 32)
    series = []
    for mode, color in (("scna-fp16", "#d4553d"), ("scna-int8", "#178f86")):
        values = []
        for model_width in widths:
            direct = micro.get((mode, "exp2", "direct", model_width), [])
            tree = micro.get((mode, "exp2", "tree", model_width), [])
            direct_time = statistics.median(item["pair_ns_per_vector"] for item in direct)
            tree_time = statistics.median(item["pair_ns_per_vector"] for item in tree)
            values.append(direct_time / tree_time)
        series.append((mode, color, values))
    ymax = max(value for _, _, values in series for value in values) * 1.12
    sx = lambda index: left + index * plot_w / 2
    sy = lambda value: top + plot_h - value / ymax * plot_h
    out = svg_begin(width, height, "Branchless Tree Paired-Vector Speedup on V81")
    for i in range(6):
        value = ymax * i / 5
        y = sy(value)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.1f}x</text>']
    out += [f'<line x1="{left}" y1="{sy(1.0):.1f}" x2="{left + plot_w}" y2="{sy(1.0):.1f}" stroke="#30343b" stroke-dasharray="5 4"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>']
    for index, model_width in enumerate(widths):
        out.append(f'<text x="{sx(index):.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">d{model_width}</text>')
    out += [f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" class="label">SCNA width</text>',
            f'<text x="20" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2})" class="label">Direct / tree speedup</text>']
    for index, (label, color, values) in enumerate(series):
        points = " ".join(f"{sx(i):.1f},{sy(value):.1f}" for i, value in enumerate(values))
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i, value in enumerate(values):
            out += [f'<circle cx="{sx(i):.1f}" cy="{sy(value):.1f}" r="4" fill="{color}"/>',
                    f'<text x="{sx(i):.1f}" y="{sy(value) - 9:.1f}" text-anchor="middle" class="tick">{value:.2f}x</text>']
        ly = 88 + index * 30
        out += [f'<line x1="610" y1="{ly}" x2="640" y2="{ly}" stroke="{color}" stroke-width="2.5"/>',
                f'<circle cx="625" cy="{ly}" r="4" fill="{color}"/>',
                f'<text x="648" y="{ly + 4}" class="legend">{label}</text>']
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    host: dict[tuple[str, str, str, int, int, int], float] = {}
    dsp: dict[tuple[str, str, str, int, int, int], list[dict[str, float]]] = defaultdict(list)
    micro: dict[tuple[str, str, str, int], list[dict[str, float]]] = defaultdict(list)

    for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        if "SCNA_EXP_BENCH" in line:
            data = fields(line)
            try:
                key = (data["mode"], data["function"], data["kernel"], int(data["width"]))
                iters = float(data["iters"])
                micro[key].append({
                    "single_ns_per_vector": float(data["elapsed_us"]) * 1000.0 / iters,
                    "pair_ns_per_vector": float(data["pair_elapsed_us"]) * 1000.0 / (2.0 * iters),
                    "rmse": float(data["rmse"]),
                    "max_abs": float(data["max_abs_error"]),
                    "dense_rmse": float(data["dense_rmse"]),
                    "dense_max_abs": float(data["dense_max_abs_error"]),
                    "direct_tree_max_abs": float(data["direct_tree_max_abs_diff"]),
                })
            except (KeyError, ValueError):
                pass
            continue
        if "phase=measure" not in line:
            continue
        data = fields(line)
        if "FIG8_ATTENTION_HOST_TIMING" in line:
            try:
                host[normalize_config(data)] = float(data["host_elapsed_us"])
            except (KeyError, ValueError):
                pass
            continue
        if "FIG8_ATTENTION_TIMERS" not in line:
            continue
        try:
            key = normalize_config(data)
            dsp[key].append({name: float(data.get(name, "0")) for name in COMPONENTS})
        except ValueError:
            pass

    samples: dict[tuple[str, str, str, int, int], list[dict[str, float]]] = defaultdict(list)
    raw_rows: list[dict[str, object]] = []
    for (mode, function, kernel, model_width, qo_len, iteration), records in sorted(dsp.items()):
        sample = {name: sum(record[name] for record in records) for name in COMPONENTS}
        sample["host_elapsed"] = host.get((mode, function, kernel, model_width, qo_len, iteration), 0.0)
        samples[(mode, function, kernel, model_width, qo_len)].append(sample)
        raw_rows.append({"mode": mode, "function": function, "kernel": kernel, "scna_width": model_width,
                         "qo_len": qo_len, "iteration": iteration, **sample})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_fields = list(raw_rows[0]) if raw_rows else ["mode", "function", "kernel", "scna_width", "qo_len", "iteration"]
    with (args.out_dir / "iteration_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw_rows)

    rows: list[dict[str, object]] = []
    for (mode, function, kernel, model_width, qo_len), values in sorted(samples.items()):
        total = [sample["profiled_total"] for sample in values]
        host_values = [sample["host_elapsed"] for sample in values]
        total_ci = median_ci(total, f"{mode}-{function}-{kernel}-{model_width}-{qo_len}-dsp")
        host_ci = median_ci(host_values, f"{mode}-{function}-{kernel}-{model_width}-{qo_len}-host")
        row: dict[str, object] = {
            "mode": mode, "function": function, "kernel": kernel, "scna_width": model_width,
            "qo_len": qo_len, "samples": len(values),
            "total_median_us": statistics.median(total), "total_p50_us": percentile(total, 0.5),
            "total_p95_us": percentile(total, 0.95), "total_ci_low_us": total_ci[0], "total_ci_high_us": total_ci[1],
            "host_median_us": statistics.median(host_values), "host_p50_us": percentile(host_values, 0.5),
            "host_p95_us": percentile(host_values, 0.95), "host_ci_low_us": host_ci[0], "host_ci_high_us": host_ci[1],
        }
        for name in COMPONENTS[1:]:
            row[f"{name}_median_us"] = statistics.median(sample[name] for sample in values)
        row["scna_share_pct"] = 100.0 * float(row["scna_exp_median_us"]) / float(row["total_median_us"])
        row["host_control_median_us"] = float(row["host_median_us"]) - float(row["total_median_us"])
        micro_values = micro.get((mode, function, kernel, model_width), [])
        for metric in ("single_ns_per_vector", "pair_ns_per_vector", "rmse", "max_abs", "dense_rmse",
                       "dense_max_abs", "direct_tree_max_abs"):
            row[f"micro_{metric}"] = statistics.median(item[metric] for item in micro_values) if micro_values else 0.0
        rows.append(row)

    baseline = {int(row["qo_len"]): float(row["total_median_us"]) for row in rows if row["mode"] == "baseline"}
    for row in rows:
        reference = baseline.get(int(row["qo_len"]), 0.0)
        row["baseline_speedup"] = reference / float(row["total_median_us"]) if reference else 0.0

    fields_out = list(rows[0]) if rows else ["mode", "function", "kernel", "scna_width", "qo_len"]
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    pareto = [row for row in rows if str(row["mode"]).startswith("scna-")]
    with (args.out_dir / "pareto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out, lineterminator="\n")
        writer.writeheader()
        writer.writerows(pareto)

    selected_labels = render_latency_svg(rows, args.out_dir / "attention_latency.svg")
    render_pareto_svg(rows, args.out_dir / "pareto_q32.svg")
    if micro:
        render_micro_speedup_svg(micro, args.out_dir / "micro_tree_speedup.svg")

    report = [
        "# V81 SCNA FlashAttention Summary", "",
        "Generated from measured per-iteration host/DSP logs. Confidence intervals are deterministic bootstrap 95% CIs of the median.", "",
        "![DSP latency](attention_latency.svg)", "", "![Accuracy-latency Pareto](pareto_q32.svg)", "",
        "![Tree microkernel speedup](micro_tree_speedup.svg)", "",
        "## Selected latency series", "",
    ]
    report.extend(f"- {label}" for label in selected_labels)
    report += ["", "## Measurements", "",
               "| Mode | Function | Kernel | Width | Qo | DSP median us | DSP p95 us | 95% CI us | Host median us | SCNA share | Baseline speedup | Dense RMSE |",
               "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(
            f'| {row["mode"]} | {row["function"]} | {row["kernel"]} | {row["scna_width"]} | {row["qo_len"]} | '
            f'{float(row["total_median_us"]):.1f} | {float(row["total_p95_us"]):.1f} | '
            f'[{float(row["total_ci_low_us"]):.1f}, {float(row["total_ci_high_us"]):.1f}] | '
            f'{float(row["host_median_us"]):.1f} | {float(row["scna_share_pct"]):.1f}% | '
            f'{float(row["baseline_speedup"]):.3f}x | {float(row["micro_dense_rmse"]):.6g} |')
    (args.out_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
