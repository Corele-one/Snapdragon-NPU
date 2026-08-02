#!/usr/bin/env python3

import argparse
import csv
import html
import math
import random
import statistics
import zlib
from collections import defaultdict
from pathlib import Path


POLICIES = ("legacy", "spin", "predictive")
COLORS = {
    "legacy": "#4e79a7",
    "spin": "#e15759",
    "predictive": "#59a14f",
}
QO_LENGTHS = (4, 8, 16, 32)
NUMERIC_FIELDS = {
    "host_cpu_requested": int,
    "host_cpu_actual": int,
    "qo_len": int,
    "kv_len": int,
    "n_heads": int,
    "n_kv_heads": int,
    "head_dim": int,
    "iteration": int,
    "host_wall_us": float,
    "host_thread_cpu_us": float,
    "wait_wall_us": float,
    "wait_thread_cpu_us": float,
    "sleep_us": float,
    "spin_us": float,
    "poll_count": float,
    "predicted_us": float,
    "prediction_error_us": float,
    "descriptor_prepare_us": float,
    "dsp_mapping_us": float,
    "dsp_validate_in_us": float,
    "dsp_compute_us": float,
    "dsp_validate_out_us": float,
    "dsp_dispatch_total_us": float,
    "ret": int,
}


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def stable_rng(label):
    return random.Random(zlib.crc32(label.encode("utf-8")))


def bootstrap_median_ci(values, label, repeats=4000):
    if not values:
        return math.nan, math.nan
    rng = stable_rng(label)
    count = len(values)
    samples = []
    for _ in range(repeats):
        draw = [values[rng.randrange(count)] for _ in range(count)]
        samples.append(statistics.median(draw))
    return quantile(samples, 0.025), quantile(samples, 0.975)


def bootstrap_ratio_ci(numerator, denominator, label, repeats=4000):
    if not numerator or not denominator:
        return math.nan, math.nan
    rng = stable_rng(label)
    n_count = len(numerator)
    d_count = len(denominator)
    ratios = []
    for _ in range(repeats):
        n_draw = [numerator[rng.randrange(n_count)] for _ in range(n_count)]
        d_draw = [denominator[rng.randrange(d_count)] for _ in range(d_count)]
        d_median = statistics.median(d_draw)
        if d_median > 0:
            ratios.append(statistics.median(n_draw) / d_median)
    return quantile(ratios, 0.025), quantile(ratios, 0.975)


def bootstrap_normalized_speedup_ci(legacy_rows, policy_rows, label, repeats=4000):
    rng = stable_rng(label)
    legacy_count = len(legacy_rows)
    policy_count = len(policy_rows)
    ratios = []
    for _ in range(repeats):
        legacy_draw = [legacy_rows[rng.randrange(legacy_count)] for _ in range(legacy_count)]
        policy_draw = [policy_rows[rng.randrange(policy_count)] for _ in range(policy_count)]
        legacy_host = statistics.median(row["host_wall_us"] for row in legacy_draw)
        legacy_dsp = statistics.median(row["dsp_dispatch_total_us"] for row in legacy_draw)
        policy_control = statistics.median(
            row["host_wall_us"] - row["dsp_dispatch_total_us"] for row in policy_draw
        )
        denominator = legacy_dsp + policy_control
        if denominator > 0:
            ratios.append(legacy_host / denominator)
    return quantile(ratios, 0.025), quantile(ratios, 0.975)


def placement_name(row):
    requested = row["host_cpu_requested"]
    if requested < 0:
        return "unpinned"
    return f"{row['cpu_role']}-cpu{requested}"


def display_placement(name):
    if name == "unpinned":
        return "Unpinned"
    role, cpu = name.rsplit("-cpu", 1)
    return f"{role.replace('-', ' ').title()} CPU{cpu}"


def load_rows(input_dir):
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(NUMERIC_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            for raw in reader:
                row = dict(raw)
                for field, converter in NUMERIC_FIELDS.items():
                    row[field] = converter(row[field])
                row["source_file"] = path.name
                row["placement"] = placement_name(row)
                rows.append(row)
    if not rows:
        raise ValueError(f"No CSV files found under {input_dir}")
    return rows


def summarize(rows):
    if any(row["mode"] != "baseline" for row in rows):
        raise ValueError("CPU scheduling experiment contains a non-baseline kernel mode")
    if any(row["ret"] != 0 for row in rows):
        raise ValueError("At least one device iteration returned an error")
    if any(
        row["host_cpu_requested"] >= 0 and row["host_cpu_actual"] != row["host_cpu_requested"]
        for row in rows
    ):
        raise ValueError("A pinned measurement executed on a different host CPU")
    measured = [row for row in rows if row["phase"] == "measure" and row["ret"] == 0]
    output_hashes = defaultdict(set)
    for row in measured:
        output_hashes[(row["placement"], row["qo_len"])].add(row["output_hash"])
    if any(len(hashes) != 1 for hashes in output_hashes.values()):
        raise ValueError("Wait policies produced different output hashes in the main matrix")
    groups = defaultdict(list)
    for row in measured:
        groups[(row["placement"], row["qo_len"], row["wait_policy"])].append(row)

    summaries = []
    for key in sorted(groups, key=lambda item: (item[0], item[1], POLICIES.index(item[2]))):
        placement, qo_len, policy = key
        group = groups[key]
        host = [row["host_wall_us"] for row in group]
        thread_cpu = [row["host_thread_cpu_us"] for row in group]
        wait = [row["wait_wall_us"] for row in group]
        sleep = [row["sleep_us"] for row in group]
        spin = [row["spin_us"] for row in group]
        polls = [row["poll_count"] for row in group]
        pred_error = [row["prediction_error_us"] for row in group]
        dispatch = [row["dsp_dispatch_total_us"] for row in group]
        control = [row["host_wall_us"] - row["dsp_dispatch_total_us"] for row in group]
        utilization = [100.0 * row["host_thread_cpu_us"] / row["host_wall_us"] for row in group if row["host_wall_us"]]
        ci_low, ci_high = bootstrap_median_ci(host, f"host:{placement}:{qo_len}:{policy}")
        control_ci_low, control_ci_high = bootstrap_median_ci(
            control, f"control:{placement}:{qo_len}:{policy}"
        )
        legacy = groups.get((placement, qo_len, "legacy"), [])
        legacy_host = [row["host_wall_us"] for row in legacy]
        legacy_dispatch = [row["dsp_dispatch_total_us"] for row in legacy]
        legacy_control = [
            row["host_wall_us"] - row["dsp_dispatch_total_us"] for row in legacy
        ]
        speedup = statistics.median(legacy_host) / statistics.median(host)
        if policy == "legacy":
            speedup_low, speedup_high = 1.0, 1.0
        else:
            speedup_low, speedup_high = bootstrap_ratio_ci(
                legacy_host, host, f"speedup:{placement}:{qo_len}:{policy}"
            )
        normalized_denominator = statistics.median(legacy_dispatch) + statistics.median(control)
        normalized_speedup = statistics.median(legacy_host) / normalized_denominator
        if policy == "legacy":
            normalized_speedup = 1.0
            normalized_low, normalized_high = 1.0, 1.0
        else:
            normalized_low, normalized_high = bootstrap_normalized_speedup_ci(
                legacy, group, f"normalized:{placement}:{qo_len}:{policy}"
            )
        legacy_amdahl = statistics.median(legacy_host) / statistics.median(legacy_dispatch)
        dispatch_shift_pct = 100.0 * (
            statistics.median(dispatch) / statistics.median(legacy_dispatch) - 1.0
        )
        legacy_control_median = statistics.median(legacy_control)
        control_median = statistics.median(control)
        summaries.append({
            "placement": placement,
            "placement_label": display_placement(placement),
            "host_cpu_requested": group[0]["host_cpu_requested"],
            "cpu_role": group[0]["cpu_role"],
            "qo_len": qo_len,
            "wait_policy": policy,
            "samples": len(group),
            "host_median_us": statistics.median(host),
            "host_p50_us": quantile(host, 0.50),
            "host_p95_us": quantile(host, 0.95),
            "host_ci95_low_us": ci_low,
            "host_ci95_high_us": ci_high,
            "host_thread_cpu_median_us": statistics.median(thread_cpu),
            "host_thread_cpu_p95_us": quantile(thread_cpu, 0.95),
            "host_cpu_utilization_median_pct": statistics.median(utilization),
            "wait_median_us": statistics.median(wait),
            "sleep_median_us": statistics.median(sleep),
            "spin_median_us": statistics.median(spin),
            "poll_median": statistics.median(polls),
            "prediction_error_p50_us": quantile(pred_error, 0.50),
            "prediction_error_p95_us": quantile(pred_error, 0.95),
            "descriptor_prepare_median_us": statistics.median(row["descriptor_prepare_us"] for row in group),
            "dsp_mapping_median_us": statistics.median(row["dsp_mapping_us"] for row in group),
            "dsp_validate_in_median_us": statistics.median(row["dsp_validate_in_us"] for row in group),
            "dsp_compute_median_us": statistics.median(row["dsp_compute_us"] for row in group),
            "dsp_validate_out_median_us": statistics.median(row["dsp_validate_out_us"] for row in group),
            "dsp_dispatch_total_median_us": statistics.median(dispatch),
            "host_control_overhead_median_us": control_median,
            "host_control_ci95_low_us": control_ci_low,
            "host_control_ci95_high_us": control_ci_high,
            "control_reduction_vs_legacy_us": legacy_control_median - control_median,
            "control_reduction_vs_legacy_pct": 100.0 * (legacy_control_median - control_median) / legacy_control_median,
            "speedup_vs_legacy": speedup,
            "speedup_ci95_low": speedup_low,
            "speedup_ci95_high": speedup_high,
            "normalized_speedup_vs_legacy": normalized_speedup,
            "normalized_speedup_ci95_low": normalized_low,
            "normalized_speedup_ci95_high": normalized_high,
            "legacy_amdahl_zero_control_limit": legacy_amdahl,
            "dsp_dispatch_shift_vs_legacy_pct": dispatch_shift_pct,
            "observed_speedup_exceeds_amdahl": int(speedup > legacy_amdahl + 1.0e-9),
            "unique_output_hashes": len({row["output_hash"] for row in group}),
        })
    return measured, summaries


def fmt(value, digits=1):
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width, height, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222;letter-spacing:0}.title{font-size:24px;font-weight:700}'
        '.subtitle{font-size:13px;fill:#555}.axis{font-size:12px}.panel{font-size:15px;font-weight:700}'
        '.legend{font-size:13px}.grid{stroke:#d9dde3;stroke-width:1}.axisline{stroke:#343a40;stroke-width:1.2}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="53" text-anchor="middle" class="subtitle">{html.escape(subtitle)}</text>',
    ]


def legend(lines, width, y=78):
    start = width / 2 - 155
    for index, policy in enumerate(POLICIES):
        x = start + index * 155
        lines.append(f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" stroke="{COLORS[policy]}" stroke-width="3"/>')
        lines.append(f'<text x="{x + 31}" y="{y + 4}" class="legend">{policy}</text>')


def nice_max(value):
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    step = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    return step * magnitude


def line_panels(path, summaries, metric, low_metric, high_metric, title, subtitle, y_label, ratio=False):
    placements = sorted({row["placement"] for row in summaries}, key=lambda name: (name != "unpinned", name))
    width = 1160
    panel_height = 235
    height = 105 + panel_height * len(placements) + 55
    lines = svg_header(width, height, title, subtitle)
    legend(lines, width)
    values = [float(row[high_metric] if high_metric else row[metric]) for row in summaries]
    y_max = nice_max(max(values) * 1.08)
    if ratio:
        y_max = max(1.2, math.ceil(max(values) * 10) / 10)
    left, right = 105, width - 45
    plot_width = right - left
    for panel_index, placement in enumerate(placements):
        top = 105 + panel_index * panel_height
        bottom = top + 175
        lines.append(f'<text x="{left}" y="{top - 12}" class="panel">{html.escape(display_placement(placement))}</text>')
        for tick in range(5):
            value = y_max * tick / 4
            y = bottom - (bottom - top) * tick / 4
            tick_label = f"{value:.2f}" if ratio else f"{value:.0f}"
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick_label}</text>')
        lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axisline"/>')
        lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axisline"/>')
        x_positions = {}
        for q_index, qo_len in enumerate(QO_LENGTHS):
            x = left + plot_width * (q_index + 0.5) / len(QO_LENGTHS)
            x_positions[qo_len] = x
            lines.append(f'<text x="{x:.1f}" y="{bottom + 20}" text-anchor="middle" class="axis">{qo_len}</text>')
        for policy in POLICIES:
            points = []
            for qo_len in QO_LENGTHS:
                row = next(item for item in summaries if item["placement"] == placement and item["qo_len"] == qo_len and item["wait_policy"] == policy)
                value = float(row[metric])
                x = x_positions[qo_len]
                y = bottom - (bottom - top) * value / y_max
                points.append((x, y, row))
            path_data = " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y, _) in enumerate(points))
            lines.append(f'<path d="{path_data}" fill="none" stroke="{COLORS[policy]}" stroke-width="2.5"/>')
            for x, y, row in points:
                if low_metric and high_metric:
                    low = float(row[low_metric])
                    high = float(row[high_metric])
                    y_low = bottom - (bottom - top) * low / y_max
                    y_high = bottom - (bottom - top) * high / y_max
                    lines.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{COLORS[policy]}" stroke-width="1.2"/>')
                    lines.append(f'<line x1="{x - 4:.1f}" y1="{y_low:.1f}" x2="{x + 4:.1f}" y2="{y_low:.1f}" stroke="{COLORS[policy]}"/>')
                    lines.append(f'<line x1="{x - 4:.1f}" y1="{y_high:.1f}" x2="{x + 4:.1f}" y2="{y_high:.1f}" stroke="{COLORS[policy]}"/>')
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS[policy]}"/>')
        center_y = (top + bottom) / 2
        lines.append(f'<text x="24" y="{center_y:.1f}" transform="rotate(-90 24 {center_y:.1f})" text-anchor="middle" class="axis">{html.escape(y_label)}</text>')
        lines.append(f'<text x="{(left + right) / 2:.1f}" y="{bottom + 40}" text-anchor="middle" class="axis">Query length qo_len (tokens)</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def control_breakdown_svg(path, summaries):
    placement = "unpinned" if any(row["placement"] == "unpinned" for row in summaries) else summaries[0]["placement"]
    selected = [row for row in summaries if row["placement"] == placement]
    width, height = 1160, 610
    lines = svg_header(width, height, "Host latency decomposition", "Median DSP dispatch plus host-side residual; unpinned placement")
    stack_colors = ("#76b7b2", "#f28e2b")
    lines.append(f'<rect x="420" y="72" width="18" height="12" fill="{stack_colors[0]}"/><text x="445" y="83" class="legend">DSP dispatch</text>')
    lines.append(f'<rect x="580" y="72" width="18" height="12" fill="{stack_colors[1]}"/><text x="605" y="83" class="legend">Host control residual</text>')
    left, right, top, bottom = 95, 1120, 110, 520
    y_max = nice_max(max(row["host_median_us"] for row in selected) * 1.1)
    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - (bottom - top) * tick / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.0f}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axisline"/>')
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axisline"/>')
    group_width = (right - left) / len(QO_LENGTHS)
    bar_width = 55
    for q_index, qo_len in enumerate(QO_LENGTHS):
        group_center = left + group_width * (q_index + 0.5)
        for policy_index, policy in enumerate(POLICIES):
            row = next(item for item in selected if item["qo_len"] == qo_len and item["wait_policy"] == policy)
            x = group_center + (policy_index - 1) * 72 - bar_width / 2
            dispatch = min(row["dsp_dispatch_total_median_us"], row["host_median_us"])
            residual = max(0.0, row["host_median_us"] - dispatch)
            dispatch_h = (bottom - top) * dispatch / y_max
            residual_h = (bottom - top) * residual / y_max
            lines.append(f'<rect x="{x:.1f}" y="{bottom - dispatch_h:.1f}" width="{bar_width}" height="{dispatch_h:.1f}" fill="{stack_colors[0]}"/>')
            lines.append(f'<rect x="{x:.1f}" y="{bottom - dispatch_h - residual_h:.1f}" width="{bar_width}" height="{residual_h:.1f}" fill="{stack_colors[1]}"/>')
            lines.append(f'<text x="{x + bar_width / 2:.1f}" y="{bottom + 18}" text-anchor="middle" class="axis">{policy[:4]}</text>')
        lines.append(f'<text x="{group_center:.1f}" y="{bottom + 43}" text-anchor="middle" class="panel">q={qo_len}</text>')
    lines.append(f'<text x="24" y="{(top + bottom) / 2}" transform="rotate(-90 24 {(top + bottom) / 2})" text-anchor="middle" class="axis">Host wall latency (us)</text>')
    lines.append(f'<text x="{(left + right) / 2}" y="{bottom + 70}" text-anchor="middle" class="axis">Wait policy grouped by query length</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prediction_error_svg(path, summaries):
    selected = [row for row in summaries if row["wait_policy"] == "predictive"]
    placements = sorted({row["placement"] for row in selected}, key=lambda name: (name != "unpinned", name))
    placement_colors = ("#4e79a7", "#59a14f", "#e15759")
    width, height = 1160, 560
    lines = svg_header(
        width, height,
        "Predictive wait error after rolling P10 calibration",
        "Point is p50; upper whisker is p95. Positive values mean the prediction woke early.",
    )
    legend_x = 245
    for index, placement in enumerate(placements):
        x = legend_x + index * 250
        lines.append(f'<line x1="{x}" y1="78" x2="{x + 24}" y2="78" stroke="{placement_colors[index]}" stroke-width="3"/>')
        lines.append(f'<text x="{x + 31}" y="82" class="legend">{html.escape(display_placement(placement))}</text>')
    left, right, top, bottom = 105, 1110, 110, 475
    raw_values = [row["prediction_error_p50_us"] for row in selected] + [row["prediction_error_p95_us"] for row in selected]
    y_min = min(-20.0, math.floor(min(raw_values) / 20.0) * 20.0)
    y_max = max(20.0, math.ceil(max(raw_values) / 20.0) * 20.0)
    y_span = y_max - y_min

    def map_y(value):
        return bottom - (bottom - top) * (value - y_min) / y_span

    for tick in range(6):
        value = y_min + y_span * tick / 5
        y = map_y(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.0f}</text>')
    zero_y = map_y(0.0)
    lines.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{right}" y2="{zero_y:.1f}" stroke="#343a40" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axisline"/>')
    lines.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axisline"/>')
    x_positions = {}
    for index, qo_len in enumerate(QO_LENGTHS):
        x = left + (right - left) * (index + 0.5) / len(QO_LENGTHS)
        x_positions[qo_len] = x
        lines.append(f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" class="axis">{qo_len}</text>')
    for placement_index, placement in enumerate(placements):
        points = []
        for qo_len in QO_LENGTHS:
            row = next(item for item in selected if item["placement"] == placement and item["qo_len"] == qo_len)
            x = x_positions[qo_len]
            p50_y = map_y(row["prediction_error_p50_us"])
            p95_y = map_y(row["prediction_error_p95_us"])
            points.append((x, p50_y))
            color = placement_colors[placement_index]
            lines.append(f'<line x1="{x:.1f}" y1="{p50_y:.1f}" x2="{x:.1f}" y2="{p95_y:.1f}" stroke="{color}" stroke-width="1.4"/>')
            lines.append(f'<line x1="{x - 5:.1f}" y1="{p95_y:.1f}" x2="{x + 5:.1f}" y2="{p95_y:.1f}" stroke="{color}"/>')
        color = placement_colors[placement_index]
        path_data = " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))
        lines.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    lines.append(f'<text x="24" y="{(top + bottom) / 2}" transform="rotate(-90 24 {(top + bottom) / 2})" text-anchor="middle" class="axis">Prediction error (us)</text>')
    lines.append(f'<text x="{(left + right) / 2}" y="{bottom + 48}" text-anchor="middle" class="axis">Query length qo_len (tokens)</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path, summaries):
    placements = sorted({row["placement"] for row in summaries}, key=lambda name: (name != "unpinned", name))
    lines = [
        "# HeteroInfer CPU scheduling summary",
        "",
        "All rows are baseline attention, 5 warmups plus 20 measured iterations. Median confidence intervals use 4,000 bootstrap resamples.",
        "",
    ]
    for placement in placements:
        lines.extend([
            f"## {display_placement(placement)}",
            "",
            "| qo_len | policy | host median us [95% CI] | observed speedup | normalized speedup | thread CPU us | CPU util % | sleep/spin us | DSP total us | DSP shift % | control us | legacy Amdahl |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for qo_len in QO_LENGTHS:
            for policy in POLICIES:
                row = next(item for item in summaries if item["placement"] == placement and item["qo_len"] == qo_len and item["wait_policy"] == policy)
                lines.append(
                    f"| {qo_len} | {policy} | {fmt(row['host_median_us'])} [{fmt(row['host_ci95_low_us'])}, {fmt(row['host_ci95_high_us'])}] "
                    f"| {fmt(row['speedup_vs_legacy'], 3)} [{fmt(row['speedup_ci95_low'], 3)}, {fmt(row['speedup_ci95_high'], 3)}] "
                    f"| {fmt(row['normalized_speedup_vs_legacy'], 3)} [{fmt(row['normalized_speedup_ci95_low'], 3)}, {fmt(row['normalized_speedup_ci95_high'], 3)}] "
                    f"| {fmt(row['host_thread_cpu_median_us'])} | {fmt(row['host_cpu_utilization_median_pct'])} "
                    f"| {fmt(row['sleep_median_us'])}/{fmt(row['spin_median_us'])} | {fmt(row['dsp_dispatch_total_median_us'])} "
                    f"| {fmt(row['dsp_dispatch_shift_vs_legacy_pct'])} | {fmt(row['host_control_overhead_median_us'])} "
                    f"| {fmt(row['legacy_amdahl_zero_control_limit'], 3)} |"
                )
        lines.append("")
    predictive = [row for row in summaries if row["wait_policy"] == "predictive"]
    spin = [row for row in summaries if row["wait_policy"] == "spin"]
    lines.extend([
        "## Automated observations",
        "",
        f"- Predictive median busy-spin range: {min(row['spin_median_us'] for row in predictive):.1f} to {max(row['spin_median_us'] for row in predictive):.1f} us.",
        f"- Spin median CPU utilization range: {min(row['host_cpu_utilization_median_pct'] for row in spin):.1f}% to {max(row['host_cpu_utilization_median_pct'] for row in spin):.1f}%.",
        f"- Unique output hashes per configuration: max {max(row['unique_output_hashes'] for row in summaries)}.",
        f"- Configurations whose observed speedup exceeds the legacy zero-control Amdahl limit: {sum(row['observed_speedup_exceeds_amdahl'] for row in summaries if row['wait_policy'] != 'legacy')} of {len([row for row in summaries if row['wait_policy'] != 'legacy'])} non-legacy configurations.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input_dir)
    measured, summaries = summarize(rows)
    expected_groups = len({row["placement"] for row in summaries}) * len(QO_LENGTHS) * len(POLICIES)
    if len(summaries) != expected_groups or any(row["samples"] != 20 for row in summaries):
        raise ValueError("The main matrix is incomplete or does not contain exactly 20 measured samples per group")

    measured_fields = list(rows[0].keys())
    summary_fields = list(summaries[0].keys())
    write_csv(args.output_dir / "iteration_samples.csv", measured, measured_fields)
    write_csv(args.output_dir / "summary.csv", summaries, summary_fields)
    write_markdown(args.output_dir / "summary.md", summaries)
    line_panels(
        args.output_dir / "host_latency.svg", summaries,
        "host_median_us", "host_ci95_low_us", "host_ci95_high_us",
        "CPU wait policy latency on SM8750P v81",
        "Median of 20 iterations; error bars are bootstrap 95% confidence intervals",
        "Host wall latency (us)",
    )
    line_panels(
        args.output_dir / "control_overhead.svg", summaries,
        "host_control_overhead_median_us", "host_control_ci95_low_us", "host_control_ci95_high_us",
        "Host control-plane residual after DSP dispatch",
        "host wall minus DSP mapping, validation, compute and output-flush total",
        "Control residual (us)",
    )
    line_panels(
        args.output_dir / "host_cpu_cost.svg", summaries,
        "host_thread_cpu_median_us", None, None,
        "CPU cost of synchronization policies",
        "Thread CPU time exposes the energy and core-occupancy cost hidden by wall latency",
        "Host thread CPU time (us)",
    )
    line_panels(
        args.output_dir / "dsp_dispatch.svg", summaries,
        "dsp_dispatch_total_median_us", None, None,
        "DSP dispatch time observed under each CPU wait policy",
        "A changing DSP total reveals SoC-level coupling and cannot be credited as synchronization-only gain",
        "DSP dispatch total (us)",
    )
    line_panels(
        args.output_dir / "speedup_vs_legacy.svg", summaries,
        "speedup_vs_legacy", "speedup_ci95_low", "speedup_ci95_high",
        "Synchronization speedup relative to legacy polling",
        "Values above 1.0 improve host wall latency; error bars are bootstrap 95% intervals",
        "Speedup (x)", ratio=True,
    )
    line_panels(
        args.output_dir / "normalized_speedup.svg", summaries,
        "normalized_speedup_vs_legacy", "normalized_speedup_ci95_low", "normalized_speedup_ci95_high",
        "Scheduling-only speedup with DSP time held at legacy",
        "Counterfactual removes observed DSP timing shifts; error bars are bootstrap 95% intervals",
        "Normalized speedup (x)", ratio=True,
    )
    control_breakdown_svg(args.output_dir / "host_control_breakdown.svg", summaries)
    prediction_error_svg(args.output_dir / "prediction_error.svg", summaries)
    print(f"rows={len(measured)} groups={len(summaries)} output={args.output_dir}")


if __name__ == "__main__":
    main()
