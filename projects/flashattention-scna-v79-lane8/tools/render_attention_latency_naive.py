#!/usr/bin/env python3
"""Render the four-evaluator latency figure with the unoptimized SCNA baseline."""

import json

from render_scna_lane8_report_figures import COLORS, RUN, line, svg, text


# The naive values are the one-worker Attention results for the reconstructed
# unoptimized stage1_dynamic_row implementation.  Source:
# ../flashattention-scna-fp16-d8-optimization-v79/results/runs/
# scna_d8_v79_20260811/summary.json
NAIVE_SCNA = {
    4: {"median_us": 1727, "median_ci_low_us": 1686, "median_ci_high_us": 1747},
    8: {"median_us": 3235, "median_ci_low_us": 3202, "median_ci_high_us": 3273},
    16: {"median_us": 6185.5, "median_ci_low_us": 6150, "median_ci_high_us": 6232},
    32: {"median_us": 12422, "median_ci_low_us": 12329.5, "median_ci_high_us": 12492},
}


def main():
    summary = json.loads((RUN / "analysis/summary.json").read_text())
    qos = [4, 8, 16, 32]
    series = [
        ("Origin-HVX", COLORS["Origin-HVX"], summary["attention"]["Origin-HVX"]),
        ("LUT-EXP", COLORS["LUT-EXP"], summary["attention"]["LUT-EXP"]),
        ("Naive SCNA d8", COLORS["SCNA serial d8"], NAIVE_SCNA),
        ("SCNA lane8 d8", COLORS["SCNA lane8 d8"], summary["attention"]["SCNA lane8 d8"]),
    ]

    left, right, top, bottom = 90, 900, 90, 440
    ymax = 14000
    body = text(left, 38, "Unoptimized naive SCNA dominates Attention latency", "title")
    body += text(
        left,
        61,
        "Median headline latency; whiskers are 95% CIs; one worker for naive SCNA",
        "sub",
    )
    for value in range(0, ymax + 1, 3500):
        y = bottom - value / ymax * (bottom - top)
        body += line(left, y, right, y)
        body += text(left - 12, y + 4, f"{value:,}", "axis", "end")
    body += line(left, top, left, bottom, "axisline")
    body += line(left, bottom, right, bottom, "axisline")
    body += text(60, 80, "Latency (µs)", "axis")

    for index, qo in enumerate(qos):
        x = left + index * (right - left) / 3
        body += text(x, bottom + 27, f"Qo={qo}", "axis", "middle")

    for series_index, (name, color, values) in enumerate(series):
        points = []
        for qo_index, qo in enumerate(qos):
            datum = values[qo] if qo in values else values[str(qo)]
            x = left + qo_index * (right - left) / 3
            y = bottom - datum["median_us"] / ymax * (bottom - top)
            ci_low_y = bottom - datum["median_ci_low_us"] / ymax * (bottom - top)
            ci_high_y = bottom - datum["median_ci_high_us"] / ymax * (bottom - top)
            stroke = f'stroke="{color}" stroke-width="2"'
            body += line(x, ci_low_y, x, ci_high_y, "", stroke)
            body += line(x - 5, ci_low_y, x + 5, ci_low_y, "", stroke)
            body += line(x - 5, ci_high_y, x + 5, ci_high_y, "", stroke)
            points.append(f"{x:.1f},{y:.1f}")
        body += (
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        for point in points:
            x, y = point.split(",")
            body += f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"/>'
        legend_x = left + series_index * 190
        body += f'<rect x="{legend_x}" y="478" width="12" height="12" rx="2" fill="{color}"/>'
        body += text(legend_x + 18, 488, name, "legend")

    svg(
        "attention_latency_naive_scna.svg",
        body,
        "Attention latency with naive SCNA",
        "Median Attention latency and 95 percent confidence intervals for four evaluators; the serial SCNA series is replaced by unoptimized naive SCNA.",
    )


if __name__ == "__main__":
    main()
