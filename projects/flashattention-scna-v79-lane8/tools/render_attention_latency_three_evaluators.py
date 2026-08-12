#!/usr/bin/env python3
"""Render Attention latency for two baselines and SCNA lane8."""

import json
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/v79/scna-lane8/20260808_194500/analysis/summary.json"
OUT = Path(
    "/mnt/d/23644/Documents/Study/Research/WeeklyMeeting/assets/"
    "2026-08-07_2026-08-12_Restructured_SCNA/attention_latency_lane8_baselines.svg"
)
W, H = 960, 540


def line(x1, y1, x2, y2, cls="grid", extra=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" {extra}/>'


def text(x, y, value, cls="axis", anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(str(value))}</text>'


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))["attention"]
    qos = [4, 8, 16, 32]
    series = [
        ("Origin-HVX", "#64748b"),
        ("LUT-EXP", "#0ea5a4"),
        ("SCNA lane8 d8", "#dc2626"),
    ]
    left, right, top, bottom = 90, 900, 90, 440
    ymax = 3200

    body = text(left, 38, "Attention latency across three evaluators", "title")
    body += text(left, 61, "Median headline latency; whiskers are 95% stratified-bootstrap CIs", "sub")
    for value in range(0, ymax + 1, 800):
        y = bottom - value / ymax * (bottom - top)
        body += line(left, y, right, y)
        body += text(left - 12, y + 4, f"{value:,}", "axis", "end")
    body += line(left, top, left, bottom, "axisline")
    body += line(left, bottom, right, bottom, "axisline")
    body += text(60, 80, "Latency (µs)", "axis")

    for index, qo in enumerate(qos):
        x = left + index * (right - left) / 3
        body += text(x, bottom + 27, f"Qo={qo}", "axis", "middle")

    legend_x = [145, 390, 625]
    for series_index, (name, color) in enumerate(series):
        points = []
        for qo_index, qo in enumerate(qos):
            datum = summary[name][str(qo)]
            x = left + qo_index * (right - left) / 3
            y = bottom - datum["median_us"] / ymax * (bottom - top)
            ci_low_y = bottom - datum["median_ci_low_us"] / ymax * (bottom - top)
            ci_high_y = bottom - datum["median_ci_high_us"] / ymax * (bottom - top)
            stroke = f'stroke="{color}" stroke-width="2"'
            body += line(x, ci_low_y, x, ci_high_y, "", stroke)
            body += line(x - 5, ci_low_y, x + 5, ci_low_y, "", stroke)
            body += line(x - 5, ci_high_y, x + 5, ci_high_y, "", stroke)
            points.append(f"{x:.1f},{y:.1f}")
        body += f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        for point in points:
            x, y = point.split(",")
            body += f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"/>'
        x = legend_x[series_index]
        body += f'<rect x="{x}" y="478" width="12" height="12" rx="2" fill="{color}"/>'
        body += text(x + 18, 488, name, "legend")

    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">Attention latency across three evaluators</title><desc id="desc">Median Attention latency and 95 percent confidence intervals for Origin-HVX, LUT-EXP, and SCNA lane8 d8.</desc>
<style>text{{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#172033}}.title{{font-size:22px;font-weight:700}}.sub{{font-size:13px;fill:#526174}}.axis{{font-size:12px;fill:#526174}}.grid{{stroke:#dbe3ed;stroke-width:1}}.axisline{{stroke:#94a3b8;stroke-width:1.2}}.legend{{font-size:13px;font-weight:600}}</style>
<rect width="100%" height="100%" fill="#ffffff"/>{body}</svg>'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
