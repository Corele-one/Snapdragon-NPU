#!/usr/bin/env python3
"""Summarize paired Qwen2.5-1.5B paper-LUT versus SCNA llama-bench JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from xml.sax.saxutils import escape


MODES = (
    ("baseline", "Original HVX exp2", "#252a31"),
    ("lut-exp", "Paper exp-LUT", "#dc2626"),
    ("scna", "SCNA exp/tree/d8", "#178f86"),
)


def load_rows(input_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode, _, _ in MODES:
        for path in sorted((input_dir / "model").glob(f"{mode}_*.jsonl")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or "samples_ts" not in row:
                    continue
                row["mode"] = mode
                row["source"] = str(path)
                rows.append(row)
    return rows


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        if n_prompt > 0 and n_gen == 0:
            phase, tokens = "prefill", n_prompt
        elif n_gen > 0 and n_prompt == 0:
            phase, tokens = "decode", n_gen
        else:
            # Combined pp+tg output does not separate the two phases and is
            # intentionally excluded from this two-axis chart.
            continue
        samples = [float(value) for value in row.get("samples_ts", [])]
        result.append({
            "mode": row["mode"], "phase": phase, "tokens": tokens,
            "samples": len(samples), "median_tps": median(samples),
            "p05_tps": percentile(samples, 0.05), "p95_tps": percentile(samples, 0.95),
            "source": row["source"],
        })
    return result


def render_svg(rows: list[dict[str, object]], out: Path) -> None:
    width, height = 1080, 510
    panels = (("prefill", "Prefill throughput", "Prompt tokens"),
              ("decode", "Decode throughput", "Generated tokens"))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0}.title{font-size:19px;font-weight:700;fill:#15181d}.axis{stroke:#30343b;stroke-width:1.2}.grid{stroke:#d9dde3}.tick{font-size:11px;fill:#3e4651}.label{font-size:13px;fill:#20242a}.legend{font-size:12px;fill:#20242a}</style>',
        '<text x="540" y="27" text-anchor="middle" class="title">Qwen2.5-1.5B: Paper exp-LUT vs SCNA on the Same V81 Device</text>',
    ]
    for panel_index, (phase, title, x_label) in enumerate(panels):
        x0, y0, w, h = 75 + panel_index * 515, 70, 380, 330
        phase_rows = [row for row in rows if row["phase"] == phase]
        xs = sorted({int(row["tokens"]) for row in phase_rows})
        ymax = max((float(row["p95_tps"]) for row in phase_rows), default=1.0) * 1.12
        sx = lambda x: x0 + (xs.index(x) * w / max(len(xs) - 1, 1))
        sy = lambda y: y0 + h - y / ymax * h
        svg.append(f'<text x="{x0 + w / 2:.1f}" y="52" text-anchor="middle" class="label">{escape(title)}</text>')
        for tick in range(6):
            value, y = ymax * tick / 5, sy(ymax * tick / 5)
            svg += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" class="grid"/>',
                    f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.1f}</text>']
        svg += [f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" class="axis"/>',
                f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" class="axis"/>']
        for x in xs:
            svg.append(f'<text x="{sx(x):.1f}" y="{y0 + h + 21}" text-anchor="middle" class="tick">{x}</text>')
        svg += [f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 44}" text-anchor="middle" class="label">{x_label}</text>',
                f'<text x="{x0 - 48}" y="{y0 + h / 2:.1f}" text-anchor="middle" transform="rotate(-90 {x0 - 48} {y0 + h / 2:.1f})" class="label">tokens/s</text>']
        for mode, label, color in MODES:
            values = sorted((row for row in phase_rows if row["mode"] == mode), key=lambda row: int(row["tokens"]))
            if not values:
                continue
            points = " ".join(f'{sx(int(row["tokens"])):.1f},{sy(float(row["median_tps"])):.1f}' for row in values)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for row in values:
                x, y = sx(int(row["tokens"])), sy(float(row["median_tps"]))
                hi, lo = sy(float(row["p95_tps"])), sy(float(row["p05_tps"]))
                svg += [f'<line x1="{x:.1f}" y1="{hi:.1f}" x2="{x:.1f}" y2="{lo:.1f}" stroke="{color}"/>',
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>']
    for index, (_, label, color) in enumerate(MODES):
        x, y = 875, 94 + index * 31
        svg += [f'<line x1="{x}" y1="{y}" x2="{x + 27}" y2="{y}" stroke="{color}" stroke-width="2.5"/>',
                f'<circle cx="{x + 13.5}" cy="{y}" r="4" fill="{color}"/>',
                f'<text x="{x + 35}" y="{y + 4}" class="legend">{escape(label)}</text>']
    svg.append('</svg>')
    out.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(load_rows(args.input_dir))
    with (args.out_dir / "qwen_throughput_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("mode", "phase", "tokens", "samples", "median_tps", "p05_tps", "p95_tps", "source"))
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        render_svg(rows, args.out_dir / "qwen_throughput.svg")
    print(f"rows={len(rows)} output={args.out_dir}")


if __name__ == "__main__":
    main()
