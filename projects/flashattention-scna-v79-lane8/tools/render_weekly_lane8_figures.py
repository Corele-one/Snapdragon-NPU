#!/usr/bin/env python3
"""Render reproducible SVG figures for the SCNA lane8 weekly report.

All numerical values come from checked-in experiment summaries. Conceptual
figures encode the implementation found in scna_exp2.c and flash_attn.c.
"""

from html import escape
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path("/mnt/d/23644/Documents/Study/Research/WeeklyMeeting/figures/scna_lane8")
W = 1200

INK = "#172033"
MUTED = "#526174"
GRID = "#dbe3ed"
NAIVE = "#64748b"
LANE8 = "#2563eb"
ACCENT = "#0f9f8f"
WARN = "#d97706"
LIGHT_BLUE = "#eff6ff"
LIGHT_TEAL = "#ecfdf5"
LIGHT_GRAY = "#f8fafc"


class Canvas:
    def __init__(self, height: int, title: str, desc: str):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
            'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>'
            '</marker></defs>',
            '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#172033}'
            '.title{font-size:26px;font-weight:700}.sub{font-size:14px;fill:#526174}'
            '.label{font-size:15px;font-weight:700}.body{font-size:14px}.small{font-size:12px;fill:#526174}'
            '.mono{font-family:"DejaVu Sans Mono",monospace;font-size:12px}</style>',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]

    def text(self, x, y, value, cls="body", anchor="start", fill=None):
        style = f' style="fill:{fill}"' if fill else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{style}>'
            f'{escape(str(value))}</text>'
        )

    def box(self, x, y, width, height, fill=LIGHT_GRAY, stroke=GRID, radius=10, sw=1.5):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=NAIVE, width=2, arrow=False, dash=None):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{width}"{marker}{dashed}/>'
        )

    def finish(self, name: str):
        self.parts.append("</svg>")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text("\n".join(self.parts), encoding="utf-8")


def render_latency():
    # Sources:
    # naive: flashattention-scna-v81/results/v81/scna/sm8750p-20260731-2300/summary/summary.csv
    # lane8: flashattention-scna-v79-lane8/results/.../analysis/summary.json
    qos = [4, 8, 16, 32]
    naive = [2847.0, 4852.5, 8946.0, 17025.5]
    lane8 = [774.0, 1087.0, 1625.5, 2923.0]
    gain = [a / b for a, b in zip(naive, lane8)]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(qos, naive, color=NAIVE, marker="o", linewidth=2.5, markersize=7,
            label="Naive SCNA (v81, 1 worker)")
    ax.plot(qos, lane8, color=LANE8, marker="s", linewidth=2.5, markersize=7,
            label="SCNA lane8 (v79, worker pool)")
    for x, y, ratio in zip(qos, lane8, gain):
        ax.annotate(f"{ratio:.2f}×", (x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", color=LANE8, fontweight="bold")
    ax.set_title("Attention latency: naive SCNA vs. SCNA lane8", loc="left", fontweight="bold", pad=32)
    ax.text(0, 1.015, "Historical end-to-end comparison; ratios include layout and system-level changes",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.set_xlabel("Query length (Qo)")
    ax.set_ylabel("DSP latency (µs)")
    ax.set_xticks(qos)
    ax.set_ylim(0, 18500)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94a3b8")
    ax.legend(frameon=False, loc="upper left")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "01_latency_comparison.svg", format="svg")
    plt.close(fig)


def render_naive():
    c = Canvas(540, "Naive SCNA execution", "Serial neuron evaluation across 64 score lanes.")
    c.text(60, 48, "Naive SCNA: 64 scores × 1 neuron per vector pass", "title")
    c.text(60, 74, "Score-level SIMD is full; the neuron dimension remains an 8-step dependency chain", "sub")

    c.text(60, 118, "HVX lane mapping", "label")
    start_x, y, cell_w, cell_h = 60, 145, 16, 42
    for lane in range(64):
        fill = "#eef2f7" if (lane // 8) % 2 == 0 else "#e2e8f0"
        c.box(start_x + lane * cell_w, y, cell_w, cell_h, fill, "#ffffff", 0, 0.5)
    for lane in [0, 1, 2, 7, 8, 15, 63]:
        c.text(start_x + lane * cell_w + cell_w / 2, y + 27, f"x{lane}", "small", "middle")
    c.text(1084, y + 65, "64 independent score lanes", "small", "end")

    nodes = [
        (60, "Load packed coeff[i]", "#f8fafc", NAIVE),
        (280, "Splat wi and bi", "#fff7ed", WARN),
        (500, "64-lane FP16 FMA", LIGHT_BLUE, LANE8),
        (720, "ReLU", LIGHT_TEAL, ACCENT),
        (940, "sum += contribution", "#f8fafc", NAIVE),
    ]
    for x, label, fill, stroke in nodes:
        c.box(x, 265, 175, 68, fill, stroke)
        c.text(x + 87.5, 305, label, "label", "middle")
    for i in range(len(nodes) - 1):
        c.line(nodes[i][0] + 175, 299, nodes[i + 1][0] - 10, 299, arrow=True)

    c.line(1115, 333, 1115, 385, stroke=WARN, width=2)
    c.line(1115, 385, 147, 385, stroke=WARN, width=2)
    c.line(147, 385, 147, 343, stroke=WARN, width=2, arrow=True)
    c.text(630, 410, "Repeat for neuron i = 0 ... 7", "label", "middle", WARN)

    c.box(250, 448, 700, 55, "#fff7ed", "#fed7aa")
    c.text(600, 471, "Critical path: sum(i) depends on sum(i−1)", "label", "middle")
    c.text(600, 491, "8 weight splats + 8 bias splats per 64-score vector", "small", "middle")
    c.finish("02_naive_execution.svg")


def render_lane8():
    c = Canvas(690, "SCNA lane8 execution", "Eight scores and eight neurons are mapped into one HVX vector.")
    c.text(60, 48, "SCNA lane8: transpose the work into 8 scores × 8 neurons", "title")
    c.text(60, 74, "Static coefficient vectors replace per-neuron splats; a reduction tree produces one result per score", "sub")

    c.text(60, 118, "1. Input expand", "label")
    c.box(60, 140, 245, 105, LIGHT_GRAY, GRID)
    c.text(182, 169, "8 source scores", "body", "middle")
    for i in range(8):
        c.box(78 + i * 27, 187, 24, 32, "#e2e8f0", "#ffffff", 2, 0.5)
        c.text(90 + i * 27, 208, f"x{i}", "small", "middle")
    c.line(315, 193, 380, 193, arrow=True)
    c.text(348, 177, "vshuff + vlut16", "small", "middle")

    c.text(395, 118, "2. Lane8 mapping", "label")
    grid_x, grid_y, cw, ch = 395, 140, 52, 28
    for score in range(8):
        for neuron in range(8):
            fill = "#dbeafe" if score % 2 == 0 else "#bfdbfe"
            c.box(grid_x + neuron * cw, grid_y + score * ch, cw, ch, fill, "#ffffff", 0, 0.5)
            c.text(grid_x + neuron * cw + cw / 2, grid_y + score * ch + 19,
                   f"x{score}/n{neuron}", "small", "middle")
    for neuron in range(8):
        c.text(grid_x + neuron * cw + cw / 2, grid_y - 8, f"n{neuron}", "small", "middle")
    for score in range(8):
        c.text(grid_x - 12, grid_y + score * ch + 19, f"x{score}", "small", "end")
    c.text(830, 365, "64 HVX lanes", "small", "end")

    c.text(875, 118, "3. Coefficient pattern", "label")
    c.box(875, 140, 265, 224, LIGHT_TEAL, "#99f6e4")
    c.text(1007, 174, "Aligned vector loads", "label", "middle")
    c.text(1007, 205, "W = [w0 ... w7] × 8", "mono", "middle")
    c.text(1007, 235, "B = [b0 ... b7] × 8", "mono", "middle")
    c.text(1007, 278, "2 vector loads", "body", "middle", ACCENT)
    c.text(1007, 303, "instead of", "small", "middle")
    c.text(1007, 328, "16 coefficient splats", "body", "middle", WARN)

    stages = [
        (70, "FP16 FMA", "8 scores × 8 neurons", LIGHT_BLUE, LANE8),
        (310, "Lane-wise ReLU", "64 contributions", LIGHT_TEAL, ACCENT),
        (550, "3-stage reduction", "1 → 2 → 4 → 8", "#fff7ed", WARN),
        (790, "Pack output block", "vdeal + mask + rotate", LIGHT_GRAY, NAIVE),
    ]
    for x, title, subtitle, fill, stroke in stages:
        c.box(x, 440, 190, 92, fill, stroke)
        c.text(x + 95, 477, title, "label", "middle")
        c.text(x + 95, 503, subtitle, "small", "middle")
    for i in range(len(stages) - 1):
        c.line(stages[i][0] + 190, 486, stages[i + 1][0] - 10, 486, arrow=True)
    c.box(1030, 440, 110, 92, "#eef2ff", "#818cf8")
    c.text(1085, 477, "8 outputs", "label", "middle")
    c.text(1085, 503, "per batch", "small", "middle")
    c.line(980, 486, 1020, 486, arrow=True)

    c.line(1085, 532, 1085, 585, stroke=LANE8, width=2)
    c.line(1085, 585, 165, 585, stroke=LANE8, width=2)
    c.line(165, 585, 165, 542, stroke=LANE8, width=2, arrow=True)
    c.text(625, 612, "Repeat 8 batches to cover x0 ... x63", "label", "middle", LANE8)
    c.box(310, 635, 580, 38, "#eff6ff", "#bfdbfe")
    c.text(600, 660, "Output: 64 SCNA values in the original score order", "body", "middle")
    c.finish("03_lane8_execution.svg")


def render_attention():
    c = Canvas(670, "SCNA lane8 in FlashAttention", "SCNA lane8 replaces Exp2 evaluation in safe and online softmax.")
    c.text(60, 48, "SCNA lane8 integration in FlashAttention", "title")
    c.text(60, 74, "QK, masking, rowsum, PV, and tiling remain unchanged; only the Exp2 evaluator is replaced", "sub")

    pipeline = [
        (55, "Q / K tiles", "Input"),
        (235, "QK", "Score matmul"),
        (415, "Mask + rowmax", "Safe softmax"),
        (625, "S − rowmax", "2 score rows"),
        (835, "SCNA lane8", "Paired evaluator"),
        (1035, "P tile", "Rowsum + store"),
    ]
    for index, (x, title, subtitle) in enumerate(pipeline):
        highlight = title == "SCNA lane8"
        c.box(x, 145, 140, 80, LIGHT_BLUE if highlight else LIGHT_GRAY,
              LANE8 if highlight else GRID, sw=2 if highlight else 1.5)
        c.text(x + 70, 179, title, "label", "middle")
        c.text(x + 70, 203, subtitle, "small", "middle")
        if index < len(pipeline) - 1:
            c.line(x + 140, 185, pipeline[index + 1][0] - 10, 185, arrow=True)

    c.text(905, 121, "Changed evaluator", "small", "middle", LANE8)
    c.line(905, 126, 905, 142, stroke=LANE8, width=2, arrow=True)

    c.text(60, 292, "Safe-softmax evaluator detail", "label")
    detail = [
        (60, "Row 0 centered scores"),
        (60, "Row 1 centered scores"),
        (325, "Validity mask"),
        (535, "Paired lane8 kernel"),
        (775, "Mask invalid lanes to 0"),
        (1010, "P rows"),
    ]
    ys = [320, 395, 357, 357, 357, 357]
    widths = [205, 205, 150, 180, 185, 125]
    for (x, label), y, width in zip(detail, ys, widths):
        highlight = label == "Paired lane8 kernel"
        c.box(x, y, width, 55, LIGHT_BLUE if highlight else LIGHT_GRAY,
              LANE8 if highlight else GRID)
        c.text(x + width / 2, y + 34, label, "body", "middle")
    c.line(265, 347, 315, 372, arrow=True)
    c.line(265, 422, 315, 382, arrow=True)
    c.line(475, 384, 525, 384, arrow=True)
    c.line(715, 384, 765, 384, arrow=True)
    c.line(960, 384, 1000, 384, arrow=True)

    c.text(60, 500, "Online-softmax rescale path", "label")
    rescale = [
        (60, "Previous max"),
        (60, "Current max"),
        (405, "m_prev − m_curr"),
        (680, "SCNA lane8 single"),
        (955, "Accumulator rescale"),
    ]
    rescale_y = [515, 585, 550, 550, 550]
    for (x, label), y in zip(rescale, rescale_y):
        highlight = "SCNA" in label
        c.box(x, y, 185, 55, LIGHT_BLUE if highlight else LIGHT_GRAY,
              LANE8 if highlight else GRID)
        c.text(x + 92.5, y + 34, label, "body", "middle")
    c.line(245, 542, 395, 570, arrow=True)
    c.line(245, 612, 395, 580, arrow=True)
    c.line(590, 577, 670, 577, arrow=True)
    c.line(865, 577, 945, 577, arrow=True)

    c.box(225, 625, 750, 32, "#ecfdf5", "#a7f3d0")
    c.text(600, 647, "Unchanged around SCNA: QK · mask · rowsum · P store · PV · output accumulation", "small", "middle")
    c.finish("04_attention_dataflow.svg")


def main():
    render_latency()
    render_naive()
    render_lane8()
    render_attention()
    print(f"Rendered 4 SVG figures to {OUT}")


if __name__ == "__main__":
    main()
