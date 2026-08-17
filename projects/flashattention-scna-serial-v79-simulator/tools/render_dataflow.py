#!/usr/bin/env python3
"""Create an editable draw.io source and matching vector/raster previews."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


UPPER = [
    ("DDR/L2\nQ: FP32", 0.4, "memory"),
    ("HVX Q load\nFP32→FP16", 2.0, "hvx"),
    ("VTCM\nQ tile", 3.6, "memory"),
    ("HMX QKᵀ\n× log₂(e)/√D", 5.2, "hmx"),
    ("HVX safe softmax\nmask + online max\n+ SCNA", 7.0, "scna"),
    ("HMX P×V\naccumulation", 9.0, "hmx"),
    ("HVX normalize\nO scale/store\nFP32", 10.8, "hvx"),
]
LOWER = [
    ("score − rowmax", 1.0),
    ("clamp to SCNA domain", 3.0),
    ("share d8 w/b\nacross row pair", 5.0),
    ("8 stages: mul + add\n+ ReLU", 7.0),
    ("reduce + pack\n≈ exp2", 9.0),
    ("rowsum + online\nrecurrence", 11.0),
]
COLORS = {"memory": "#D9EAF7", "hvx": "#56B4E9", "hmx": "#009E73", "scna": "#E69F00"}


def drawio(out: Path):
    mxfile = Element("mxfile", host="app.diagrams.net", modified="2026-08-13T00:00:00.000Z", agent="Codex", version="24.7.17")
    diagram = SubElement(mxfile, "diagram", id="scna-attention", name="SCNA Attention Dataflow")
    model = SubElement(diagram, "mxGraphModel", dx="1200", dy="700", grid="1", gridSize="10", page="1", pageWidth="1600", pageHeight="900")
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")
    for idx, (label, x, kind) in enumerate(UPPER, 2):
        cell = SubElement(root, "mxCell", id=str(idx), value=html.escape(label).replace("\n", "&lt;br&gt;"),
                          style=f"rounded=1;whiteSpace=wrap;html=1;fillColor={COLORS[kind]};strokeColor=#263238;fontSize=13;fontStyle=1;",
                          vertex="1", parent="1")
        SubElement(cell, "mxGeometry", x=str(60 + x * 105), y="100", width="145", height="72", **{"as": "geometry"})
        if idx > 2:
            edge = SubElement(root, "mxCell", id=f"e{idx}", style="edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;",
                              edge="1", parent="1", source=str(idx - 1), target=str(idx))
            SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
    for idx, (label, x) in enumerate(LOWER, 20):
        cell = SubElement(root, "mxCell", id=str(idx), value=html.escape(label).replace("\n", "&lt;br&gt;"),
                          style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FCE8C3;strokeColor=#B35C00;fontSize=13;",
                          vertex="1", parent="1")
        SubElement(cell, "mxGeometry", x=str(60 + x * 105), y="340", width="160", height="68", **{"as": "geometry"})
        if idx > 20:
            edge = SubElement(root, "mxCell", id=f"e{idx}", style="edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeWidth=2;strokeColor=#B35C00;",
                              edge="1", parent="1", source=str(idx - 1), target=str(idx))
            SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
    note = SubElement(root, "mxCell", id="40", value="SCNA changes exp2 evaluation only; QK/PV, K/V loading and tiling remain unchanged. No K/V tile-reuse claim.",
                      style="shape=note;whiteSpace=wrap;html=1;fillColor=#FFF4CC;strokeColor=#B38F00;fontSize=12;", vertex="1", parent="1")
    SubElement(note, "mxGeometry", x="380", y="500", width="850", height="65", **{"as": "geometry"})
    ElementTree(mxfile).write(out, encoding="utf-8", xml_declaration=True)


def render(out_dir: Path):
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(0.2, 5.75, "Attention dataflow on Hexagon v79", fontsize=14, weight="bold")
    ax.text(0.2, 3.05, "Serial SCNA exp2 evaluator (zoom-in)", fontsize=12, weight="bold", color="#9A4B00")
    for label, x, kind in UPPER:
        box = FancyBboxPatch((x, 4.25), 1.42, 0.82, boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor=COLORS[kind], edgecolor="#263238", linewidth=1.1)
        ax.add_patch(box); ax.text(x + 0.71, 4.66, label, ha="center", va="center", fontsize=7.7, weight="bold")
    for left, right in zip(UPPER, UPPER[1:]):
        ax.add_patch(FancyArrowPatch((left[1] + 1.42, 4.66), (right[1], 4.66), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.3, color="#37474F"))
    ax.text(0.4, 5.25, "K/V FP16 → HVX load/scatter → VTCM Kᵀ/V tiles", fontsize=8.5, color="#37474F")
    for label, x in LOWER:
        box = FancyBboxPatch((x, 1.45), 1.55, 0.82, boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor="#FCE8C3", edgecolor="#B35C00", linewidth=1.1)
        ax.add_patch(box); ax.text(x + 0.775, 1.86, label, ha="center", va="center", fontsize=8.2)
    for left, right in zip(LOWER, LOWER[1:]):
        ax.add_patch(FancyArrowPatch((left[1] + 1.55, 1.86), (right[1], 1.86), arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.3, color="#B35C00"))
    ax.add_patch(FancyArrowPatch((7.72, 4.22), (7.72, 2.4), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, linestyle="--", color="#B35C00"))
    ax.text(6.35, 3.30, "zoom into SCNA exp2", fontsize=8, color="#9A4B00")
    ax.text(0.35, 0.55, "Scope: SCNA replaces exp2 inside safe softmax. QK/PV, K/V loading and tiling are unchanged; no K/V tile reuse is claimed.",
            fontsize=8.5, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF4CC", "edgecolor": "#B38F00"})
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out_dir / f"attention_scna_dataflow.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    drawio(args.out_dir / "attention_scna_dataflow.drawio")
    render(args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
