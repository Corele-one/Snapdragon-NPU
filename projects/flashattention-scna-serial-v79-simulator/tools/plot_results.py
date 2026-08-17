#!/usr/bin/env python3
"""Generate publication-friendly diagnostic charts only from summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None

VARIANTS = [
    "stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
    "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized",
]
MODES = ["origin", "exp-lut", "stage1", "optimized"]
COLORS = {"origin": "#6B7280", "exp-lut": "#E69F00", "stage1": "#CC79A7", "optimized": "#0072B2"}
MARKERS = {"origin": "o", "exp-lut": "s", "stage1": "^", "optimized": "D"}


def save_all(fig, out: Path, stem: str):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if sns:
        sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "figure.titlesize": 10})

    values = [data["micro"][name]["pair_elapsed_us"]["median"] / 1000.0 for name in VARIANTS]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    bars = ax.bar(np.arange(len(VARIANTS)), values, color="#0072B2", edgecolor="black", hatch="//", linewidth=0.6)
    ax.set_xticks(np.arange(len(VARIANTS)), [name.replace("_", "\n") for name in VARIANTS])
    ax.set_ylabel("Pair latency (us / 1k iterations)")
    ax.set_title("Serial SCNA d8 optimization ladder — simulator diagnostic")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save_all(fig, args.out_dir, "01_scna_micro_ladder")

    sweep = [item for item in data["attention"].values() if item["kv"] == 64 and item["heads"] == 12 and item["head_dim"] == 128]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for mode in MODES:
        items = sorted((item for item in sweep if item["mode"] == mode), key=lambda item: item["qo"])
        if not items:
            continue
        x = [item["qo"] for item in items]
        y = [item["metrics"]["kernel_us"]["median"] for item in items]
        low = [ym - item["metrics"]["kernel_us"]["min"] for ym, item in zip(y, items)]
        high = [item["metrics"]["kernel_us"]["max"] - ym for ym, item in zip(y, items)]
        ax.errorbar(x, y, yerr=[low, high], label=mode, color=COLORS[mode], marker=MARKERS[mode],
                    linewidth=1.6, capsize=2, linestyle="-" if mode != "stage1" else "--")
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.set_xlabel("Qo length")
    ax.set_ylabel("DSP qtimer kernel latency (us)")
    ax.set_title("Attention latency, KV=64, H=12/KVH=2/D=128, 1 worker")
    ax.legend(ncol=2)
    save_all(fig, args.out_dir, "02_attention_latency")

    components = ["q_load_us", "k_load_us", "v_load_us", "qk_dot_us", "safe_sm_us", "core_acc_us", "o_scale_us", "o_store_us"]
    labels = [name.removesuffix("_us") for name in components]
    items = sorted((item for item in sweep if item["mode"] == "optimized"), key=lambda item: item["qo"])
    if items:
        fig, ax = plt.subplots(figsize=(6.8, 3.9))
        bottom = np.zeros(len(items))
        palette = plt.get_cmap("tab20c")(np.linspace(0.05, 0.9, len(components)))
        hatches = ("//", "\\\\", "..", "xx", "++", "oo", "--", "||")
        for label, component, color, hatch in zip(labels, components, palette, hatches):
            vals = np.array([item["metrics"][component]["median"] for item in items])
            ax.bar([item["qo"] for item in items], vals, bottom=bottom, label=label, color=color,
                   edgecolor="white", linewidth=0.4, hatch=hatch)
            bottom += vals
        ax.set_xticks([item["qo"] for item in items])
        ax.set_xlabel("Qo length")
        ax.set_ylabel("Summed component time (us)")
        ax.set_title("Optimized serial SCNA Attention component breakdown")
        ax.legend(ncol=4, fontsize=7, loc="upper left")
        save_all(fig, args.out_dir, "03_attention_components")
    print(args.out_dir)


if __name__ == "__main__":
    main()
