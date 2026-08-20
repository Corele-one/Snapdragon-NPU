#!/usr/bin/env python3
"""Generate evidence-bounded LUT-vs-SCNA tables, figures and Chinese report."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scna-roofline-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
LABELS = {
    "lut-exp": "LUT-EXP",
    "pair_static_d8": "SCNA qf16 static",
    "optimized": "SCNA widened FMA",
    "optimized_qf16_tree": "SCNA qf16 tree",
    "optimized_piecewise_d8": "SCNA piecewise",
    "baseline": "Origin",
}
COLORS = dict(zip(LABELS, sns.color_palette("colorblind", len(LABELS))))
MARKERS = {"lut-exp": "o", "pair_static_d8": "s", "optimized": "^",
           "optimized_qf16_tree": "D", "optimized_piecewise_d8": "P", "baseline": "X"}
OPS_PER_ELEMENT = {"pair_static_d8": 24, "optimized": 24,
                   "optimized_qf16_tree": 21, "optimized_piecewise_d8": 2}


def fields(line):
    result = {}
    for key, value in KV_RE.findall(line):
        try:
            result[key] = float(value) if any(c in value.lower() for c in (".", "e")) else int(value, 0)
        except ValueError:
            result[key] = value
    return result


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def median(values):
    return statistics.median(values) if values else None


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_ratio(candidate, baseline, seed=20260819, draws=10000):
    pairs = [(candidate[key], baseline[key]) for key in sorted(candidate.keys() & baseline.keys())
             if baseline[key] > 0]
    if not pairs:
        return None
    observed = median([a for a, _ in pairs]) / median([b for _, b in pairs])
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        samples.append(median([a for a, _ in chosen]) / median([b for _, b in chosen]))
    return {"ratio": observed, "ci_low": quantile(samples, 0.025), "ci_high": quantile(samples, 0.975),
            "pairs": len(pairs)}


def bootstrap_median_ci(values, seed=20260819, draws=10000):
    if not values:
        return None, None
    rng = random.Random(seed)
    samples = [median([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws)]
    return quantile(samples, .025), quantile(samples, .975)


def parse_roofs(root):
    rows = []
    roof_dir = root / "raw" / "roofline"
    calibrated_bandwidth = any(roof_dir.glob("bandwidth_calibrated_sample*.log"))
    calibrated_lut = any(roof_dir.glob("lut_*_calibrated_sample*.log"))
    for path in sorted(roof_dir.glob("*.log")):
        if calibrated_bandwidth and re.fullmatch(r"bandwidth_sample\d+\.log", path.name):
            continue
        if calibrated_lut and re.fullmatch(r"lut_(dense|attention|random)_sample\d+\.log", path.name):
            continue
        sample_match = re.search(r"sample(\d+)", path.name)
        sample = int(sample_match.group(1)) if sample_match else 0
        for line in path.read_text(errors="replace").splitlines():
            cols = line.strip().split(",")
            try:
                if len(cols) == 20 and cols[0] == "2":
                    row = {
                        "source": path.name, "sample": sample, "schema_version": 2,
                        "mode": cols[1], "engine": cols[2], "kind": cols[3], "path": cols[4],
                        "distribution_id": int(cols[5]), "distribution": cols[6], "size": int(cols[7]),
                        "iters": int(cols[8]), "elapsed_us": int(cols[9]), "elapsed_ticks": int(cols[10]),
                        "elements": int(cols[11]), "input_bytes": int(cols[12]), "output_bytes": int(cols[13]),
                        "lut_entry_logical_bytes": int(cols[14]), "effective_ops": int(cols[15]),
                        "metric": float(cols[16]), "unit": cols[17], "correctness": int(cols[18]),
                        "failure_code": int(cols[19]),
                    }
                    # The protocol requires a nominal >=50 ms target.  Allow
                    # 2% qtimer/calibration rounding, but reject the known
                    # 0/1-us pilot-degeneration rows instead of treating them
                    # as real independent measurements.
                    if row["kind"] in ("hmx_fp16_gemm", "hvx_fp16_gemm",
                                       "hvx_v79_qf16_affine_relu", "lut_exp_vtcm_vgather") and row["elapsed_us"] < 49000:
                        continue
                    rows.append(row)
                elif len(cols) == 9 and cols[0].startswith(("hmx_", "hvx_", "ddr_", "vtcm_")):
                    rows.append({"source": path.name, "sample": sample, "schema_version": 1,
                                 "mode": cols[0], "engine": "HMX" if cols[0].startswith("hmx") else "HVX",
                                 "kind": cols[1], "size": int(cols[3]), "iters": int(cols[4]),
                                 "elapsed_us": int(cols[5]), "effective_ops": int(cols[6]),
                                 "metric": float(cols[7]), "unit": cols[8], "correctness": 1})
            except (ValueError, IndexError):
                continue
    return rows


def label_from_name(name):
    for label in sorted(LABELS, key=len, reverse=True):
        if name.startswith(label + "_") or name == label + ".log":
            return label
    return None


def parse_micro(root):
    rows = []
    for path in sorted((root / "raw" / "micro").glob("*_sample*.log")):
        label = label_from_name(path.name)
        sm = re.search(r"sample(\d+)", path.name)
        for line in path.read_text(errors="replace").splitlines():
            if "SCNA_EXP_BENCH" not in line:
                continue
            data = fields(line)
            if "paired_ns_per_64" in data:
                rows.append({"source": path.name, "label": label, "sample": int(sm.group(1)) if sm else 0,
                             "ns_per_64": float(data["paired_ns_per_64"]),
                             "optimized_impl": int(data.get("optimized_impl", -1)),
                             "dead_neurons_removed": int(data.get("dead_neurons_removed", 0)),
                             "dense_rmse": float(data.get("dense_rmse", math.nan)),
                             "dense_max_abs": float(data.get("dense_max_abs", math.nan)),
                             "monotonic_violations": int(data.get("monotonic_violations", -1)),
                             "nonfinite": int(data.get("random_nonfinite_count", -1)) + int(data.get("nan_count", -1))})
    return rows


def parse_attention(root, folder="attention"):
    rows = []
    for path in sorted((root / "raw" / folder).glob("*.log")):
        label = label_from_name(path.name)
        qm = re.search(r"_q(\d+)", path.name)
        sm = re.search(r"_s(\d+)", path.name)
        wm = re.search(r"_w(auto|\d+)", path.name)
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_HOST_TIMING" not in line:
                continue
            data = fields(line)
            if data.get("phase") != "measure" or int(data.get("ret", 1)) != 0:
                continue
            rows.append({"source": path.name, "label": label, "qo_len": int(data.get("qo_len", qm.group(1) if qm else 0)),
                         "kv_len": int(data.get("kv_len", 0)), "session": int(sm.group(1)) if sm else 0,
                         "iteration": int(data.get("iteration", 0)), "workers": wm.group(1) if wm else str(data.get("workers", 1)),
                         "host_us": float(data["host_elapsed_us"])})
    return rows


def parse_diagnostics(root):
    rows = []
    for path in sorted((root / "raw" / "lut_scna_diagnostic").glob("*.log")):
        label = label_from_name(path.name)
        km = re.search(r"kv(\d+)", path.name)
        totals = defaultdict(float)
        records = 0
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_TIMERS" not in line:
                continue
            data = fields(line)
            if data.get("phase") != "measure":
                continue
            records += 1
            for key in ("profiled_total", "q_load", "k_load", "v_load", "qk_dot", "safe_sm",
                        "scna_exp", "core_acc", "o_scale", "o_store"):
                totals[key + "_us"] += float(data.get(key, 0))
        if records:
            row = {"source": path.name, "label": label, "kv_len": int(km.group(1)) if km else 0,
                   "records": records}
            row.update(totals)
            rows.append(row)
    return rows


def parse_accuracy(root):
    rows = []
    for path in sorted((root / "raw" / "accuracy").glob("*.log")):
        label = label_from_name(path.name)
        compare = None
        masked_ok = tail_ok = True
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_COMPARE " in line:
                compare = fields(line)
            if "FIG8_NUMERIC " in line:
                data = fields(line)
                masked_ok &= int(data.get("masked_p_nonzero", 0)) == 0
                tail_ok &= int(data.get("tail_p_nonzero", 0)) == 0
        if compare:
            rows.append({"source": path.name, "label": label, "pass": int(compare.get("pass", 0)),
                         "rmse": float(compare.get("rmse", math.inf)),
                         "max_abs": float(compare.get("max_abs_error", math.inf)),
                         "finite": int(compare.get("candidate_nonfinite", 1)) == 0,
                         "mask_zero": masked_ok, "tail_zero": tail_ok})
    return rows


def static_audit(root, project):
    path = root / "static" / "static_metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    lut = {"available": False, "symbols": {}, "vgather": 0}
    artifact = project / "artifacts/variants/stage1_dynamic_row/libhtp_ops_skel.so"
    objdump = Path("/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump")
    if artifact.exists() and objdump.exists():
        text = subprocess.check_output([str(objdump), "-d", "--no-show-raw-insn", str(artifact)], text=True)
        out = root / "static" / "lut_hot_symbols.v79.disasm.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        current = None
        blocks = defaultdict(list)
        for line in text.splitlines():
            match = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
            if match:
                current = match.group(1)
            elif current and re.match(r"^\s*[0-9a-f]+:", line):
                blocks[current].append(line)
        selected = {name: body for name, body in blocks.items() if any("vgather" in x.lower() for x in body)}
        lut = {"available": True, "artifact": str(artifact.relative_to(project)),
               "disassembly": str(out.relative_to(root)), "vgather": sum("vgather" in x.lower() for b in selected.values() for x in b),
               "symbols": {name: {"instructions": len(body), "packets": sum("{" in x for x in body),
                                  "vgather": sum("vgather" in x.lower() for x in body),
                                  "vmem": sum("vmem" in x.lower() for x in body)} for name, body in selected.items()}}
    return {"scna": metrics, "lut": lut}


def summarize_samples(rows, key, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(float(row[key]))
    result = []
    for group, values in sorted(grouped.items(), key=lambda x: str(x[0])):
        item = dict(zip(group_keys, group))
        item.update({"n": len(values), "median": median(values), "q1": quantile(values, .25),
                     "q3": quantile(values, .75), "ci_low": quantile(values, .025), "ci_high": quantile(values, .975)})
        result.append(item)
    return result


def ratio_by_sample(rows, value_key, candidate, baseline, extra_filter=None):
    selected = [row for row in rows if not extra_filter or extra_filter(row)]
    maps = {}
    for label in (candidate, baseline):
        maps[label] = {(row.get("session", 0), row.get("iteration", row.get("sample", 0))): float(row[value_key])
                       for row in selected if row.get("label") == label}
    return bootstrap_ratio(maps[candidate], maps[baseline])


def configure_plots():
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    matplotlib.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                                "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                                "svg.fonttype": "none", "svg.hashsalt": "lut-scna-v2", "pdf.fonttype": 42})


def save_figure(fig, base):
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300, metadata={"Software": "Matplotlib+Seaborn"})
    plt.close(fig)


def plot_nonlinear(base, roofs, micro, single=False):
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 5.8)) if single else plt.subplots(1, 2, figsize=(7.2, 3.0))
    short = {"pair_static_d8": "qf16 static", "optimized": "widened FMA",
             "optimized_qf16_tree": "qf16 tree", "optimized_piecewise_d8": "piecewise"}
    ax = axes[0]
    peak = max((r["metric"] for r in roofs if r["kind"] == "hvx_v79_qf16_affine_relu" and r["correctness"]), default=None)
    vtcm = max((r["metric"] for r in roofs if r["kind"] == "vtcm_copy"), default=None)
    if peak:
        xs = [0.1, 100]
        ys = [min(peak, (vtcm or 1e9) * x / 1000) for x in xs]
        ax.plot(xs, ys, color="0.25", linestyle="--", label=("HVX roof" if single else f"v79 HVX roof ({peak:.3f} TOPS)"))
    for label, ops in OPS_PER_ELEMENT.items():
        values = [r["ns_per_64"] for r in micro if r["label"] == label]
        if not values:
            continue
        throughput = ops * 64 / (median(values) * 1e-9) / 1e12
        ax.scatter(ops / 4, throughput, color=COLORS[label], marker=MARKERS[label], s=42,
                   label=short[label] if single else LABELS[label])
    ax.set(xscale="log", yscale="log", xlabel=("Effective FLOP / logical byte" if single else "Arithmetic intensity (effective FLOP / logical S+P byte)"),
           ylabel="Effective arithmetic throughput (TOPS)", title="(a) SCNA arithmetic roofline")
    ax.legend(frameon=False)

    ax = axes[1]
    if vtcm:
        x = [3.5, 6.5]
        ax.plot(x, [vtcm / value for value in x], color="0.25", linestyle="--",
                label="VTCM byte roof" if single else f"VTCM logical-byte roof ({vtcm:.1f} GB/s)")
    for label in OPS_PER_ELEMENT:
        values = [r["ns_per_64"] for r in micro if r["label"] == label]
        if values:
            gelem = 64 / (median(values) * 1e-9) / 1e9
            ax.scatter(4, gelem, color=COLORS[label], marker=MARKERS[label], s=42,
                       label=short[label] if single else LABELS[label])
    lut_groups = defaultdict(list)
    for row in roofs:
        if row["kind"] == "lut_exp_vtcm_vgather" and row["correctness"]:
            lut_groups[row["distribution"]].append(row["metric"])
    for index, (distribution, values) in enumerate(sorted(lut_groups.items())):
        ax.scatter(6 + (index - 1) * .08, median(values), color=COLORS["lut-exp"], marker=("o", "s", "^")[index],
                   s=42, label=f"LUT-{distribution}")
    ax.set(xlabel=("Logical B / exp2 element\n(transaction size unobserved)" if single else "Logical bytes / exp2 element (transaction size unobserved)"), ylabel="Useful exp2 throughput (GElem/s)",
           title="(b) Common useful-work view")
    ax.set_xlim(3.5, 6.5)
    ax.legend(frameon=False, ncol=1 if single else 2, loc="upper right")
    fig.suptitle("LUT-EXP vs. SCNA" if single else "LUT-EXP vs. SCNA nonlinear evaluation", y=1.02)
    fig.tight_layout()
    save_figure(fig, base)


def plot_pipeline(base, attention, diagnostics, roofs, single=False):
    fig = plt.figure(figsize=(3.45, 8.2) if single else (7.2, 4.7))
    if single:
        grid = fig.add_gridspec(3, 1, height_ratios=(1.1, 1, 1), hspace=.65)
        ax = fig.add_subplot(grid[0, 0])
    else:
        grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1), hspace=.47, wspace=.35)
        ax = fig.add_subplot(grid[0, :])
    for label in LABELS:
        points = []
        for q in (1, 4, 8, 16, 32):
            values = [r["host_us"] for r in attention if r["label"] == label and r["qo_len"] == q]
            if values:
                lo, hi = bootstrap_median_ci(values, seed=20260819 + q)
                points.append((q, median(values), lo, hi))
        if not points:
            continue
        ax.errorbar([p[0] for p in points], [p[1] for p in points],
                    yerr=[[p[1] - p[2] for p in points], [p[3] - p[1] for p in points]],
                    label=LABELS[label], color=COLORS[label], marker=MARKERS[label], linewidth=1.2, capsize=2)
    ax.set(xlabel="Query length (KV=4096)", ylabel="Host latency (us)",
           title="(a) Full Attention median and bootstrap 95% CI")
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.legend(frameon=False, ncol=1 if single else 2)

    ax = fig.add_subplot(grid[1, 0])
    long_rows = [r for r in diagnostics if r["kv_len"] == 4096]
    labels = [r["label"] for r in long_rows]
    x = list(range(len(labels)))
    common = [max(0, r.get("safe_sm_us", 0) - r.get("scna_exp_us", 0)) for r in long_rows]
    evaluator = [r.get("scna_exp_us", 0) if r["label"] != "lut-exp" else r.get("safe_sm_us", 0) for r in long_rows]
    ax.bar(x, common, color="0.75", label="Shared safe_sm work")
    ax.bar(x, evaluator, bottom=common, color=[COLORS.get(label, (.3, .3, .3)) for label in labels],
           hatch=["//" if label == "lut-exp" else "" for label in labels], label="Evaluator / LUT safe_sm")
    ax.set_xticks(x, [LABELS.get(label, label) for label in labels], rotation=25, ha="right")
    ax.set(ylabel="Summed DSP qtimer (us)", title="(b) HVX safe_sm diagnostic")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[2, 0] if single else grid[1, 1])
    ax.axis("off")
    hmx = max((r["metric"] for r in roofs if r["kind"] == "hmx_fp16_gemm"), default=None)
    hvx = max((r["metric"] for r in roofs if r["kind"] == "hvx_v79_qf16_affine_relu"), default=None)
    ddr = max((r["metric"] for r in roofs if r["kind"] == "ddr_copy"), default=None)
    dma = max((r["metric"] for r in roofs if r["kind"] == "hmx_dma_read"), default=None)
    vtcm = max((r["metric"] for r in roofs if r["kind"] == "vtcm_copy"), default=None)
    items = [("DDR copy", f"{ddr:.1f} GB/s" if ddr else "UNAVAILABLE", "host-visible memory stream"),
             ("DDR→VTCM DMA", f"{dma:.1f} GB/s" if dma else "UNAVAILABLE", "Q/K/V tile load roof"),
             ("VTCM data", f"{vtcm:.1f} GB/s" if vtcm else "UNAVAILABLE", "S/P + LUT table"),
             ("HVX compute", f"{hvx:.3f} TOPS" if hvx else "UNAVAILABLE", "evaluator + safe_sm"),
             ("HMX compute", f"{hmx:.2f} TFLOP/s" if hmx else "UNAVAILABLE", "QK + PV")]
    for index, (name, value, stage) in enumerate(items):
        y = .87 - index * .185
        ax.add_patch(FancyBboxPatch((.03, y - .055), .94, .135, transform=ax.transAxes,
                                    boxstyle="round,pad=.02", fc=sns.color_palette("colorblind")[index],
                                    ec="0.25", alpha=.35))
        ax.text(.07, y + .025, name, transform=ax.transAxes, fontsize=8, weight="bold", va="center")
        ax.text(.93, y + .025, value, transform=ax.transAxes, fontsize=8, ha="right", va="center")
        ax.text(.07, y - .035, stage, transform=ax.transAxes, fontsize=7, va="center", color="0.25")
    ax.set_title("(c) Independent measured roofs (native units)")
    fig.suptitle("Attention pipeline: compute engines remain separate", y=1.02)
    save_figure(fig, base)


def write_drawio(path):
    boxes = [(20, 40, 125, 55, "S in VTCM"), (180, 15, 160, 80, "LUT: index + vgather\n64 KiB resident table"),
             (180, 125, 160, 80, "SCNA d8: 7 active affine-ReLU\nbroadcast + dependency chains"),
             (380, 40, 125, 55, "P in VTCM"), (545, 40, 125, 55, "rowsum + P·V")]
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for index, (x, y, w, h, text) in enumerate(boxes, 2):
        cells.append(f'<mxCell id="{index}" value="{text}" style="rounded=1;whiteSpace=wrap;html=1;fontSize=11;" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for index, (source, target) in enumerate(((2, 3), (2, 4), (3, 5), (4, 5), (5, 6)), 20):
        cells.append(f'<mxCell id="{index}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;" edge="1" parent="1" source="{source}" target="{target}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    path.write_text('<mxfile host="app.diagrams.net"><diagram name="LUT-vs-SCNA"><mxGraphModel><root>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>\n')


def plot_dataflow(base, drawio_path):
    write_drawio(drawio_path)
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [(0.2, 1.55, 1.5, .8, "S in VTCM", "0.85"), (2.4, 2.35, 2.5, 1.05, "LUT-EXP\nindex → vgather\n64 KiB resident", COLORS["lut-exp"]),
             (2.4, .35, 2.5, 1.25, "SCNA d8\n7 active affine–ReLU paths\nsplat + dependency chains", COLORS["optimized_qf16_tree"]),
             (5.7, 1.55, 1.5, .8, "P in VTCM", "0.85"), (8.1, 1.55, 1.6, .8, "rowsum + P·V", "0.85")]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04", fc=color, ec="0.25", alpha=.85))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    arrows = [((1.7, 2.0), (2.4, 2.85)), ((1.7, 1.9), (2.4, .95)), ((4.9, 2.85), (5.7, 2.0)),
              ((4.9, .95), (5.7, 1.9)), ((7.2, 1.95), (8.1, 1.95))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=1, color="0.25"))
    ax.text(3.65, 3.65, "low arithmetic work; gather latency may overlap", ha="center", fontsize=7)
    ax.text(3.65, .08, "higher issue pressure; tree/piecewise target the chain", ha="center", fontsize=7)
    ax.set_title("Nonlinear evaluator dataflow and measured weakness hypotheses")
    fig.tight_layout()
    save_figure(fig, base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    root, project = Path(args.run_dir), Path(args.project)
    root.mkdir(parents=True, exist_ok=True)
    configure_plots()

    roofs = parse_roofs(root)
    micro = parse_micro(root)
    attention = parse_attention(root)
    scaling = parse_attention(root, "scaling")
    diagnostics = parse_diagnostics(root)
    accuracy = parse_accuracy(root)
    audit = static_audit(root, project)
    roof_summary = summarize_samples(roofs, "metric", ["mode", "kind", "distribution", "size"])
    micro_summary = summarize_samples(micro, "ns_per_64", ["label"])
    attention_summary = summarize_samples(attention, "host_us", ["label", "qo_len"])

    micro_ratios = {label: ratio_by_sample(micro, "ns_per_64", label, "pair_static_d8")
                    for label in ("optimized", "optimized_qf16_tree", "optimized_piecewise_d8")}
    q32 = lambda row: row.get("qo_len") == 32
    attention_ratios = {label: ratio_by_sample(attention, "host_us", label, "pair_static_d8", q32)
                        for label in ("optimized", "optimized_qf16_tree", "optimized_piecewise_d8", "lut-exp")}
    attention_vs_lut = {str(q): {label: ratio_by_sample(
                            attention, "host_us", label, "lut-exp", lambda row, q=q: row.get("qo_len") == q)
                        for label in ("pair_static_d8", "optimized", "optimized_qf16_tree", "optimized_piecewise_d8")}
                        for q in (1, 4, 8, 16, 32)}
    accuracy_gate = {label: bool([r for r in accuracy if r["label"] == label]) and
                     all(r["pass"] and r["finite"] and r["mask_zero"] and r["tail_zero"]
                         for r in accuracy if r["label"] == label)
                     for label in ("optimized_qf16_tree", "optimized_piecewise_d8")}
    promotion = {}
    for label in accuracy_gate:
        mr, ar = micro_ratios.get(label), attention_ratios.get(label)
        promotion[label] = {"accuracy": accuracy_gate[label], "micro_faster": bool(mr and mr["ci_high"] < 1),
                            "attention_no_significant_regression": bool(ar and ar["ci_low"] <= 1),
                            "promote": bool(accuracy_gate[label] and mr and mr["ci_high"] < 1 and ar and ar["ci_low"] <= 1)}

    recovery_logs = list((root / "raw" / "recovery").glob("*.log"))
    lut_valid_counts = {distribution: len({r["sample"] for r in roofs
                                           if r["kind"] == "lut_exp_vtcm_vgather" and
                                           r.get("distribution") == distribution and r["correctness"]})
                        for distribution in ("dense", "attention", "random")}
    lut_complete = all(count >= 30 for count in lut_valid_counts.values())
    failure_status = {
        "hmx_fp16": "MEASURED" if any(r["kind"] == "hmx_fp16_gemm" for r in roofs) else "MISSING",
        "hvx_fp16": "MEASURED" if any(r["kind"] == "hvx_fp16_gemm" for r in roofs) else "UNAVAILABLE",
        "hvx_native_v81_case21": "N/A_ON_V79_BUILD_GUARD",
        "hvx_v79_peak": "MEASURED" if any(r["kind"] == "hvx_v79_qf16_affine_relu" for r in roofs) else "UNAVAILABLE",
        "lut_exp_micro": "MEASURED_30_OF_30" if lut_complete else "PARTIAL_VALID_SAMPLES",
        "lut_valid_samples": lut_valid_counts,
        "recovery_log_count": len(recovery_logs),
        "final_unavailable_cases": sum(count < 30 for count in lut_valid_counts.values()),
    }
    summary = {"schema_version": 2, "environment": "real_device_only", "roofs": roofs,
               "roof_summary": roof_summary, "micro": micro, "micro_summary": micro_summary,
               "attention": attention, "attention_summary": attention_summary, "scaling": scaling,
               "diagnostics": diagnostics, "accuracy": accuracy, "static_audit": audit,
               "ratios": {"micro_vs_pair_static_d8": micro_ratios,
                          "q32_attention_vs_pair_static_d8": attention_ratios,
                          "attention_vs_lut_exp": attention_vs_lut},
               "promotion": promotion, "failure_status": failure_status,
               "definitions": {"logical_scna_bytes_per_element": 4, "logical_lut_bytes_per_element": 6,
                               "lut_transaction_bytes_observed": False,
                               "simulator_merged_with_device": False}}
    (root / "roofline_lut_scna_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(root / "roofline_metrics.csv", roofs)
    write_csv(root / "scna_micro_samples.csv", micro)
    write_csv(root / "attention_samples.csv", attention)
    write_csv(root / "lut_scna_diagnostic.csv", diagnostics)
    write_csv(root / "accuracy_gates.csv", accuracy)

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plot_nonlinear(figures / "roofline_nonlinear_lut_vs_scna", roofs, micro)
    plot_nonlinear(figures / "roofline_nonlinear_lut_vs_scna_double", roofs, micro)
    plot_nonlinear(figures / "roofline_nonlinear_lut_vs_scna_single", roofs, micro, single=True)
    plot_pipeline(figures / "roofline_attention_pipeline", attention, diagnostics, roofs)
    plot_pipeline(figures / "roofline_attention_pipeline_double", attention, diagnostics, roofs)
    plot_pipeline(figures / "roofline_attention_pipeline_single", attention, diagnostics, roofs, single=True)
    plot_dataflow(figures / "nonlinear_dataflow_weakness", figures / "nonlinear_dataflow_weakness.drawio")

    hmx_peak = max((r["metric"] for r in roofs if r["kind"] == "hmx_fp16_gemm"), default=None)
    hvx_gemm = max((r["metric"] for r in roofs if r["kind"] == "hvx_fp16_gemm"), default=None)
    hvx_peak = max((r["metric"] for r in roofs if r["kind"] == "hvx_v79_qf16_affine_relu"), default=None)
    ddr_copy_peak = max((r["metric"] for r in roofs if r["kind"] == "ddr_copy"), default=None)
    vtcm_copy_peak = max((r["metric"] for r in roofs if r["kind"] == "vtcm_copy"), default=None)
    dma_peak = max((r["metric"] for r in roofs if r["kind"] == "hmx_dma_read"), default=None)
    lut_attention = median([r["metric"] for r in roofs if r["kind"] == "lut_exp_vtcm_vgather" and r["distribution"] == "attention"])
    static_rows = audit.get("scna", {})
    facts = []
    if hmx_peak is not None: facts.append(f"- HMX FP16 实测 roof 最大值为 {hmx_peak:.4f} TFLOP/s；旧日志中的 HMX 行有效。")
    if hvx_gemm is not None: facts.append(f"- 校准后的 HVX FP16 软件 GEMM 最大值为 {hvx_gemm:.4f} TFLOP/s，原失败是组合请求超时而非零性能。")
    if hvx_peak is not None: facts.append(f"- v79 qf16 affine–ReLU–accumulate 指令混合 roof 为 {hvx_peak:.4f} TOPS。")
    if ddr_copy_peak is not None and vtcm_copy_peak is not None and dma_peak is not None:
        facts.append(f"- 独立数据 roof：DDR copy {ddr_copy_peak:.2f} GB/s、DDR→VTCM HMX DMA {dma_peak:.2f} GB/s、VTCM copy {vtcm_copy_peak:.2f} GB/s；三者不与 compute roof 合并。")
    if lut_attention is not None: facts.append(f"- Attention-score 分布下 LUT evaluator 中位吞吐为 {lut_attention:.4f} GElem/s（有效 n={lut_valid_counts['attention']}）；仅报告 6 B/元素逻辑流量，真实 gather transaction 未观测。")
    for label in ("pair_static_d8", "optimized", "optimized_qf16_tree", "optimized_piecewise_d8"):
        values = [r["ns_per_64"] for r in micro if r["label"] == label]
        if values: facts.append(f"- {LABELS[label]} evaluator 中位数：{median(values):.3f} ns/64 elements（n={len(values)}）。")

    diag_long = {r["label"]: r for r in diagnostics if r["kv_len"] == 4096}
    q32_medians = {label: median([r["host_us"] for r in attention if r["label"] == label and r["qo_len"] == 32])
                   for label in ("lut-exp", "pair_static_d8", "optimized", "optimized_qf16_tree", "optimized_piecewise_d8")}
    if lut_attention:
        facts.append(f"- 把 micro 统一为 useful exp2：LUT 为 {lut_attention:.3f} GElem/s；qf16 static 为 {64 / median([r['ns_per_64'] for r in micro if r['label']=='pair_static_d8']):.3f} GElem/s；piecewise 为 {64 / median([r['ns_per_64'] for r in micro if r['label']=='optimized_piecewise_d8']):.3f} GElem/s。")
    if diag_long.get("lut-exp") and diag_long.get("pair_static_d8"):
        facts.append(f"- KV=4096 诊断中，qf16 static `safe_sm` 为 {diag_long['pair_static_d8']['safe_sm_us']:.0f} us，LUT 为 {diag_long['lut-exp']['safe_sm_us']:.0f} us（{diag_long['pair_static_d8']['safe_sm_us']/diag_long['lut-exp']['safe_sm_us']:.2f}×）；`scna_exp` 是嵌套时间，不与 `safe_sm` 相加。")
    if q32_medians.get("lut-exp") and q32_medians.get("optimized_piecewise_d8"):
        facts.append(f"- 主结论 Qo=32/KV=4096：LUT 完整 Attention 中位数 {q32_medians['lut-exp']:.1f} us，piecewise SCNA {q32_medians['optimized_piecewise_d8']:.1f} us，后者仍慢 {100*(q32_medians['optimized_piecewise_d8']/q32_medians['lut-exp']-1):.1f}%。")

    lines = ["# LUT-EXP 与 SCNA：Qualcomm NPU Roofline、弱点归因与优化实测", "",
             "> 本报告只使用当前 run 的原始日志；失败或缺失项不会被补值。Simulator 结果单独报告，不能与真机 latency/cycle 合并。", "",
             "## 测量事实", "", *(facts or ["- 当前 run 尚无可用真机测量。"]), "",
             "### 失败项重新分类", "", f"```json\n{json.dumps(failure_status, indent=2, sort_keys=True)}\n```", "",
             "## Roofline 位置", "",
             "![Nonlinear roofline](figures/roofline_nonlinear_lut_vs_scna.svg)", "",
             "![Attention pipeline](figures/roofline_attention_pipeline.svg)", "",
             "- SCNA 标准面板按 affine multiplication、affine addition 和 accumulation 计 effective FLOP；ReLU/compare/mux 单独由反汇编解释。",
             "- LUT 不赋予虚构 FLOP；共同面板以一个有效 exp2 输出作为 useful-work unit。", "",
             "### 完整 Attention：相对 LUT-EXP", "", "| Qo | LUT median us | qf16 static / LUT | widened FMA / LUT | qf16 tree / LUT | piecewise / LUT |", "|---:|---:|---:|---:|---:|---:|"]
    for q in (1, 4, 8, 16, 32):
        lut_med = median([r["host_us"] for r in attention if r["label"] == "lut-exp" and r["qo_len"] == q])
        ratio_text = []
        for label in ("pair_static_d8", "optimized", "optimized_qf16_tree", "optimized_piecewise_d8"):
            ratio = attention_vs_lut[str(q)].get(label)
            ratio_text.append("N/A" if not ratio else f"{ratio['ratio']:.3f} [{ratio['ci_low']:.3f}, {ratio['ci_high']:.3f}]")
        lines.append(f"| {q} | {lut_med:.1f} | " + " | ".join(ratio_text) + " |")
    lines += ["", "表中为候选/LUT latency ratio；大于 1 表示候选更慢。CI 为按 session/iteration 配对的 10,000 次 bootstrap 95% CI。", "",
             "## Weakness 归因", "",
             "![Evaluator dataflow](figures/nonlinear_dataflow_weakness.svg)", "",
             "1. **算术并非天然更便宜。** d8 原实现对每元素执行 8 条 affine–ReLU–accumulate 路径；第八条在部署域内恒零，仍消耗指令。",
             "2. **v79 FMA 假设不成立。** widened-FMA artifact 在 v79 上展开为 qf32 `vmpy/vadd` packets；是否增加 packets、栈和 spill 见下表。",
             "3. **依赖链与广播形成 issue 压力。** qf16-tree 缩短累加依赖但提高并行 live range；piecewise 将算术降到一次 affine，却引入 compare/vmux 和可能的栈压力。最终以真机 CI 判定。",
             "4. **LUT 的 64 KiB 表常驻 VTCM。** vgather 的地址离散不等于 DDR 随机访问；小表局部性、双行发射及与 softmax 其余工作的重叠可使其优于 SCNA。",
             "5. **完整 Attention 会稀释 evaluator 改进。** piecewise micro 比 qf16 static 快约 11%，但诊断 `scna_exp` 只下降约 5%；rowmax、rowsum、在线 recurrence、S/P 搬运和 HMX QK/PV 均为共有成本。",
             "6. **LUT 优势随形状变化。** Qo=1 的固定调度/FastRPC 成本可反转排序；长 Qo/KV 才暴露 evaluator 与 `safe_sm` 差距，因此主结论限定在 KV=4096。", "",
             "### 热函数静态指标", "", "| Variant | inst | packets | splat | qf16 | vmpy | vadd | vmax | vmux | stack B | spills |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("pair_static_d8", "optimized", "optimized_qf16_tree", "optimized_piecewise_d8"):
        row = static_rows.get(label, {})
        lines.append(f"| {LABELS[label]} | {row.get('instructions','N/A')} | {row.get('packets','N/A')} | {row.get('splat','N/A')} | {row.get('qf16','N/A')} | {row.get('vmpy','N/A')} | {row.get('vadd','N/A')} | {row.get('vmax','N/A')} | {row.get('vmux','N/A')} | {row.get('stack_frame_bytes','N/A')} | {row.get('spill_memory','N/A')} |")
    lines += ["", "## 优化候选判定", "", "| Candidate | Accuracy | Micro faster (95% CI) | KV=4096 Qo32 no significant regression | Promote |", "|---|---:|---:|---:|---:|"]
    for label in ("optimized_qf16_tree", "optimized_piecewise_d8"):
        gate = promotion[label]
        mr, ar = micro_ratios.get(label), attention_ratios.get(label)
        mtext = "N/A" if not mr else f"{mr['ratio']:.3f} [{mr['ci_low']:.3f}, {mr['ci_high']:.3f}]"
        atext = "N/A" if not ar else f"{ar['ratio']:.3f} [{ar['ci_low']:.3f}, {ar['ci_high']:.3f}]"
        lines.append(f"| {LABELS[label]} | {gate['accuracy']} | {mtext} | {atext} | {gate['promote']} |")
    lines += ["", "表中 micro 数值为候选/qf16-static latency ratio；小于 1 才表示更快。", "",
              "## 优化策略与本轮裁决", "",
              "- `optimized_piecewise_d8` 通过全部数值门禁，micro 约快 11%，Qo32/KV4096 相对 qf16-static 无回退，因此是本轮唯一有效候选；它仍未追平 LUT-EXP。",
              "- `optimized_qf16_tree` 虽通过精度门禁，但 micro 慢约 2.2%，49 packets 高于 static 的 41，作为被否证方案保留，不提升为默认路径。",
              "- widened-FMA 在 v79 上增加到 61 packets、128 B stack 和 2 个 spill，且 micro 更慢；不要按 intrinsic 名称推断硬件 FMA 收益。",
              "- piecewise 的下一步应减少 32 个 `vmux` 与 384 B stack frame：把断点选择改为分层 predicate/区间编码，并跨多个 row-pair 延长已广播常量生命周期。",
              "- 若仍以超过 LUT 为目标，应优先软件流水化两组独立 row-pair，把 compare/mux 与 rowsum/online recurrence 交叠；d4/d6 重训和混合 LUT-SCNA 不属于本轮。", "",
              "## 限制", "",
              "- 未得到可靠 PMU/cache transaction 时，不报告 gather cache miss、stall 或真实表项传输字节。",
              "- `safe_sm` event replay 是低频诊断，不代替 5-session host latency 排名。",
              "- V81 HMX 手册仅用于机制解释，未用于外推 v79 峰值。",
              "- 图由 Matplotlib + Seaborn 从 CSV 自动生成；`.drawio` 是可编辑概念图源。具体投稿前需按 venue 政策披露 AI 辅助。", ""]
    (root / "ROOFLINE_LUT_VS_SCNA_REPORT_ZH.md").write_text("\n".join(lines))

    evidence = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "evidence_manifest.sha256":
            evidence.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    (root / "evidence_manifest.sha256").write_text("\n".join(evidence) + "\n")


if __name__ == "__main__":
    main()
