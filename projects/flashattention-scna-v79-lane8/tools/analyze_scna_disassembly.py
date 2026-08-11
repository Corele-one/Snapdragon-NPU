#!/usr/bin/env python3
"""Disassemble the three SCNA kernels and count instruction pressure categories."""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


FUNCTIONS = (
    "hvx_scna_exp2_serial_d8_vhf",
    "hvx_scna_exp2_lane8_d8_vhf",
    "hvx_scna_exp2_lane8_sequential_pair_d8_vhf",
    "hvx_scna_exp2_lane8_split4_pair_d8_vhf",
    "hvx_scna_exp2_lane8_pack_once_pair_d8_vhf",
    "hvx_scna_exp2_lane8_pair_d8_vhf",
)
LABEL = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")
INSN = re.compile(r"^\s*[0-9a-f]+:\s+(?:\{\s*)?(.*?)(?:\s*\})?\s*$")


def category(text):
    mnemonic_match = re.search(r"=\s*([a-z][a-z0-9.]*)\(|\b(call|jump|dealloc_return)\b", text)
    mnemonic = (mnemonic_match.group(1) or mnemonic_match.group(2)) if mnemonic_match else "scalar"
    if mnemonic.startswith(("vlut", "vlalign", "vdeal", "vshuff", "vror", "valign")):
        return "shuffle_reduction", mnemonic
    if mnemonic.startswith(("vmpy", "vadd", "vmax", "vcmp", "vmux", "vxor", "vand", "vor")):
        return "vector_compute", mnemonic
    if "vmem" in text:
        return "vector_load_store", "vmem"
    if mnemonic in {"call", "jump", "dealloc_return"}:
        return "control", mnemonic
    return "scalar_other", mnemonic


def svg(rows, path):
    cats = ("vector_compute", "shuffle_reduction", "vector_load_store", "scalar_other", "control")
    colors = ("#3366cc", "#dc3912", "#ff9900", "#109618", "#990099")
    width, height, left, top = 980, 440, 90, 45
    chart_h = 300
    max_total = max(sum(row[cat] for cat in cats) for row in rows)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<text x="490" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">SCNA v79 static instruction categories</text>']
    bar_w = 170
    for index, row in enumerate(rows):
        x = left + index * 290
        y = top + chart_h
        for cat, color in zip(cats, colors):
            h = chart_h * row[cat] / max_total
            y -= h
            pieces.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{color}"/>')
        pieces.append(f'<text x="{x + bar_w/2}" y="{top + chart_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="13">{row["variant"]}</text>')
        pieces.append(f'<text x="{x + bar_w/2}" y="{y - 6:.2f}" text-anchor="middle" font-family="sans-serif" font-size="13">{row["instructions"]}</text>')
    lx = 90
    for cat, color in zip(cats, colors):
        pieces.append(f'<rect x="{lx}" y="400" width="14" height="14" fill="{color}"/>')
        pieces.append(f'<text x="{lx + 19}" y="412" font-family="sans-serif" font-size="12">{cat}</text>')
        lx += 170
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    disassembly = subprocess.check_output(
        [str(args.objdump), "-d", "--no-show-raw-insn", str(args.binary)], text=True
    )
    (args.out_dir / "libhtp_ops_skel.disasm.txt").write_text(disassembly, encoding="utf-8")
    selected = {name: [] for name in FUNCTIONS}
    current = None
    for line in disassembly.splitlines():
        label = LABEL.match(line)
        if label:
            current = label.group(1) if label.group(1) in selected else None
            continue
        if current and INSN.match(line):
            selected[current].append(line)

    rows = []
    for name in FUNCTIONS:
        counts, mnemonics = Counter(), Counter()
        spills = 0
        packet_sizes, current_packet = Counter(), 0
        for line in selected[name]:
            match = INSN.match(line)
            text = match.group(1)
            cat, mnemonic = category(text)
            counts[cat] += 1
            mnemonics[mnemonic] += 1
            spills += int("r29" in text and "mem" in text)
            if "{" in line:
                if current_packet:
                    packet_sizes[current_packet] += 1
                current_packet = 1
            else:
                current_packet += 1
            if "}" in line:
                packet_sizes[current_packet] += 1
                current_packet = 0
        variant = name.replace("hvx_scna_exp2_", "").replace("_d8_vhf", "")
        variant = {
            "lane8": "lane8-single",
            "lane8_pair": "current-pair",
            "lane8_sequential_pair": "sequential-pair-wrapper",
            "lane8_split4_pair": "split4-pair",
            "lane8_pack_once_pair": "pack-once-pair",
        }.get(variant, variant)
        row = {
            "function": name, "variant": variant, "instructions": len(selected[name]), "spill_memory_ops": spills,
            **{cat: counts[cat] for cat in ("vector_compute", "shuffle_reduction", "vector_load_store", "scalar_other", "control")},
            "packet_slot_distribution": dict(sorted(packet_sizes.items())),
            "top_mnemonics": dict(mnemonics.most_common(20)),
        }
        rows.append(row)
    if any(row["instructions"] == 0 for row in rows):
        raise SystemExit("one or more SCNA symbols were not found in disassembly")
    (args.out_dir / "scna_disassembly_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "scna_disassembly_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["variant", "instructions", "vector_compute", "shuffle_reduction", "vector_load_store", "scalar_other", "control", "spill_memory_ops"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    svg(rows, args.out_dir / "scna_instruction_categories.svg")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
