#!/usr/bin/env python3
"""Collect v79 hot-symbol disassembly and auditable instruction classes."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

VARIANTS = [
    "stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
    "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized",
]
HOT_SYMBOL = {
    "stage1_dynamic_row": ("stage1_dynamic_row", "hvx_scna_exp2_pair_vhf", "hvx_scna_exp2_vhf"),
    "prepare_once_row": ("prepared_dynamic_row", "hvx_scna_exp2_pair_vhf", "hvx_scna_exp2_vhf"),
    "pair_shared_dynamic": ("pair_shared_dynamic_qf16", "hvx_scna_exp2"),
    "pair_static_d8": ("pair_static_d8_qf16", "hvx_scna_exp2"),
    "pair_d8_fma_noinline": ("pair_static_d8_fma_noinline", "hvx_scna_exp2"),
    "pair_d8_fma_inline": ("hvx_scna_exp2_pair_vhf", "hvx_scna_exp2_vhf"),
    "optimized": ("pair_static_d8_fma_noinline", "hvx_scna_exp2_pair_vhf", "hvx_scna_exp2_vhf"),
}
ATTENTION_SYMBOLS = (
    "simple_flash_attn_f16_core",
    "hmx_mat_mul_fp16_core",
    "hmx_unit_acquire",
    "hmx_unit_release",
)


def instruction_metrics(lines):
    low = [line.lower() for line in lines]
    frames = [int(x, 0) for line in low for x in re.findall(r"allocframe\(#(0x[0-9a-f]+|\d+)\)", line)]
    return {
        "instructions": len(lines),
        "packets": sum("{" in line for line in lines),
        "branches": sum(bool(re.search(r"\b(jump|loop[01]|if\s*\(|dealloc_return)\b", line)) for line in low),
        "calls": sum("call" in line for line in low),
        "returns": sum("dealloc_return" in line or bool(re.search(r"\bjumpr\b", line)) for line in low),
        "vector_load_store": sum(bool(re.search(r"\b(vmem|mem[bhwd])", line)) for line in low),
        "splat_broadcast": sum("vsplat" in line for line in low),
        "shuffle_permute": sum(any(token in line for token in ("vshuff", "valign", "vdeal", "vdelta")) for line in low),
        "vector_mux_compare": sum(any(token in line for token in ("vmux", "vcmp")) for line in low),
        "vector_scatter": sum("vscatter" in line for line in low),
        "qf16_or_convert": sum("qf16" in line or "vconvert" in line for line in low),
        "fp16_multiply_fma": sum(("vmpy" in line or "vmpyacc" in line) and ".hf" in line for line in low),
        "vector_add_max": sum(("vadd" in line or "vmax" in line) for line in low),
        "predicate_ops": sum(bool(re.search(r"\bp[0-3]\b", line)) for line in low),
        "hmx_load_store": sum("mxmem" in line for line in low),
        "hmx_accumulator_control": sum("mxclracc" in line or "acc:" in line for line in low),
        "spill_memory": sum(bool(re.search(r"mem[dhw]\(r(29|30)", line)) for line in low),
        "stack_frame_bytes": max(frames, default=0),
        "code_bytes": 4 * len(lines),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    out = args.run_dir.resolve() / "static"
    out.mkdir(parents=True, exist_ok=True)
    sdk = Path(os.environ.get("HEXAGON_SDK_ROOT", "/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0"))
    tools = Path(os.environ.get("HEXAGON_SIM_CORE", sdk / "tools/HEXAGON_Tools/19.0.07")) / "Tools/bin"
    objdump = tools / "hexagon-llvm-objdump"
    readelf = tools / "hexagon-readelf"
    summary = {}
    for variant in VARIANTS:
        library = project / "artifacts/variants" / variant / "scna_sim.so"
        if not library.exists():
            summary[variant] = {"missing": True}
            continue
        text = subprocess.check_output([str(objdump), "-d", "--no-show-raw-insn", str(library)], text=True)
        disasm = out / f"{variant}.v79.disasm.txt"
        disasm.write_text(text)
        blocks, current = {}, None
        for line in text.splitlines():
            found = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
            if found:
                current = found.group(1)
                blocks[current] = []
            elif current and re.match(r"^\s*[0-9a-f]+:", line):
                blocks[current].append(line)
        selected = {name: body for name, body in blocks.items() if any(token in name for token in HOT_SYMBOL[variant])}
        hot = out / f"{variant}.hot.disasm.txt"
        hot.write_text("\n\n".join([f"<{name}>:\n" + "\n".join(body) for name, body in sorted(selected.items())]) + "\n")
        lines = [line for body in selected.values() for line in body]
        header = subprocess.check_output([str(readelf), "-h", str(library)], text=True, stderr=subprocess.STDOUT)
        row = instruction_metrics(lines)
        row.update({
            "artifact": str(library.relative_to(project)),
            "hot_disassembly": str(hot.relative_to(args.run_dir.resolve())),
            "machine_hexagon": "Hexagon" in header,
            "symbols": {name: instruction_metrics(body) for name, body in sorted(selected.items())},
        })
        summary[variant] = row
        if variant == "optimized":
            attention = {name: body for name, body in blocks.items() if name in ATTENTION_SYMBOLS}
            attention_path = out / "attention_hmx_hvx.hot.disasm.txt"
            attention_path.write_text(
                "\n\n".join([f"<{name}>:\n" + "\n".join(body) for name, body in sorted(attention.items())]) + "\n"
            )
            attention_lines = [line for body in attention.values() for line in body]
            summary["attention_hmx_hvx"] = {
                **instruction_metrics(attention_lines),
                "artifact": str(library.relative_to(project)),
                "hot_disassembly": str(attention_path.relative_to(args.run_dir.resolve())),
                "symbols": {name: instruction_metrics(body) for name, body in sorted(attention.items())},
                "scope_note": "Static code for selected Attention/HMX symbols; not a dynamic execution count.",
            }
    (out / "static_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(out / "static_metrics.json")


if __name__ == "__main__":
    main()
