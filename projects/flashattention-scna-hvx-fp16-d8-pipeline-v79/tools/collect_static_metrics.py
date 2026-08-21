#!/usr/bin/env python3
"""Collect auditable v79 code-generation metrics for every independent DSP library."""
import argparse
import json
import re
import subprocess
from pathlib import Path

VARIANTS = [
    "static_d8_ref", "d7_serial", "d7_scalar_w", "d7_pairret_noinline",
    "d7_pairret_inline", "d7_quad_pipeline", "d7_prebroadcast",
    "qf16_tree_control", "piecewise_control", "combined_confirm",
]
OBJDUMP = "/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"
EVALUATOR_SYMBOL = {
    "static_d8_ref": ("pair_static_d8_qf16",),
    "d7_serial": ("hvx_scna_exp2_pair_vhf",),
    "d7_scalar_w": ("hvx_scna_exp2_pair_vhf",),
    "d7_pairret_noinline": ("hvx_scna_exp2_pair_hot_return_vhf",),
    "d7_pairret_inline": ("hvx_scna_exp2_pair_vhf",),
    "d7_quad_pipeline": ("hvx_scna_exp2_pair_vhf",),
    "d7_prebroadcast": ("hvx_scna_exp2_pair_vhf",),
    "qf16_tree_control": ("hvx_scna_exp2_pair_vhf", "pair_static_d8_qf16_tree"),
    "piecewise_control": ("hvx_scna_exp2_pair_vhf", "pair_piecewise_d8"),
    "combined_confirm": ("hvx_scna_exp2_pair_vhf",),
}
CALLER_SYMBOL = "simple_flash_attn_f16_core"


def metric(lines):
    low = [line.lower() for line in lines]
    frames = [int(x, 0) for line in low for x in re.findall(r"allocframe\(#(0x[0-9a-f]+|\d+)\)", line)]
    return {
        "instructions": len(lines),
        "packets": sum("{" in line for line in lines),
        "branches": sum(bool(re.search(r"\b(jump|loop[01]|if\s*\(|dealloc_return)\b", line)) for line in low),
        "calls": sum("call" in line for line in low),
        "returns": sum(bool("dealloc_return" in line or re.search(r"\bjumpr\b", line)) for line in low),
        "splat": sum("vsplat" in line for line in low),
        "qf16": sum("qf16" in line or "vconvert" in line for line in low),
        # Hexagon's disassembler expands Q6_Vhf_vmpyacc_VhfVhfVhf into
        # widened .qf32 vmpy/vadd packets.  Count the IEEE-half input form,
        # while keeping qf16/convert as a separate dependency-chain metric.
        "fp16_fma": sum("vmpy" in line and ".hf" in line and ".qf32" in line for line in low),
        "vmpy": sum("vmpy" in line for line in low),
        "vadd": sum("vadd" in line for line in low),
        "vmax": sum("vmax" in line for line in low),
        "vmux": sum("vmux" in line for line in low),
        "vgather": sum("vgather" in line for line in low),
        "spill_memory": sum(bool(re.search(r"mem[dhw]\(r(29|30)", line)) for line in low),
        "stack_frame_bytes": max(frames, default=0),
        "code_bytes": 4 * len(lines),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    project, out = Path(args.project), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant in VARIANTS:
        library = project / "artifacts" / "variants" / variant / "libhtp_ops_skel.so"
        if not library.exists():
            summary[variant] = {"missing": True, "artifact": str(library)}
            continue
        text = subprocess.check_output([OBJDUMP, "-d", "--no-show-raw-insn", str(library)], text=True)
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
        selected = {name: body for name, body in blocks.items()
                    if any(token in name for token in EVALUATOR_SYMBOL[variant])}
        caller = {name: body for name, body in blocks.items() if CALLER_SYMBOL in name}
        lines = [line for body in selected.values() for line in body]
        row = metric(lines)
        row.update({
            "artifact": str(library.relative_to(project)),
            "disassembly": str(disasm.relative_to(out.parent)),
            "symbols": {name: metric(body) for name, body in sorted(selected.items())},
            "caller": metric([line for body in caller.values() for line in body]),
            "caller_symbols": {name: metric(body) for name, body in sorted(caller.items())},
            "mv79_verified": "-mv79" in (library.parent / "compile_flags.txt").read_text(errors="replace")
                              if (library.parent / "compile_flags.txt").exists() else False,
        })
        row["effective_pair_splat"] = row["splat"]
        row["scalar_weight_multiply"] = sum("vmpy" in line.lower() and ",r" in line.lower()
                                               for line in lines)
        summary[variant] = row
    (out / "static_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
