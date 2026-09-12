#!/usr/bin/env python3
"""Audit the frozen pair-return evaluator and absence of experimental kernels."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


SYMBOL = "hvx_scna_exp2_pair_hot_return_vhf"


def extract_symbol(disassembly: str) -> list[str]:
    lines = []
    active = False
    for line in disassembly.splitlines():
        match = re.match(r"^([0-9a-f]+) <([^>]+)>:$", line)
        if match:
            if active:
                break
            active = match.group(2) == SYMBOL
            continue
        if active and re.match(r"^\s*[0-9a-f]+:", line):
            lines.append(line)
    if not lines:
        raise ValueError(f"symbol {SYMBOL} not found")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disassembly-output", type=Path, required=True)
    args = parser.parse_args()

    disassembly = subprocess.run(
        [str(args.objdump), "-d", str(args.artifact)], check=True, text=True, stdout=subprocess.PIPE
    ).stdout
    args.disassembly_output.parent.mkdir(parents=True, exist_ok=True)
    args.disassembly_output.write_text(disassembly)
    lines = extract_symbol(disassembly)
    packet_count = sum("{" in line for line in lines)
    frame_lines = [line for line in lines if re.search(r"\b(?:allocframe|deallocframe|r29|sp)\b", line)]
    spill_lines = [line for line in lines if re.search(r"mem[dw]\(r29|mem[dw]\(sp", line)]

    implementation_files = [
        args.source_root / "include/dsp/scna_exp2_hot.h",
        args.source_root / "src/dsp/ops/scna_exp2.c",
        args.source_root / "src/dsp/ops/flash_attn.c",
    ]
    banned = (
        "two_neuron", "short_constant", "prebroadcast", "quad_hot", "two_accumulator",
        "piecewise_exp", "scale_probe", "static_d8_hot", "optimized_4row",
    )
    banned_hits = []
    for path in implementation_files:
        text = path.read_text(errors="replace").lower()
        for token in banned:
            if token in text:
                banned_hits.append(f"{path.relative_to(args.source_root)}:{token}")

    result = {
        "schema_version": 1,
        "symbol": SYMBOL,
        "instructions": len(lines),
        "packets": packet_count,
        "spill_instructions": len(spill_lines),
        "frame_bytes": 0 if not frame_lines else None,
        "frame_related_lines": frame_lines,
        "banned_experimental_kernel_hits": banned_hits,
    }
    result["pass"] = (
        result["instructions"] == 112
        and result["packets"] == 36
        and result["spill_instructions"] == 0
        and result["frame_bytes"] == 0
        and not banned_hits
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
