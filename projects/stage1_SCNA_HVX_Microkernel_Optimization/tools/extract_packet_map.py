#!/usr/bin/env python3
"""Extract auditable packet-level facts from a Hexagon disassembly symbol."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def symbol_lines(path: Path, token: str) -> list[str]:
    selected = []
    active = False
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if match:
            if active:
                break
            active = token in match.group(1)
            continue
        if active and re.match(r"^\s*[0-9a-f]+:", line):
            selected.append(line)
    if not selected:
        raise ValueError(f"symbol containing {token!r} not found in {path}")
    return selected


def packets(lines: list[str]) -> list[list[str]]:
    result: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        instruction = re.sub(r"^\s*[0-9a-f]+:\s*", "", line).strip()
        starts = instruction.startswith("{")
        ends = instruction.endswith("}")
        instruction = instruction.removeprefix("{").removesuffix("}").strip()
        parts = [part.strip() for part in instruction.split(";\t") if part.strip()]
        if starts:
            if current is not None:
                raise ValueError("nested packet")
            current = []
        if current is None:
            raise ValueError(f"instruction outside packet: {line}")
        current.extend(parts)
        if ends:
            result.append(current)
            current = None
    if current is not None:
        raise ValueError("unterminated packet")
    return result


def categories(instruction: str) -> list[str]:
    low = instruction.lower()
    result = []
    if "vmpy" in low or "mpyi" in low:
        result.append("multiply")
    if "vadd" in low or re.search(r"\badd\(", low):
        result.append("add/FMA")
    if "vcmp" in low or re.search(r"\bcmp\.", low):
        result.append("compare")
    if any(op in low for op in ("vdeal", "vshuff", "valign", "vlalign", "vdelta", "vrdelta")):
        result.append("permute/shuffle")
    if "bitsplit" in low:
        result.append("scalar-bit-split")
    if "vsplat" in low:
        result.append("splat")
    if "mem" in low and "=" in low:
        result.append("load")
    if re.search(r"v\d+\.hf\s*=\s*v\d+\.qf16", low):
        result.append("conversion")
    if "vmax" in low or "vmux" in low:
        result.append("compare/select")
    if "vxor" in low or (re.search(r"v\d+\.hf\s*=\s*v\d+\.qf16", low)):
        result.append("scalar/vector move")
    if "jump" in low or "jumpr" in low:
        result.append("branch/control")
    return result or ["other"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disassembly", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    packet_rows = []
    histogram = Counter()
    for index, packet in enumerate(packets(symbol_lines(args.disassembly, args.symbol))):
        kinds = sorted({kind for instruction in packet for kind in categories(instruction)})
        histogram[len(packet)] += 1
        packet_rows.append({
            "packet": f"P{index}",
            "instruction_count": len(packet),
            "instructions": packet,
            "categories": kinds,
            "has_hvx_multiply": any("vmpy" in instruction.lower() for instruction in packet),
            "has_any_multiply": any("vmpy" in instruction.lower() or "mpyi" in instruction.lower()
                                    for instruction in packet),
            "has_hvx_permute": any(any(op in instruction.lower() for op in
                                       ("vdeal", "vshuff", "valign", "vlalign", "vdelta", "vrdelta"))
                                   for instruction in packet),
            "has_splat": any("vsplat" in instruction.lower() for instruction in packet),
        })
    document = {
        "schema_version": 1,
        "source": str(args.disassembly),
        "symbol": args.symbol,
        "total_instructions": sum(row["instruction_count"] for row in packet_rows),
        "total_packets": len(packet_rows),
        "instructions_per_packet": sum(row["instruction_count"] for row in packet_rows) / len(packet_rows),
        "hvx_multiply_packets": sum(row["has_hvx_multiply"] for row in packet_rows),
        "any_multiply_packets": sum(row["has_any_multiply"] for row in packet_rows),
        "hvx_permute_packets": sum(row["has_hvx_permute"] for row in packet_rows),
        "splat_packets": sum(row["has_splat"] for row in packet_rows),
        "single_instruction_packets": histogram[1],
        "multi_instruction_packets": len(packet_rows) - histogram[1],
        "packet_fill_histogram": dict(sorted(histogram.items())),
        "packets": packet_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: value for key, value in document.items() if key != "packets"}, indent=2))


if __name__ == "__main__":
    main()
