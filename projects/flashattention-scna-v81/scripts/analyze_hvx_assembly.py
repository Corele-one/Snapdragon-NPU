#!/usr/bin/env python3
"""Summarize packet and instruction mix for the compiled SCNA HVX kernels."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


SYMBOL_RE = re.compile(
    r"^hvx_scna_(?P<family>tree|exp2)(?P<pair>_pair)?(?P<int8>_int8)?_d(?P<width>8|16|32)_vhf$"
)
INSTRUCTION_RE = re.compile(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+){4}[0-9a-f]{8}\s+(?P<asm>.*)$")
CATEGORIES = {
    "vector_load": re.compile(r"\b(?:v\d+(?::\d+)?\s*=\s*)?vmemu?\b"),
    "vector_store": re.compile(r"\bvmem[^=]*="),
    "multiply": re.compile(r"\bvmpy(?:acc)?\b"),
    "lookup": re.compile(r"\bvlut16(?:or)?\b"),
    "permute_shift": re.compile(r"\bv(?:shuff|deal|pack|unpack|l?align|asl|lsr|asr|ror)\b"),
    "compare_mux_relu": re.compile(r"\bv(?:cmp|mux|max)\b"),
    "vector_arithmetic": re.compile(r"\bv(?:add|sub|or|and|xor)\b"),
    "branch": re.compile(r"\b(?:jump|jumpr|call)\b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tool-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout


def read_symbols(nm: Path, binary: Path) -> dict[str, int]:
    output = run([str(nm), "--print-size", "--size-sort", str(binary)])
    symbols: dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        match = SYMBOL_RE.fullmatch(parts[3])
        if match:
            symbols[parts[3]] = int(parts[1], 16)
    return symbols


def analyze_symbol(objdump: Path, binary: Path, symbol: str, code_bytes: int) -> tuple[dict[str, object], str]:
    output = run([str(objdump), f"--disassemble-symbols={symbol}", str(binary)])
    match = SYMBOL_RE.fullmatch(symbol)
    assert match is not None
    instructions: list[str] = []
    packet_lengths: list[int] = []
    packet_length = 0
    for line in output.splitlines():
        instruction = INSTRUCTION_RE.match(line)
        if instruction is None:
            continue
        asm = instruction.group("asm").lower()
        instructions.append(asm)
        if "{" in asm:
            packet_length = 0
        packet_length += 1
        if "}" in asm:
            packet_lengths.append(packet_length)
            packet_length = 0
    if packet_length:
        packet_lengths.append(packet_length)

    row: dict[str, object] = {
        "symbol": symbol,
        "kernel": "tree" if match.group("family") == "tree" else "direct",
        "precision": "int8" if match.group("int8") else "fp16",
        "vector_form": "pair" if match.group("pair") else "single",
        "width": int(match.group("width")),
        "code_bytes": code_bytes,
        "instructions": len(instructions),
        "packets": len(packet_lengths),
        "instructions_per_packet": len(instructions) / len(packet_lengths) if packet_lengths else 0.0,
        "max_packet_instructions": max(packet_lengths, default=0),
    }
    for name, pattern in CATEGORIES.items():
        row[name] = sum(bool(pattern.search(asm)) for asm in instructions)
    clean_output = "\n".join(line.rstrip() for line in output.splitlines())
    return row, clean_output


def ratio(numerator: float, denominator: float) -> str:
    return f"{numerator / denominator:.2f}x" if denominator else "n/a"


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    paired = [row for row in rows if row["vector_form"] == "pair"]
    by_key = {(row["precision"], row["width"], row["kernel"]): row for row in paired}
    report = [
        "# V81 SCNA HVX Static Assembly Check",
        "",
        "Counts come from the final packetized v81 ELF. `vlut16` is used only for branchless register-indexed tree traversal; it is not the LUT exponential backend.",
        "",
        "| Precision | Width | Direct bytes | Tree bytes | Size reduction | Direct inst. | Tree inst. | Inst. reduction | Direct mul | Tree mul | Tree lookup | Tree inst./packet |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for precision in ("fp16", "int8"):
        for width in (8, 16, 32):
            direct = by_key[(precision, width, "direct")]
            tree = by_key[(precision, width, "tree")]
            report.append(
                "| {precision} | d{width} | {direct_bytes} | {tree_bytes} | {size_ratio} | "
                "{direct_inst} | {tree_inst} | {inst_ratio} | {direct_mul} | {tree_mul} | "
                "{tree_lookup} | {packet_fill:.2f} |".format(
                    precision=precision.upper(), width=width,
                    direct_bytes=direct["code_bytes"], tree_bytes=tree["code_bytes"],
                    size_ratio=ratio(float(direct["code_bytes"]), float(tree["code_bytes"])),
                    direct_inst=direct["instructions"], tree_inst=tree["instructions"],
                    inst_ratio=ratio(float(direct["instructions"]), float(tree["instructions"])),
                    direct_mul=direct["multiply"], tree_mul=tree["multiply"],
                    tree_lookup=tree["lookup"], packet_fill=float(tree["instructions_per_packet"]),
                )
            )
    report += [
        "",
        "The table is a static resource-pressure check, not a cycle model. Runtime qtimer measurements remain authoritative for latency because cache state, call overhead, and instruction issue constraints are device-dependent.",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    nm = args.tool_dir / "hexagon-nm"
    objdump = args.tool_dir / "hexagon-llvm-objdump"
    symbols = read_symbols(nm, args.binary)
    if len(symbols) != 24:
        raise SystemExit(f"expected 24 SCNA kernels, found {len(symbols)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    disassembly: list[str] = []
    for symbol, code_bytes in sorted(symbols.items()):
        row, output = analyze_symbol(objdump, args.binary, symbol, code_bytes)
        rows.append(row)
        disassembly.append(output.rstrip())

    rows.sort(key=lambda row: (str(row["precision"]), str(row["vector_form"]), int(row["width"]), str(row["kernel"])))
    with (args.out_dir / "instruction_mix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "scna_kernels.s").write_text("\n\n".join(disassembly) + "\n", encoding="utf-8")
    write_report(rows, args.out_dir / "summary.md")


if __name__ == "__main__":
    main()
