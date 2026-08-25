#!/usr/bin/env python3
"""Compare Stage 2.5 v79 disassembly with deterministic static gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def symbol_lines(path: Path, token: str) -> list[str]:
    current = ""
    selected: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if match:
            current = match.group(1)
        elif token in current and re.match(r"^\s*[0-9a-f]+:", line):
            selected.append(line)
    return selected


def metrics(path: Path, token: str) -> dict:
    lines = symbol_lines(path, token)
    low = [line.lower() for line in lines]
    frames = [int(value, 0) for line in low
              for value in re.findall(r"allocframe\(#(0x[0-9a-f]+|\d+)\)", line)]
    return {
        "instructions": len(lines),
        "packets": sum("{" in line for line in lines),
        "instructions_per_packet": len(lines) / max(1, sum("{" in line for line in lines)),
        "stack_frame_bytes": max(frames, default=0),
        "stack_references": sum(bool(re.search(r"mem[bdhw]\(r(?:29|30)(?:[+#)]|$)", line)) for line in low),
        "scalar_divide": sum(any(op in line for op in ("sdiv", "udiv", "dfixup")) for line in low),
        "scalar_modulo_helpers": sum("mod" in line and "call" in line for line in low),
        "divmod_helper_calls": sum("call" in line and re.search(r"<(?:__hexagon_)?[us]?(?:div|mod)", line) is not None
                                   for line in low),
        "vector_loads": sum("mem" in line and "=" in line for line in low),
        "unaligned_vector_loads": sum("vmemu" in line for line in low),
        "calls": sum("call" in line for line in low),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--baseline", default="stage2_final")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = {}
    for directory in sorted(path for path in args.artifact_root.iterdir() if path.is_dir()):
        disassemblies = list(directory.glob("*.v79.disasm.txt"))
        if len(disassemblies) == 1:
            rows[directory.name] = {
                "core": metrics(disassemblies[0], "simple_flash_attn_f16_core"),
                "worker": metrics(disassemblies[0], "simple_flash_attn_worker"),
            }
    if args.baseline not in rows:
        raise SystemExit(f"missing baseline disassembly: {args.baseline}")
    baseline = rows[args.baseline]
    for label, row in rows.items():
        for section in ("core", "worker"):
            row[section]["delta_vs_stage2_final"] = {
                key: row[section][key] - baseline[section][key]
                for key in ("instructions", "packets", "stack_frame_bytes", "stack_references",
                            "scalar_divide", "scalar_modulo_helpers", "divmod_helper_calls",
                            "vector_loads", "unaligned_vector_loads")
            }
    payload = {"schema_version": 1, "baseline": args.baseline, "variants": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
