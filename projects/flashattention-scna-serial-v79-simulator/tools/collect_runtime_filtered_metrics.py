#!/usr/bin/env python3
"""Validate relocated hot-function PC-filter captures; never promote empty captures."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

VARIANTS = ["stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
            "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized"]
PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")

def number(value): return int(value, 0)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); result = {}
    for variant in VARIANTS:
        log = args.run_dir / "raw/detailed" / f"{variant}.log"; metrics = args.run_dir / "metrics/detailed" / variant
        text = log.read_text(errors="replace") if log.exists() else ""
        lines = [line for line in text.splitlines() if line.startswith("RUNTIME_FILTER_RESULT")]
        meta = dict(PAIR.findall(lines[-1])) if lines else {}
        packet = metrics / "packet_analyze.txt"; ihist = metrics / "ihist.txt"; pmu = metrics / "pmu.txt"
        packet_text = packet.read_text(errors="replace") if packet.exists() else ""
        addresses = [int(x, 16) for x in re.findall(r"(?:0x)?([0-9a-fA-F]{8,})", packet_text)]
        start = number(meta["pc_start"]) if "pc_start" in meta else 0; end = number(meta["pc_end"]) if "pc_end" in meta else -1
        in_range = [address for address in addresses if start <= address <= end]
        ihist_nonzero = any(int(x) > 0 for x in re.findall(r"\b(\d+)\b", ihist.read_text(errors="replace") if ihist.exists() else ""))
        validated = meta.get("status") == "PASS" and bool(in_range) and ihist_nonzero
        result[variant] = {**meta, "validated": validated, "packet_addresses_in_range": len(in_range),
                           "ihist_nonzero": ihist_nonzero, "pmu_present": pmu.exists() and pmu.stat().st_size > 0,
                           "authority": "dynamic" if validated else "static_disassembly",
                           "note": "Filtered dynamic evidence requires both in-range packet PCs and non-zero ihist samples."}
    out = args.run_dir / "metrics/detailed/runtime_filtered_metrics.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(out)

if __name__ == "__main__": main()
