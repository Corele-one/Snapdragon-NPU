#!/usr/bin/env python3
"""Validate the fixed 180-case attention matrix and q32 determinism gate."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def fields(line: str) -> dict[str, str]:
    return dict(KV_RE.findall(line))


def analyze(root: Path) -> dict:
    rows = []
    expected_names = {
        f"{mask}_q{q}_kv{kv}_d{dim}_seed{seed}.log"
        for mask in ("full", "causal", "padding")
        for q in (1, 4, 8, 16, 32)
        for kv in (4093, 4096)
        for dim in (64, 128)
        for seed in ("figure8_fixed", "20260810", "20260811")
    }
    actual = {path.name for path in (root / "raw").glob("*.log")}
    missing = sorted(expected_names - actual)
    extra = sorted(actual - expected_names)
    if missing or extra:
        raise ValueError(f"correctness file set mismatch: missing={missing}, extra={extra}")

    for path in sorted((root / "raw").glob("*.log")):
        compare = []
        host = []
        numeric = []
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_COMPARE " in line:
                compare.append(fields(line))
            elif "FIG8_ATTENTION_HOST_TIMING " in line and "phase=measure" in line:
                host.append(fields(line))
            elif "FIG8_NUMERIC " in line:
                numeric.append(fields(line))
        if len(compare) != 1 or len(host) != 1:
            raise ValueError(f"expected one compare and one measured host line in {path}")
        c, h = compare[0], host[0]
        rmse = float(c["rmse"])
        max_abs = float(c["max_abs_error"])
        nonfinite = int(c["candidate_nonfinite"]) + int(c["reference_nonfinite"])
        masked = sum(int(row.get("masked_p_nonzero", "0")) for row in numeric)
        tail = sum(int(row.get("tail_p_nonzero", "0")) for row in numeric)
        ok = (
            h.get("ret") == "0"
            and c.get("pass") == "1"
            and nonfinite == 0
            and masked == 0
            and tail == 0
            and rmse <= 0.002
            and max_abs <= 0.01
        )
        rows.append({
            "file": path.name,
            "rmse": rmse,
            "max_abs_error": max_abs,
            "nonfinite": nonfinite,
            "masked_p_nonzero": masked,
            "tail_p_nonzero": tail,
            "pass": ok,
        })

    checksum_rows = []
    determinism_files = sorted((root / "determinism").glob("q32_repeat*.log"))
    if len(determinism_files) != 10:
        raise ValueError(f"expected 10 determinism logs, found {len(determinism_files)}")
    for path in determinism_files:
        measured_host = []
        measured_checksum = []
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_HOST_TIMING " in line and "phase=measure" in line:
                measured_host.append(fields(line))
            elif "FIG8_ATTENTION_CHECKSUM " in line and "phase=measure" in line:
                measured_checksum.append(fields(line))
        if len(measured_host) != 1 or measured_host[0].get("ret") != "0" or len(measured_checksum) != 1:
            raise ValueError(f"invalid determinism log {path}")
        checksum_rows.append(measured_checksum[0]["checksum"])

    result = {
        "schema_version": 1,
        "cases": len(rows),
        "passed_cases": sum(row["pass"] for row in rows),
        "max_rmse": max(row["rmse"] for row in rows),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "nonfinite": sum(row["nonfinite"] for row in rows),
        "masked_p_nonzero": sum(row["masked_p_nonzero"] for row in rows),
        "tail_p_nonzero": sum(row["tail_p_nonzero"] for row in rows),
        "determinism_processes": len(checksum_rows),
        "determinism_unique_checksums": sorted(set(checksum_rows)),
        "pass": len(rows) == 180 and all(row["pass"] for row in rows) and len(set(checksum_rows)) == 1,
        "rows": rows,
    }
    if not result["pass"]:
        raise ValueError("correctness or determinism gate failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
