#!/usr/bin/env python3
"""Summarize SCNA KV-pipeline byte-equivalence and FP32 correctness gates."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PAIR_RE = re.compile(r"(\w+)=([^\s]+)")


def fields(line: str) -> dict[str, str]:
    return dict(PAIR_RE.findall(line))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pipeline_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    current: dict[str, str] = {}
    for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        data = fields(line)
        if "FIG8_ATTENTION_CONFIG" in line:
            current = data
        elif "FIG8_ATTENTION_PIPELINE_COMPARE" in line:
            pipeline_rows.append({
                "mode": data.get("mode", ""), "function": data.get("function", ""),
                "kernel": data.get("kernel", ""), "width": int(data.get("width", "0")),
                "mask_mode": current.get("mask_mode", ""), "kv_len": int(data.get("kv_len", "0")),
                "head_dim": int(data.get("head_dim", "0")),
                "byte_mismatches": int(data.get("byte_mismatches", "-1")),
                "rmse": float(data.get("rmse", "nan")), "max_abs_error": float(data.get("max_abs_error", "nan")),
                "candidate_nonfinite": int(data.get("candidate_nonfinite", "-1")),
                "alternate_nonfinite": int(data.get("alternate_nonfinite", "-1")),
                "gate": data.get("gate", "missing"), "ret": int(data.get("ret", "-1")),
            })
        elif "FIG8_ATTENTION_COMPARE" in line and data.get("reference_mode") == "host-fp32":
            reference_rows.append({
                "mode": data.get("candidate_mode", ""), "function": data.get("scna_function", ""),
                "kernel": data.get("scna_kernel", ""), "pipeline": data.get("scna_pipeline", ""),
                "width": int(data.get("scna_width", "0")), "mask_mode": current.get("mask_mode", ""),
                "kv_len": int(data.get("kv_len", "0")), "head_dim": int(data.get("head_dim", "0")),
                "rmse": float(data.get("rmse", "nan")),
                "max_abs_error": float(data.get("max_abs_error", "nan")),
                "candidate_nonfinite": int(data.get("candidate_nonfinite", "-1")),
                "reference_nonfinite": int(data.get("reference_nonfinite", "-1")),
                "ret": int(data.get("ret", "-1")),
            })
        elif "FIG8_NUMERIC" in line and data.get("kv_head") == "0":
            kv_len = int(current.get("kv_len", "0"))
            mask_mode = current.get("mask_mode", "")
            requires_zero = kv_len % 64 != 0 or mask_mode in ("padding", "causal")
            last_bits = int(data.get("p0_last_bits", "-1"), 0)
            probe_rows.append({
                "mode": current.get("mode", ""), "function": current.get("scna_function", ""),
                "kernel": current.get("scna_kernel", ""), "width": int(current.get("scna_width", "0")),
                "mask_mode": mask_mode, "kv_len": kv_len, "head_dim": int(current.get("head_dim", "0")),
                "p0_first_bits": data.get("p0_first_bits", ""), "p0_last_bits": data.get("p0_last_bits", ""),
                "requires_zero": int(requires_zero), "gate": "pass" if not requires_zero or last_bits == 0 else "fail",
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "pipeline_compare.csv", pipeline_rows)
    write_csv(args.out_dir / "fp32_reference.csv", reference_rows)
    write_csv(args.out_dir / "mask_tail_probes.csv", probe_rows)

    failures = sum(row["gate"] != "pass" or row["ret"] != 0 for row in pipeline_rows)
    failures += sum(row["ret"] != 0 or row["candidate_nonfinite"] != 0 or row["reference_nonfinite"] != 0
                    for row in reference_rows)
    failures += sum(row["gate"] != "pass" for row in probe_rows)
    max_rmse = max((float(row["rmse"]) for row in reference_rows), default=0.0)
    max_abs = max((float(row["max_abs_error"]) for row in reference_rows), default=0.0)
    max_mismatches = max((int(row["byte_mismatches"]) for row in pipeline_rows), default=0)
    report = [
        "# V81 SCNA KV Pipeline Correctness", "",
        f"- Pipeline on/off comparisons: {len(pipeline_rows)}",
        f"- Host FP32 comparisons: {len(reference_rows)}",
        f"- Mask/tail probes: {len(probe_rows)}",
        f"- Maximum pipeline byte mismatches: {max_mismatches}",
        f"- Maximum FP32 RMSE: {max_rmse:.7g}",
        f"- Maximum FP32 absolute error: {max_abs:.7g}",
        f"- Gate failures: {failures}", "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise SystemExit(f"pipeline correctness failures: {failures}")


if __name__ == "__main__":
    main()
