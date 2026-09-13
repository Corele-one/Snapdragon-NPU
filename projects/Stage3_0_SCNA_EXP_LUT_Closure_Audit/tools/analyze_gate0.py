#!/usr/bin/env python3
"""Validate the Stage 3.0 correctness gate and emit a hard PASS/AUDIT_INVALID verdict."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = json.loads((PROJECT / "experiment_spec.json").read_text())


def fp16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def float32_bits(bits: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0]


def load(path: Path) -> list[dict[str, object]]:
    values = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        value = json.loads(line)
        for item in value.values():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"non-finite JSON number at {path}:{number}")
        values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight_path = args.results_dir / "gate0_preflight_verdict.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text())
        if preflight.get("verdict") == "AUDIT_INVALID":
            verdict = {
                **preflight,
                "reasons": [preflight.get("reason", "frozen reproducer failed")],
                "matrix_records": 0,
                "matrix_unique_cases": 0,
                "quality_failure_count": sum(int(value.get("pass", 0)) != 1
                                             for value in preflight.get("comparisons", [])),
                "max_rowsum_ulp": preflight.get("max_rowsum_ulp"),
                "worst_failures": [value for value in preflight.get("comparisons", [])
                                   if int(value.get("pass", 0)) != 1],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
            rows = "\n".join(
                f"| {value.get('mode')} | {value.get('rmse')} | {value.get('max_abs_error')} | {value.get('pass')} |"
                for value in preflight.get("comparisons", []))
            args.output.with_suffix(".md").write_text(
                "# Stage 3.0 Gate 0\n\n## 裁决\n\n`AUDIT_INVALID`\n\n"
                "冻结复现点已经触发预注册 fail-fast；2160-case 矩阵及所有后续轨均未执行。\n\n"
                "| mode | RMSE | max-abs | pass |\n|---|---:|---:|---:|\n"
                f"{rows}\n")
            print(json.dumps(verdict, indent=2, sort_keys=True))
            return 3
    raw = args.results_dir / "raw/gate0.jsonl"
    values = load(raw)
    cfg = SPEC["gate0"]
    gates = cfg["gates"]
    reasons: list[str] = []
    matrix = [v for v in values if v.get("record_type") == "attention_correctness" and v.get("kind") == "matrix"]
    identities = {(v.get("mode"), v.get("mask"), v.get("q"), v.get("kv"), v.get("head_dim"), v.get("seed")) for v in matrix}
    if len(matrix) != cfg["expected_cases"] or len(identities) != cfg["expected_cases"]:
        reasons.append(f"Gate 0 matrix incomplete or duplicated: records={len(matrix)}, unique={len(identities)}, expected={cfg['expected_cases']}")
    quality_failures = [v for v in matrix if float(v.get("rmse", math.inf)) > gates["rmse_max"] or
                        float(v.get("max_abs_error", math.inf)) > gates["max_abs_max"] or
                        int(v.get("candidate_nonfinite", 1)) > gates["nonfinite_max"] or
                        int(v.get("reference_nonfinite", 1)) > gates["nonfinite_max"]]
    if quality_failures:
        reasons.append(f"{len(quality_failures)} matrix cases exceed RMSE/max-abs/nonfinite limits")
    sentinels = [v for v in values if v.get("record_type") == "attention_correctness" and v.get("kind") == "long-sentinel"]
    expected_sentinels = len(cfg["long_context_sentinels"]["masks"]) * len(cfg["long_context_sentinels"]["q"]) * len(cfg["modes"])
    if len(sentinels) != expected_sentinels or any(int(v.get("pass", 0)) != 1 for v in sentinels):
        reasons.append(f"long-context sentinel failed or incomplete: {len(sentinels)}/{expected_sentinels}")
    numeric = [v for v in values if v.get("record_type") == "attention_numeric" and v.get("kind") in {"matrix", "long-sentinel"}]
    if not numeric or any(int(v.get("masked_p_nonzero", 1)) != 0 or int(v.get("tail_p_nonzero", 1)) != 0 for v in numeric):
        reasons.append("mask/tail zero invariant failed or is missing")
    rowsums = [v for v in values if v.get("record_type") == "attention_numeric_block" and
               v.get("kind") in {"matrix", "long-sentinel"} and int(str(v.get("p_scalar_sum_bits", "0x0")), 16) != 0]
    ulps = []
    for value in rowsums:
        expected = fp16_bits(float32_bits(str(value["p_scalar_sum_bits"])))
        actual = int(str(value["rowsum0_bits"]), 16)
        ulps.append(abs(actual - expected))
    if not ulps or max(ulps) > gates["rowsum_fp16_ulp_max"]:
        reasons.append(f"rowsum ULP invariant failed or is missing: max={max(ulps) if ulps else 'missing'}")
    checksum_groups: dict[tuple, set[str]] = defaultdict(set)
    for value in values:
        if value.get("record_type") == "attention_checksum" and value.get("kind") == "determinism" and value.get("phase") == "measure":
            key = (value.get("mode"), value.get("mask"), value.get("q"), value.get("kv"), value.get("head_dim"), value.get("seed"))
            checksum_groups[key].add(str(value.get("checksum")))
    expected_groups = len(cfg["modes"]) * len(cfg["determinism"]["cases"])
    if len(checksum_groups) != expected_groups or any(len(group) != 1 for group in checksum_groups.values()):
        reasons.append(f"determinism failed or incomplete: {len(checksum_groups)}/{expected_groups} groups")
    worst = sorted(quality_failures, key=lambda v: (float(v.get("rmse", 0)), float(v.get("max_abs_error", 0))), reverse=True)[:20]
    verdict = {
        "schema_version": 1,
        "verdict": "AUDIT_INVALID" if reasons else "PASS",
        "reasons": reasons,
        "matrix_records": len(matrix),
        "matrix_unique_cases": len(identities),
        "quality_failure_count": len(quality_failures),
        "max_rowsum_ulp": max(ulps) if ulps else None,
        "worst_failures": [{key: value.get(key) for key in ("mode", "mask", "q", "kv", "head_dim", "seed", "rmse", "max_abs_error")} for value in worst],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    report = args.output.with_suffix(".md")
    details = "\n".join(f"- {reason}" for reason in reasons) or "- 所有预注册 Gate 0 条件通过。"
    report.write_text(f"# Stage 3.0 Gate 0\n\n## 裁决\n\n`{verdict['verdict']}`\n\n## 检查\n\n{details}\n\n"
                      f"- matrix: {len(matrix)}/{cfg['expected_cases']}\n- quality failures: {len(quality_failures)}\n"
                      f"- max rowsum ULP: {verdict['max_rowsum_ulp']}\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if not reasons else 3


if __name__ == "__main__":
    raise SystemExit(main())
