#!/usr/bin/env python3
"""Freeze the evidence that separates HMX matmul from frozen Attention failure."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path

from model_lineage import file_record, stable_case_id


KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
PPL = re.compile(r"Final estimate: PPL = ([0-9.eE+-]+)")
MODEL_MATMUL = "MODEL_MATMUL_AUDIT_JSON "


def convert(value: str):
    try:
        return int(value, 0)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def kv_record(line: str) -> dict[str, object]:
    return {key: convert(value) for key, value in KEY_VALUE.findall(line)}


def parse_ppl(path: Path) -> float:
    matches = PPL.findall(path.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"PPL marker missing: {path}")
    return float(matches[-1])


def parse_model_matmul(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if MODEL_MATMUL not in line:
            continue
        record = json.loads(line.split(MODEL_MATMUL, 1)[1])
        record["line_number"] = line_number
        records.append(record)
    if not records:
        raise RuntimeError(f"model matmul markers missing: {path}")
    return records


def parse_attention(paths: list[Path]) -> list[dict[str, object]]:
    records = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if "FIG8_ATTENTION_COMPARE " not in line:
                continue
            record = kv_record(line)
            record.update({"record_type": "attention_correctness_diagnostic",
                           "source_log": str(path.resolve()), "line_number": line_number})
            records.append(record)
    if not records:
        raise RuntimeError("attention comparison markers missing")
    return records


def fp16_bits(value: object) -> float:
    return struct.unpack("<e", struct.pack("<H", int(value) & 0xFFFF))[0]


def fp32_bits(value: object) -> float:
    return struct.unpack("<f", struct.pack("<I", int(value) & 0xFFFFFFFF))[0]


def parse_numeric(path: Path) -> dict[str, object]:
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if "FIG8_NUMERIC " not in line:
            continue
        record = kv_record(line)
        if int(record.get("score_count", 0)) <= 0:
            continue
        record.update({"record_type": "attention_rowsum_diagnostic",
                       "source_log": str(path.resolve()), "line_number": line_number,
                       "rowsum_fp16": fp16_bits(record["rowsum0_bits"]),
                       "expected_probability_sum_fp32": fp32_bits(record["p_expected_sum_bits"]),
                       "masked_and_tail_zero": (int(record.get("masked_p_nonzero", -1)) == 0 and
                                                  int(record.get("tail_p_nonzero", -1)) == 0)})
        return record
    raise RuntimeError(f"usable FIG8_NUMERIC marker missing: {path}")


def append_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            identity = {key: record.get(key) for key in
                        ("record_type", "candidate_mode", "mask_mode", "qo_len", "kv_len",
                         "tensor", "divergence_ordinal", "execution") if key in record}
            record.setdefault("case_id", stable_case_id(identity))
            record.setdefault("attempt", 1)
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-flash-log", type=Path, required=True)
    parser.add_argument("--no-flash-log", type=Path, required=True)
    parser.add_argument("--model-matmul-log", type=Path, required=True)
    parser.add_argument("--attention-log", type=Path, action="append", required=True)
    parser.add_argument("--numeric-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic evidence: {output}")
    raw = output / "raw"
    logs = output / "logs"
    raw.mkdir(parents=True)
    logs.mkdir()

    sources = [args.custom_flash_log, args.no_flash_log, args.model_matmul_log,
               *args.attention_log, args.numeric_log]
    source_records = []
    for index, source in enumerate(sources):
        source = source.resolve()
        target = logs / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        source_records.append({"record_type": "diagnostic_source", "original": file_record(source),
                               "frozen_copy": file_record(target)})

    custom_ppl = parse_ppl(args.custom_flash_log)
    no_flash_ppl = parse_ppl(args.no_flash_log)
    model_records = parse_model_matmul(args.model_matmul_log)
    first_failed = next((record for record in model_records if not record.get("pass")), None)
    if first_failed is None:
        raise RuntimeError("model matmul audit contains no divergence")
    earlier = [record for record in model_records if record["line_number"] < first_failed["line_number"]]
    attention = parse_attention(args.attention_log)
    numeric = parse_numeric(args.numeric_log)

    model_summary = {
        "record_type": "model_path_isolation",
        "execution": "custom_flash_on_vs_off",
        "custom_flash_ppl": custom_ppl,
        "no_custom_flash_ppl": no_flash_ppl,
        "first_divergent_tensor": first_failed["tensor"],
        "first_divergent_output": first_failed.get("output"),
        "first_divergent_relative_l2": first_failed.get("relative_l2"),
        "preceding_model_matmuls": len(earlier),
        "preceding_model_matmuls_all_pass": bool(earlier) and all(r.get("pass") for r in earlier),
    }
    common_modes = {str(record.get("candidate_mode")) for record in attention
                    if int(record.get("qo_len", -1)) == 32 and int(record.get("kv_len", -1)) == 32 and
                    str(record.get("mask_mode")) == "causal" and int(record.get("pass", 1)) == 0}
    summary = {
        "schema_version": 1,
        "verdict": "AUDIT_INVALID",
        "failed_boundary": "frozen_flash_attention_state_update",
        "model_path_isolation": model_summary,
        "short_kv_failed_modes": sorted(common_modes),
        "short_kv_failure_is_common_to_evaluators": {"baseline", "lut-exp", "scna-fp16"}.issubset(common_modes),
        "rowsum": {
            "stored_fp16": numeric["rowsum_fp16"],
            "scalar_expected_fp32": numeric["expected_probability_sum_fp32"],
            "masked_and_tail_zero": numeric["masked_and_tail_zero"],
        },
        "interpretation": (
            "The repaired HMX matmul path is not the first divergent stage. The frozen fused "
            "Attention row-sum/state-update path fails at short KV for baseline, EXP-LUT and SCNA. "
            "Evaluator comparison is therefore not identifiable without an out-of-scope change."
        ),
        "sources": [record["frozen_copy"] for record in source_records],
    }
    records = [*source_records, model_summary, *attention, numeric,
               {"record_type": "model_chain_diagnostic_summary", **summary}]
    append_jsonl(raw / "model_chain_diagnostic.jsonl", records)
    (output / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
