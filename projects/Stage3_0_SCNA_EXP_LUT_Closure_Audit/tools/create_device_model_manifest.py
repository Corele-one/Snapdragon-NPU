#!/usr/bin/env python3
"""Freeze the independently validated, device-derived Stage 2.9.1 model pair."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from model_lineage import build_id, file_record, git_commit, sha256


PROJECT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ("cpu_f16", "hmx_f16", "cpu_quant", "hmx_quant")


def jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_identity(path: Path) -> str:
    try:
        return build_id(path)
    except RuntimeError:
        # Some Android NDK links omit a GNU build-id.  A content-addressed
        # identity is stronger than leaving the executable unidentified.
        return "sha256:" + sha256(path)


def device_records(paths: dict[str, str]) -> dict[str, dict[str, object]]:
    quoted = " ".join(paths.values())
    hashes = subprocess.run(["adb", "shell", f"sha256sum {quoted}"], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    sizes = subprocess.run(["adb", "shell", "stat -c '%s %n' " + quoted], check=True, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    by_path_hash = {line.split()[1]: line.split()[0] for line in hashes.splitlines()}
    by_path_size = {line.split()[1]: int(line.split()[0]) for line in sizes.splitlines()}
    return {key: {"sha256": by_path_hash[path], "size": by_path_size[path]} for key, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--cpu-validation", type=Path, required=True)
    parser.add_argument("--hmx-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    base = json.loads(args.base_manifest.read_text())
    cpu_records = jsonl(args.cpu_validation)
    hmx_records = jsonl(args.hmx_validation)
    for branch, records in (("cpu", cpu_records), ("hmx", hmx_records)):
        summaries = [row for row in records if row.get("record_type") == "device_quant_lineage_summary"]
        tensors = [row for row in records if row.get("record_type") == "device_quant_tensor"]
        if len(summaries) != 1 or summaries[0].get("pass") is not True or len(tensors) != 338:
            raise RuntimeError(f"{branch} device quant lineage is incomplete or failed")
        if any(row.get("pass") is not True or row.get("branch") != branch for row in tensors):
            raise RuntimeError(f"{branch} contains a failed/mislabeled tensor")
    if {row["tensor"] for row in cpu_records if row.get("record_type") == "device_quant_tensor"} != {
            row["tensor"] for row in hmx_records if row.get("record_type") == "device_quant_tensor"}:
        raise RuntimeError("CPU/HMX tensor inventories differ")

    directory = str(base["device"]["directory"])
    paths = {
        "cpu_f16": f"{directory}/qwen2.5-1.5b-stage291-cpu-f16.gguf",
        "hmx_f16": f"{directory}/qwen2.5-1.5b-stage291-hmx-f16.gguf",
        "cpu_quant": f"{directory}/qwen2.5-1.5b-stage291-device-cpu-iq4nl-q8.gguf",
        "hmx_quant": f"{directory}/qwen2.5-1.5b-stage291-device-hmx-iq4nl-q8.gguf",
    }
    actual = device_records(paths)
    for key in ("cpu_f16", "hmx_f16"):
        if actual[key]["sha256"] != base["models"][key]["sha256"]:
            raise RuntimeError(f"device {key} no longer matches the validated host model")

    tool_dir = PROJECT / "artifacts/device_model_tools"
    llama_tree = Path("/home/corleone/code/Snapdragon-NPU/projects/flashattention/src/llama.cpp-npu-htp-backend")
    quantizer = tool_dir / "llama-quantize"
    models: dict[str, object] = {
        "cpu_f16": base["models"]["cpu_f16"],
        "hmx_f16": base["models"]["hmx_f16"],
        "cpu_quant": {**actual["cpu_quant"], "bytes": actual["cpu_quant"]["size"],
                      "path": None, "device_derived": True,
                      "source_f16_sha256": actual["cpu_f16"]["sha256"]},
        "hmx_quant": {**actual["hmx_quant"], "bytes": actual["hmx_quant"]["size"],
                      "path": None, "device_derived": True,
                      "source_f16_sha256": actual["hmx_f16"]["sha256"]},
    }
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "derivation": "on-device from bit-exact host-validated F16 pair",
        "base_manifest": file_record(args.base_manifest.resolve()),
        "repair_spec": base["repair_spec"],
        "source": base["source"],
        "toolchain": {
            **base["toolchain"],
            "device_llama_commit": git_commit(llama_tree),
            "device_quantizer": {**file_record(quantizer), "build_id": build_identity(quantizer)},
            "device_hmx_permuter": {**file_record(tool_dir / "gguf-hmx-permute"),
                                     "build_id": build_identity(tool_dir / "gguf-hmx-permute")},
            "device_quant_validator": {**file_record(tool_dir / "gguf-quant-lineage"),
                                        "build_id": build_identity(tool_dir / "gguf-quant-lineage")},
        },
        "commands": {
            "cpu_quantize": "REPACK_FOR_HVX unset; llama-quantize cpu_f16 cpu_quant 514 4",
            "hmx_quantize": "REPACK_FOR_HVX=1 llama-quantize hmx_f16 hmx_quant 514 4",
        },
        "environments": {"cpu_quantize": {"REPACK_FOR_HVX": "unset"},
                         "hmx_quantize": {"REPACK_FOR_HVX": "1"}},
        "lineage": {"requantization": False,
                    "cpu_quant_input_sha256": actual["cpu_f16"]["sha256"],
                    "hmx_quant_input_sha256": actual["hmx_f16"]["sha256"]},
        "models": models,
        "device": {"directory": directory, "models": paths},
        "tensor_validation": {
            "pass": True, "tensor_count": 338, "failed_count": 0,
            "cpu_jsonl": file_record(args.cpu_validation.resolve()),
            "hmx_jsonl": file_record(args.hmx_validation.resolve()),
            "f16_pair_validation": base["tensor_validation"],
        },
        "derivation_evidence": {
            "hmx_f16": file_record(PROJECT / "generated/device_hmx_f16_derivation.jsonl"),
            "cpu_quantize": file_record(PROJECT / "generated/device_cpu_quant_derivation.log"),
            "hmx_quantize": file_record(PROJECT / "generated/device_hmx_quant_derivation.log"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "cpu_quant_sha256": actual["cpu_quant"]["sha256"],
                      "hmx_quant_sha256": actual["hmx_quant"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
