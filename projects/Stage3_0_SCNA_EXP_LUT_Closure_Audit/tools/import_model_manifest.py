#!/usr/bin/env python3
"""Import the immutable Stage 2.9 model pair without copying model data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT.parent / "Archived/Stage2_9_SCNA_EXP_LUT_Value_Audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "generated/model_pair_import.json")
    args = parser.parse_args()
    source_path = ARCHIVE / "generated/models/qwen2.5-1.5b-stage291/model_pair_manifest.json"
    source = json.loads(source_path.read_text())
    model_root = source_path.parent
    imported: dict[str, object] = {}
    for key, record in source["models"].items():
        path = model_root / Path(record["path"]).name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise SystemExit(f"missing or size-mismatched imported model: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise SystemExit(f"hash mismatch for imported model: {path}")
        imported[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    validation = source["tensor_validation"]
    validation_path = model_root / "tensor_validation.jsonl"
    if validation.get("pass") is not True or not validation_path.is_file():
        raise SystemExit("Stage 2.9 tensor validation is not a passing immutable input")
    if sha256(validation_path) != validation["validation_jsonl"]["sha256"]:
        raise SystemExit("tensor-validation hash mismatch")
    output = {
        "schema_version": 1,
        "copy_policy": "reference-only; model bytes are not copied into Stage 3.0",
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256(source_path),
        "models": imported,
        "device": source["device"],
        "tensor_validation": {
            **validation,
            "pass": True,
            "tensor_count": int(validation["tensor_count"]),
            "validation_jsonl": {
                **validation["validation_jsonl"],
                "path": str(validation_path),
            },
        },
        "lineage": source["lineage"],
        "source": source["source"],
    }
    if output["tensor_validation"]["tensor_count"] != 338:
        raise SystemExit("unexpected tensor validation cardinality")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
