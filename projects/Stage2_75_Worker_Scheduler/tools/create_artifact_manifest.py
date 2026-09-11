#!/usr/bin/env python3
"""Create a validated, deterministic manifest for SCNA DSP artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_kv(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def build_manifest(artifact_root: Path) -> dict:
    artifacts = []
    for directory in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        binary = directory / "libhtp_ops_skel.so"
        build_id_path = directory / "build_id.txt"
        flags_path = directory / "compile_flags.txt"
        if not (binary.is_file() and build_id_path.is_file() and flags_path.is_file()):
            continue
        build = parse_kv(build_id_path)
        flags = flags_path.read_text(errors="replace").strip()
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if "-mv79" not in flags:
            raise ValueError(f"missing -mv79 in {flags_path}")
        artifacts.append({
            "kernel_impl": build.get("kernel_impl", directory.name),
            "kernel_impl_id": int(build["kernel_impl_id"]),
            "runtime_variant": build.get("runtime_variant"),
            "sha256": digest,
            "size_bytes": binary.stat().st_size,
            "binary": str(binary),
            "compile_flags_file": str(flags_path),
            "compile_flags": flags,
        })
    if not artifacts:
        raise ValueError(f"no complete artifacts under {artifact_root}")
    if len({row["sha256"] for row in artifacts}) != len(artifacts):
        raise ValueError("DSP artifacts do not have unique SHA256 values")
    return {"schema_version": 3, "architecture": "v79", "artifacts": artifacts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = build_manifest(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
