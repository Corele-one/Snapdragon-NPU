#!/usr/bin/env python3
"""Create deterministic source/artifact hashes and verify the reference snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GENERATED_PREFIXES = ("android_", "hexagon_")


def source_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(GENERATED_PREFIXES) for part in path.relative_to(root).parts)
        and "__pycache__" not in path.parts
    )


def tree_manifest(root: Path) -> dict:
    entries = []
    aggregate = hashlib.sha256()
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": relative, "sha256": digest, "size": path.stat().st_size})
        aggregate.update(relative.encode() + b"\0" + digest.encode() + b"\n")
    return {"file_count": len(entries), "tree_sha256": aggregate.hexdigest(), "files": entries}


def artifact_manifest(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "sha256.txt"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = tree_manifest(args.project / "reference/flashattention/htp-ops-lib-main")
    authority = tree_manifest(args.authority)
    optimized = tree_manifest(args.project / "src/htp-ops-lib-main")
    reference_matches = reference == authority
    result = {
        "schema_version": 1,
        "target": "Hexagon v79 only",
        "build_parameters": {
            "android": "build_cmake android",
            "hexagon": "build_cmake hexagon DSP_ARCH=v79 FIGURE8_ENABLE_PROFILE_TIMERS=ON FIGURE8_ENABLE_LUT_EXP=OFF",
        },
        "reference_matches_flashattention_authority": reference_matches,
        "reference_source": reference,
        "optimized_source": optimized,
        "artifacts": artifact_manifest(args.project / "artifacts"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "reference_matches_flashattention_authority": reference_matches,
        "reference_tree_sha256": reference["tree_sha256"],
        "optimized_tree_sha256": optimized["tree_sha256"],
        "artifact_count": len(result["artifacts"]),
    }, indent=2))
    if not reference_matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
