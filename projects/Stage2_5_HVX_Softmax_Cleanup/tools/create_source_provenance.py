#!/usr/bin/env python3
"""Verify and record the Stage 2 allowlisted import without altering Stage 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_PROJECT = PROJECT.parent / "Stage2_HVX_Softmax_Dataflow"
OUTPUT = PROJECT / "source_import_manifest.json"

ALLOWLIST = (
    "src/htp-ops-lib-main",
    "scripts/build.sh",
    "scripts/build_baselines.sh",
    "scripts/build_candidate.sh",
    "scripts/deploy_and_smoke.sh",
    "scripts/run_baseline_experiment.sh",
    "scripts/run_candidate_experiment.sh",
    "scripts/run_component_profile.sh",
    "scripts/use_hexagon_sdk_6_6.sh",
    "tools/analyze_baseline.py",
    "tools/analyze_candidate.py",
    "tools/create_artifact_manifest.py",
    "tools/extract_packet_map.py",
)

EXCLUDED_PARTS = {
    "artifacts", "results", "reports", "build", "__pycache__", ".venv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_for(entry: str) -> list[Path]:
    base = SOURCE_PROJECT / entry
    if base.is_file():
        return [base]
    return sorted(
        path for path in base.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS or part.startswith("hexagon_") or part.startswith("android_")
                    for part in path.relative_to(SOURCE_PROJECT).parts)
    )


def main() -> None:
    records = []
    for entry in ALLOWLIST:
        for source in files_for(entry):
            relative = source.relative_to(SOURCE_PROJECT)
            imported = PROJECT / relative
            if not imported.is_file():
                raise SystemExit(f"missing imported file: {relative}")
            source_hash = sha256(source)
            imported_hash = sha256(imported)
            if source_hash != imported_hash:
                raise SystemExit(f"import changed before provenance capture: {relative}")
            records.append({
                "path": relative.as_posix(),
                "source_sha256": source_hash,
                "imported_sha256": imported_hash,
                "bytes": source.stat().st_size,
            })

    payload = {
        "schema_version": 1,
        "source_project": str(SOURCE_PROJECT),
        "destination_project": str(PROJECT),
        "excluded": sorted(EXCLUDED_PARTS | {"hexagon_*", "android_*"}),
        "files": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"verified {len(records)} imported files -> {OUTPUT}")


if __name__ == "__main__":
    main()

