#!/usr/bin/env python3
"""Hash every frozen input, tool source, artifact and formal result."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from model_lineage import file_record


PROJECT = Path(__file__).resolve().parents[1]


def referenced_local_files(value: object) -> list[Path]:
    """Collect existing local files named by nested model-manifest records."""
    found: list[Path] = []
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path.startswith("/"):
            candidate = Path(path)
            if candidate.is_file():
                found.append(candidate)
        for child in value.values():
            found.extend(referenced_local_files(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(referenced_local_files(child))
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--extra-results-dir", type=Path, action="append", default=[])
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite final manifest: {output}")
    model_manifest = json.loads(args.model_manifest.read_text())
    files: list[Path] = [args.spec.resolve(), (PROJECT / "model_repair_spec.json").resolve(),
                        (PROJECT / "audit_rerun_spec.json").resolve(), args.model_manifest.resolve(),
                        *referenced_local_files(model_manifest)]
    for root in (PROJECT / "tools", PROJECT / "scripts", PROJECT / "tests", PROJECT / "templates",
                 PROJECT / "requirements", PROJECT / "docs", PROJECT / "manifests",
                 PROJECT / "src/htp-ops-lib-main"):
        files.extend(path for path in root.rglob("*") if path.is_file() and
                     not any(part.startswith(("android_", "hexagon_", "__pycache__")) for part in path.parts))
    for root in (PROJECT / "artifacts", args.results_dir.resolve(),
                 *(path.resolve() for path in args.extra_results_dir)):
        files.extend(path for path in root.rglob("*") if path.is_file() and path.resolve() != output)
    if args.reports_dir:
        files.extend(path for path in args.reports_dir.resolve().rglob("*") if path.is_file())
    dataset_path = Path(json.loads(args.spec.read_text())["model"]["dataset"])
    if dataset_path.is_file():
        files.append(dataset_path)
    manifest = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
                "files": [file_record(path) for path in sorted(set(files))],
                "device_models": model_manifest.get("models", {})}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
