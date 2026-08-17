#!/usr/bin/env python3
"""Record source/tool provenance without copying Qualcomm binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(*command: str) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    run_dir = args.run_dir.resolve()
    sdk = Path(os.environ.get("HEXAGON_SDK_ROOT", "/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0"))
    tools = Path(os.environ.get("HEXAGON_SIM_CORE", sdk / "tools/HEXAGON_Tools/19.0.07"))
    files = []
    source_suffixes = {".c", ".cc", ".cpp", ".h", ".hpp", ".idl", ".cmake", ".py", ".sh", ".json", ".md"}
    for base in (project / "src", project / "scripts", project / "tools", project / "docs", project / ".vscode"):
        for path in sorted(
            p for p in base.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
            and not any(part.startswith("hexagon_") for part in p.relative_to(project).parts)
            and (p.suffix in source_suffixes or p.name == "CMakeLists.txt")
        ):
            files.append({"path": str(path.relative_to(project)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for path in (
        project / "README.md", project / "requirements.txt", project / "experiment_spec.source.json",
        project / "experiment_spec.simulator.json",
    ):
        files.append({"path": str(path.relative_to(project)), "sha256": sha256(path), "bytes": path.stat().st_size})
    artifacts = []
    for path in sorted((project / "artifacts").rglob("*.so")):
        artifacts.append({"path": str(path.relative_to(project)), "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": output("git", "-C", str(project), "rev-parse", "HEAD"),
        "sdk_root": str(sdk),
        "tools_root": str(tools),
        "hexagon_clang_version": output(str(tools / "Tools/bin/hexagon-clang"), "--version").splitlines(),
        "simulator_model": os.environ.get("HEXAGON_SIM_MODEL", "v79na_1"),
        "files": files,
        "artifacts": artifacts,
        "proprietary_binaries_copied": False,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(run_dir / "manifest.json")


if __name__ == "__main__":
    main()
