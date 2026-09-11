#!/usr/bin/env python3
"""Create deterministic source/artifact provenance for the Stage 2.9 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_build_id(path: Path) -> str:
    proc = subprocess.run(["readelf", "-n", str(path)], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    match = re.search(r"Build ID:\s*([0-9a-f]+)", proc.stdout)
    return match.group(1) if match else f"sha256:{sha256(path)}"


def tree_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not any(
        part.startswith(("android_", "hexagon_")) for part in p.relative_to(root).parts
    )):
        rel = path.relative_to(root).as_posix().encode()
        digest.update(len(rel).to_bytes(4, "little"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256(path)))
        count += 1
    return digest.hexdigest(), count


def selected_digest(project: Path, entries: tuple[str, ...]) -> tuple[str, int]:
    """Hash only the audit harness, excluding generated evidence and binaries."""
    digest = hashlib.sha256()
    paths: list[Path] = []
    for entry in entries:
        path = project / entry
        if path.is_dir():
            paths.extend(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            paths.append(path)
    count = 0
    for path in sorted(set(paths)):
        rel = path.relative_to(project).as_posix().encode()
        digest.update(len(rel).to_bytes(4, "little"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256(path)))
        count += 1
    return digest.hexdigest(), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    source = project / "src/htp-ops-lib-main"
    source_hash, source_files = tree_digest(source)
    baseline = project.parent / "Stage2_75_Worker_Scheduler/artifacts/final_stage3_baseline/libhtp_ops_skel.so"
    artifacts: dict[str, dict[str, object]] = {}
    for flavor in ("fair_combined", "lut_only", "scna_only"):
        binary = project / "artifacts" / flavor / "libhtp_ops_skel.so"
        if binary.is_file():
            build_id = project / "artifacts" / flavor / "build_id.txt"
            artifacts[flavor] = {"path": str(binary), "bytes": binary.stat().st_size,
                                 "sha256": sha256(binary),
                                 "content_build_id": artifact_build_id(binary),
                                 "build_id": build_id.read_text().strip() if build_id.is_file() else None}
    host_artifacts: dict[str, dict[str, object]] = {}
    host_dir = project / "artifacts/host"
    if host_dir.is_dir():
        for path in sorted(p for p in host_dir.iterdir() if p.is_file()):
            host_artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path),
                                         "build_id": artifact_build_id(path)}
    model_host_artifacts: dict[str, dict[str, object]] = {}
    model_host_dir = project / "artifacts/model_host"
    if model_host_dir.is_dir():
        for path in sorted(p for p in model_host_dir.iterdir() if p.is_file()):
            model_host_artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path),
                                               "build_id": artifact_build_id(path)}
    llama_bridge = project / "work/llama.cpp-stage29/ggml/src/ggml-htp"
    llama_bridge_hash, llama_bridge_files = tree_digest(llama_bridge) if llama_bridge.is_dir() else (None, 0)
    harness_entries = ("scripts", "tools", "tests", "templates", "requirements", "docs", "README.md", ".gitignore",
                       "experiment_spec.json", "model_repair_spec.json", "audit_rerun_spec.json")
    harness_hash, harness_files = selected_digest(project, harness_entries)
    manifest = {
        "schema_version": 1,
        "architecture": "v79",
        "frozen_configuration": {
            "runtime_variant": "pair_static_d8",
            "kernel_impl": "d7_pairret_noinline",
            "kernel_impl_id": 3,
            "softmax_impl": "fused_state_update",
            "softmax_impl_id": 2,
            "scheduler": "stage2_75_adaptive",
        },
        "source": {"path": str(source), "tree_sha256": source_hash, "files": source_files},
        "stage2_75_baseline": {
            "path": str(baseline),
            "sha256": sha256(baseline) if baseline.is_file() else None,
        },
        "artifacts": artifacts,
        "host_artifacts": host_artifacts,
        "model_host_artifacts": model_host_artifacts,
        "llama_backend_bridge": {"path": str(llama_bridge), "tree_sha256": llama_bridge_hash,
                                  "files": llama_bridge_files},
        "audit_harness": {
            "included": list(harness_entries),
            "tree_sha256": harness_hash,
            "files": harness_files,
        },
        "experiment_spec": {
            "path": str(project / "experiment_spec.json"),
            "sha256": sha256(project / "experiment_spec.json"),
        },
    }
    out = project / "manifests/build_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
