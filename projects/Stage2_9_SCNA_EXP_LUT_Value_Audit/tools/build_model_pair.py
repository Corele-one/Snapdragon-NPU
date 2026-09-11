#!/usr/bin/env python3
"""Build a versioned CPU/HMX model pair from one immutable HF source."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_lineage import (build_id, command_output, environment_snapshot, file_record,
                           git_commit, git_lfs_oid, sha256)


PROJECT = Path(__file__).resolve().parents[1]


def run(argv: list[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        proc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(shlex.quote, argv))}; see {log}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT / "model_repair_spec.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-venv", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true",
                        help="write the manifest for already completed, non-empty outputs")
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    src = Path(spec["source"]["hf_dir"])
    tree = (PROJECT / spec["toolchain"]["llama_tree"]).resolve()
    output = (args.output_dir or PROJECT / spec["outputs"]["root"]).resolve()
    if output.exists() and any(output.iterdir()) and not args.finalize_existing:
        raise RuntimeError(f"refusing to overwrite non-empty model directory: {output}")
    output.mkdir(parents=True, exist_ok=False) if not output.exists() else None
    logs = output / "logs"

    actual_commit = git_commit(src)
    if actual_commit != spec["source"]["git_commit"]:
        raise RuntimeError(f"HF commit mismatch: {actual_commit}")
    source_tensor = src / spec["source"]["safetensors"]
    source_hash = sha256(source_tensor)
    lfs_oid, lfs_size = git_lfs_oid(src, spec["source"]["safetensors"])
    if source_hash != spec["source"]["safetensors_sha256"] or source_hash != lfs_oid:
        raise RuntimeError("safetensors hash does not match repair spec and Git-LFS OID")
    if source_tensor.stat().st_size != lfs_size:
        raise RuntimeError("safetensors size does not match Git-LFS pointer")
    if git_commit(tree) != spec["toolchain"]["llama_commit"]:
        raise RuntimeError("llama.cpp commit mismatch")

    # Do not resolve this path: venv/bin/python is normally a symlink to the
    # system interpreter.  Resolving it would make uv install into /usr and
    # would also derive the wrong venv root below.
    venv_python = PROJECT / spec["toolchain"]["python"]
    venv_root = venv_python.parent.parent
    requirements = (PROJECT / spec["toolchain"]["requirements"]).resolve()
    if not args.skip_venv:
        uv = command_output(["sh", "-c", "command -v uv"])
        venv_root.parent.mkdir(parents=True, exist_ok=True)
        if not venv_python.exists():
            run([uv, "venv", str(venv_root), "--python", "3.10"], logs / "venv.log")
        run([uv, "pip", "install", "--python", str(venv_python), "-r", str(requirements)],
            logs / "requirements.log")
    if not venv_python.is_file():
        raise RuntimeError(f"conversion Python missing: {venv_python}")

    names = spec["outputs"]
    cpu_f16, hmx_f16 = output / names["cpu_f16"], output / names["hmx_f16"]
    cpu_quant, hmx_quant = output / names["cpu_quant"], output / names["hmx_quant"]
    quantizer = Path(spec["toolchain"]["quantizer"])
    commands = [
        [str(venv_python), str(tree / spec["toolchain"]["cpu_converter"]), str(src),
         "--outfile", str(cpu_f16), "--outtype", "f16"],
        [str(venv_python), str(tree / spec["toolchain"]["hmx_converter"]), str(src),
         "--outfile", str(hmx_f16), "--outtype", "f16"],
        [str(quantizer), str(cpu_f16), str(cpu_quant), spec["toolchain"]["quantizer_type"]],
        [str(quantizer), str(hmx_f16), str(hmx_quant), spec["toolchain"]["quantizer_type"]],
    ]
    conversion_env = dict(os.environ)
    local_gguf = str(tree / "gguf-py")
    conversion_env["PYTHONPATH"] = local_gguf + (
        os.pathsep + conversion_env["PYTHONPATH"] if conversion_env.get("PYTHONPATH") else ""
    )
    if args.finalize_existing:
        missing = [str(path) for path in (cpu_f16, hmx_f16, cpu_quant, hmx_quant)
                   if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"cannot finalize: missing or empty model outputs: {missing}")
    else:
        run(commands[0], logs / "convert_cpu_f16.log", env=conversion_env)
        run(commands[1], logs / "convert_hmx_f16.log", env=conversion_env)
        cpu_env = dict(os.environ)
        cpu_env.pop("REPACK_FOR_HVX", None)
        run(commands[2], logs / "quantize_cpu.log", env=cpu_env)
        hmx_env = {**os.environ, "REPACK_FOR_HVX": "1"}
        run(commands[3], logs / "quantize_hmx.log", env=hmx_env)

    uv = command_output(["sh", "-c", "command -v uv"])
    pip_freeze = command_output([uv, "pip", "freeze", "--python", str(venv_python)]).splitlines()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repair_spec": file_record(spec_path),
        "source": {"git_commit": actual_commit, "lfs_oid_sha256": lfs_oid,
                   "lfs_size": lfs_size, "safetensors": file_record(source_tensor)},
        "toolchain": {
            "llama_commit": git_commit(tree),
            "cpu_converter": file_record(tree / spec["toolchain"]["cpu_converter"]),
            "hmx_converter": file_record(tree / spec["toolchain"]["hmx_converter"]),
            "quantizer": {**file_record(quantizer), "build_id": build_id(quantizer)},
            "requirements_lock": file_record(requirements), "pip_freeze": pip_freeze,
            "python": command_output([str(venv_python), "--version"]),
            "platform": platform.platform(),
        },
        "commands": [shlex.join(command) for command in commands],
        "environments": {
            "common": environment_snapshot(["PATH", "LD_LIBRARY_PATH"]),
            "conversion": {"PYTHONPATH_prepend": local_gguf},
            "hmx_quantize": {"REPACK_FOR_HVX": "1"},
            "cpu_quantize": {"REPACK_FOR_HVX": "unset"},
        },
        "lineage": {
            "cpu_quant_input_sha256": sha256(cpu_f16),
            "hmx_quant_input_sha256": sha256(hmx_f16),
            "requantization": False,
        },
        "models": {key: file_record(path) for key, path in {
            "cpu_f16": cpu_f16, "hmx_f16": hmx_f16,
            "cpu_quant": cpu_quant, "hmx_quant": hmx_quant}.items()},
        "device": {
            "directory": f"/data/local/tmp/stage2_9_model_pair_{source_hash[:12]}",
            "models": {key: f"/data/local/tmp/stage2_9_model_pair_{source_hash[:12]}/{path.name}"
                       for key, path in {"cpu_f16": cpu_f16, "hmx_f16": hmx_f16,
                                         "cpu_quant": cpu_quant, "hmx_quant": hmx_quant}.items()},
        },
        "tensor_validation": {"status": "pending", "path": names["tensor_validation"]},
    }
    manifest_path = output / names["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
