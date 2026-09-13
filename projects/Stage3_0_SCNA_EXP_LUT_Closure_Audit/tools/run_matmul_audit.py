#!/usr/bin/env python3
"""Run the Stage 2.9.1 end-to-end HMX matmul oracle matrix."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from model_lineage import build_id, sha256, stable_case_id


PROJECT = Path(__file__).resolve().parents[1]
MARKER = "MATMUL_AUDIT_JSON "


def artifact_build_id(path: Path) -> str:
    """Return an ELF build-id, or an explicit content-addressed fallback.

    GNU readelf cannot walk the Qualcomm LLVM metadata note in some DSP ELFs,
    so those artifacts use their full SHA256 as a non-ambiguous build id.
    """
    try:
        return build_id(path)
    except RuntimeError:
        return f"sha256:{sha256(path)}"


def compatible_paths(dtype: str, m: int, k: int, n: int) -> list[str]:
    if dtype == "f16":
        return ["sequential"]
    actual = "pipeline" if m >= 128 and k <= n else "sequential"
    paths = ["production", actual]
    if m >= 128 and k > n and k < 16384 and k % 256 == 0:
        paths.append("output_stationary")
    return paths


def parse_record(text: str) -> dict[str, object]:
    matches = [line.split(MARKER, 1)[1] for line in text.splitlines() if MARKER in line]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {MARKER.strip()} record, got {len(matches)}")
    return json.loads(matches[0])


def append(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT / "model_repair_spec.json")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    out_dir = args.results_dir.resolve()
    raw = out_dir / "raw/matmul.jsonl"
    if raw.exists() and raw.stat().st_size:
        raise RuntimeError(f"refusing to overwrite matmul evidence: {raw}")
    logs = out_dir / "logs/matmul"
    deploy = subprocess.run([str(PROJECT / "scripts/deploy.sh"), "--flavor", "fair_combined"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "deploy.log").write_text(deploy.stdout)
    if deploy.returncode:
        raise RuntimeError("matmul audit deployment failed")
    artifact = PROJECT / "artifacts/fair_combined/libhtp_ops_skel.so"
    host = PROJECT / "artifacts/host/htp_ops_test"
    provenance = {"spec_sha256": sha256(spec_path), "runner_sha256": sha256(Path(__file__)),
                  "dsp_sha256": sha256(artifact),
                  "dsp_build_id": artifact_build_id(artifact), "host_sha256": sha256(host),
                  "host_build_id": artifact_build_id(host),
                  "matmul_source_sha256": sha256(PROJECT / "src/htp-ops-lib-main/src/dsp/ops/mat_mul.c"),
                  "executor_source_sha256": sha256(PROJECT / "src/htp-ops-lib-main/src/dsp/op_executor.cc")}

    cfg = spec["matmul_audit"]
    cases: list[tuple[str, str, int, int, int, int]] = []
    qwen_shapes = [(m, k, n) for m, (k, n) in itertools.product(cfg["m"], cfg["kn"])]
    for dtype in cfg["dtypes"]:
        shapes = qwen_shapes + [tuple(cfg["single_tile_cases"][dtype])]
        for (m, k, n), seed in itertools.product(shapes, cfg["seeds"]):
            if dtype != "f16" and k % 256:
                raise RuntimeError(f"invalid quant audit shape: dtype={dtype}, k={k}")
            for path in compatible_paths(dtype, m, k, n):
                cases.append((dtype, path, m, k, n, seed))
    if args.smoke:
        cases = [("f16", "sequential", 32, 1536, 256, 29101),
                 ("iq4_nl", "sequential", 32, 1536, 256, 29101),
                 ("q8_0", "sequential", 32, 1536, 256, 29101)]

    failures = 0
    checksum_by_case: dict[str, str] = {}
    for index, (dtype, path, m, k, n, seed) in enumerate(cases):
        identity = {"record_type": "matmul_value_audit", "dtype": dtype, "path": path,
                    "m": m, "k": k, "n": n, "seed": seed}
        case_id = stable_case_id(identity)
        for repetition in (1, 2):
            argv = [str(PROJECT / "scripts/device_exec.sh"), "--matmul-audit", "--dtype", dtype,
                    "--execution-path", path, "--m", str(m), "--k", str(k), "--n", str(n),
                    "--seed", str(seed)]
            final_record = None
            for attempt in (1, 2):
                proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      timeout=180)
                log = logs / f"{index:04d}_{case_id}_r{repetition}_a{attempt}.log"
                log.write_text(proc.stdout)
                try:
                    record = parse_record(proc.stdout)
                except (RuntimeError, json.JSONDecodeError) as exc:
                    record = {**identity, "pass": False, "infrastructure_error": str(exc)}
                expected_path = ({"sequential": 1, "pipeline": 2, "output_stationary": 3}.get(path)
                                 if path != "production" else (2 if m >= 128 and k <= n else 1))
                if record.get("actual_path") != expected_path:
                    record.update({"pass": False, "path_error":
                                   f"expected actual_path={expected_path}, got {record.get('actual_path')}"})
                record.update(identity)
                record.update(provenance)
                record.update({"case_id": case_id, "repetition": repetition, "attempt": attempt,
                               "returncode": proc.returncode})
                append(raw, record)
                final_record = record
                if proc.returncode == 0 and record.get("pass") is True:
                    break
            assert final_record is not None
            if final_record.get("pass") is not True:
                failures += 1
                continue
            checksum = str(final_record.get("output_checksum"))
            previous = checksum_by_case.setdefault(case_id, checksum)
            if previous != checksum:
                append(raw, {**identity, **provenance, "record_type": "matmul_checksum_instability",
                             "case_id": case_id, "first": previous, "second": checksum, "pass": False})
                failures += 1
    append(raw, {"record_type": "matmul_matrix_summary", **provenance, "cases": len(cases),
                 "repetitions": 2, "failures": failures, "pass": failures == 0})
    print(f"matmul cases={len(cases)} failures={failures}")
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
