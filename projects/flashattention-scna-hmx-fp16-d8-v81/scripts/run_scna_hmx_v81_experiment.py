#!/usr/bin/env python3
"""Run the versioned v81 FP16 d8 SCNA gates and paired experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "experiment_spec.json"
PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
PREFIXES = (
    "FIG8_ATTENTION_CONFIG", "FIG8_ATTENTION_HOST_TIMING", "FIG8_ATTENTION_TIMERS",
    "FIG8_ATTENTION_COMPARE", "FIG8_NUMERIC", "SCNA_EXP_BENCH",
)


def run(command: list[str], *, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, check=check)


def adb_shell(command: str, *, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["adb", "shell", command], timeout=timeout, check=check)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fields(line: str) -> dict[str, str]:
    return dict(PAIR_RE.findall(line))


def parse_records(output: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        marker = next((value for value in PREFIXES if value in line), None)
        if marker:
            records.append({"record": marker, "fields": parse_fields(line), "raw": line})
    return records


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def device_state() -> dict[str, Any]:
    script = r'''max=0; sensor=""; for z in /sys/class/thermal/thermal_zone*; do
t=$(cat "$z/type" 2>/dev/null); case "$t" in cpu-*|cpuss-*|cpu_therm)
v=$(cat "$z/temp" 2>/dev/null); [ -n "$v" ] && [ "$v" -gt "$max" ] && max="$v" && sensor="$t";; esac; done
echo max_cpu_temp_millic=$max; echo max_cpu_temp_sensor=$sensor
for p in /sys/devices/system/cpu/cpufreq/policy*; do
echo cpu_freq_$(basename "$p")=$(cat "$p/scaling_cur_freq" 2>/dev/null); done
echo soc_model=$(getprop ro.soc.model); echo product_model=$(getprop ro.product.model)'''
    result = adb_shell(script, timeout=15, check=False)
    values = parse_fields(result.stdout)
    temp = int(values.get("max_cpu_temp_millic", "0") or 0)
    return {**values, "max_cpu_temp_c": temp / 1000.0 if temp else None,
            "dsp_frequency_observable": False,
            "requested_dsp_power_mode": "HAP_DCVS_V2_PERFORMANCE_MODE"}


def wait_device_state_near(target_c: float, tolerance_c: float, max_wait_s: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + max_wait_s
    state = device_state()
    while state["max_cpu_temp_c"] is not None and abs(float(state["max_cpu_temp_c"]) - target_c) > tolerance_c:
        if time.monotonic() >= deadline:
            return state
        time.sleep(0.5)
        state = device_state()
    return state


def mode_args(mode: str) -> str:
    if mode.startswith("scna-"):
        return "--scna-width 8 --scna-function exp2 --scna-kernel direct --kv-pipeline off"
    return ""


def remote_command(spec: dict[str, Any], mode: str, *, qo_len: int, kv_len: int,
                   head_dim: int, mask_mode: str, warmup: int, iterations: int,
                   compare: bool) -> str:
    remote = spec["remote_dir"]
    extra = mode_args(mode)
    compare_arg = "--compare-reference" if compare else ""
    return (
        f"cd '{remote}' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' "
        f"./htp_ops_test --figure8-attn --mode '{mode}' {extra} --qo-len {qo_len} "
        f"--kv-len {kv_len} --n-heads 12 --n-kv-heads 2 --head-dim {head_dim} "
        f"--mask-mode {mask_mode} --warmup {warmup} --iters {iterations} --no-events {compare_arg}"
    )


def verify_single_binary(spec: dict[str, Any], expected: dict[str, str]) -> None:
    remote = spec["remote_dir"]
    result = adb_shell(
        f"sha256sum '{remote}/htp_ops_test' '{remote}/libhtp_ops.so' "
        f"'{remote}/cdsp/libhtp_ops_skel.so'", timeout=15)
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            found[Path(parts[1]).name] = parts[0]
    ensure_digest_match(expected, found)


def ensure_digest_match(expected: dict[str, str], found: dict[str, str]) -> None:
    for name, digest in expected.items():
        if found.get(name) != digest:
            raise RuntimeError(f"mixed/stale binary rejected: {name}: expected {digest}, got {found.get(name)}")


def collect_manifest(spec: dict[str, Any], out: Path) -> dict[str, Any]:
    htp = ROOT / "src/htp-ops-lib-main"
    artifacts = {
        "htp_ops_test": htp / "android_ReleaseG_aarch64/ship/htp_ops_test",
        "libhtp_ops.so": htp / "android_ReleaseG_aarch64/ship/libhtp_ops.so",
        "libhtp_ops_skel.so": htp / "hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so",
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing build artifacts: " + ", ".join(missing))
    digests = {name: sha256(path) for name, path in artifacts.items()}
    status = run(["git", "status", "--short"], check=False).stdout.splitlines()
    git_head = run(["git", "rev-parse", "HEAD"], check=False).stdout.strip()
    device = adb_shell(
        "getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.release; "
        "getprop ro.build.fingerprint", timeout=15).stdout.splitlines()
    params = ROOT / "src/htp-ops-lib-main/include/dsp/scna_params.h"
    source_params = ROOT.parent / "flashattention-scna-v81/src/htp-ops-lib-main/include/dsp/scna_params.h"
    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project": str(ROOT),
        "architecture": "v81",
        "hexagon_sdk": "6.6.0.0",
        "hexagon_tools": "19.0.07",
        "compile_flags": spec["compiler_flags"],
        "artifacts_sha256": digests,
        "scna_params_sha256": sha256(params),
        "scna_params_source": str(source_params),
        "scna_params_source_sha256": sha256(source_params),
        "git_status": status,
        "git_head": git_head,
        "git_note": "The independent project is intentionally untracked in the parent repository during this experiment.",
        "device": {
            "serial": run(["adb", "get-serialno"]).stdout.strip(),
            "product_model": device[0] if len(device) > 0 else "unknown",
            "soc_model": device[1] if len(device) > 1 else "unknown",
            "android": device[2] if len(device) > 2 else "unknown",
            "fingerprint": device[3] if len(device) > 3 else "unknown",
        },
        "power": {
            "requested_mode": "HAP_DCVS_V2_PERFORMANCE_MODE",
            "target_corner": "HAP_DCVS_VCORNER_TURBO_L3",
            "dsp_frequency_observable_from_android": False,
        },
        "spec": spec,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def collect_disassembly(out: Path) -> None:
    tool = Path("/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump")
    binary = ROOT / "src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so"
    result = run([str(tool), "-d", "--demangle", str(binary)], timeout=120)
    full = out / "verification/disassembly_full.txt"
    full.write_text(result.stdout, encoding="utf-8")
    wanted = ("scna_hmx_fp16_d8_affine_relu_kernel", "scna_hmx_fp16_d8_reduce_kernel",
              "scna_hmx_fp16_d8_hybrid_reduce_hvx")
    lines = result.stdout.splitlines()
    excerpts: list[str] = []
    for index, line in enumerate(lines):
        if any(name in line for name in wanted):
            excerpts.extend(lines[index:index + 45])
            excerpts.append("")
    text = "\n".join(excerpts) + "\n"
    required = ("activation.hf", "weight.hf", "bias = mxmem2", "cvt.hf", "mxmem")
    evidence = {item: item in text for item in required}
    evidence["hybrid_hvx_vadd"] = "vadd" in text
    (out / "verification/scna_hmx_symbols.disasm.txt").write_text(text, encoding="utf-8")
    (out / "verification/disassembly_gate.json").write_text(
        json.dumps({"pass": all(evidence.values()), "evidence": evidence}, indent=2, sort_keys=True) + "\n")
    if not all(evidence.values()):
        raise RuntimeError(f"HMX disassembly gate failed: {evidence}")


def collect_micro(spec: dict[str, Any], out: Path, quick: bool) -> None:
    cfg = spec["microkernel"]
    samples = 2 if quick else int(cfg["samples"])
    path = out / "raw/microkernel.jsonl"
    for mode in spec["correctness"]["modes"]:
        iterations = int(cfg["iterations"][mode])
        if quick:
            iterations = min(iterations, 2000)
        for sample in range(samples):
            state = device_state()
            remote = spec["remote_dir"]
            command = (f"cd '{remote}' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' "
                       f"./htp_ops_test --scna-exp-bench --mode '{mode}' --scna-width 8 "
                       f"--warmup 5 --iters {iterations}")
            result = adb_shell(command, timeout=180, check=False)
            records = parse_records(result.stdout)
            bench = next((record for record in records if record["record"] == "SCNA_EXP_BENCH"), None)
            row = {"mode": mode, "sample": sample, "iterations": iterations, "device_state": state,
                   "returncode": result.returncode, "record": bench, "status": "pass"}
            if bench is None or result.returncode != 0:
                row["status"] = "failed"
            else:
                fields = bench["fields"]
                elapsed_ms = float(fields["elapsed_us"]) / 1000.0
                row["elapsed_ms"] = elapsed_ms
                row["duration_gate"] = elapsed_ms >= float(cfg["minimum_sample_ms"])
                if float(fields["implementation_rmse"]) > float(cfg["implementation_rmse_limit"]) or \
                        float(fields["implementation_max_abs_error"]) > float(cfg["implementation_max_abs_error_limit"]):
                    row["status"] = "failed"
            append_jsonl(path, row)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    evidence: dict[str, Any] = {}
    for mode in spec["correctness"]["modes"][-2:]:
        mode_rows = [row for row in rows if row["mode"] == mode and row.get("record")]
        checks: list[dict[str, Any]] = []
        for row in mode_rows:
            fields = row["record"]["fields"]
            checks.append({
                "implementation_rmse": float(fields["implementation_rmse"]),
                "implementation_max_abs_error": float(fields["implementation_max_abs_error"]),
                "random_samples": int(fields.get("random_samples", 0)),
                "random_implementation_rmse": float(fields.get("random_implementation_rmse", "inf")),
                "random_implementation_max_abs_error": float(fields.get("random_implementation_max_abs_error", "inf")),
                "tail_implementation_max_abs_error": float(fields.get("tail_implementation_max_abs_error", "inf")),
                "monotonic_violations": int(fields["monotonic_violations"]),
                "negative_count": int(fields["negative_count"]), "nan_count": int(fields["nan_count"]),
                "y_zero": float(fields["y_zero"]), "pair_max_abs_diff": float(fields["pair_max_abs_diff"]),
            })
        evidence[mode] = checks
    rmse_limit = float(cfg["implementation_rmse_limit"])
    max_limit = float(cfg["implementation_max_abs_error_limit"])
    passed = bool(evidence) and all(checks for checks in evidence.values()) and all(
        item["implementation_rmse"] <= rmse_limit and item["implementation_max_abs_error"] <= max_limit and
        item["random_samples"] == 4096 and item["random_implementation_rmse"] <= rmse_limit and
        item["random_implementation_max_abs_error"] <= max_limit and
        item["tail_implementation_max_abs_error"] <= max_limit and item["monotonic_violations"] == 0 and
        item["negative_count"] == 0 and item["nan_count"] == 0 and item["y_zero"] == 1.0 and
        item["pair_max_abs_diff"] == 0.0 for checks in evidence.values() for item in checks)
    gate = {"pass": passed, "rmse_limit": rmse_limit, "max_abs_error_limit": max_limit,
            "coverage": ["boundary", "relu-fold/dense-grid", "fixed-seed-random", "tail-lengths-1-7-31-32-33-63",
                         "32x8-layout", "two-pass-direct-reload"], "evidence": evidence}
    (out / "verification/hmx_numeric_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("HMX numeric gate failed; performance collection is stopped")


def collect_correctness(spec: dict[str, Any], out: Path, quick: bool) -> None:
    cfg = spec["correctness"]
    modes = cfg["modes"] if not quick else cfg["modes"][-2:]
    masks = cfg["mask_mode"] if not quick else ["full", "causal"]
    kvs = cfg["kv_len"] if not quick else [4093]
    dims = cfg["head_dim"] if not quick else [128]
    path = out / "raw/correctness.jsonl"
    log_dir = out / "raw/correctness_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        for mask in masks:
            for kv_len in kvs:
                for head_dim in dims:
                    command = remote_command(spec, mode, qo_len=int(cfg["qo_len"]), kv_len=int(kv_len),
                                             head_dim=int(head_dim), mask_mode=mask, warmup=1,
                                             iterations=1, compare=True) + " --numeric-debug"
                    result = adb_shell(command, timeout=120, check=False)
                    stem = f"{mode}_mask-{mask}_kv-{kv_len}_d-{head_dim}"
                    (log_dir / f"{stem}.log").write_text(result.stdout, encoding="utf-8")
                    records = parse_records(result.stdout)
                    compare = next((record for record in records if record["record"] == "FIG8_ATTENTION_COMPARE"), None)
                    status = "pass"
                    numeric = [r for r in records if r["record"] == "FIG8_NUMERIC"]
                    if result.returncode != 0 or compare is None or compare["fields"].get("ret") != "0":
                        status = "failed"
                    elif (float(compare["fields"]["rmse"]) > float(cfg["rmse_limit"]) or
                          float(compare["fields"]["max_abs_error"]) > float(cfg["max_abs_error_limit"]) or
                          int(compare["fields"]["candidate_nonfinite"]) != 0):
                        status = "failed"
                    expect_tail_zero = int(kv_len) == 4093 or mask in ("padding", "causal")
                    if expect_tail_zero and (not numeric or any(
                            int(record["fields"].get("p0_last_bits", "-1"), 16) != 0 for record in numeric)):
                        status = "failed"
                    append_jsonl(path, {"mode": mode, "mask_mode": mask, "kv_len": kv_len,
                                       "head_dim": head_dim, "returncode": result.returncode,
                                       "status": status, "compare": compare,
                                       "tail_zero_required": expect_tail_zero, "numeric": numeric})


def latin_order(modes: list[str], session: int) -> list[str]:
    # Fixed cyclic Latin square: each mode occupies each order position once.
    return modes[session % len(modes):] + modes[:session % len(modes)]


def collect_performance(spec: dict[str, Any], out: Path, digests: dict[str, str], quick: bool,
                        resume: bool = False) -> None:
    cfg = spec["performance"]
    sessions = 1 if quick else int(cfg["sessions"])
    warmup = 1 if quick else int(cfg["warmup"])
    iterations = 2 if quick else int(cfg["iterations"])
    q_values = cfg["qo_len"] if not quick else [4, 32]
    raw = out / "raw/performance.jsonl"
    log_dir = out / "raw/performance_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    existing = ([json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
                if resume and raw.is_file() else [])
    valid_pairs = {(int(row["qo_len"]), int(row["session"])) for row in existing if row.get("session_valid")}
    for qo_len in q_values:
        for session in range(sessions):
            if (int(qo_len), session) in valid_pairs:
                continue
            accepted = False
            old_attempts = [int(row["attempt"]) for row in existing
                            if int(row["qo_len"]) == int(qo_len) and int(row["session"]) == session]
            attempt_start = max(old_attempts, default=-1) + 1
            for attempt in range(attempt_start, attempt_start + int(cfg["max_attempts_per_session"])):
                verify_single_binary(spec, digests)
                order = latin_order(list(spec["modes"]), session)
                starts: list[float] = []
                attempt_rows: list[dict[str, Any]] = []
                for order_index, mode in enumerate(order):
                    if order_index == 0:
                        state = device_state()
                        target_temperature = state["max_cpu_temp_c"]
                    elif target_temperature is not None:
                        state = wait_device_state_near(
                            float(target_temperature), float(cfg["max_start_temperature_span_c"]) / 2.0)
                    else:
                        state = device_state()
                    if state["max_cpu_temp_c"] is not None:
                        starts.append(float(state["max_cpu_temp_c"]))
                    command = remote_command(spec, mode, qo_len=int(qo_len), kv_len=int(cfg["kv_len"]),
                                             head_dim=int(cfg["head_dim"]), mask_mode=str(cfg["mask_mode"]),
                                             warmup=warmup, iterations=iterations, compare=False)
                    result = adb_shell(command, timeout=180, check=False)
                    stem = f"q{qo_len}_s{session}_a{attempt}_o{order_index}_{mode}"
                    (log_dir / f"{stem}.log").write_text(result.stdout, encoding="utf-8")
                    parsed = parse_records(result.stdout)
                    measures = [r for r in parsed if r["record"] == "FIG8_ATTENTION_HOST_TIMING" and
                                r["fields"].get("phase") == "measure"]
                    timers = [r for r in parsed if r["record"] == "FIG8_ATTENTION_TIMERS" and
                              r["fields"].get("phase") == "measure"]
                    status = "pass" if result.returncode == 0 and len(measures) == iterations and timers else "failed"
                    attempt_rows.append({"qo_len": qo_len, "session": session, "attempt": attempt,
                                         "order_index": order_index, "mode": mode, "order": order,
                                         "start_state": state, "returncode": result.returncode,
                                         "status": status, "host_measurements": measures, "timers": timers})
                span = max(starts) - min(starts) if starts else None
                temperature_valid = span is not None and span <= float(cfg["max_start_temperature_span_c"])
                complete = all(row["status"] == "pass" for row in attempt_rows)
                accepted = bool(temperature_valid and complete)
                for row in attempt_rows:
                    row["temperature_span_c"] = span
                    row["session_valid"] = accepted
                    append_jsonl(raw, row)
                if accepted:
                    break
            if not accepted:
                print(f"warning: q={qo_len} session={session} has no valid attempt", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="result directory name; default is UTC timestamp plus binary hash")
    parser.add_argument("--quick", action="store_true", help="small plumbing check, not a reportable experiment")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--phase", choices=("all", "micro", "correctness", "performance"), default="all")
    parser.add_argument("--resume-performance", action="store_true",
                        help="append only missing valid performance sessions in an existing run-id")
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not args.skip_build:
        run([str(ROOT / "scripts/build.sh"), "--dsp-arch", "v81"], timeout=300)
    if not args.skip_deploy:
        run([str(ROOT / "scripts/deploy_and_smoke.sh"), "--mode", "ping"], timeout=120)
    binary = ROOT / "src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so"
    run_id = args.run_id or f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{sha256(binary)[:12]}"
    out = ROOT / "results" / run_id
    for child in ("raw", "verification", "figures", "summary"):
        (out / child).mkdir(parents=True, exist_ok=True)
    manifest = collect_manifest(spec, out)
    verify_single_binary(spec, manifest["artifacts_sha256"])
    collect_disassembly(out)
    if args.phase in ("all", "micro"):
        collect_micro(spec, out, args.quick)
    if args.phase in ("all", "correctness"):
        collect_correctness(spec, out, args.quick)
    if args.phase in ("all", "performance"):
        collect_performance(spec, out, manifest["artifacts_sha256"], args.quick, args.resume_performance)
    print(out)


if __name__ == "__main__":
    main()
