#!/usr/bin/env python3
"""Matched sustained-load thermal audit; deliberately makes no energy claim."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from model_lineage import stable_case_id


PROJECT = Path(__file__).resolve().parents[1]
SPEC = json.loads((PROJECT / "experiment_spec.json").read_text())
TEMP = re.compile(r"temperature:\s*(-?\d+)")
BATCH = re.compile(r"BATCH mode=(\S+) index=(\d+) elapsed_ns=(\d+) ret=(\d+)")


def shell(command: str, timeout: int = 30) -> str:
    return subprocess.run(["adb", "shell", command], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout, check=True).stdout


def battery_temp() -> float:
    match = TEMP.search(shell("dumpsys battery"))
    if not match:
        raise RuntimeError("battery temperature unavailable")
    return int(match.group(1)) / 10.0


def sensor_record(mode: str, elapsed: float) -> dict[str, object]:
    battery = shell("dumpsys battery")
    thermal = shell("dumpsys thermalservice")
    match = TEMP.search(battery)
    return {
        "record_type": "thermal_sensor",
        "mode": mode,
        "elapsed_seconds": elapsed,
        "battery_temperature_c": int(match.group(1)) / 10.0 if match else None,
        "thermal_status": next((line.strip() for line in thermal.splitlines()
                                if "Thermal Status" in line or "mStatus" in line), "unknown"),
        "thermal_snapshot": [line.strip() for line in thermal.splitlines()
                             if "Temperature{" in line or "mValue" in line][:32],
        "energy_claim_allowed": False,
    }


def append(path: Path, record: dict[str, object]) -> None:
    identity = {key: record.get(key) for key in
                ("record_type", "mode", "event", "sample", "batch") if key in record}
    record.setdefault("case_id", stable_case_id(identity))
    record.setdefault("attempt", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def wait_for_matched_temperature(target: float, tolerance: float, timeout: int = 900) -> float:
    deadline = time.monotonic() + timeout
    while True:
        value = battery_temp()
        if abs(value - target) <= tolerance:
            return value
        if time.monotonic() >= deadline:
            raise RuntimeError(f"unable to match start temperature {target:.1f}C; current {value:.1f}C")
        time.sleep(15)


def workload_command(remote: str, mode: str, seconds: int) -> str:
    args = ["--figure8-attn", "--mode", mode, "--scna-variant", "pair_static_d8", "--workers", "auto",
            "--q-task-rows", "auto", "--mask-mode", "causal", "--qo-len", "32", "--kv-len", "4096",
            "--n-heads", "12", "--n-kv-heads", "2", "--head-dim", "128", "--seed", "12029",
            "--warmup", "5", "--iters", "1000", "--no-events"]
    rendered = " ".join(shlex.quote(value) for value in args)
    return (
        f"cd {remote}; end=$(( $(date +%s) + {seconds} )); i=0; "
        f"while [ $(date +%s) -lt $end ]; do t0=$(date +%s%N); "
        f"LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test {rendered} >/dev/null 2>&1; "
        f"r=$?; t1=$(date +%s%N); echo BATCH mode={mode} index=$i elapsed_ns=$((t1-t0)) ret=$r; "
        f"i=$((i+1)); [ $r -eq 0 ] || exit $r; done"
    )


def main() -> int:
    global SPEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT / "experiment_spec.json")
    parser.add_argument("--out-dir", "--results-dir", dest="out_dir", type=Path,
                        default=PROJECT / "results/formal")
    parser.add_argument("--duration", type=int)
    args = parser.parse_args()
    SPEC = json.loads(args.spec.resolve().read_text())
    if args.duration is None:
        args.duration = SPEC["thermal"]["duration_seconds"]
    raw = args.out_dir / "raw/thermal.jsonl"
    logs = args.out_dir / "logs/thermal"
    logs.mkdir(parents=True, exist_ok=True)
    remote = "/data/local/tmp/stage2_9_value_audit_fair_combined"
    subprocess.run([str(PROJECT / "scripts/deploy.sh"), "--flavor", "fair_combined", "--remote-dir", remote],
                   check=True, stdout=subprocess.PIPE, text=True)
    modes = ["lut-exp", "scna-fp16"]
    first_start = battery_temp()
    tolerance = SPEC["thermal"]["start_temperature_delta_c_max"]
    for run_index, mode in enumerate(modes):
        start_temp = first_start if run_index == 0 else wait_for_matched_temperature(first_start, tolerance)
        append(raw, {"record_type": "thermal_run", "mode": mode, "event": "start",
                     "temperature_c": start_temp, "duration_seconds": args.duration,
                     "energy_claim_allowed": False})
        process = subprocess.Popen(["adb", "shell", workload_command(remote, mode, args.duration)],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        started = time.monotonic()
        sample_index = 0
        while process.poll() is None:
            record = sensor_record(mode, time.monotonic() - started)
            record["sample"] = sample_index
            append(raw, record)
            sample_index += 1
            time.sleep(SPEC["thermal"]["sample_period_seconds"])
        output = process.communicate()[0]
        (logs / f"{mode}_batches.log").write_text(output)
        if process.returncode:
            raise RuntimeError(f"thermal workload failed for {mode}: {process.returncode}")
        for batch_mode, index, elapsed_ns, ret in BATCH.findall(output):
            append(raw, {"record_type": "thermal_batch", "mode": batch_mode, "batch": int(index),
                         "requests": 1000, "elapsed_us": int(elapsed_ns) / 1000.0, "ret": int(ret),
                         "energy_claim_allowed": False})
        append(raw, {"record_type": "thermal_run", "mode": mode, "event": "end",
                     "temperature_c": battery_temp(), "energy_claim_allowed": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
