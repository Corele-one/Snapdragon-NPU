#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
remote_dir="/data/local/tmp/stage2_75_scheduler_v79"
run_id="${1:-20260827_stage275_correctness_formal_v1}"
out="$project_dir/results/runs/$run_id"
artifact="$project_dir/artifacts/variants/fused_state_update/libhtp_ops_skel.so"
host_ship="$project_dir/src/htp-ops-lib-main/android_ReleaseG_aarch64/ship"

mkdir -p "$out"/{raw/correctness,raw/production_correctness,raw/determinism,evidence,recovery}
[[ -s "$artifact" ]] || { echo "Missing $artifact" >&2; exit 1; }

adb get-state >/dev/null
"$script_dir/deploy_and_smoke.sh" --mode ping --kernel-impl d7_pairret_noinline \
  --remote-dir "$remote_dir" >"$out/evidence/deploy.log" 2>&1
adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >>"$out/evidence/deploy.log" 2>&1
adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >>"$out/evidence/deploy.log" 2>&1

{
  echo "schema_version=1"
  echo "run_id=$run_id"
  echo "matrix=3 masks x 2 q lengths x 2 kv lengths x 2 head dimensions x 3 seeds = 72"
  echo "production_extension=3 masks x 3 q lengths x 2 kv lengths x 2 head dimensions x 3 seeds = 108"
  echo "scheduler=q_task_rows:auto,workers:auto"
  echo "kernel=fused_state_update"
  echo "captured_at=$(date -u +%FT%TZ)"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.build.fingerprint
  sha256sum "$artifact" "$host_ship/htp_ops_test"
} >"$out/evidence/manifest.txt"

run_case() {
  local mask="$1" q="$2" kv="$3" dim="$4" seed="$5" category="$6"
  local log="$out/raw/$category/${mask}_q${q}_kv${kv}_d${dim}_seed${seed}.log"
  if [[ -s "$log" ]] && grep -q 'FIG8_ATTENTION_COMPARE .*pass=1' "$log"; then
    echo "resume: ${log#$out/}"
    return
  fi
  local status=0
  set +e
  timeout 180s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' \
    ./htp_ops_test --figure8-attn --mode scna-fp16 --scna-variant pair_static_d8 \
    --workers auto --q-task-rows auto --scna-width 8 --mask-mode '$mask' \
    --qo-len '$q' --kv-len '$kv' --n-heads 12 --n-kv-heads 2 --head-dim '$dim' \
    --warmup 1 --iters 1 --seed '$seed' --compare-reference --numeric-debug --no-events" \
    >"$log" 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 ]] || ! grep -q 'FIG8_ATTENTION_COMPARE .*pass=1' "$log"; then
    {
      echo "exit_code=$status"
      tail -100 "$log"
      adb get-state
    } >"$out/recovery/$(basename "${log%.log}").log" 2>&1 || true
    echo "FAILED: ${log#$out/}" >&2
    return 1
  fi
  echo "pass: ${log#$out/}"
}

for mask in full causal padding; do
  for q in 1 4; do
    for kv in 4093 4096; do
      for dim in 64 128; do
        for seed in figure8_fixed 20260810 20260811; do
          run_case "$mask" "$q" "$kv" "$dim" "$seed" correctness
        done
      done
    done
  done
done

# The inherited 72-case gate only exercises q1/q4.  Exercise every adaptive
# production row choice as an extension so r8/r16 correctness is direct evidence.
for mask in full causal padding; do
  for q in 8 16 32; do
    for kv in 4093 4096; do
      for dim in 64 128; do
        for seed in figure8_fixed 20260810 20260811; do
          run_case "$mask" "$q" "$kv" "$dim" "$seed" production_correctness
        done
      done
    done
  done
done

# Repeat the final production path in independent processes.  Only measured
# checksums are compared; warmup checksums are deliberately excluded.
for repeat in $(seq 1 10); do
  log="$out/raw/determinism/q32_kv4096_repeat${repeat}.log"
  timeout 180s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' \
    ./htp_ops_test --figure8-attn --mode scna-fp16 --scna-variant pair_static_d8 \
    --workers auto --q-task-rows auto --scna-width 8 --mask-mode full \
    --qo-len 32 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 \
    --warmup 1 --iters 1 --seed figure8_fixed --no-events" >"$log" 2>&1
  grep -q 'FIG8_ATTENTION_HOST_TIMING .*phase=measure .*ret=0' "$log"
done

checksum_count="$({ rg 'FIG8_ATTENTION_CHECKSUM .*phase=measure' "$out/raw/determinism"/*.log || true; } \
  | sed -E 's/.*checksum=//' | sort -u | wc -l)"
[[ "$checksum_count" -eq 1 ]] || { echo "Determinism failed: $checksum_count checksums" >&2; exit 1; }

python3 - "$out" <<'PY'
import json, re, sys
from pathlib import Path

out = Path(sys.argv[1])
pattern = re.compile(r"FIG8_ATTENTION_COMPARE .*rmse=([^ ]+).*max_abs_error=([^ ]+).*candidate_nonfinite=(\d+).*reference_nonfinite=(\d+).*pass=(\d+)")
rows = []
for category in ("correctness", "production_correctness"):
  for path in sorted((out / f"raw/{category}").glob("*.log")):
    matches = pattern.findall(path.read_text(errors="replace"))
    if len(matches) != 1:
        raise SystemExit(f"expected one comparison in {path}, found {len(matches)}")
    rmse, max_abs, cand_nf, ref_nf, passed = matches[0]
    rows.append({"category": category, "file": str(path), "rmse": float(rmse),
                 "max_abs_error": float(max_abs), "candidate_nonfinite": int(cand_nf),
                 "reference_nonfinite": int(ref_nf), "pass": int(passed)})
checksums = []
for path in sorted((out / "raw/determinism").glob("*.log")):
    line = next(x for x in path.read_text(errors="replace").splitlines()
                if "FIG8_ATTENTION_CHECKSUM" in x and "phase=measure" in x)
    checksums.append(line.rsplit("checksum=", 1)[1])
summary = {
    "schema_version": 1,
    "required_cases": sum(x["category"] == "correctness" for x in rows),
    "production_extension_cases": sum(x["category"] == "production_correctness" for x in rows),
    "cases": len(rows),
    "passed": sum(x["pass"] for x in rows),
    "max_rmse": max(x["rmse"] for x in rows),
    "max_abs_error": max(x["max_abs_error"] for x in rows),
    "candidate_nonfinite": sum(x["candidate_nonfinite"] for x in rows),
    "reference_nonfinite": sum(x["reference_nonfinite"] for x in rows),
    "determinism_repeats": len(checksums),
    "determinism_unique_checksums": sorted(set(checksums)),
    "rows": rows,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
PY

echo "completed: $out"
