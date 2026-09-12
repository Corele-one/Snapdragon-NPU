#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
remote_dir="${SCNA_REMOTE_DIR:-/data/local/tmp/scna_optimization_v79}"
out="$project_dir/results/v79/correctness"

mkdir -p "$out/raw" "$out/determinism" "$out/evidence" "$out/recovery"
adb get-state >/dev/null
"$script_dir/deploy.sh" scna "$remote_dir" >"$out/evidence/deploy.log" 2>&1

{
  echo "captured_at=$(date -u +%FT%TZ)"
  echo "matrix=3 masks x 5 q x 2 kv x 2 dimensions x 3 seeds = 180"
  echo "scheduler=adaptive; q<=8:4,q<=16:8,otherwise:16"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.soc.model
  adb shell getprop ro.build.fingerprint
  sha256sum "$project_dir"/artifacts/scna/*
} >"$out/evidence/manifest.txt"
cp "$project_dir/experiment_spec.json" "$out/evidence/experiment_spec.json"

complete_case() {
  [[ -s "$1" ]] && grep -q 'FIG8_ATTENTION_COMPARE .*pass=1' "$1"
}

run_case() {
  local mask="$1" q="$2" kv="$3" dim="$4" seed="$5"
  local log="$out/raw/${mask}_q${q}_kv${kv}_d${dim}_seed${seed}.log"
  if complete_case "$log"; then
    echo "resume: ${log#$project_dir/}"
    return
  fi
  local attempt status
  for attempt in 1 2; do
    set +e
    timeout 180s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' \
      ./htp_ops_test --figure8-attn --mode scna-fp16 --mask-mode '$mask' \
      --qo-len '$q' --kv-len '$kv' --n-heads 12 --n-kv-heads 2 --head-dim '$dim' \
      --warmup 1 --iters 1 --seed '$seed' --compare-reference --numeric-debug --no-events" \
      >"$log" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]] && complete_case "$log"; then
      echo "pass: ${log#$project_dir/}"
      return
    fi
    {
      echo "exit_code=$status"
      tail -100 "$log"
      adb get-state
    } >"$out/recovery/$(basename "${log%.log}")_attempt${attempt}.log" 2>&1 || true
  done
  echo "FAILED: ${log#$project_dir/}" >&2
  return 1
}

adb shell dumpsys thermalservice >"$out/evidence/thermal_before.txt" 2>&1 || true
for mask in full causal padding; do
  for q in 1 4 8 16 32; do
    for kv in 4093 4096; do
      for dim in 64 128; do
        for seed in figure8_fixed 20260810 20260811; do
          run_case "$mask" "$q" "$kv" "$dim" "$seed"
        done
      done
    done
  done
done

for repeat in $(seq 1 10); do
  log="$out/determinism/q32_repeat${repeat}.log"
  if [[ -s "$log" ]] && grep -q 'FIG8_ATTENTION_HOST_TIMING .*phase=measure .*ret=0' "$log"; then
    echo "resume: ${log#$project_dir/}"
    continue
  fi
  timeout 180s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' \
    ./htp_ops_test --figure8-attn --mode scna-fp16 --mask-mode full \
    --qo-len 32 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 \
    --warmup 1 --iters 1 --seed figure8_fixed --no-events" >"$log" 2>&1
  grep -q 'FIG8_ATTENTION_HOST_TIMING .*phase=measure .*ret=0' "$log"
done
adb shell dumpsys thermalservice >"$out/evidence/thermal_after.txt" 2>&1 || true
python3 "$project_dir/tools/analyze_correctness.py" --input "$out" --output "$out/summary.json"

