#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
remote_dir="${SCNA_REMOTE_DIR:-/data/local/tmp/scna_optimization_v79}"
out="$project_dir/results/v79"
sessions=5
warmup=5
iters=20

mkdir -p "$out"/{baseline,lut_exp,scna,diagnostic,evidence,recovery}

mode_name() {
  case "$1" in
    baseline) echo baseline ;;
    lut_exp) echo lut-exp ;;
    scna) echo scna-fp16 ;;
    *) return 2 ;;
  esac
}

bundle_name() {
  [[ "$1" == scna ]] && echo scna || echo reference
}

is_complete() {
  local log="$1" mode="$2"
  [[ -s "$log" ]] || return 1
  [[ "$(grep -c "FIG8_ATTENTION_HOST_TIMING mode=$mode .*phase=measure .*ret=0" "$log" || true)" -eq "$iters" ]]
}

run_one() {
  local label="$1" q="$2" session="$3" mode log attempt status
  mode="$(mode_name "$label")"
  log="$out/$label/q${q}_s${session}.log"
  if is_complete "$log" "$mode"; then
    echo "resume: ${log#$project_dir/}"
    return
  fi
  for attempt in 1 2; do
    set +e
    timeout 180s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' \
      ./htp_ops_test --figure8-attn --mode '$mode' --qo-len '$q' --kv-len 4096 \
      --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup '$warmup' --iters '$iters' --no-events" \
      >"$log" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]] && is_complete "$log" "$mode"; then
      echo "done: ${log#$project_dir/}"
      return
    fi
    {
      echo "exit_code=$status"
      tail -100 "$log"
      adb get-state
    } >"$out/recovery/${label}_q${q}_s${session}_attempt${attempt}.log" 2>&1 || true
  done
  echo "FAILED: ${log#$project_dir/}" >&2
  return 1
}

adb get-state >/dev/null
{
  echo "captured_at=$(date -u +%FT%TZ)"
  echo "sessions=$sessions warmup=$warmup iters=$iters expected_samples=1200"
  echo "baseline_source=reference/flashattention/htp-ops-lib-main"
  echo "scna_source=src/htp-ops-lib-main"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.soc.model
  adb shell getprop ro.build.fingerprint
  sha256sum "$project_dir"/artifacts/reference/* "$project_dir"/artifacts/scna/*
} >"$out/evidence/manifest.txt"
cp "$project_dir/experiment_spec.json" "$out/evidence/experiment_spec.json"
cp "$0" "$out/evidence/run_benchmark.sh"
adb shell dumpsys thermalservice >"$out/evidence/thermal_before.txt" 2>&1 || true

# Rotate the primary mode per session: B-L-S, L-S-B, S-B-L, ...
orders=("baseline lut_exp scna" "lut_exp scna baseline" "scna baseline lut_exp")
current_bundle=""
for session in $(seq 1 "$sessions"); do
  read -r -a labels <<<"${orders[$(((session - 1) % 3))]}"
  for label in "${labels[@]}"; do
    wanted_bundle="$(bundle_name "$label")"
    if [[ "$wanted_bundle" != "$current_bundle" ]]; then
      "$script_dir/deploy.sh" "$wanted_bundle" "$remote_dir" \
        >>"$out/evidence/deploy_session${session}.log" 2>&1
      current_bundle="$wanted_bundle"
    fi
    for q in 4 8 16 32; do
      run_one "$label" "$q" "$session"
    done
  done
done

# Diagnostic runs are intentionally separate from ranking samples and retain events.
for label in baseline lut_exp scna; do
  wanted_bundle="$(bundle_name "$label")"
  if [[ "$wanted_bundle" != "$current_bundle" ]]; then
    "$script_dir/deploy.sh" "$wanted_bundle" "$remote_dir" >>"$out/evidence/deploy_diagnostic.log" 2>&1
    current_bundle="$wanted_bundle"
  fi
  mode="$(mode_name "$label")"
  log="$out/diagnostic/${label}_q32.log"
  timeout 180s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' \
    ./htp_ops_test --figure8-attn --mode '$mode' --qo-len 32 --kv-len 4096 \
    --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup 1 --iters 3" >"$log" 2>&1
done

adb shell dumpsys thermalservice >"$out/evidence/thermal_after.txt" 2>&1 || true
python3 "$project_dir/tools/analyze_results.py" --input "$out"

