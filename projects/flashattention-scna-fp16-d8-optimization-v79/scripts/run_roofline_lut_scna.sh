#!/usr/bin/env bash
# Matched real-device remeasurement for LUT-EXP, SCNA and their hardware roofs.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
report_python="${ROOFLINE_PYTHON:-$project_dir/.venv-roofline/bin/python}"
if ! "$report_python" -c 'import matplotlib, seaborn' >/dev/null 2>&1; then
  report_python="$project_dir/../flashattention-scna-serial-v79-simulator/.venv/bin/python"
fi
if ! "$report_python" -c 'import matplotlib, seaborn' >/dev/null 2>&1; then
  echo "Matplotlib/Seaborn environment unavailable; set ROOFLINE_PYTHON" >&2
  exit 1
fi
remote_dir="${REMOTE_DIR:-/data/local/tmp/scna_fp16_d8_roofline_v79}"
run_id="${RUN_ID:-lut_scna_v79_$(date +%Y%m%dT%H%M%S)}"
out="$project_dir/results/runs/$run_id"
device="cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test"
seeds=(figure8_fixed 20260810 20260811)
comparisons=(pair_static_d8 optimized optimized_qf16_tree optimized_piecewise_d8 lut-exp)

mkdir -p "$out/raw/roofline" "$out/raw/micro" "$out/raw/attention" \
  "$out/raw/accuracy" "$out/raw/scaling" "$out/raw/lut_scna_diagnostic" \
  "$out/raw/recovery" "$out/static" "$out/evidence"

complete() { [[ -s "$1" ]] && grep -Eq "$2" "$1"; }

run_remote() {
  local log="$1" marker="$2"; shift 2
  if complete "$log" "$marker"; then echo "resume: ${log#$out/}"; return 0; fi
  local attempt status recovery
  for attempt in 1 2; do
    set +e
    timeout 40s adb shell "$device $*" >"$log" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]] && complete "$log" "$marker"; then
      echo "done: ${log#$out/}"
      return 0
    fi
    recovery="$out/raw/recovery/$(basename "${log%.log}")_attempt${attempt}.log"
    {
      echo "failed_log=${log#$out/}"
      echo "attempt=$attempt"
      echo "exit_code=$status"
      echo "expected_marker=$marker"
      adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./scna_env_smoke"
    } >"$recovery" 2>&1 || true
  done
  echo "unavailable after retry: ${log#$out/}" >&2
  return 1
}

deploy_artifact() {
  local artifact_name="$1"
  local artifact="$project_dir/artifacts/variants/$artifact_name/libhtp_ops_skel.so"
  [[ -f "$artifact" ]] || { echo "missing artifact: $artifact" >&2; exit 1; }
  adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >/dev/null
  adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >/dev/null
}

runtime_variant() {
  case "$1" in
    optimized_qf16_tree|optimized_piecewise_d8) echo optimized ;;
    lut-exp) echo stage1_dynamic_row ;;
    *) echo "$1" ;;
  esac
}

runtime_mode() { [[ "$1" == lut-exp ]] && echo lut-exp || echo scna-fp16; }

attention_args() {
  local label="$1" workers="$2" q="$3" kv="$4" warmup="$5" iters="$6" seed="$7" mask="${8:-full}" dim="${9:-128}"
  printf '%s' "--figure8-attn --mode $(runtime_mode "$label") --scna-variant $(runtime_variant "$label") --workers $workers --scna-width 8 --mask-mode $mask --qo-len $q --kv-len $kv --n-heads 12 --n-kv-heads 2 --head-dim $dim --warmup $warmup --iters $iters --seed $seed --no-events"
}

adb get-state >/dev/null
if [[ ! -f "$project_dir/artifacts/variants/optimized_piecewise_d8/libhtp_ops_skel.so" ]]; then
  "$script_dir/build_all_variants.sh"
fi
"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" --mode ping --scna-variant stage1_dynamic_row \
  >"$out/raw/deploy.log" 2>&1

{
  echo "schema_version=2"
  echo "run_id=$run_id"
  echo "captured_at=$(date -u +%FT%TZ)"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.product.device
  adb shell getprop ro.board.platform
  adb shell getprop ro.soc.model
  adb shell getprop ro.build.fingerprint
  for artifact in stage1_dynamic_row pair_static_d8 optimized optimized_qf16_tree optimized_piecewise_d8; do
    sha256sum "$project_dir/artifacts/variants/$artifact/libhtp_ops_skel.so"
  done
} >"$out/evidence/device_manifest.txt"
cp "$0" "$out/evidence/run_roofline_lut_scna.sh"

deploy_artifact stage1_dynamic_row
for sample in $(seq 1 30); do
  run_remote "$out/raw/roofline/hmx_fp16_sample${sample}.log" '^2,hmx_fp16,' \
    --roofline-hmx-fp16-bench --target-ms 50 --warmup 2 --iters 1 || true
  run_remote "$out/raw/roofline/hvx_fp16_sample${sample}.log" '^2,hvx_fp16,' \
    --roofline-hvx-fp16-bench --target-ms 50 --warmup 2 --iters 1 || true
  run_remote "$out/raw/roofline/hvx_v79_peak_sample${sample}.log" '^2,hvx_v79_peak,' \
    --roofline-hvx-v79-peak-bench --target-ms 150 --warmup 2 --iters 1 || true
  for distribution in dense attention random; do
    run_remote "$out/raw/roofline/lut_${distribution}_calibrated_sample${sample}.log" '^2,lut_exp,' \
      --lut-exp-bench --distribution "$distribution" --target-ms 200 --warmup 4 --iters 1 \
      --qo-len 32 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 --seed figure8_fixed || true
  done
  # The first pilot showed that 200 iterations left the fastest VTCM rows below
  # 50 ms.  6000 keeps every bandwidth row above 50 ms while the independent
  # request remains below the five-second ceiling.
  run_remote "$out/raw/roofline/bandwidth_calibrated_sample${sample}.log" '^mode,kind' \
    --roofline-bandwidth-bench --bench-bytes 1048576 --warmup 5 --iters 6000 || true
done

for label in pair_static_d8 optimized optimized_qf16_tree optimized_piecewise_d8; do
  deploy_artifact "$label"
  variant="$(runtime_variant "$label")"
  pilot="$out/raw/micro/${label}_pilot.log"
  run_remote "$pilot" 'SCNA_EXP_BENCH' --scna-exp-bench --scna-variant "$variant" --scna-width 8 --warmup 5 --iters 200000
  elapsed="$(sed -n 's/.*pair_elapsed_us=\([0-9][0-9]*\).*/\1/p' "$pilot" | tail -1)"
  [[ -n "$elapsed" && "$elapsed" -gt 0 ]] || { echo "micro calibration failed: $label" >&2; exit 1; }
  calibrated=$(( (80000 * 200000 + elapsed - 1) / elapsed ))
  (( calibrated < 200000 )) && calibrated=200000
  for sample in $(seq 1 30); do
    run_remote "$out/raw/micro/${label}_sample${sample}.log" 'SCNA_EXP_BENCH' \
      --scna-exp-bench --scna-variant "$variant" --scna-width 8 --warmup 5 --iters "$calibrated"
  done
done

for session in $(seq 1 5); do
  seed="${seeds[$(((session - 1) % ${#seeds[@]}))]}"
  for offset in "${!comparisons[@]}"; do
    index=$(((offset + session - 1) % ${#comparisons[@]}))
    label="${comparisons[$index]}"
    carrier="$label"; [[ "$label" == lut-exp ]] && carrier=stage1_dynamic_row
    deploy_artifact "$carrier"
    for q in 1 4 8 16 32; do
      args="$(attention_args "$label" 1 "$q" 4096 5 20 "$seed")"
      run_remote "$out/raw/attention/${label}_q${q}_s${session}.log" 'phase=measure iteration=19 .*ret=0' "$args"
    done
  done
done

for label in optimized_qf16_tree optimized_piecewise_d8; do
  deploy_artifact "$label"
  for mask in full causal padding; do
    for q in 1 4; do
      for kv in 4093 4096; do
        for dim in 64 128; do
          for seed in "${seeds[@]}"; do
            args="$(attention_args "$label" 1 "$q" "$kv" 1 1 "$seed" "$mask" "$dim")"
            log="$out/raw/accuracy/${label}_${mask}_q${q}_kv${kv}_d${dim}_seed${seed}.log"
            run_remote "$log" 'FIG8_ATTENTION_COMPARE .*pass=1' "$args --compare-reference --numeric-debug"
          done
        done
      done
    done
  done
done

for kv in 64 4096; do
  for label in "${comparisons[@]}"; do
    carrier="$label"; [[ "$label" == lut-exp ]] && carrier=stage1_dynamic_row
    deploy_artifact "$carrier"
    args="$(attention_args "$label" 1 32 "$kv" 1 1 figure8_fixed)"
    run_remote "$out/raw/lut_scna_diagnostic/${label}_kv${kv}.log" 'FIG8_ATTENTION_TIMERS' "$args --events"
  done
done

for label in "${comparisons[@]}"; do
  carrier="$label"; [[ "$label" == lut-exp ]] && carrier=stage1_dynamic_row
  deploy_artifact "$carrier"
  for workers in 1 2 3 4 5 6 auto; do
    args="$(attention_args "$label" "$workers" 32 4096 5 20 figure8_fixed)"
    run_remote "$out/raw/scaling/${label}_w${workers}.log" 'phase=measure iteration=19 .*ret=0' "$args"
  done
done

"$report_python" "$project_dir/tools/verify_scna_piecewise_fp16.py" --json-out "$out/evidence/scna_fp16_exhaustive.json"
"$report_python" "$project_dir/tools/collect_static_metrics.py" --project "$project_dir" --out-dir "$out/static"
"$report_python" "$project_dir/tools/generate_roofline_lut_scna_report.py" --run-dir "$out" --project "$project_dir"
echo "completed: $out/ROOFLINE_LUT_VS_SCNA_REPORT_ZH.md"
