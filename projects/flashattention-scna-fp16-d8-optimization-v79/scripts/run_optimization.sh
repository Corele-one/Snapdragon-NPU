#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
remote_dir="${REMOTE_DIR:-/data/local/tmp/scna_fp16_d8_optimization_v79}"
run_id="${RUN_ID:-$(date +%Y%m%dT%H%M%S)}"
out="$project_dir/results/runs/$run_id"
variants=(stage1_dynamic_row prepare_once_row pair_shared_dynamic pair_static_d8 pair_d8_fma_noinline pair_d8_fma_inline)
all_variants=("${variants[@]}" optimized)
seeds=(figure8_fixed 20260810 20260811)
device="cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test"

mkdir -p "$out/raw/micro" "$out/raw/attention" "$out/raw/diagnostic" "$out/raw/accuracy" "$out/raw/scaling" "$out/static"

complete() { [[ -s "$1" ]] && grep -q "$2" "$1"; }
run_case() {
  local log="$1" marker="$2"; shift 2
  if complete "$log" "$marker"; then echo "resume: ${log#$out/}"; return; fi
  if ! adb shell "$device $*" >"$log" 2>&1; then
    echo "failed: ${log#$out/}" >&2; tail -30 "$log" >&2; return 1
  fi
  complete "$log" "$marker" || { echo "missing marker in ${log#$out/}" >&2; tail -30 "$log" >&2; return 1; }
  echo "done: ${log#$out/}"
}
deploy_variant() {
  local variant="$1"
  local artifact="$project_dir/artifacts/variants/$variant/libhtp_ops_skel.so"
  [[ -f "$artifact" ]] || { echo "missing $artifact" >&2; exit 1; }
  adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >/dev/null
  adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >/dev/null
  echo "deployed: $variant"
}
attention_args() {
  printf '%s' "--figure8-attn --mode $1 --scna-variant $2 --workers $3 --scna-width 8 --mask-mode $4 --qo-len $5 --kv-len $6 --n-heads 12 --n-kv-heads 2 --head-dim $7 --warmup $8 --iters $9 --seed ${10} --no-events"
}

"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" --mode ping --scna-variant stage1_dynamic_row >"$out/raw/deploy.log" 2>&1
{
  echo "run_id=$run_id"; echo "date_utc=$(date -u +%FT%TZ)"; echo "spec_sha256=$(sha256sum "$project_dir/experiment_spec.json" | awk '{print $1}')"
  adb get-serialno; adb shell getprop ro.product.model; adb shell getprop ro.build.fingerprint; adb shell getprop ro.board.platform
  for variant in "${all_variants[@]}"; do sha256sum "$project_dir/artifacts/variants/$variant/libhtp_ops_skel.so"; done
} >"$out/manifest.txt"

# Micro samples. Each pilot calibrates one independent sample to >=50 ms.
for variant in "${variants[@]}"; do
  deploy_variant "$variant"
  pilot="$out/raw/micro/${variant}_pilot.log"
  run_case "$pilot" "SCNA_EXP_BENCH" "--scna-exp-bench --scna-variant $variant --scna-width 8 --warmup 5 --iters 200000"
  elapsed="$(sed -n 's/.*pair_elapsed_us=\([0-9][0-9]*\).*/\1/p' "$pilot" | tail -1)"
  [[ -n "$elapsed" && "$elapsed" -gt 0 ]] || { echo "micro calibration failed: $variant" >&2; exit 1; }
  calibrated=$(( (50000 * 200000 + elapsed - 1) / elapsed )); (( calibrated < 200000 )) && calibrated=200000
  for sample in $(seq 1 30); do
    run_case "$out/raw/micro/${variant}_sample${sample}.log" "SCNA_EXP_BENCH" \
      "--scna-exp-bench --scna-variant $variant --scna-width 8 --warmup 5 --iters $calibrated"
  done
done

# Five balanced sessions: rotate variant order, fixed 5 warmup + 20 measured.
for session in $(seq 1 5); do
  seed="${seeds[$(((session-1)%3))]}"
  for offset in "${!variants[@]}"; do
    index=$(((offset+session-1)%${#variants[@]})); variant="${variants[$index]}"; deploy_variant "$variant"
    for q in 1 4 8 16 32; do
      args="$(attention_args scna-fp16 "$variant" 1 full "$q" 4096 128 5 20 "$seed")"
      run_case "$out/raw/attention/${variant}_q${q}_s${session}.log" "phase=measure iteration=19 .*ret=0" "$args"
    done
  done
done

# Select call policy from paired single-worker DSP totals, then rebuild only optimized if required.
policy="$(python3 "$project_dir/tools/select_inline_policy.py" --run-dir "$out")"
if [[ "$policy" == inline ]]; then
  "$script_dir/build.sh" --variant optimized --optimized-inline 1 --dsp-only
  dsp_ship=("$htp_dir"/hexagon_ReleaseG_toolv*_v79/ship)
  cp -f "${dsp_ship[0]}/libhtp_ops_skel.so" "$project_dir/artifacts/variants/optimized/libhtp_ops_skel.so"
  printf 'variant=optimized\nbuild_id=6\noptimized_inline=1\n' >"$project_dir/artifacts/variants/optimized/build_id.txt"
  sha256sum "$project_dir/artifacts/variants/optimized/libhtp_ops_skel.so" >"$project_dir/artifacts/variants/optimized/sha256.txt"
fi
echo "inline_policy=$policy" >> "$out/manifest.txt"
echo "optimized_final_sha256=$(sha256sum "$project_dir/artifacts/variants/optimized/libhtp_ops_skel.so" | awk '{print $1}')" >> "$out/manifest.txt"

# Final optimized artifact gets the same 30-sample micro protocol.
deploy_variant optimized
pilot="$out/raw/micro/optimized_pilot.log"
run_case "$pilot" "SCNA_EXP_BENCH" "--scna-exp-bench --scna-variant optimized --scna-width 8 --warmup 5 --iters 200000"
elapsed="$(sed -n 's/.*pair_elapsed_us=\([0-9][0-9]*\).*/\1/p' "$pilot" | tail -1)"
[[ -n "$elapsed" && "$elapsed" -gt 0 ]] || { echo "micro calibration failed: optimized" >&2; exit 1; }
calibrated=$(( (50000 * 200000 + elapsed - 1) / elapsed )); (( calibrated < 200000 )) && calibrated=200000
for sample in $(seq 1 30); do
  run_case "$out/raw/micro/optimized_sample${sample}.log" "SCNA_EXP_BENCH" \
    "--scna-exp-bench --scna-variant optimized --scna-width 8 --warmup 5 --iters $calibrated"
done

# Optimized main matrix.
deploy_variant optimized
for session in $(seq 1 5); do
  seed="${seeds[$(((session-1)%3))]}"
  for q in 1 4 8 16 32; do
    args="$(attention_args scna-fp16 optimized 1 full "$q" 4096 128 5 20 "$seed")"
    run_case "$out/raw/attention/optimized_q${q}_s${session}.log" "phase=measure iteration=19 .*ret=0" "$args"
  done
done

# Horizontal baselines use the stage1 library only as a build carrier; SCNA is disabled.
deploy_variant stage1_dynamic_row
for session in $(seq 1 5); do
  seed="${seeds[$(((session-1)%3))]}"
  for q in 1 4 8 16 32; do
    for mode in baseline lut-exp; do
      args="$(attention_args "$mode" stage1_dynamic_row 1 full "$q" 4096 128 5 20 "$seed")"
      run_case "$out/raw/attention/${mode}_q${q}_s${session}.log" "phase=measure iteration=19 .*ret=0" "$args"
    done
  done
done

# Low-frequency diagnostic replay (timers are excluded from the main conclusion).
for variant in "${all_variants[@]}"; do
  deploy_variant "$variant"
  args="$(attention_args scna-fp16 "$variant" 1 full 4 4096 128 1 1 figure8_fixed)"
  run_case "$out/raw/diagnostic/${variant}_q4.log" "FIG8_ATTENTION_TIMERS" "$args --events"
done

# One low-frequency multi-worker event replay supplies an auditable worker
# timeline.  It is diagnostic only and never enters the main timing result.
deploy_variant optimized
args="$(attention_args scna-fp16 optimized auto full 32 4096 128 1 1 figure8_fixed)"
run_case "$out/raw/diagnostic/optimized_q32_auto_timeline.log" "FIG8_ATTENTION_EVENT_COUNT" "$args --events"

# Full registered correctness pressure set.
for variant in "${all_variants[@]}"; do
  deploy_variant "$variant"
  for mask in full causal padding; do for q in 1 4; do for kv in 4093 4096; do for dim in 64 128; do for seed in "${seeds[@]}"; do
    log="$out/raw/accuracy/${variant}_${mask}_q${q}_kv${kv}_d${dim}_seed${seed}.log"
    args="$(attention_args scna-fp16 "$variant" 1 "$mask" "$q" "$kv" "$dim" 1 1 "$seed")"
    run_case "$log" "FIG8_ATTENTION_COMPARE .*pass=1" "$args --compare-reference --numeric-debug"
  done; done; done; done; done
done

# Unified worker scaling, including auto. Origin/EXP-LUT carried by stage1; optimized uses its own artifact.
for mode in baseline lut-exp; do
  deploy_variant stage1_dynamic_row
  for workers in 1 2 3 4 5 6 auto; do
    args="$(attention_args "$mode" stage1_dynamic_row "$workers" full 32 4096 128 5 20 figure8_fixed)"
    run_case "$out/raw/scaling/${mode}_w${workers}.log" "phase=measure iteration=19 .*ret=0" "$args"
  done
done
deploy_variant optimized
for workers in 1 2 3 4 5 6 auto; do
  args="$(attention_args scna-fp16 optimized "$workers" full 32 4096 128 5 20 figure8_fixed)"
  run_case "$out/raw/scaling/optimized_w${workers}.log" "phase=measure iteration=19 .*ret=0" "$args"
done

python3 "$project_dir/tools/collect_static_metrics.py" --project "$project_dir" --out-dir "$out/static"
python3 "$project_dir/tools/generate_optimization_report.py" --run-dir "$out" --spec "$project_dir/experiment_spec.json"
python3 "$project_dir/tools/verify_experiment.py" --run-dir "$out" --project "$project_dir"
echo "completed: $out/REPORT.md"
