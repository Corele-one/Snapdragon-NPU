#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/sim_common.sh"
require_sim_runtime
kind="${1:---smoke}"
[[ "$kind" == --smoke || "$kind" == --sweep || "$kind" == --all-serial ]] || {
  echo "Usage: $0 --smoke|--sweep|--all-serial" >&2; exit 2;
}
run_id="${SCNA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
run_dir="$project_dir/results/runs/$run_id"
failed=0

run_attention_case() {
  local mode="$1" variant="$2" workers="$3" qo="$4" kv="$5" heads="$6" kv_heads="$7" dim="$8" warmup="$9"
  shift 9
  local iters="$1" tail="$2"
  local artifact_variant="$variant"
  [[ "$mode" == origin || "$mode" == exp-lut ]] && artifact_variant=optimized
  [[ "$mode" == stage1 ]] && artifact_variant=stage1_dynamic_row
  [[ "$mode" == optimized ]] && artifact_variant=optimized
  local module="$project_dir/artifacts/variants/$artifact_variant/scna_sim.so"
  local worker_label="w${workers}"
  local label="${mode}_${variant}_${worker_label}_q${qo}_kv${kv}_h${heads}_kh${kv_heads}_d${dim}"
  [[ -f "$module" ]] || { echo "Missing artifact: $module" >&2; return 1; }
  local args=(attention --mode "$mode")
  [[ "$mode" == serial ]] && args+=(--variant "$variant")
  args+=(--workers "$workers" --qo "$qo" --kv "$kv" --heads "$heads" --kv-heads "$kv_heads"
    --head-dim "$dim" --warmup "$warmup" --iters "$iters" --tail-check "$tail")
  run_sim_case "$module" "$run_dir/raw/attention/${label}.log" "$run_dir/metrics/attention/$label" \
    "${args[@]}"
}

record_case() { run_attention_case "$@" || failed=1; }

if [[ "$kind" == --smoke ]]; then
  record_case optimized optimized 1 1 64 2 1 64 1 1 0
  record_case optimized optimized 1 3 65 2 1 64 1 1 1
elif [[ "$kind" == --sweep ]]; then
  for mode in origin exp-lut stage1 optimized; do
    for qo in 1 4 8 16 32; do
      variant=none
      [[ "$mode" == stage1 ]] && variant=stage1_dynamic_row
      [[ "$mode" == optimized ]] && variant=optimized
      record_case "$mode" "$variant" 1 "$qo" 64 12 2 128 1 5 0
    done
  done
else
  variants=(stage1_dynamic_row prepare_once_row pair_shared_dynamic pair_static_d8 pair_d8_fma_noinline pair_d8_fma_inline optimized)
  for mode in origin exp-lut; do
    for qo in 1 4 8 16 32; do
      record_case "$mode" none 1 "$qo" 64 12 2 128 1 5 0
    done
  done
  for variant in "${variants[@]}"; do
    for qo in 1 4 8 16 32; do
      record_case serial "$variant" 1 "$qo" 64 12 2 128 1 5 0
    done
    record_case serial "$variant" 1 3 65 2 1 64 1 1 1
  done
  for qo in 1 4 8 16 32; do
    record_case serial optimized auto "$qo" 64 12 2 128 1 5 0
  done
fi
rg 'ATTENTION_(SMOKE_RESULT|VERIFY|TIMER)|SIM_PROCESS_RESULT' "$run_dir/raw/attention"
exit "$failed"
