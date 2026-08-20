#!/usr/bin/env bash
# Matched v79 simulator diagnostic at Qo=32 for short and controlled-long KV.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
source "$script_dir/sim_common.sh"
require_sim_runtime

run_id="${SCNA_RUN_ID:-lut_scna_kv_v79_$(date +%Y%m%dT%H%M%S)}"
run_dir="$project_dir/results/runs/$run_id"
native_source="${SCNA_NATIVE_SOURCE_RUN:-$project_dir/results/runs/scna_combined_resource_v79_20260818_164158}"
mkdir -p "$run_dir/raw/attention" "$run_dir/metrics/attention" "$run_dir/evidence"

[[ -f "$project_dir/artifacts/variants/optimized/scna_sim.so" ]] || "$script_dir/build_all.sh"

run_case() {
  local label="$1" mode="$2" variant="$3" artifact="$4" kv="$5"
  local log="$run_dir/raw/attention/${label}_kv${kv}.log"
  local module="$project_dir/artifacts/variants/$artifact/scna_sim.so"
  local args=(attention --mode "$mode")
  [[ "$mode" == serial ]] && args+=(--variant "$variant")
  args+=(--workers 1 --qo 32 --kv "$kv" --heads 12 --kv-heads 2 --head-dim 128
    --warmup 0 --iters 1 --tail-check 0 --resource-audit 1 --trace-events 8192)
  if [[ -s "$log" ]] && rg -q 'ATTENTION_VERIFY status=PASS' "$log" &&
     rg -q 'SIM_PROCESS_RESULT exit_code=0' "$log"; then
    echo "resume: ${log#$run_dir/}"
    return
  fi
  run_sim_case "$module" "$log" "$run_dir/metrics/attention/${label}_kv${kv}" "${args[@]}"
  rg -q 'ATTENTION_VERIFY status=PASS' "$log"
  rg -q 'SIM_PROCESS_RESULT exit_code=0' "$log"
  echo "done: ${log#$run_dir/}"
}

for kv in 64 512; do
  run_case origin origin none optimized "$kv"
  run_case exp-lut exp-lut none optimized "$kv"
  run_case stage1 serial stage1_dynamic_row stage1_dynamic_row "$kv"
  run_case optimized serial optimized optimized "$kv"
done

{
  echo "schema_version=2"
  echo "scope=Hexagon v79 simulator diagnostic only; not Snapdragon performance"
  echo "captured_at=$(date -u +%FT%TZ)"
  echo "model=v79na_1"
  echo "native_source_run=$native_source"
  "$sim_bin" --version 2>&1 | head -1 || true
  sha256sum "$project_dir/artifacts/variants/optimized/scna_sim.so" \
    "$project_dir/artifacts/variants/stage1_dynamic_row/scna_sim.so"
} > "$run_dir/evidence/manifest.txt"

"$project_dir/.venv/bin/python" "$project_dir/tools/generate_lut_scna_roofline_diagnostic.py" \
  --run-dir "$run_dir" --native-source-run "$native_source"
echo "$run_dir"
