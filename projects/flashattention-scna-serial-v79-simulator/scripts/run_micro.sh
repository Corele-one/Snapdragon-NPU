#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/sim_common.sh"
require_sim_runtime
variants=(stage1_dynamic_row prepare_once_row pair_shared_dynamic pair_static_d8 pair_d8_fma_noinline pair_d8_fma_inline optimized)
selected=()
if [[ ${1:-} == --all ]]; then selected=("${variants[@]}"); shift; elif [[ ${1:-} == --variant && $# -ge 2 ]]; then selected=("$2"); shift 2; else echo "Usage: $0 --all|--variant NAME" >&2; exit 2; fi
warmup="${SCNA_MICRO_WARMUP:-5}"
iters="${SCNA_MICRO_ITERS:-1000}"
processes="${SCNA_MICRO_PROCESSES:-2}"
run_id="${SCNA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
run_dir="$project_dir/results/runs/$run_id"
for variant in "${selected[@]}"; do
  id=-1
  for index in "${!variants[@]}"; do [[ "${variants[$index]}" == "$variant" ]] && id="$index"; done
  [[ $id -ge 0 ]] || { echo "Unknown variant: $variant" >&2; exit 2; }
  module="$project_dir/artifacts/variants/$variant/scna_sim.so"
  [[ -f "$module" ]] || { echo "Missing artifact: $module" >&2; exit 1; }
  for ((process=0; process<processes; ++process)); do
    run_sim_case "$module" "$run_dir/raw/micro/${variant}_p${process}.log" \
      "$run_dir/metrics/micro/${variant}_p${process}" micro --variant "$id" --warmup "$warmup" --iters "$iters"
  done
done
rg 'SCNA_SIM_RESULT|SIM_PROCESS_RESULT' "$run_dir/raw/micro"
