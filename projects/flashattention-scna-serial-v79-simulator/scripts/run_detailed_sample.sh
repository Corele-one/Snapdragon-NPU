#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/sim_common.sh"
require_sim_runtime
run_id="${SCNA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
run_dir="$project_dir/results/runs/$run_id"
module="$project_dir/artifacts/variants/optimized/scna_sim.so"
nm_bin="$tools_root/Tools/bin/hexagon-nm"
read -r start size _ < <("$nm_bin" -S "$module" | awk '$4 == "pair_static_d8_fma_noinline" {print $1, $2, $3}')
[[ -n ${start:-} && -n ${size:-} ]] || { echo "Unable to locate optimized SCNA symbol" >&2; exit 1; }
end="$(printf '%x' "$((16#$start + 16#$size - 4))")"
export SCNA_SIM_TIMING=1 SCNA_SIM_DETAILED=1 SCNA_SIM_PCFILTER="0x${start}-0x${end}"
export SCNA_SIM_CASE_TIMEOUT_SEC="${SCNA_SIM_CASE_TIMEOUT_SEC:-300}"
run_sim_case "$module" "$run_dir/raw/detailed/optimized_micro_timing.log" \
  "$run_dir/metrics/detailed/optimized_micro_timing" micro --variant 6 --warmup 0 --iters 1
rg 'SCNA_SIM_RESULT|Total: Insns|SIM_PROCESS_RESULT' "$run_dir/raw/detailed/optimized_micro_timing.log"
