#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/sim_common.sh"
require_sim_runtime
run_id="${SCNA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
run_dir="$project_dir/results/runs/$run_id"
module="$project_dir/artifacts/variants/optimized/scna_sim.so"
[[ -f "$module" ]] || { echo "Build optimized first: scripts/build_all.sh --variant optimized" >&2; exit 1; }
run_sim_case "$module" "$run_dir/raw/probe.log" "$run_dir/metrics/probe" probe
rg 'SIM_CAPABILITY|SIM_PROCESS_RESULT' "$run_dir/raw/probe.log"
