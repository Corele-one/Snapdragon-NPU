#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/sim_common.sh"
[[ $# -eq 2 && $1 == --skel ]] || { echo "Usage: $0 --skel /path/to/lib*_skel.so" >&2; exit 2; }
skel="$(realpath -- "$2")"
[[ -f "$skel" ]] || { echo "Missing skel: $skel" >&2; exit 1; }
run_id="${SCNA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
evidence="$project_dir/results/runs/$run_id/evidence"
mkdir -p "$evidence"
set +e
run_sim_case "$skel" "$evidence/fastrpc_skel_load.log" "$evidence/fastrpc_skel_metrics"
launcher_rc=$?
set -e
nm_bin="$tools_root/Tools/bin/hexagon-nm"
if "$nm_bin" "$skel" | rg ' main$' > "$evidence/original_skel_main_symbol.txt"; then
  main_symbol_found=1
else
  main_symbol_found=0
fi
printf 'skel=%s\nlauncher_rc=%d\nmain_symbol_found=%d\n' \
  "$skel" "$launcher_rc" "$main_symbol_found" > "$evidence/fastrpc_boundary_summary.txt"
rg -n -i 'main|load|error|SIM_PROCESS_RESULT' "$evidence/fastrpc_skel_load.log" || true
cat "$evidence/fastrpc_boundary_summary.txt"
[[ $launcher_rc -ne 0 && $main_symbol_found -eq 0 ]]
