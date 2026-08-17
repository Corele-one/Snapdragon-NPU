#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/sim_common.sh"
require_sim_runtime
run_id="${SCNA_RUN_ID:?SCNA_RUN_ID must name the all-serial run}"
run_dir="$project_dir/results/runs/$run_id"
nm_bin="$tools_root/Tools/bin/hexagon-nm"
variants=(stage1_dynamic_row prepare_once_row pair_shared_dynamic pair_static_d8 pair_d8_fma_noinline pair_d8_fma_inline optimized)
symbols=(stage1_dynamic_row prepared_dynamic_row pair_shared_dynamic_qf16 pair_static_d8_qf16 pair_static_d8_fma_noinline hvx_scna_exp2_pair_vhf pair_static_d8_fma_noinline)

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"; symbol="${symbols[$index]}"
  completed_log="$run_dir/raw/detailed/${variant}.log"
  if [[ -f "$completed_log" ]] && grep -q '^RUNTIME_FILTER_RESULT ' "$completed_log"; then
    echo "Keeping completed runtime-filter sample: $variant"
    continue
  fi
  module="$project_dir/artifacts/variants/$variant/scna_sim.so"
  source_log="$(find "$run_dir/raw/attention" -name "serial_${variant}_w1_*.log" -print -quit)"
  base="$(awk '/load 0x0 ->/{print $4; exit}' "$source_log")"
  # Read nm to EOF: under pipefail, an early awk exit can surface hexagon-nm's SIGPIPE as exit 74.
  symbol_line="$($nm_bin -S -C --defined-only "$module" | awk -v name="$symbol" '$4 == name && !found {print; found=1}')"
  if [[ -z "$base" || -z "$symbol_line" ]]; then
    printf 'RUNTIME_FILTER_RESULT status=FAIL variant=%s reason=missing_base_or_symbol base=%s symbol=%s\n' "$variant" "${base:-none}" "$symbol" \
      > "$run_dir/raw/detailed/${variant}.log"
    continue
  fi
  address="$(awk '{print $1}' <<<"$symbol_line")"; size="$(awk '{print $2}' <<<"$symbol_line")"
  base_hex="${base#0x}"
  start="$(printf '0x%x' "$((16#$base_hex + 16#$address))")"
  end="$(printf '0x%x' "$((16#$base_hex + 16#$address + 16#$size - 4))")"
  export SCNA_SIM_TIMING=1 SCNA_SIM_DETAILED=1 SCNA_SIM_PCFILTER="${start}-${end}"
  metrics="$run_dir/metrics/detailed/$variant"
  log="$run_dir/raw/detailed/${variant}.log"
  mkdir -p "$(dirname -- "$log")"
  set +e
  run_sim_case "$module" "$log" "$metrics" micro --warmup 0 --iters 1
  rc=$?
  set -e
  printf 'RUNTIME_FILTER_RESULT status=%s variant=%s symbol=%s link_address=0x%s size=0x%s runtime_base=%s pc_start=%s pc_end=%s exit_code=%d\n' \
    "$([[ $rc -eq 0 ]] && echo PASS || echo FAIL)" "$variant" "$symbol" "$address" "$size" "$base" "$start" "$end" "$rc" >> "$log"
done
