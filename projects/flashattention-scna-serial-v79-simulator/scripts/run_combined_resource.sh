#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
source "$script_dir/sim_common.sh"
require_sim_runtime

source_run="$project_dir/results/runs/scna_sim_v79_20260813_r1"
primary_run="$project_dir/results/runs/scna_all_serial_v79_20260813_r1"
run_id="${SCNA_RUN_ID:-scna_combined_resource_v79_$(date +%Y%m%d_%H%M%S)}"
run_dir="$project_dir/results/runs/$run_id"
variants=(stage1_dynamic_row prepare_once_row pair_shared_dynamic pair_static_d8 pair_d8_fma_noinline pair_d8_fma_inline optimized)
mkdir -p "$run_dir/raw/resource" "$run_dir/raw/native" "$run_dir/metrics/native" "$run_dir/evidence"

{
  printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'uname='; uname -a
  printf 'cpu_model='; awk -F: '/model name/{sub(/^ /,"",$2); print $2; exit}' /proc/cpuinfo
  printf 'logical_cpus='; getconf _NPROCESSORS_ONLN
  printf 'memory='; awk '/MemTotal/{print $2 " " $3}' /proc/meminfo
  printf 'simulator='; "$sim_bin" --version 2>&1 | head -1 || true
  printf 'sdk_root=%s\ntools_root=%s\nmodel=v79na_1\n' "$sdk_root" "$tools_root"
} > "$run_dir/evidence/environment.txt"
sha256sum "$source_run/summary.json" "$primary_run/summary_all_serial.json" \
  "$source_run/manifest.json" "$primary_run/manifest.json" > "$run_dir/evidence/source_hashes.sha256"

"$script_dir/build_all.sh"

run_audit() {
  local mode="$1" variant="$2" artifact="$3"
  local module="$project_dir/artifacts/variants/$artifact/scna_sim.so"
  local log="$run_dir/raw/resource/${mode}_${variant}_q32.log"
  local args=(attention --mode "$mode")
  [[ "$mode" == serial ]] && args+=(--variant "$variant")
  args+=(--workers 1 --qo 32 --kv 64 --heads 12 --kv-heads 2 --head-dim 128
    --warmup 0 --iters 1 --resource-audit 1 --trace-events 4096)
  run_sim_case "$module" "$log" "$run_dir/metrics/resource/${mode}_${variant}" attention \
    "${args[@]:1}"
}

run_audit origin none optimized
run_audit exp-lut none optimized
for variant in "${variants[@]}"; do run_audit serial "$variant" "$variant"; done

optimized_module="$project_dir/artifacts/variants/optimized/scna_sim.so"
optimized_log="$run_dir/raw/resource/serial_optimized_q32.log"
nm_bin="$tools_root/Tools/bin/hexagon-nm"
base="$(awk '/load 0x0 ->/{print $4; exit}' "$optimized_log")"
symbol_line="$($nm_bin -S -C --defined-only "$optimized_module" | awk '$4 == "simple_flash_attn_f16_core" && !found {print; found=1}')"
if [[ -n "$base" && -n "$symbol_line" ]]; then
  address="$(awk '{print $1}' <<<"$symbol_line")"; size="$(awk '{print $2}' <<<"$symbol_line")"
  base_hex="${base#0x}"
  start="$(printf '0x%x' "$((16#$base_hex + 16#$address))")"
  end="$(printf '0x%x' "$((16#$base_hex + 16#$address + 16#$size - 4))")"
  # Minimal PC trace is still address-resolved evidence and avoids the full
  # register dump emitted by --pctrace on every packet.
  export SCNA_SIM_PCFILTER="${start}-${end}" SCNA_SIM_MEMTRACE=1 SCNA_SIM_COPROCTRACE=1 SCNA_SIM_PCTRACE=min
  set +e
  run_sim_case "$optimized_module" "$run_dir/raw/native/optimized_pcfilter_trace.log" \
    "$run_dir/metrics/native/pcfilter" attention --mode serial --variant optimized --workers 1 \
    --qo 32 --kv 64 --heads 12 --kv-heads 2 --head-dim 128 --warmup 0 --iters 1
  filtered_rc=$?
  set -e
  unset SCNA_SIM_PCFILTER SCNA_SIM_MEMTRACE SCNA_SIM_COPROCTRACE SCNA_SIM_PCTRACE
  printf 'pc_start=%s\npc_end=%s\nruntime_base=%s\nlink_address=0x%s\nsymbol_size=0x%s\nexit_code=%d\n' \
    "$start" "$end" "$base" "$address" "$size" "$filtered_rc" > "$run_dir/evidence/native_pc_range.txt"
else
  printf 'status=UNAVAILABLE\nreason=missing_runtime_base_or_symbol\n' > "$run_dir/evidence/native_pc_range.txt"
fi

export SCNA_SIM_TIMING_NODBC=1 SCNA_SIM_DETAILED=1
set +e
run_sim_case "$optimized_module" "$run_dir/raw/native/optimized_unfiltered_timing.log" \
  "$run_dir/metrics/native/unfiltered" attention --mode serial --variant optimized --workers 1 \
  --qo 32 --kv 64 --heads 12 --kv-heads 2 --head-dim 128 --warmup 0 --iters 1
timing_rc=$?
set -e
unset SCNA_SIM_TIMING_NODBC SCNA_SIM_DETAILED
printf 'exit_code=%d\n' "$timing_rc" > "$run_dir/evidence/native_unfiltered_exit.txt"

"$project_dir/.venv/bin/python" "$project_dir/tools/analyze_combined_resources.py" \
  --source-run "$source_run" --primary-run "$primary_run" --run-dir "$run_dir"
"$project_dir/.venv/bin/python" "$project_dir/tools/plot_combined_report.py" --run-dir "$run_dir"
"$project_dir/.venv/bin/python" "$project_dir/tools/render_combined_flows.py" --run-dir "$run_dir"
"$project_dir/.venv/bin/python" "$project_dir/tools/generate_combined_report.py" --run-dir "$run_dir"
"$project_dir/.venv/bin/python" "$project_dir/tools/generate_combined_tutorial.py" --run-dir "$run_dir"
"$project_dir/.venv/bin/pytest" -q "$project_dir/tests/test_analyze_combined_resources.py"
"$project_dir/.venv/bin/python" "$project_dir/tools/validate_combined_report.py" --run-dir "$run_dir"
"$project_dir/.venv/bin/python" "$project_dir/tools/validate_combined_tutorial.py" --run-dir "$run_dir"
printf '%s\n' "$run_dir"
