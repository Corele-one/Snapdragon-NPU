#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
sdk_root="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0}"
tools_root="${HEXAGON_SIM_CORE:-$sdk_root/tools/HEXAGON_Tools/19.0.07}"
sim_bin="$tools_root/Tools/bin/hexagon-sim"
run_main_build="$sdk_root/libs/run_main_on_hexagon/hexagon_ReleaseG_toolv19_v79"
run_main_bin="$run_main_build/ship/run_main_on_hexagon_sim"
runelf="$sdk_root/rtos/qurt/computev79/sdksim_bin/runelf.pbn"
timeout_seconds="${SCNA_SIM_CASE_TIMEOUT_SEC:-600}"
target_lib_dir="$tools_root/Tools/target/hexagon/lib/v79/G0/pic"

require_sim_runtime() {
  local required=("$sim_bin" "$run_main_bin" "$run_main_build/q6ss.cfg" "$run_main_build/osam.cfg" "$runelf" "$target_lib_dir/libc++.so.1")
  for path in "${required[@]}"; do
    [[ -e "$path" ]] || { echo "Missing simulator runtime component: $path" >&2; return 1; }
  done
}

run_sim_case() {
  local module="$1" log="$2" metrics_dir="$3"
  shift 3
  mkdir -p "$(dirname -- "$log")" "$metrics_dir"
  local started ended rc had_errexit=0
  local sim_options=(-mv79na_1 --simulated_returnval --usefs "$target_lib_dir")
  if [[ ${SCNA_SIM_TIMING:-0} == 1 ]]; then
    sim_options+=(--timing --pmu_statsfile "$metrics_dir/pmu.txt")
  fi
  if [[ ${SCNA_SIM_TIMING_NODBC:-0} == 1 ]]; then
    sim_options+=(--timing_nodbc --pmu_statsfile "$metrics_dir/pmu.txt")
  fi
  if [[ ${SCNA_SIM_DETAILED:-0} == 1 ]]; then
    sim_options+=(--ihist "$metrics_dir/ihist.txt" --packet_analyze "$metrics_dir/packet_analyze.txt")
  fi
  if [[ -n ${SCNA_SIM_PCFILTER:-} ]]; then sim_options+=(--pcfilter "$SCNA_SIM_PCFILTER"); fi
  if [[ ${SCNA_SIM_MEMTRACE:-0} == 1 ]]; then sim_options+=(--memtrace "$metrics_dir/memtrace.txt"); fi
  if [[ ${SCNA_SIM_COPROCTRACE:-0} == 1 ]]; then sim_options+=(--coproctrace "$metrics_dir/coproctrace.txt"); fi
  if [[ ${SCNA_SIM_PCTRACE:-0} == 1 ]]; then sim_options+=(--pctrace "$metrics_dir/pctrace.txt"); fi
  if [[ ${SCNA_SIM_PCTRACE:-0} == min ]]; then sim_options+=(--pctrace_min "$metrics_dir/pctrace.txt"); fi
  [[ $- == *e* ]] && had_errexit=1
  started="$(date +%s%N)"
  set +e
  timeout --preserve-status "$timeout_seconds" "$sim_bin" "${sim_options[@]}" \
    --cosim_file "$run_main_build/q6ss.cfg" --l2tcm_base 0xd800 --subsystem_base 0xFC90 \
    --rtos "$run_main_build/osam.cfg" "$runelf" -- "$run_main_bin" -- "$module" "$@" \
    >"$log" 2>&1
  rc=$?
  if [[ $had_errexit -eq 1 ]]; then set -e; else set +e; fi
  ended="$(date +%s%N)"
  printf 'SIM_PROCESS_RESULT exit_code=%d wall_ns=%s module=%s args=%q\n' \
    "$rc" "$((ended - started))" "$module" "$*" >> "$log"
  return "$rc"
}
