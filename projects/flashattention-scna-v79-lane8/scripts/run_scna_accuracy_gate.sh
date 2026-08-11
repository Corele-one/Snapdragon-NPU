#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
result_dir="${1:-$project_dir/results/v79/scna-lane8/accuracy-first-$(date +%Y%m%d_%H%M%S)}"
remote_dir="${2:-/data/local/tmp/scna_v79_accuracy}"
mkdir -p "$result_dir/accuracy/raw" "$result_dir/evaluator" "$result_dir/provenance"

device_cmd=(adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test")
summary="$result_dir/accuracy/case_status.tsv"
printf 'stage\tmode\tlayout\tseed\tmask\tqo_len\tkv_len\thead_dim\texit_code\tcompare_pass\tlayout_pass\tmask_tail_pass\tchecksum_unique_count\tlog\n' > "$summary"

"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" --mode ping \
  >"$result_dir/provenance/deploy.log" 2>&1 || exit 2
adb devices -l > "$result_dir/provenance/adb_devices.txt"
adb shell getprop > "$result_dir/provenance/device_getprop.txt" 2>&1
dsp_binary="$project_dir/src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v79/ship/libhtp_ops_skel.so"
host_binary="$project_dir/src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/htp_ops_test"
coefficient_header="$project_dir/src/htp-ops-lib-main/include/dsp/scna_params.h"
dsp_sha256="$(sha256sum "$dsp_binary" | awk '{print $1}')"
host_sha256="$(sha256sum "$host_binary" | awk '{print $1}')"
header_sha256="$(sha256sum "$coefficient_header" | awk '{print $1}')"
provenance_line="SCNA_PROVENANCE binary_sha256=$dsp_sha256 host_sha256=$host_sha256 header_sha256=$header_sha256 isa=v79 sdk=6.6 tools=19.0.07"
printf '%s\n' "$provenance_line" > "$result_dir/provenance/runtime_provenance.txt"

run_evaluator() {
  local layout="$1"
  local log="$result_dir/evaluator/${layout}.log"
  "${device_cmd[@]} --scna-exp-bench --scna-layout '$layout' --scna-width 8 --warmup 5 --iters 200000" \
    >"$log" 2>&1
  sed -i "1i$provenance_line" "$log"
  grep -q 'monotonic_violations=0 negative_count=0 nan_count=0' "$log" &&
    grep -q 'canonical_oracle_mismatches=0 paired_single_mismatches=0' "$log" &&
    grep -q 'reciprocal_pass=1' "$log" &&
    { [[ "$layout" != lane8 ]] || grep -q 'lane_oracle_mismatches=0' "$log"; }
}

run_level() {
  local stage="$1" mode="$2" layout="$3" repetitions="$4" compare_layout="$5"
  local failures=0
  for seed in figure8_fixed 20260810 20260811; do
    for mask in full padding causal; do
      for qo in 1 4; do
        for kv in 4093 4096; do
          for dim in 64 128; do
            local case_id="${stage}_${seed}_${mask}_q${qo}_k${kv}_d${dim}"
            local log="$result_dir/accuracy/raw/${case_id}.log"
            local layout_arg=""
            [[ "$compare_layout" == 1 ]] && layout_arg="--compare-scna-layout"
            "${device_cmd[@]} --figure8-attn --mode '$mode' --scna-layout '$layout' --scna-width 8 --seed '$seed' --mask-mode '$mask' --qo-len '$qo' --kv-len '$kv' --n-heads 12 --n-kv-heads 2 --head-dim '$dim' --warmup 1 --iters '$repetitions' --no-events --compare-reference --numeric-debug $layout_arg" \
              >"$log" 2>&1
            local cmd_exit=$?
            sed -i "1i$provenance_line" "$log"
            local compare_pass=0 layout_pass=1 mask_tail_pass=0 checksum_unique_count=0
            grep -q 'FIG8_ATTENTION_COMPARE .* pass=1$' "$log" && compare_pass=1
            if [[ "$compare_layout" == 1 ]]; then
              layout_pass=0
              grep -q 'FIG8_ATTENTION_LAYOUT_COMPARE .* pass=1$' "$log" && layout_pass=1
            fi
            if grep 'FIG8_NUMERIC .*kv_head=0 ' "$log" | grep -qvE 'masked_p_nonzero=0 tail_p_nonzero=0'; then
              mask_tail_pass=0
            elif grep -q 'FIG8_NUMERIC .*kv_head=0 ' "$log"; then
              mask_tail_pass=1
            fi
            checksum_unique_count="$(grep 'FIG8_ATTENTION_CHECKSUM .*phase=measure' "$log" | sed -n 's/.*checksum=//p' | sort -u | wc -l)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$stage" "$mode" "$layout" "$seed" "$mask" "$qo" "$kv" "$dim" "$cmd_exit" \
              "$compare_pass" "$layout_pass" "$mask_tail_pass" "$checksum_unique_count" "$log" >> "$summary"
            if [[ "$cmd_exit" != 0 || "$compare_pass" != 1 || "$layout_pass" != 1 ||
                  "$mask_tail_pass" != 1 || "$checksum_unique_count" != 1 ]]; then
              failures=$((failures + 1))
            fi
          done
        done
      done
    done
  done
  printf 'ACCURACY_STAGE_RESULT stage=%s cases=72 failures=%d pass=%d\n' \
    "$stage" "$failures" "$((failures == 0))" | tee "$result_dir/accuracy/${stage}_result.txt"
  [[ "$failures" == 0 ]]
}

if ! run_level baseline baseline serial 1 0; then
  printf 'ACCURACY_GATE_STOP failed_stage=baseline phase2_started=0\n' | tee "$result_dir/accuracy/gate_result.txt"
  exit 10
fi
if ! run_evaluator serial || ! run_level scna_serial scna-fp16 serial 1 0; then
  printf 'ACCURACY_GATE_STOP failed_stage=scna_serial phase2_started=0\n' | tee "$result_dir/accuracy/gate_result.txt"
  exit 11
fi
if ! run_evaluator lane8 || ! run_level scna_lane8 scna-fp16 lane8 10 1; then
  printf 'ACCURACY_GATE_STOP failed_stage=scna_lane8 phase2_started=0\n' | tee "$result_dir/accuracy/gate_result.txt"
  exit 12
fi
printf 'ACCURACY_GATE_PASS total_cases=216 phase2_allowed=1\n' | tee "$result_dir/accuracy/gate_result.txt"
