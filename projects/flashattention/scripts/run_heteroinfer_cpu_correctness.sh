#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/heteroinfer_cpu_common.sh"

STAMP=${STAMP:-20260801}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/results/v81/heteroinfer-cpu/stage1-correctness-$STAMP}
RAW_DIR="$OUT_ROOT/raw"
CSV_DIR="$OUT_ROOT/device-csv"
BIN_DIR="$OUT_ROOT/output-bin"
SUMMARY_DIR="$OUT_ROOT/summary"
mkdir -p "$RAW_DIR" "$CSV_DIR" "$BIN_DIR" "$SUMMARY_DIR"

if [[ ${SKIP_BUILD:-0} != 1 ]]; then
  build_v81
  deploy_v81
fi
verify_baseline_kernel
discover_cpu_profiles

CASES=(
  "full-q4-kv4096-h128:full:4:4096:128"
  "causal-q8-kv4093-h128:causal:8:4093:128"
  "padding-q16-kv4093-h64:padding:16:4093:64"
)
POLICIES=(legacy spin predictive)

printf 'case,wait_policy,sha256,bytes,reference_gate,byte_equal_to_legacy\n' > "$SUMMARY_DIR/correctness.csv"
: > "$RAW_DIR/all.log"

for case_spec in "${CASES[@]}"; do
  IFS=: read -r case_name mask_mode qo_len kv_len head_dim <<< "$case_spec"
  legacy_bin=""
  for policy in "${POLICIES[@]}"; do
    stem="${case_name}-${policy}"
    remote_csv="${stem}.csv"
    remote_bin="${stem}.bin"
    log_path="$RAW_DIR/${stem}.log"
    run_attention "$log_path" \
      --figure8-attn --mode baseline --wait-policy "$policy" --host-cpu "$LOW_CPU" \
      --host-sync-calibration 5 --mask-mode "$mask_mode" --qo-len "$qo_len" --kv-len "$kv_len" \
      --n-heads 12 --n-kv-heads 2 --head-dim "$head_dim" --warmup 5 --iters 1 --no-events \
      --compare-reference --csv-out "$remote_csv" --output-bin "$remote_bin"
    cat "$log_path" >> "$RAW_DIR/all.log"
    pull_remote_file "$remote_csv" "$CSV_DIR/$remote_csv"
    pull_remote_file "$remote_bin" "$BIN_DIR/$remote_bin"

    if ! rg -q 'FIG8_ATTENTION_COMPARE .*gate=pass' "$log_path"; then
      echo "FP32 reference gate failed for $stem" >&2
      exit 1
    fi
    if [[ "$policy" == legacy ]]; then
      legacy_bin="$BIN_DIR/$remote_bin"
      byte_equal=1
    elif cmp -s "$legacy_bin" "$BIN_DIR/$remote_bin"; then
      byte_equal=1
    else
      byte_equal=0
      echo "Wait-policy output mismatch for $case_name: legacy vs $policy" >&2
      exit 1
    fi
    hash=$(sha256sum "$BIN_DIR/$remote_bin" | awk '{ print $1 }')
    bytes=$(stat -c '%s' "$BIN_DIR/$remote_bin")
    printf '%s,%s,%s,%s,pass,%s\n' "$case_name" "$policy" "$hash" "$bytes" "$byte_equal" \
      >> "$SUMMARY_DIR/correctness.csv"
  done
done

{
  printf 'adb_serial=%s\n' "$ADB_SERIAL"
  printf 'device_model=%s\n' "$("${ADB[@]}" shell getprop ro.product.model | tr -d '\r')"
  printf 'device_soc=%s\n' "$("${ADB[@]}" shell getprop ro.soc.model | tr -d '\r')"
  printf 'low_cpu=%s\nlow_freq_khz=%s\n' "$LOW_CPU" "$LOW_FREQ_KHZ"
  printf 'high_cpu=%s\nhigh_freq_khz=%s\n' "$HIGH_CPU" "$HIGH_FREQ_KHZ"
  printf 'flash_attn_sha256=%s\n' "$BASELINE_FLASH_SHA256"
  printf 'git_head=%s\n' "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  printf 'git_dirty_files=%s\n' "$(git -C "$PROJECT_ROOT" status --porcelain | wc -l)"
  printf 'host_test_source_sha256=%s\n' "$(sha256sum "$HTP_ROOT/src/host/test.c" | awk '{ print $1 }')"
  printf 'dsp_dispatch_source_sha256=%s\n' "$(sha256sum "$HTP_ROOT/src/dsp/op_executor.cc" | awk '{ print $1 }')"
  printf 'profile_header_sha256=%s\n' "$(sha256sum "$HTP_ROOT/include/op_reg.h" | awk '{ print $1 }')"
  printf 'host_binary_sha256=%s\n' "$(sha256sum "$HTP_ROOT/android_ReleaseG_aarch64/ship/htp_ops_test" | awk '{ print $1 }')"
  printf 'dsp_binary_sha256=%s\n' "$(sha256sum "$HTP_ROOT/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so" | awk '{ print $1 }')"
  printf 'dsp_arch=v81\nhexagon_sdk=6.6.0.0\nhexagon_tools=19.0.07\n'
  printf 'warmup=5\niters=1\n'
} > "$OUT_ROOT/provenance.txt"

echo "Correctness results: $OUT_ROOT"
