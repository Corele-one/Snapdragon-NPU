#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/heteroinfer_cpu_common.sh"

STAMP=${STAMP:-20260801}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/results/v81/heteroinfer-cpu/stage1-main-$STAMP}
RAW_DIR="$OUT_ROOT/raw"
CSV_DIR="$OUT_ROOT/device-csv"
SUMMARY_DIR="$OUT_ROOT/summary"
mkdir -p "$RAW_DIR" "$CSV_DIR" "$SUMMARY_DIR"

if [[ ${SKIP_BUILD:-0} != 1 ]]; then
  build_v81
  deploy_v81
fi
verify_baseline_kernel
discover_cpu_profiles

WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}
CALIBRATION=${CALIBRATION:-20}
QO_LENGTHS=(4 8 16 32)
POLICY_ORDERS=(
  "legacy spin predictive"
  "spin predictive legacy"
  "predictive legacy spin"
)

: > "$RAW_DIR/all.log"
run_index=0
for qo_len in "${QO_LENGTHS[@]}"; do
  for profile_spec in "${CPU_PROFILES[@]}"; do
    IFS=: read -r placement host_cpu <<< "$profile_spec"
    read -r -a policies <<< "${POLICY_ORDERS[$((run_index % ${#POLICY_ORDERS[@]}))]}"
    for policy in "${policies[@]}"; do
      stem="baseline-${placement}-${policy}-q${qo_len}"
      remote_csv="${stem}.csv"
      log_path="$RAW_DIR/${stem}.log"
      cpu_args=()
      if [[ "$host_cpu" != -1 ]]; then
        cpu_args=(--host-cpu "$host_cpu")
      fi
      echo "Running $stem" >&2
      run_attention "$log_path" \
        --figure8-attn --mode baseline --wait-policy "$policy" "${cpu_args[@]}" \
        --host-sync-calibration "$CALIBRATION" --mask-mode full --qo-len "$qo_len" --kv-len 4096 \
        --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup "$WARMUP" --iters "$ITERS" \
        --no-events --csv-out "$remote_csv"
      cat "$log_path" >> "$RAW_DIR/all.log"
      pull_remote_file "$remote_csv" "$CSV_DIR/$remote_csv"
      measured=$(awk -F, 'NR > 1 && $12 == "measure" && $30 == 0 { n++ } END { print n + 0 }' \
        "$CSV_DIR/$remote_csv")
      if [[ "$measured" != "$ITERS" ]]; then
        echo "Incomplete result for $stem: expected $ITERS measured rows, got $measured" >&2
        exit 1
      fi
      sleep "${INTER_RUN_SLEEP_SECONDS:-0.10}"
    done
    run_index=$((run_index + 1))
  done
done

{
  printf 'adb_serial=%s\n' "$ADB_SERIAL"
  printf 'device_model=%s\n' "$("${ADB[@]}" shell getprop ro.product.model | tr -d '\r')"
  printf 'device_soc=%s\n' "$("${ADB[@]}" shell getprop ro.soc.model | tr -d '\r')"
  printf 'android_release=%s\n' "$("${ADB[@]}" shell getprop ro.build.version.release | tr -d '\r')"
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
  printf 'warmup=%s\niters=%s\ncalibration_window=%s\n' "$WARMUP" "$ITERS" "$CALIBRATION"
  printf 'qo_len=4,8,16,32\nkv_len=4096\nn_heads=12\nn_kv_heads=2\nhead_dim=128\n'
  printf 'policy_order=rotating_latin_order\n'
  "${ADB[@]}" shell 'cat /sys/class/power_supply/battery/temp 2>/dev/null; cat /sys/class/power_supply/battery/capacity 2>/dev/null' \
    | tr -d '\r' | awk 'NR == 1 { print "battery_temp_deci_c=" $0 } NR == 2 { print "battery_capacity_percent=" $0 }'
} > "$OUT_ROOT/provenance.txt"

python3 "$SCRIPT_DIR/analyze_heteroinfer_cpu.py" --input-dir "$CSV_DIR" --output-dir "$SUMMARY_DIR"
echo "Main results: $OUT_ROOT"
