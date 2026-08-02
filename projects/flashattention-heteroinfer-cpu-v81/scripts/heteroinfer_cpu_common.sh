#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
HTP_ROOT="$PROJECT_ROOT/src/htp-ops-lib-main"
REMOTE_ROOT=${REMOTE_ROOT:-/data/local/tmp/flashattention_heteroinfer_cpu}
ADB_SERIAL=${ADB_SERIAL:-}
BASELINE_FLASH_SHA256=6a8b025bda0cbaa36c3b48e3d379b7502631cdd584a7571e926e6633483356cb

if [[ -z "$ADB_SERIAL" ]]; then
  ADB_SERIAL=$(adb devices | awk 'NR > 1 && $2 == "device" { print $1; exit }')
fi
if [[ -z "$ADB_SERIAL" ]]; then
  echo "No adb device is connected" >&2
  exit 1
fi

ADB=(adb -s "$ADB_SERIAL")

verify_baseline_kernel() {
  local actual
  actual=$(sha256sum "$HTP_ROOT/src/dsp/ops/flash_attn.c" | awk '{ print $1 }')
  if [[ "$actual" != "$BASELINE_FLASH_SHA256" ]]; then
    echo "Baseline flash_attn.c changed: expected $BASELINE_FLASH_SHA256, got $actual" >&2
    exit 1
  fi
}

build_v81() {
  verify_baseline_kernel
  if [[ ! -f "$HTP_ROOT/android_ReleaseG_aarch64/CMakeCache.txt" ||
        ! -f "$HTP_ROOT/hexagon_ReleaseG_toolv19_v81/CMakeCache.txt" ]]; then
    echo "Configured SDK 6.6 Android/v81 build directories are required" >&2
    exit 1
  fi
  cmake --build "$HTP_ROOT/android_ReleaseG_aarch64" --parallel "${BUILD_JOBS:-8}"
  cmake --build "$HTP_ROOT/hexagon_ReleaseG_toolv19_v81" --parallel "${BUILD_JOBS:-8}" --verbose \
    > "$HTP_ROOT/hexagon_ReleaseG_toolv19_v81/build-v81.log" 2>&1
  if ! rg -q -- '-mv81' "$HTP_ROOT/hexagon_ReleaseG_toolv19_v81/build-v81.log" &&
     ! rg -q -- '-mv81' "$HTP_ROOT/hexagon_ReleaseG_toolv19_v81/build.ninja"; then
    echo "The final DSP build does not contain -mv81" >&2
    exit 1
  fi
}

deploy_v81() {
  "${ADB[@]}" shell "mkdir -p '$REMOTE_ROOT/cdsp' '$REMOTE_ROOT/dsp'"
  "${ADB[@]}" push "$HTP_ROOT/android_ReleaseG_aarch64/ship/htp_ops_test" "$REMOTE_ROOT/htp_ops_test" >/dev/null
  "${ADB[@]}" push "$HTP_ROOT/android_ReleaseG_aarch64/ship/libhtp_ops.so" "$REMOTE_ROOT/libhtp_ops.so" >/dev/null
  "${ADB[@]}" push "$HTP_ROOT/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so" \
    "$REMOTE_ROOT/cdsp/libhtp_ops_skel.so" >/dev/null
  "${ADB[@]}" push "$HTP_ROOT/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so" \
    "$REMOTE_ROOT/dsp/libhtp_ops_skel.so" >/dev/null
  "${ADB[@]}" shell "chmod 755 '$REMOTE_ROOT/htp_ops_test'"
}

discover_cpu_profiles() {
  local topology
  topology=$("${ADB[@]}" shell \
    'for c in /sys/devices/system/cpu/cpu[0-9]*; do n=${c##*cpu}; f=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0); echo "$n $f"; done' \
    | tr -d '\r' | sort -n)
  LOW_CPU=$(awk '$2 > 0 { if (min == 0 || $2 < min) { min=$2; cpu=$1 } } END { print cpu }' <<< "$topology")
  HIGH_CPU=$(awk '$2 > max { max=$2; cpu=$1 } END { print cpu }' <<< "$topology")
  LOW_FREQ_KHZ=$(awk -v cpu="$LOW_CPU" '$1 == cpu { print $2 }' <<< "$topology")
  HIGH_FREQ_KHZ=$(awk -v cpu="$HIGH_CPU" '$1 == cpu { print $2 }' <<< "$topology")
  if [[ -z "$LOW_CPU" || -z "$HIGH_CPU" ]]; then
    echo "Could not discover device CPU frequency domains" >&2
    exit 1
  fi
  CPU_PROFILES=("unpinned:-1" "low-capacity-cpu${LOW_CPU}:${LOW_CPU}")
  if [[ "$HIGH_CPU" != "$LOW_CPU" ]]; then
    CPU_PROFILES+=("high-capacity-cpu${HIGH_CPU}:${HIGH_CPU}")
  fi
  export LOW_CPU HIGH_CPU LOW_FREQ_KHZ HIGH_FREQ_KHZ
}

run_attention() {
  local log_path=$1
  shift
  "${ADB[@]}" shell "cd '$REMOTE_ROOT' && export LD_LIBRARY_PATH=. && export DSP_LIBRARY_PATH='./cdsp;./dsp;.' && ./htp_ops_test $*" \
    > "$log_path" 2>&1
}

pull_remote_file() {
  local remote_path=$1
  local local_path=$2
  "${ADB[@]}" pull "$REMOTE_ROOT/$remote_path" "$local_path" >/dev/null
}
