#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/reference/original_flashattention/htp-ops-lib-main"
remote_dir="/data/local/tmp/stage2_75_original_v79"
run_id="original_reproduction_$(date -u +%Y%m%dT%H%M%SZ)"
sessions=5
warmup=5
iters=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --quick) sessions=2; warmup=2; iters=3; shift ;;
    --help|-h) echo "Usage: $0 [--run-id ID] [--remote-dir PATH] [--quick]"; exit 0 ;;
    *) exit 2 ;;
  esac
done

host_ship="$htp_dir/android_ReleaseG_aarch64/ship"
dsp_ship="$htp_dir/hexagon_ReleaseG_toolv19_v79/ship"
out="$project_dir/results/runs/$run_id"
mkdir -p "$out"/{raw/attention,evidence}

for artifact in "$host_ship/htp_ops_test" "$host_ship/libhtp_ops.so" "$dsp_ship/libhtp_ops_skel.so"; do
  [[ -s "$artifact" ]] || { echo "Missing original artifact: $artifact" >&2; exit 1; }
done

adb get-state >/dev/null
adb shell "mkdir -p '$remote_dir/cdsp' '$remote_dir/dsp'"
adb push "$host_ship/htp_ops_test" "$remote_dir/" >/dev/null
adb push "$host_ship/libhtp_ops.so" "$remote_dir/" >/dev/null
adb push "$dsp_ship/libhtp_ops_skel.so" "$remote_dir/cdsp/" >/dev/null
adb push "$dsp_ship/libhtp_ops_skel.so" "$remote_dir/dsp/" >/dev/null
adb shell "chmod 755 '$remote_dir/htp_ops_test'"

{
  echo "schema_version=1"
  echo "run_id=$run_id"
  echo "sessions=$sessions"
  echo "warmup=$warmup"
  echo "iters=$iters"
  echo "captured_at=$(date -u +%FT%TZ)"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.soc.model
  sha256sum "$host_ship/htp_ops_test" "$host_ship/libhtp_ops.so" "$dsp_ship/libhtp_ops_skel.so"
} >"$out/evidence/manifest.txt"

for session in $(seq 1 "$sessions"); do
  for mode in baseline lut-exp; do
    label="original_${mode}"
    log="$out/raw/attention/${label}_q32_kv4096_s${session}.log"
    if [[ -s "$log" ]] && grep -q "phase=measure iteration=$((iters - 1)).*ret=0" "$log"; then
      echo "resume: ${log#$out/}"
      continue
    fi
    timeout 120s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test \
      --figure8-attn --mode '$mode' --qo-len 32 --kv-len 4096 --n-heads 12 --n-kv-heads 2 \
      --head-dim 128 --warmup '$warmup' --iters '$iters' --no-events" >"$log" 2>&1
    grep -q "phase=measure iteration=$((iters - 1)).*ret=0" "$log"
    echo "done: ${log#$out/}"
  done
done

echo "completed: $out"
