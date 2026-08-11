#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRE_REMOTE="${PRE_REMOTE:-/data/local/tmp/flashattention_numeric_perf_pre_v81}"
POST_REMOTE="${POST_REMOTE:-/data/local/tmp/flashattention_numeric_perf_post_v81}"
OUT_DIR="${OUT_DIR:-$ROOT/results/v81/scna/numeric-fix-performance-$(date +%Y%m%d-%H%M%S)}"
WARMUP="${WARMUP:-5}"
ITERS_PER_LEG="${ITERS_PER_LEG:-15}"
SESSIONS="${SESSIONS:-3}"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/provenance" "$OUT_DIR/analysis"

temperature_tenths_c() {
  adb shell dumpsys battery 2>/dev/null | awk '/temperature:/{print $2; exit}' | tr -d '\r'
}

mode_args() {
  case "$1" in
    baseline) printf '%s' '--mode baseline' ;;
    lut-exp) printf '%s' '--mode lut-exp' ;;
    scna-d8) printf '%s' '--mode scna-fp16 --scna-function exp2 --scna-kernel direct --scna-width 8 --scna-pipeline off' ;;
    *) return 2 ;;
  esac
}

run_leg() {
  local session="$1" qo="$2" mode="$3" leg="$4" revision="$5" remote="$6"
  local log="$OUT_DIR/raw/s${session}_q${qo}_${mode}_leg${leg}_${revision}.log"
  local tmp="${log}.tmp"
  local args
  args="$(mode_args "$mode")"
  {
    printf 'NUMERIC_FIX_RUN revision=%s session=%s qo_len=%s mode=%s leg=%s warmup=%s measured=%s temp_before_tenths_c=%s\n' \
      "$revision" "$session" "$qo" "$mode" "$leg" "$WARMUP" "$ITERS_PER_LEG" "$(temperature_tenths_c)"
    adb shell "cd $remote && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;.' ./htp_ops_test \
      --figure8-attn $args --qo-len $qo --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 \
      --warmup $WARMUP --iters $ITERS_PER_LEG --no-events --compare-reference"
    printf 'NUMERIC_FIX_RUN_END revision=%s session=%s qo_len=%s mode=%s leg=%s temp_after_tenths_c=%s\n' \
      "$revision" "$session" "$qo" "$mode" "$leg" "$(temperature_tenths_c)"
  } >"$tmp" 2>&1
  local measured
  measured="$(grep -Ec '^FIG8_ATTENTION_HOST_TIMING .*phase=measure .*ret=0$' "$tmp" || true)"
  if [[ "$measured" != "$ITERS_PER_LEG" ]] || ! grep -Eq '^FIG8_ATTENTION_COMPARE .*ret=0$' "$tmp"; then
    mv "$tmp" "$log"
    printf 'invalid run: %s measured=%s expected=%s\n' "$log" "$measured" "$ITERS_PER_LEG" >&2
    return 1
  fi
  mv "$tmp" "$log"
  printf 'complete: session=%s q=%s mode=%s leg=%s revision=%s\n' "$session" "$qo" "$mode" "$leg" "$revision"
}

for ((session = 1; session <= SESSIONS; ++session)); do
  if (( session % 2 == 0 )); then
    qo_values=(32 16 8 4)
    modes=(scna-d8 baseline)
  else
    qo_values=(4 8 16 32)
    modes=(baseline scna-d8)
  fi
  for qo in "${qo_values[@]}"; do
    for mode in "${modes[@]}"; do
      run_leg "$session" "$qo" "$mode" 1 pre "$PRE_REMOTE"
      run_leg "$session" "$qo" "$mode" 2 post "$POST_REMOTE"
      run_leg "$session" "$qo" "$mode" 3 post "$POST_REMOTE"
      run_leg "$session" "$qo" "$mode" 4 pre "$PRE_REMOTE"
    done
  done
done

adb devices -l >"$OUT_DIR/provenance/adb-devices.txt"
adb shell getprop >"$OUT_DIR/provenance/device-properties.txt"
adb shell "sha256sum $PRE_REMOTE/htp_ops_test $PRE_REMOTE/cdsp/libhtp_ops_skel.so \
  $POST_REMOTE/htp_ops_test $POST_REMOTE/cdsp/libhtp_ops_skel.so" \
  >"$OUT_DIR/provenance/remote-binary-sha256.txt"
git -C "$ROOT" status --short -- . >"$OUT_DIR/provenance/git-status.txt"

python3 "$ROOT/tools/analyze_numeric_fix_performance.py" --input-dir "$OUT_DIR/raw" --output-dir "$OUT_DIR/analysis"
printf 'numeric-fix ABBA matrix complete: %s\n' "$OUT_DIR"
