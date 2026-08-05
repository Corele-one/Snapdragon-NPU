#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/flashattention_scna_pipeline_fair}"
OUT_DIR="${OUT_DIR:-$ROOT/results/v81/scna/stage6-pipeline-fair-$(date +%Y%m%d-%H%M%S)}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
MAX_RETRIES="${MAX_RETRIES:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-90}"

mkdir -p "$OUT_DIR/raw/perf" "$OUT_DIR/raw/correctness" \
  "$OUT_DIR/summary" "$OUT_DIR/correctness" "$OUT_DIR/provenance"

run_remote() {
  timeout "${TIMEOUT_SECONDS}s" adb shell \
    "cd $REMOTE_DIR && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;.' ./htp_ops_test $*"
}

run_logged() {
  local log_path="$1"
  local iteration_offset="$2"
  local attempt status temp_log
  shift 2
  temp_log="${log_path}.untagged"
  for ((attempt = 1; attempt <= MAX_RETRIES; ++attempt)); do
    status=0
    run_remote "$@" >"$temp_log" 2>&1 || status=$?
    if [[ "$status" == "0" ]] && grep -q '^FIG8_ATTENTION_HOST_TIMING .*phase=measure .*ret=0$' "$temp_log"; then
      perl -pe "s/iteration=(\\d+)/'iteration='.(\$1+$iteration_offset)/ge" "$temp_log" >"$log_path"
      rm "$temp_log"
      printf 'complete: %s (attempt %d)\n' "$(basename "$log_path")" "$attempt"
      return 0
    fi
    printf 'retry: %s attempt=%d status=%d\n' "$(basename "$log_path")" "$attempt" "$status" >&2
    sleep 1
  done
  mv "$temp_log" "${log_path}.failed"
  return 1
}

run_perf_case() {
  local session="$1"
  local offset="$2"
  local qo_len="$3"
  local mode="$4"
  local pipeline="$5"
  local stem="${session}_${mode}_pipe${pipeline}_q${qo_len}"
  local mode_args=(--mode "$mode")
  local compare_args=()
  if [[ "$mode" == "scna-fp16" ]]; then
    mode_args+=(--scna-function exp --scna-kernel tree --scna-width 8)
  fi
  if [[ "$pipeline" == "on" ]]; then
    compare_args+=(--compare-pipeline --compare-reference)
  fi
  run_logged "$OUT_DIR/raw/perf/${stem}.log" "$offset" \
    --figure8-attn "${mode_args[@]}" --kv-pipeline "$pipeline" \
    --qo-len "$qo_len" --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 \
    --warmup "$WARMUP" --iters "$ITERS" --no-events "${compare_args[@]}"
}

run_perf_matrix() {
  local session="$1"
  local offset="$2"
  local order="$3"
  local qo_len config mode pipeline
  local configs=(
    "baseline off"
    "baseline on"
    "lut-exp off"
    "lut-exp on"
    "scna-fp16 off"
    "scna-fp16 on"
  )
  if [[ "$order" == "reverse" ]]; then
    configs=(
      "scna-fp16 on"
      "scna-fp16 off"
      "lut-exp on"
      "lut-exp off"
      "baseline on"
      "baseline off"
    )
  fi
  for qo_len in 4 8 16 32; do
    for config in "${configs[@]}"; do
      read -r mode pipeline <<<"$config"
      run_perf_case "$session" "$offset" "$qo_len" "$mode" "$pipeline"
    done
  done
}

run_correctness() {
  local mode="$1"
  local mask="$2"
  local kv_len="$3"
  local head_dim="$4"
  local mode_args=(--mode "$mode")
  if [[ "$mode" == "scna-fp16" ]]; then
    mode_args+=(--scna-function exp --scna-kernel tree --scna-width 8)
  fi
  run_logged "$OUT_DIR/raw/correctness/${mode}_${mask}_kv${kv_len}_h${head_dim}.log" 0 \
    --figure8-attn "${mode_args[@]}" --kv-pipeline on --mask-mode "$mask" \
    --qo-len 4 --kv-len "$kv_len" --n-heads 12 --n-kv-heads 2 --head-dim "$head_dim" \
    --warmup 1 --iters 1 --no-events --numeric-debug --compare-pipeline --compare-reference
}

run_perf_matrix forward 0 forward
run_perf_matrix reverse 100 reverse

for mode in baseline lut-exp scna-fp16; do
  run_correctness "$mode" full 4093 128
  run_correctness "$mode" padding 4093 128
  run_correctness "$mode" causal 4093 128
  run_correctness "$mode" full 4096 64
done

find "$OUT_DIR/raw/perf" -name '*.log' -print0 | sort -z | xargs -0 cat >"$OUT_DIR/raw/perf/all.log"
find "$OUT_DIR/raw/correctness" -name '*.log' -print0 | sort -z | xargs -0 cat >"$OUT_DIR/raw/correctness/all.log"

adb shell getprop >"$OUT_DIR/provenance/device.txt"
adb shell "cd $REMOTE_DIR && sha256sum htp_ops_test libhtp_ops.so cdsp/libhtp_ops_skel.so" \
  >"$OUT_DIR/provenance/binary-sha256.txt"
rg -- '-mv81|-DFIGURE8_ENABLE_LUT_EXP=1' \
  "$ROOT/src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v81/build.ninja" \
  >"$OUT_DIR/provenance/build-gates.txt"
git -C "$ROOT" rev-parse HEAD >"$OUT_DIR/provenance/git-head.txt"
git -C "$ROOT" status --short -- . >"$OUT_DIR/provenance/git-status.txt"

python3 "$ROOT/scripts/analyze_scna_pipeline.py" \
  --input "$OUT_DIR/raw/perf/all.log" --out-dir "$OUT_DIR/summary"
python3 "$ROOT/scripts/analyze_scna_pipeline_correctness.py" \
  --input "$OUT_DIR/raw/correctness/all.log" --out-dir "$OUT_DIR/correctness"

printf 'fair KV pipeline experiment complete: %s\n' "$OUT_DIR"
