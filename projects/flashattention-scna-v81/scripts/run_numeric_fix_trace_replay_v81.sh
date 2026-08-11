#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRE_REMOTE="${PRE_REMOTE:-/data/local/tmp/flashattention_numeric_perf_pre_v81}"
POST_REMOTE="${POST_REMOTE:-/data/local/tmp/flashattention_numeric_perf_post_v81}"
OUT_DIR="${OUT_DIR:?set OUT_DIR to the numeric-fix result directory}"

run_trace() {
  local revision="$1" mode="$2" remote="$3" args="$4"
  local raw_dir="$OUT_DIR/trace/$revision/$mode/raw"
  local trace_dir="$OUT_DIR/trace/$revision/$mode/perfetto"
  mkdir -p "$raw_dir" "$trace_dir"
  adb shell "cd $remote && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;.' ./htp_ops_test \
    --figure8-attn $args --qo-len 32 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 \
    --warmup 1 --iters 3 --compare-reference" >"$raw_dir/raw_q32.log" 2>&1
  grep -q '^FIG8_ATTENTION_COMPARE .*ret=0$' "$raw_dir/raw_q32.log"
  grep -q '^FIG8_ATTENTION_EVENT_COUNT .*phase=measure .*overflow=0' "$raw_dir/raw_q32.log"
  python3 "$ROOT/tools/generate_figure8_perfetto_trace.py" --input-dir "$raw_dir" --out-dir "$trace_dir"
}

run_trace pre baseline "$PRE_REMOTE" '--mode baseline'
run_trace post baseline "$POST_REMOTE" '--mode baseline'
run_trace pre scna-d8 "$PRE_REMOTE" '--mode scna-fp16 --scna-function exp2 --scna-kernel direct --scna-width 8 --scna-pipeline off'
run_trace post scna-d8 "$POST_REMOTE" '--mode scna-fp16 --scna-function exp2 --scna-kernel direct --scna-width 8 --scna-pipeline off'

printf 'numeric-fix trace replay complete: %s/trace\n' "$OUT_DIR"
