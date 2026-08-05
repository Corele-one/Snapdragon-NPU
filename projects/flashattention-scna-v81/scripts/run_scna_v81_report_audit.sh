#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/flashattention_scna_report_audit}"
OUT_DIR="${OUT_DIR:-$ROOT/results/v81/scna/report-audit-$(date +%Y%m%d-%H%M%S)}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/provenance"

run_case() {
  local session="$1"
  local offset="$2"
  local name="$3"
  local raw_log="$OUT_DIR/raw/${session}_${name}.log"
  local untagged_log="${raw_log}.untagged"
  shift 3

  adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test $*" \
    >"$untagged_log" 2>&1
  perl -pe "s/iteration=(\\d+)/'iteration='.(\$1+$offset)/ge" "$untagged_log" >"$raw_log"
  rm "$untagged_log"
  printf 'complete: %s %s\n' "$session" "$name"
}

run_matrix() {
  local session="$1"
  local offset="$2"
  local order="$3"
  local qo_len common

  for qo_len in 4 8 16 32; do
    common="--figure8-attn --qo-len $qo_len --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup $WARMUP --iters $ITERS --no-events --compare-reference"
    if [[ "$order" == "reverse" ]]; then
      run_case "$session" "$offset" "q${qo_len}_exp_fp16_tree_d8_pipeon" --mode scna-fp16 --scna-function exp --scna-kernel tree --scna-width 8 --scna-pipeline on $common
      run_case "$session" "$offset" "q${qo_len}_exp_fp16_tree_d8_pipeoff" --mode scna-fp16 --scna-function exp --scna-kernel tree --scna-width 8 --scna-pipeline off $common
      run_case "$session" "$offset" "q${qo_len}_exp2_fp16_tree_d32" --mode scna-fp16 --scna-function exp2 --scna-kernel tree --scna-width 32 --scna-pipeline off $common
      run_case "$session" "$offset" "q${qo_len}_exp2_fp16_direct_d32" --mode scna-fp16 --scna-function exp2 --scna-kernel direct --scna-width 32 --scna-pipeline off $common
      run_case "$session" "$offset" "q${qo_len}_lut_exp" --mode lut-exp $common
      run_case "$session" "$offset" "q${qo_len}_baseline" --mode baseline $common
    else
      run_case "$session" "$offset" "q${qo_len}_baseline" --mode baseline $common
      run_case "$session" "$offset" "q${qo_len}_lut_exp" --mode lut-exp $common
      run_case "$session" "$offset" "q${qo_len}_exp2_fp16_direct_d32" --mode scna-fp16 --scna-function exp2 --scna-kernel direct --scna-width 32 --scna-pipeline off $common
      run_case "$session" "$offset" "q${qo_len}_exp2_fp16_tree_d32" --mode scna-fp16 --scna-function exp2 --scna-kernel tree --scna-width 32 --scna-pipeline off $common
      run_case "$session" "$offset" "q${qo_len}_exp_fp16_tree_d8_pipeoff" --mode scna-fp16 --scna-function exp --scna-kernel tree --scna-width 8 --scna-pipeline off $common
      run_case "$session" "$offset" "q${qo_len}_exp_fp16_tree_d8_pipeon" --mode scna-fp16 --scna-function exp --scna-kernel tree --scna-width 8 --scna-pipeline on $common
    fi
  done
}

# The reverse pass receives a disjoint iteration range so the analyzer cannot
# accidentally merge two DSP records into one sample.
run_matrix forward 0 forward
run_matrix reverse 100 reverse

find "$OUT_DIR/raw" -name '*.log' -print0 | sort -z | xargs -0 cat >"$OUT_DIR/raw/all.log"
adb shell getprop >"$OUT_DIR/provenance/device.txt"
adb shell "cd $REMOTE_DIR && sha256sum htp_ops_test dsp/libhtp_ops_skel.so" >"$OUT_DIR/provenance/binary-sha256.txt"
git -C "$ROOT" rev-parse HEAD >"$OUT_DIR/provenance/git-head.txt"
git -C "$ROOT" status --short -- . >"$OUT_DIR/provenance/git-status.txt"

python3 "$ROOT/scripts/analyze_scna_pipeline.py" \
  --input "$OUT_DIR/raw/all.log" --out-dir "$OUT_DIR/summary"
python3 "$ROOT/scripts/analyze_scna_pipeline_correctness.py" \
  --input "$OUT_DIR/raw/all.log" --out-dir "$OUT_DIR/correctness"
python3 "$ROOT/docs/stage-reports/render_scna_route_report.py"

printf 'audit complete: %s\n' "$OUT_DIR"
