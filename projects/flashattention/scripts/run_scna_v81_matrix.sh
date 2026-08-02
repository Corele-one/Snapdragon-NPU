#!/usr/bin/env bash
set -euo pipefail

# By default, run the standalone binary deployed using README.md's layout.
# TEST_CMD may override this with a simple local launcher command.
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/figure8_attn}"
OUT_DIR="${OUT_DIR:-results/v81/scna/$(date +%Y%m%d-%H%M%S)}"
MICRO_WARMUP="${MICRO_WARMUP:-20}"
MICRO_ITERS="${MICRO_ITERS:-1000}"
mkdir -p "$OUT_DIR"
REMOTE_OUT_DIR="$REMOTE_DIR/scna_results/$(basename "$OUT_DIR")"
if [[ -z "${TEST_CMD:-}" ]]; then
  adb shell "mkdir -p $REMOTE_OUT_DIR"
fi

run_test() {
  if [[ -n "${TEST_CMD:-}" ]]; then
    # shellcheck disable=SC2086
    $TEST_CMD "$@"
  else
    adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test $*"
  fi
}

csv_path() {
  if [[ -n "${TEST_CMD:-}" ]]; then
    printf '%s/%s' "$OUT_DIR" "$1"
  else
    printf '%s/%s' "$REMOTE_OUT_DIR" "$1"
  fi
}

for mode in scna-fp16 scna-int8; do
  for width in 8 16 32; do
    log="$OUT_DIR/${mode}_d${width}_exp2.log"
    run_test --scna-exp2-bench --mode "$mode" --scna-width "$width" --warmup "$MICRO_WARMUP" --iters "$MICRO_ITERS" \
      --csv-out "$(csv_path "${mode}_d${width}_exp2.csv")" >"$log" 2>&1
  done
done

for mode in baseline lut-exp scna-fp16 scna-int8; do
  for width in 8 16 32; do
    if [[ "$mode" == "baseline" || "$mode" == "lut-exp" ]] && [[ "$width" != 16 ]]; then
      continue
    fi
    for qo_len in 4 8 16 32; do
      log="$OUT_DIR/${mode}_d${width}_q${qo_len}.log"
      run_test --figure8-attn --mode "$mode" --scna-width "$width" --qo-len "$qo_len" \
        --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup 5 --iters 20 \
        --no-events --csv-out "$(csv_path "${mode}_d${width}_q${qo_len}.csv")" >"$log" 2>&1
    done
  done
done

cat "$OUT_DIR"/*.log >"$OUT_DIR/attention_raw.log"
python3 scripts/analyze_scna_attention.py --input "$OUT_DIR/attention_raw.log" --out-dir "$OUT_DIR/summary"
if [[ -z "${TEST_CMD:-}" ]]; then
  adb pull "$REMOTE_OUT_DIR" "$OUT_DIR/device_csv"
fi
