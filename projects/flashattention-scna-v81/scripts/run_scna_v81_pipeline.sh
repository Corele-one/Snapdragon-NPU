#!/usr/bin/env bash
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/flashattention_scna_tree}"
OUT_DIR="${OUT_DIR:-results/v81/scna/pipeline-$(date +%Y%m%d-%H%M%S)}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
MAX_RETRIES="${MAX_RETRIES:-3}"
mkdir -p "$OUT_DIR/raw" "$OUT_DIR/retries" "$OUT_DIR/device-csv" "$OUT_DIR/summary"
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
    printf '%s/device-csv/%s' "$OUT_DIR" "$1"
  else
    printf '%s/%s' "$REMOTE_OUT_DIR" "$1"
  fi
}

complete_log() {
  local log_path="$1"
  local measured
  [[ -s "$log_path" ]] || return 1
  measured="$(grep -Ec '^FIG8_ATTENTION_HOST_TIMING .*phase=measure .*ret=0$' "$log_path" || true)"
  [[ "$measured" -ge "$ITERS" ]]
}

run_logged() {
  local log_path="$1"
  local stem
  local attempt
  local status
  shift
  stem="$(basename "${log_path%.log}")"

  if complete_log "$log_path"; then
    printf 'resume: %s already complete\n' "$stem"
    return 0
  fi
  if [[ -e "$log_path" ]]; then
    cp "$log_path" "$OUT_DIR/retries/${stem}.incomplete.log"
  fi
  for ((attempt = 1; attempt <= MAX_RETRIES; ++attempt)); do
    status=0
    run_test "$@" >"${log_path}.tmp" 2>&1 || status=$?
    mv "${log_path}.tmp" "$log_path"
    if [[ "$status" == "0" ]] && complete_log "$log_path"; then
      printf 'complete: %s (attempt %d)\n' "$stem" "$attempt"
      return 0
    fi
    cp "$log_path" "$OUT_DIR/retries/${stem}.attempt-${attempt}.log"
    printf 'retry: %s attempt=%d status=%d\n' "$stem" "$attempt" "$status" >&2
    sleep 1
  done
  printf 'failed after %d attempts: %s\n' "$MAX_RETRIES" "$stem" >&2
  return 1
}

for qo_len in 4 8 16 32; do
  stem="attention_baseline_q${qo_len}"
  run_logged "$OUT_DIR/raw/${stem}.log" \
    --figure8-attn --mode baseline --qo-len "$qo_len" --kv-len 4096 \
    --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup "$WARMUP" --iters "$ITERS" --no-events \
    --csv-out "$(csv_path "${stem}.csv")"
done

for function in exp2 exp; do
  for mode in scna-fp16 scna-int8; do
    for kernel in direct tree; do
      for width in 8 16 32; do
        for qo_len in 4 8 16 32; do
          for pipeline in off on; do
            stem="attention_${function}_${mode}_${kernel}_d${width}_pipe${pipeline}_q${qo_len}"
            run_logged "$OUT_DIR/raw/${stem}.log" \
              --figure8-attn --mode "$mode" --scna-function "$function" --scna-kernel "$kernel" \
              --scna-width "$width" --scna-pipeline "$pipeline" --qo-len "$qo_len" --kv-len 4096 \
              --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup "$WARMUP" --iters "$ITERS" --no-events \
              --csv-out "$(csv_path "${stem}.csv")"
          done
        done
      done
    done
  done
done

find "$OUT_DIR/raw" -maxdepth 1 -name '*.log' ! -name 'all.log' -print0 | sort -z | xargs -0 cat >"$OUT_DIR/raw/all.log"
python3 scripts/analyze_scna_pipeline.py --input "$OUT_DIR/raw/all.log" --out-dir "$OUT_DIR/summary"

if [[ -z "${TEST_CMD:-}" ]]; then
  adb pull "$REMOTE_OUT_DIR/." "$OUT_DIR/device-csv" >/dev/null
fi

printf 'SCNA v81 pipeline matrix complete: %s\n' "$OUT_DIR"
