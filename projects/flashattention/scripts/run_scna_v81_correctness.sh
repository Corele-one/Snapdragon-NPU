#!/usr/bin/env bash
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/flashattention_scna_tree}"
OUT_DIR="${OUT_DIR:-results/v81/scna/correctness-$(date +%Y%m%d-%H%M%S)}"
MAX_RETRIES="${MAX_RETRIES:-3}"
mkdir -p "$OUT_DIR/raw" "$OUT_DIR/retries" "$OUT_DIR/summary"

run_test() {
  if [[ -n "${TEST_CMD:-}" ]]; then
    # shellcheck disable=SC2086
    $TEST_CMD "$@"
  else
    adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test $*"
  fi
}

correctness_complete() {
  local log_path="$1"
  local require_tree_gate="$2"

  [[ -s "$log_path" ]] || return 1
  grep -Eq '^FIG8_ATTENTION_COMPARE .*reference_mode=host-fp32 .*ret=0$' "$log_path" || return 1
  if [[ "$require_tree_gate" == "1" ]]; then
    grep -Eq '^FIG8_ATTENTION_DIRECT_TREE_COMPARE .*gate=pass ret=0$' "$log_path" || return 1
  fi
}

run_logged() {
  local log_path="$1"
  local require_tree_gate="$2"
  local stem
  local attempt
  local status
  shift 2
  stem="$(basename "${log_path%.log}")"

  if correctness_complete "$log_path" "$require_tree_gate"; then
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
    if [[ "$status" == "0" ]] && correctness_complete "$log_path" "$require_tree_gate"; then
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

cases=(
  "full 4093 128"
  "padding 4093 128"
  "causal 4093 128"
  "full 4096 64"
)

for case_spec in "${cases[@]}"; do
  read -r mask kv_len head_dim <<<"$case_spec"
  stem="baseline_${mask}_kv${kv_len}_h${head_dim}"
  run_logged "$OUT_DIR/raw/${stem}.log" 0 \
    --figure8-attn --mode baseline --mask-mode "$mask" --qo-len 4 --kv-len "$kv_len" \
    --n-heads 12 --n-kv-heads 2 --head-dim "$head_dim" --warmup 1 --iters 1 --no-events \
    --numeric-debug --compare-reference
done

for function in exp2 exp; do
  for mode in scna-fp16 scna-int8; do
    for kernel in direct tree; do
      for width in 8 16 32; do
        for case_spec in "${cases[@]}"; do
          read -r mask kv_len head_dim <<<"$case_spec"
          stem="${function}_${mode}_${kernel}_d${width}_${mask}_kv${kv_len}_h${head_dim}"
          extra=()
          if [[ "$kernel" == "tree" ]]; then
            extra+=(--compare-direct-tree)
          fi
          run_logged "$OUT_DIR/raw/${stem}.log" "$([[ "$kernel" == "tree" ]] && printf 1 || printf 0)" \
            --figure8-attn --mode "$mode" --scna-function "$function" --scna-kernel "$kernel" \
            --scna-width "$width" --mask-mode "$mask" --qo-len 4 --kv-len "$kv_len" \
            --n-heads 12 --n-kv-heads 2 --head-dim "$head_dim" --warmup 1 --iters 1 --no-events \
            --numeric-debug --compare-reference "${extra[@]}"
        done
      done
    done
  done
done

find "$OUT_DIR/raw" -maxdepth 1 -name '*.log' ! -name 'all.log' -print0 | sort -z | xargs -0 cat >"$OUT_DIR/raw/all.log"
python3 scripts/analyze_scna_correctness.py --input "$OUT_DIR/raw/all.log" --out-dir "$OUT_DIR/summary"
printf 'SCNA v81 correctness matrix complete: %s\n' "$OUT_DIR"
