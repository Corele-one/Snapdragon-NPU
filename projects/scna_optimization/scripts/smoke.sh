#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remote_dir="${SCNA_REMOTE_DIR:-/data/local/tmp/scna_optimization_v79}"
selection="${1:-all}"

case "$selection" in
  all) bundles=(scna reference) ;;
  scna|reference) bundles=("$selection") ;;
  *) echo "Usage: $0 [all|scna|reference]" >&2; exit 2 ;;
esac

run_attention() {
  local mode="$1"
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' \
    ./htp_ops_test --figure8-attn --mode '$mode' --qo-len 4 --kv-len 4096 \
    --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup 1 --iters 1 --no-events" 2>&1 \
    | grep 'FIG8_ATTENTION_HOST_TIMING' | grep 'phase=measure' | grep 'ret=0'
}

for bundle in "${bundles[@]}"; do
  "$script_dir/deploy.sh" "$bundle" "$remote_dir"
  if [[ "$bundle" == scna ]]; then
    adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./scna_env_smoke"
    run_attention scna-fp16
  else
    run_attention baseline
    run_attention lut-exp
  fi
done

echo "smoke_passed=$selection"
