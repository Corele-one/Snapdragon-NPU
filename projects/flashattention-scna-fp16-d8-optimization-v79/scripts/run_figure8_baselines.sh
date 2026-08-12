#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
remote_dir="/data/local/tmp/scna_v79"
out_dir="$project_dir/results/local/v79"
qo_lens="${QO_LENS:-4 8 16 32}"
kv_len="${KV_LEN:-4096}"
warmup="${WARMUP:-5}"
iters="${ITERS:-20}"

usage() {
  cat <<'EOF'
Usage: run_figure8_baselines.sh

Collects baseline and LUT-exp Figure 8 timings at qo_len=4,8,16,32 by
default, then generates parser and comparison outputs under results/local/v79.

Optional environment variables: QO_LENS, KV_LEN, WARMUP, ITERS.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

for pair in "KV_LEN=$kv_len" "WARMUP=$warmup" "ITERS=$iters"; do
  name="${pair%%=*}"
  value="${pair#*=}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done

# Deploy once and verify the low-level FastRPC path before collecting timings.
"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" --mode ping

for mode in baseline lut-exp; do
  mode_dir="${mode/lut-exp/lut_exp}"
  mkdir -p "$out_dir/$mode_dir"
  for qo_len in $qo_lens; do
    [[ "$qo_len" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid QO_LENS item: $qo_len" >&2; exit 2; }
    log="$out_dir/$mode_dir/raw_q${qo_len}.log"
    adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --figure8-attn --mode '$mode' --qo-len '$qo_len' --kv-len '$kv_len' --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup '$warmup' --iters '$iters' --no-events" \
      2>&1 | tee "$log"
  done
done

python3 "$project_dir/tools/parse_figure8_attention_timers.py" \
  --input-dir "$out_dir/baseline" --out-dir "$out_dir/baseline"
python3 "$project_dir/tools/parse_figure8_attention_timers.py" \
  --input-dir "$out_dir/lut_exp" --out-dir "$out_dir/lut_exp"
python3 "$project_dir/tools/compare_figure8_lut_exp.py" \
  --baseline-summary "$out_dir/baseline/attention_timers_summary.json" \
  --lut-exp-summary "$out_dir/lut_exp/attention_timers_summary.json" \
  --out-dir "$out_dir/lut_exp"

echo "Baseline analysis written to: $out_dir"
