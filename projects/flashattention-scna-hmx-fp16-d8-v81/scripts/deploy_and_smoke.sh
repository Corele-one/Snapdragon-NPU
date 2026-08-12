#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
dsp_arch="v81"
remote_dir="/data/local/tmp/scna_hmx_fp16_d8_v81"
mode="ping"
qo_len=4
kv_len=4096
n_heads=12
n_kv_heads=2
head_dim=128
warmup=1
iters=1

usage() {
  cat <<'EOF' >&2
Usage: deploy_and_smoke.sh [options]

Deploys the FastRPC environment and both FlashAttention baseline artifacts.

Options:
  --mode ping|baseline|lut-exp|scna-hvx-fp16-d8|scna-hmx-fp16-d8-*
  --dsp-arch v81                DSP build target (default: v81)
  --remote-dir PATH             Device deployment directory
  --qo-len N --kv-len N         Figure 8 attention shape (baseline/lut-exp)
  --n-heads N --n-kv-heads N --head-dim N
  --warmup N --iters N          Figure 8 repetitions (default: 1/1)
EOF
}

require_positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid positive integer for $1: $2" >&2
    exit 2
  }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      mode="$2"; shift 2 ;;
    --dsp-arch)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      dsp_arch="$2"; shift 2 ;;
    --remote-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      remote_dir="$2"; shift 2 ;;
    --qo-len|--kv-len|--n-heads|--n-kv-heads|--head-dim|--warmup|--iters)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      require_positive_int "$1" "$2"
      case "$1" in
        --qo-len) qo_len="$2" ;;
        --kv-len) kv_len="$2" ;;
        --n-heads) n_heads="$2" ;;
        --n-kv-heads) n_kv_heads="$2" ;;
        --head-dim) head_dim="$2" ;;
        --warmup) warmup="$2" ;;
        --iters) iters="$2" ;;
      esac
      shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      usage; exit 2 ;;
  esac
done

[[ "$dsp_arch" == "v81" ]] || { echo "This project is v81-only." >&2; exit 2; }

case "$mode" in
  ping|baseline|lut-exp|scna-hvx-fp16-d8|scna-hmx-fp16-d8-hybrid|scna-hmx-fp16-d8-two-pass|\
  scna-hmx-fp16-d8-hybrid-vtranspose|scna-hmx-fp16-d8-two-pass-vtranspose|\
  scna-hmx-fp16-d8-hybrid-batch4|scna-hmx-fp16-d8-two-pass-batch4|\
  scna-hmx-fp16-d8-hybrid-direct-p|scna-hmx-fp16-d8-two-pass-direct-p|\
  scna-hmx-fp16-d8-hybrid-attn-pipeline|scna-hmx-fp16-d8-two-pass-attn-pipeline) ;;
  *) echo "Unsupported --mode: $mode" >&2; exit 2 ;;
esac

host_ship="$htp_dir/android_ReleaseG_aarch64/ship"
dsp_ship_candidates=("$htp_dir"/hexagon_ReleaseG_toolv*_"$dsp_arch"/ship)
dsp_ship="${dsp_ship_candidates[0]:-}"

for artifact in htp_ops_test libhtp_ops.so; do
  [[ -e "$host_ship/$artifact" ]] || {
    echo "Missing Android artifact: $host_ship/$artifact. Run scripts/build.sh first." >&2
    exit 1
  }
done
[[ ${#dsp_ship_candidates[@]} -eq 1 && -f "$dsp_ship/libhtp_ops_skel.so" ]] || {
  echo "Missing or ambiguous DSP artifacts for $dsp_arch. Run scripts/build.sh --dsp-arch $dsp_arch first." >&2
  exit 1
}

adb get-state >/dev/null
adb shell "mkdir -p '$remote_dir/cdsp' '$remote_dir/dsp'"
for artifact in htp_ops_test libhtp_ops.so; do
  adb push "$host_ship/$artifact" "$remote_dir/"
done
for artifact in libhtp_ops_skel.so; do
  adb push "$dsp_ship/$artifact" "$remote_dir/cdsp/"
  adb push "$dsp_ship/$artifact" "$remote_dir/dsp/"
  adb push "$dsp_ship/$artifact" "$remote_dir/cdsp/libhtp_ops_skel_bp4.so"
  adb push "$dsp_ship/$artifact" "$remote_dir/dsp/libhtp_ops_skel_bp4.so"
done
adb shell "chmod 755 '$remote_dir/htp_ops_test'"

if [[ "$mode" == "ping" ]]; then
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --scna-exp-bench --mode scna-hvx-fp16-d8 --scna-width 8 --warmup 1 --iters 1"
else
  extra_scna=""
  if [[ "$mode" == scna-* ]]; then extra_scna="--scna-width 8 --scna-function exp2 --scna-kernel direct --kv-pipeline off"; fi
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --figure8-attn --mode '$mode' $extra_scna --qo-len '$qo_len' --kv-len '$kv_len' --n-heads '$n_heads' --n-kv-heads '$n_kv_heads' --head-dim '$head_dim' --warmup '$warmup' --iters '$iters' --no-events"
fi
