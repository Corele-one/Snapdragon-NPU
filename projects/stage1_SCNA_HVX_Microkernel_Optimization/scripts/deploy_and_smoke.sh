#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
dsp_arch="v79"
remote_dir="/data/local/tmp/scna_v79"
mode="ping"
scna_variant="stage1_dynamic_row"
kernel_impl="static_d8_ref"
scna_width=8
workers=1
mask_mode="full"
print_events=0
compare_reference=0
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
  --mode ping|baseline|lut-exp|scna-fp16|scna-exp-bench
  --dsp-arch v79                 DSP build target (this experiment is v79-only)
  --kernel-impl IMPL --workers auto|1..6 --scna-width 8
  --mask-mode full|causal|padding
  --events                       Print event-level DSP qtimer rows for trace replay
  --compare-reference            Run host FP32 correctness reference after attention
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
    --scna-variant)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      scna_variant="$2"; shift 2 ;;
    --kernel-impl)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      kernel_impl="$2"; shift 2 ;;
    --workers)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      if [[ "$2" == auto ]]; then workers=auto; else require_positive_int "$1" "$2"; workers="$2"; fi
      shift 2 ;;
    --scna-width)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      require_positive_int "$1" "$2"; scna_width="$2"; shift 2 ;;
    --mask-mode)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      mask_mode="$2"; shift 2 ;;
    --events)
      print_events=1; shift ;;
    --compare-reference)
      compare_reference=1; shift ;;
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

case "$mode" in
  ping|baseline|lut-exp|scna-fp16|scna-exp-bench) ;;
  *) echo "Unsupported --mode: $mode" >&2; exit 2 ;;
esac
[[ "$dsp_arch" == "v79" ]] || { echo "This optimization experiment only accepts --dsp-arch v79" >&2; exit 2; }
[[ "$scna_width" == "8" ]] || { echo "Only d8 is in the registered experiment" >&2; exit 2; }
[[ "$workers" == auto || "$workers" -le 6 ]] || { echo "--workers must be auto or in [1,6]" >&2; exit 2; }
case "$kernel_impl" in
  static_d8_ref|d7_serial|d7_scalar_w|d7_pairret_noinline|d7_pairret_inline|d7_quad_pipeline|d7_prebroadcast|qf16_tree_control|piecewise_control|combined_confirm|d7_pairret_two_acc|d7_pairret_short_const|d7_pairret_two_neuron|d7_pairret_scale_probe) ;;
  *) echo "Unsupported --kernel-impl: $kernel_impl" >&2; exit 2 ;;
esac
[[ "$mask_mode" == "full" || "$mask_mode" == "causal" || "$mask_mode" == "padding" ]] || {
  echo "--mask-mode must be full, causal, or padding" >&2; exit 2;
}

host_ship="$htp_dir/android_ReleaseG_aarch64/ship"
dsp_ship_candidates=("$htp_dir"/hexagon_ReleaseG_toolv*_"$dsp_arch"/ship)
dsp_ship="${dsp_ship_candidates[0]:-}"
variant_artifact="$project_dir/artifacts/variants/$kernel_impl/libhtp_ops_skel.so"

for artifact in scna_env_smoke libscna_env.so htp_ops_test libhtp_ops.so; do
  [[ -e "$host_ship/$artifact" ]] || {
    echo "Missing Android artifact: $host_ship/$artifact. Run scripts/build.sh first." >&2
    exit 1
  }
done
[[ ${#dsp_ship_candidates[@]} -eq 1 && -f "$dsp_ship/libscna_env_skel.so" && -f "$dsp_ship/libhtp_ops_skel.so" ]] || {
  echo "Missing or ambiguous DSP artifacts for $dsp_arch. Run scripts/build.sh --dsp-arch $dsp_arch first." >&2
  exit 1
}
[[ -f "$variant_artifact" ]] || {
  echo "Missing independent DSP artifact: $variant_artifact. Run scripts/build_all_variants.sh first." >&2
  exit 1
}

adb get-state >/dev/null
adb shell "mkdir -p '$remote_dir/cdsp' '$remote_dir/dsp'"
for artifact in scna_env_smoke libscna_env.so htp_ops_test libhtp_ops.so; do
  adb push "$host_ship/$artifact" "$remote_dir/"
done
for artifact in libscna_env_skel.so; do
  adb push "$dsp_ship/$artifact" "$remote_dir/cdsp/"
  adb push "$dsp_ship/$artifact" "$remote_dir/dsp/"
done
adb push "$variant_artifact" "$remote_dir/cdsp/libhtp_ops_skel.so"
adb push "$variant_artifact" "$remote_dir/dsp/libhtp_ops_skel.so"
adb shell "chmod 755 '$remote_dir/scna_env_smoke' '$remote_dir/htp_ops_test'"

if [[ "$mode" == "ping" ]]; then
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./scna_env_smoke"
elif [[ "$mode" == "scna-exp-bench" ]]; then
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --scna-exp-bench --scna-variant pair_static_d8 --scna-width '$scna_width' --warmup '$warmup' --iters '$iters'"
else
  event_arg="--no-events"
  [[ "$print_events" == "1" ]] && event_arg=""
  reference_arg=""
  [[ "$compare_reference" == "1" ]] && reference_arg="--compare-reference"
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --figure8-attn --mode '$mode' --scna-variant pair_static_d8 --workers '$workers' --scna-width '$scna_width' --mask-mode '$mask_mode' --qo-len '$qo_len' --kv-len '$kv_len' --n-heads '$n_heads' --n-kv-heads '$n_kv_heads' --head-dim '$head_dim' --warmup '$warmup' --iters '$iters' $event_arg $reference_arg"
fi
