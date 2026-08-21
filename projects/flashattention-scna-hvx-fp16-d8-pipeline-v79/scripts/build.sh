#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
dsp_arch="v79"
variant="stage1_dynamic_row"
optimized_inline=0
optimized_impl="fma"
kernel_impl="static_d8_ref"
dsp_only=0

usage() {
  echo "Usage: $0 [--variant NAME] [--kernel-impl NAME] [--dsp-only]  # v79 only" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dsp-arch)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      dsp_arch="$2"
      shift 2
      ;;
    --variant)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      variant="$2"; shift 2 ;;
    --optimized-inline)
      [[ $# -ge 2 && ( "$2" == 0 || "$2" == 1 ) ]] || { usage; exit 2; }
      optimized_inline="$2"; shift 2 ;;
    --optimized-impl)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      optimized_impl="$2"; shift 2 ;;
    --kernel-impl)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      kernel_impl="$2"; shift 2 ;;
    --dsp-only)
      dsp_only=1; shift ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ "$dsp_arch" == "v79" ]] || { echo "This experiment is v79-only" >&2; exit 2; }
case "$variant" in
  stage1_dynamic_row) variant_id=0 ;;
  prepare_once_row) variant_id=1 ;;
  pair_shared_dynamic) variant_id=2 ;;
  pair_static_d8) variant_id=3 ;;
  pair_d8_fma_noinline) variant_id=4 ;;
  pair_d8_fma_inline) variant_id=5 ;;
  optimized) variant_id=6 ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
case "$optimized_impl" in
  fma) optimized_impl_id=0 ;;
  qf16-tree) optimized_impl_id=1 ;;
  piecewise-d8) optimized_impl_id=2 ;;
  *) echo "Unknown optimized implementation: $optimized_impl" >&2; exit 2 ;;
esac
case "$kernel_impl" in
  static_d8_ref) kernel_impl_id=0 ;;
  d7_serial) kernel_impl_id=1 ;;
  d7_scalar_w) kernel_impl_id=2 ;;
  d7_pairret_noinline) kernel_impl_id=3 ;;
  d7_pairret_inline) kernel_impl_id=4 ;;
  d7_quad_pipeline) kernel_impl_id=5 ;;
  d7_prebroadcast) kernel_impl_id=6 ;;
  qf16_tree_control) kernel_impl_id=7 ;;
  piecewise_control) kernel_impl_id=8 ;;
  combined_confirm) kernel_impl_id=9 ;;
  *) echo "Unknown kernel implementation: $kernel_impl" >&2; exit 2 ;;
esac
if [[ "$variant" != optimized && "$optimized_impl" != fma ]]; then
  echo "--optimized-impl is only valid with --variant optimized" >&2
  exit 2
fi

# Qualcomm's setup script reads optional undefined variables.  Temporarily
# disable nounset while sourcing it, then restore this script's strict mode.
set +u
# shellcheck source=use_hexagon_sdk_6_6.sh
source "$script_dir/use_hexagon_sdk_6_6.sh"
set -u
command -v build_cmake >/dev/null || {
  echo "Hexagon SDK setup did not provide build_cmake." >&2
  exit 1
}

cd "$htp_dir"
if [[ "$dsp_only" == 0 ]]; then
  build_cmake android
fi
build_cmake hexagon "DSP_ARCH=$dsp_arch" \
  FIGURE8_ENABLE_PROFILE_TIMERS=ON \
  FIGURE8_ENABLE_LUT_EXP=OFF \
  "SCNA_BUILD_VARIANT=$variant_id" \
  "SCNA_OPTIMIZED_INLINE=$optimized_inline" \
  "SCNA_OPTIMIZED_IMPL=$optimized_impl_id" \
  "SCNA_KERNEL_IMPL=$kernel_impl_id"

echo "Built variant=$variant build_id=$variant_id kernel_impl=$kernel_impl($kernel_impl_id) for $dsp_arch."
