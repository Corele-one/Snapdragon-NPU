#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
requested="all"
dsp_only=0

usage() {
  echo "Usage: $0 [--flavor all|fair_combined|lut_only|scna_only] [--dsp-only]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor) [[ $# -ge 2 ]] || { usage; exit 2; }; requested="$2"; shift 2 ;;
    --dsp-only) dsp_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

case "$requested" in
  all) flavors=(fair_combined lut_only scna_only) ;;
  fair_combined|lut_only|scna_only) flavors=("$requested") ;;
  *) echo "Unknown flavor: $requested" >&2; exit 2 ;;
esac

set +u
source "$script_dir/use_hexagon_sdk_6_6.sh"
set -u
command -v build_cmake >/dev/null || { echo "Hexagon SDK setup did not provide build_cmake" >&2; exit 1; }

mkdir -p "$project_dir/artifacts/host"
cd "$htp_dir"
if [[ "$dsp_only" == 0 ]]; then
  build_cmake android
  cp -a android_ReleaseG_aarch64/ship/htp_ops_test android_ReleaseG_aarch64/ship/libhtp_ops.so \
    "$project_dir/artifacts/host/"
  if [[ -f android_ReleaseG_aarch64/ship/libscna_env.so ]]; then
    cp -a android_ReleaseG_aarch64/ship/libscna_env.so android_ReleaseG_aarch64/ship/scna_env_smoke \
      "$project_dir/artifacts/host/"
  fi
fi

for flavor in "${flavors[@]}"; do
  case "$flavor" in
    fair_combined) flavor_id=0 ;;
    lut_only) flavor_id=1 ;;
    scna_only) flavor_id=2 ;;
  esac
  build_cmake hexagon DSP_ARCH=v79 \
    FIGURE8_ENABLE_PROFILE_TIMERS=ON \
    FIGURE8_ENABLE_LUT_EXP=OFF \
    SCNA_BUILD_VARIANT=3 \
    SCNA_OPTIMIZED_INLINE=0 \
    SCNA_OPTIMIZED_IMPL=0 \
    SCNA_KERNEL_IMPL=3 \
    STAGE2_SOFTMAX_IMPL=2 \
    STAGE2_ENABLE_FINE_TIMERS=OFF \
    "AUDIT_BUILD_FLAVOR=$flavor_id"
  out_dir="$project_dir/artifacts/$flavor"
  mkdir -p "$out_dir"
  cp -a hexagon_ReleaseG_toolv19_v79/ship/libhtp_ops_skel.so "$out_dir/"
  if [[ -f hexagon_ReleaseG_toolv19_v79/ship/libscna_env_skel.so ]]; then
    cp -a hexagon_ReleaseG_toolv19_v79/ship/libscna_env_skel.so "$out_dir/"
  fi
  {
    echo "architecture=v79"
    echo "audit_build_flavor=$flavor"
    echo "audit_build_flavor_id=$flavor_id"
    echo "runtime_variant=pair_static_d8"
    echo "kernel_impl=d7_pairret_noinline"
    echo "kernel_impl_id=3"
    echo "softmax_impl=fused_state_update"
    echo "softmax_impl_id=2"
    echo "scheduler=stage2_75_adaptive"
  } >"$out_dir/build_id.txt"
  (cd "$out_dir" && sha256sum libhtp_ops_skel.so >sha256.txt)
done

python3 "$project_dir/tools/create_manifest.py" --project "$project_dir"
echo "Stage2.9 artifacts built: ${flavors[*]}"
