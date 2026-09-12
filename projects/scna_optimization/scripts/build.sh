#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"

set +u
# shellcheck source=use_hexagon_sdk_6_6.sh
source "$script_dir/use_hexagon_sdk_6_6.sh"
set -u
command -v build_cmake >/dev/null || { echo "build_cmake is unavailable" >&2; exit 1; }

build_tree() {
  local source_dir="$1"
  (
    cd "$source_dir"
    build_cmake android
    build_cmake hexagon DSP_ARCH=v79 FIGURE8_ENABLE_PROFILE_TIMERS=ON FIGURE8_ENABLE_LUT_EXP=OFF
  )
}

build_tree "$project_dir/reference/flashattention/htp-ops-lib-main"
build_tree "$project_dir/src/htp-ops-lib-main"

reference_android="$project_dir/reference/flashattention/htp-ops-lib-main/android_ReleaseG_aarch64/ship"
reference_dsp=("$project_dir/reference/flashattention/htp-ops-lib-main"/hexagon_ReleaseG_toolv*_v79/ship)
optimized_android="$project_dir/src/htp-ops-lib-main/android_ReleaseG_aarch64/ship"
optimized_dsp=("$project_dir/src/htp-ops-lib-main"/hexagon_ReleaseG_toolv*_v79/ship)
[[ ${#reference_dsp[@]} -eq 1 && ${#optimized_dsp[@]} -eq 1 ]] || {
  echo "Expected exactly one v79 DSP build directory per source tree" >&2
  exit 1
}

mkdir -p "$project_dir/artifacts/reference" "$project_dir/artifacts/scna"
cp "$reference_android/htp_ops_test" "$reference_android/libhtp_ops.so" "$project_dir/artifacts/reference/"
cp "${reference_dsp[0]}/libhtp_ops_skel.so" "$project_dir/artifacts/reference/"
cp "$optimized_android/htp_ops_test" "$optimized_android/libhtp_ops.so" "$project_dir/artifacts/scna/"
cp "$optimized_android/scna_env_smoke" "$optimized_android/libscna_env.so" "$project_dir/artifacts/scna/"
cp "${optimized_dsp[0]}/libhtp_ops_skel.so" "${optimized_dsp[0]}/libscna_env_skel.so" "$project_dir/artifacts/scna/"

(
  cd "$project_dir"
  sha256sum artifacts/reference/* > artifacts/reference/sha256.txt
  sha256sum artifacts/scna/* > artifacts/scna/sha256.txt
)
echo "Built reference and frozen SCNA bundles for v79."
