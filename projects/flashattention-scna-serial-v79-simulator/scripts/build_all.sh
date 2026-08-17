#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
source_dir="$project_dir/src/htp-ops-lib-main"
artifact_root="$project_dir/artifacts/variants"
variants=(stage1_dynamic_row prepare_once_row pair_shared_dynamic pair_static_d8 pair_d8_fma_noinline pair_d8_fma_inline optimized)
requested="all"

if [[ $# -gt 0 ]]; then
  [[ $# -eq 2 && "$1" == "--variant" ]] || { echo "Usage: $0 [--variant NAME]" >&2; exit 2; }
  requested="$2"
fi

set +u
source "$script_dir/use_hexagon_sdk_6_6.sh"
set -u
command -v build_cmake >/dev/null || { echo "build_cmake is unavailable" >&2; exit 1; }

mkdir -p "$artifact_root"
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  [[ "$requested" == all || "$requested" == "$variant" ]] || continue
  inline=0
  [[ "$variant" == pair_d8_fma_inline ]] && inline=1
  cd "$source_dir"
  build_cmake hexagon DSP_ARCH=v79 SCNA_SIMULATOR=ON FIGURE8_ENABLE_PROFILE_TIMERS=ON \
    FIGURE8_ENABLE_LUT_EXP=OFF CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "SCNA_BUILD_VARIANT=$index" "SCNA_OPTIMIZED_INLINE=$inline"
  ship_dirs=("$source_dir"/hexagon_ReleaseG_toolv*_v79/ship)
  [[ ${#ship_dirs[@]} -eq 1 && -f "${ship_dirs[0]}/libhtp_ops_skel.so" ]] || {
    echo "Missing unique v79 DSP artifact for $variant" >&2; exit 1;
  }
  out="$artifact_root/$variant"
  mkdir -p "$out"
  cp -f "${ship_dirs[0]}/libhtp_ops_skel.so" "$out/scna_sim.so"
  sha256sum "$out/scna_sim.so" > "$out/sha256.txt"
  printf 'variant=%s\nbuild_id=%s\noptimized_inline=%s\n' "$variant" "$index" "$inline" > "$out/build_id.txt"
  file "$out/scna_sim.so" > "$out/file.txt"
done

[[ "$requested" == all ]] || [[ -f "$artifact_root/$requested/scna_sim.so" ]] || {
  echo "Unknown variant: $requested" >&2; exit 2;
}
echo "Built simulator variants under $artifact_root"
