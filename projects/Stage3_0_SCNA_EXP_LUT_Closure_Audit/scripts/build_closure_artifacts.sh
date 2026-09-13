#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
requested="${1:-all}"

case "$requested" in
  all|exp_lut_system|scna_system|fair_combined|lut_only|scna_only) ;;
  *) echo "Usage: $0 [all|exp_lut_system|scna_system|fair_combined|lut_only|scna_only]" >&2; exit 2 ;;
esac

set +u
source "$script_dir/use_hexagon_sdk_6_6.sh"
set -u
command -v build_cmake >/dev/null || { echo "build_cmake is unavailable" >&2; exit 1; }

fair_tree="$project_dir/src/htp-ops-lib-main"
exp_tree="$project_dir/systems/exp_lut_system/htp-ops-lib-main"
scna_tree="$project_dir/systems/scna_system/htp-ops-lib-main"

mapfile -t helper_hashes < <(sha256sum \
  "$fair_tree/include/dsp/flash_attn_rowsum.h" \
  "$exp_tree/include/dsp/flash_attn_rowsum.h" \
  "$scna_tree/include/dsp/flash_attn_rowsum.h" | awk '{print $1}')
[[ "${helper_hashes[0]}" == "${helper_hashes[1]}" && "${helper_hashes[0]}" == "${helper_hashes[2]}" ]] || {
  echo "shared rowsum helper differs between systems" >&2; exit 1;
}

build_host() {
  local tree="$1" out="$2"
  (cd "$tree" && build_cmake android)
  mkdir -p "$out"
  cp -a "$tree/android_ReleaseG_aarch64/ship/htp_ops_test" \
    "$tree/android_ReleaseG_aarch64/ship/libhtp_ops.so" "$out/"
  for optional in libscna_env.so scna_env_smoke; do
    [[ ! -f "$tree/android_ReleaseG_aarch64/ship/$optional" ]] || \
      cp -a "$tree/android_ReleaseG_aarch64/ship/$optional" "$out/"
  done
}

build_dsp() {
  local tree="$1" out="$2" flavor="$3" kind="$4"
  shift 4
  (cd "$tree" && build_cmake hexagon DSP_ARCH=v79 FIGURE8_ENABLE_PROFILE_TIMERS=ON \
    FIGURE8_ENABLE_LUT_EXP=OFF "$@")
  local ship=("$tree"/hexagon_ReleaseG_toolv*_v79/ship)
  [[ ${#ship[@]} -eq 1 ]] || { echo "expected one v79 ship directory for $tree" >&2; exit 1; }
  mkdir -p "$out"
  cp -a "${ship[0]}/libhtp_ops_skel.so" "$out/"
  [[ ! -f "${ship[0]}/libscna_env_skel.so" ]] || cp -a "${ship[0]}/libscna_env_skel.so" "$out/"
  {
    echo "audit=stage3_0_closure"
    echo "artifact=$flavor"
    echo "system_kind=$kind"
    echo "architecture=v79"
    echo "sdk=6.6.0.0"
    echo "rowsum_helper_sha256=${helper_hashes[0]}"
  } >"$out/build_id.txt"
  (cd "$out" && sha256sum lib*.so >sha256.txt)
}

build_fair_host=0
for flavor in fair_combined lut_only scna_only; do
  [[ "$requested" == all || "$requested" == "$flavor" ]] || continue
  if [[ $build_fair_host -eq 0 ]]; then
    build_host "$fair_tree" "$project_dir/artifacts/host"
    build_fair_host=1
  fi
  case "$flavor" in fair_combined) flavor_id=0 ;; lut_only) flavor_id=1 ;; scna_only) flavor_id=2 ;; esac
  build_dsp "$fair_tree" "$project_dir/artifacts/$flavor" "$flavor" fair-attribution \
    SCNA_BUILD_VARIANT=3 SCNA_OPTIMIZED_INLINE=0 SCNA_OPTIMIZED_IMPL=0 SCNA_KERNEL_IMPL=3 \
    STAGE2_SOFTMAX_IMPL=2 STAGE2_ENABLE_FINE_TIMERS=OFF "AUDIT_BUILD_FLAVOR=$flavor_id"
done

if [[ "$requested" == all || "$requested" == exp_lut_system ]]; then
  build_host "$exp_tree" "$project_dir/artifacts/exp_lut_system"
  build_dsp "$exp_tree" "$project_dir/artifacts/exp_lut_system" exp_lut_system original-flashattention
fi

if [[ "$requested" == all || "$requested" == scna_system ]]; then
  build_host "$scna_tree" "$project_dir/artifacts/scna_system"
  build_dsp "$scna_tree" "$project_dir/artifacts/scna_system" scna_system optimized-scna \
    SCNA_BUILD_VARIANT=3 SCNA_OPTIMIZED_INLINE=0 SCNA_OPTIMIZED_IMPL=0 SCNA_KERNEL_IMPL=3 \
    STAGE2_SOFTMAX_IMPL=2 STAGE2_ENABLE_FINE_TIMERS=OFF
fi

python3 "$project_dir/tools/create_manifest.py" --project "$project_dir" >/dev/null
echo "Stage 3.0 artifacts built: $requested"
