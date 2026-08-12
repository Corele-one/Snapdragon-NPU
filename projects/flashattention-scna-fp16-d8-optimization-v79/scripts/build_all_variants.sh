#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
artifact_root="$project_dir/artifacts/variants"
variants=(stage1_dynamic_row prepare_once_row pair_shared_dynamic pair_static_d8 pair_d8_fma_noinline pair_d8_fma_inline optimized)
final_policy="${FINAL_POLICY:-noinline}"

[[ "$final_policy" == inline || "$final_policy" == noinline ]] || {
  echo "FINAL_POLICY must be inline or noinline" >&2; exit 2;
}
mkdir -p "$artifact_root"

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  inline=0
  [[ "$variant" == pair_d8_fma_inline ]] && inline=1
  [[ "$variant" == optimized && "$final_policy" == inline ]] && inline=1
  args=(--variant "$variant" --optimized-inline "$inline")
  (( index > 0 )) && args+=(--dsp-only)
  "$script_dir/build.sh" "${args[@]}"

  dsp_ship=("$htp_dir"/hexagon_ReleaseG_toolv*_v79/ship)
  [[ ${#dsp_ship[@]} -eq 1 && -f "${dsp_ship[0]}/libhtp_ops_skel.so" ]] || {
    echo "Missing unique v79 DSP artifact for $variant" >&2; exit 1;
  }
  out="$artifact_root/$variant"
  mkdir -p "$out"
  cp -f "${dsp_ship[0]}/libhtp_ops_skel.so" "$out/"
  sha256sum "$out/libhtp_ops_skel.so" > "$out/sha256.txt"
  printf 'variant=%s\nbuild_id=%s\noptimized_inline=%s\n' "$variant" "$index" "$inline" > "$out/build_id.txt"
  rg -m1 '^  FLAGS = .* -mv79' "${dsp_ship[0]%/ship}/build.ninja" > "$out/compile_flags.txt"
  rg -A2 -m1 'build .*scna_exp2.c.obj:' "${dsp_ship[0]%/ship}/build.ninja" >> "$out/compile_flags.txt"
done

echo "Built ${#variants[@]} independent v79 DSP libraries under $artifact_root"
