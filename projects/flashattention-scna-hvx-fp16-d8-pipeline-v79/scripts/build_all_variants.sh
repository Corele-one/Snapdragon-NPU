#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
artifact_root="$project_dir/artifacts/variants"
kernel_impls=(
  static_d8_ref
  d7_serial
  d7_scalar_w
  d7_pairret_noinline
  d7_pairret_inline
  d7_quad_pipeline
  d7_prebroadcast
  qf16_tree_control
  piecewise_control
  combined_confirm
)
mkdir -p "$artifact_root"

for index in "${!kernel_impls[@]}"; do
  kernel_impl="${kernel_impls[$index]}"
  args=(--variant pair_static_d8 --kernel-impl "$kernel_impl")
  (( index > 0 )) && args+=(--dsp-only)
  "$script_dir/build.sh" "${args[@]}"

  dsp_ship=("$htp_dir"/hexagon_ReleaseG_toolv*_v79/ship)
  [[ ${#dsp_ship[@]} -eq 1 && -f "${dsp_ship[0]}/libhtp_ops_skel.so" ]] || {
    echo "Missing unique v79 DSP artifact for $kernel_impl" >&2; exit 1;
  }
  out="$artifact_root/$kernel_impl"
  mkdir -p "$out"
  cp -f "${dsp_ship[0]}/libhtp_ops_skel.so" "$out/"
  sha256sum "$out/libhtp_ops_skel.so" > "$out/sha256.txt"
  printf 'schema_version=3\nruntime_variant=pair_static_d8\nkernel_impl=%s\nkernel_impl_id=%s\n' \
    "$kernel_impl" "$index" > "$out/build_id.txt"
  rg -m1 '^  FLAGS = .* -mv79' "${dsp_ship[0]%/ship}/build.ninja" > "$out/compile_flags.txt"
  rg -A2 -m1 'build .*scna_exp2.c.obj:' "${dsp_ship[0]%/ship}/build.ninja" >> "$out/compile_flags.txt"
done
python3 "$project_dir/tools/create_artifact_manifest.py" \
  --artifact-root "$artifact_root" --output "$artifact_root/manifest.json"
echo "Built ${#kernel_impls[@]} independently hashed v79 implementations under $artifact_root"
