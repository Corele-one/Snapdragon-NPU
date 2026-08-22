#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
artifact_root="$project_dir/artifacts/variants"

mkdir -p "$artifact_root"

build_and_archive() {
  local kernel_impl="$1"
  local kernel_impl_id="$2"
  shift 2

  "$script_dir/build.sh" --variant pair_static_d8 --kernel-impl "$kernel_impl" "$@"

  local dsp_ship_candidates=("$htp_dir"/hexagon_ReleaseG_toolv*_v79/ship)
  [[ ${#dsp_ship_candidates[@]} -eq 1 && -f "${dsp_ship_candidates[0]}/libhtp_ops_skel.so" ]] || {
    echo "Missing unique v79 DSP artifact for $kernel_impl" >&2
    exit 1
  }

  local out="$artifact_root/$kernel_impl"
  mkdir -p "$out"
  cp -f "${dsp_ship_candidates[0]}/libhtp_ops_skel.so" "$out/"
  sha256sum "$out/libhtp_ops_skel.so" >"$out/sha256.txt"
  printf 'schema_version=1\nruntime_variant=pair_static_d8\nkernel_impl=%s\nkernel_impl_id=%s\n' \
    "$kernel_impl" "$kernel_impl_id" >"$out/build_id.txt"
  rg -m1 '^  FLAGS = .* -mv79' "${dsp_ship_candidates[0]%/ship}/build.ninja" >"$out/compile_flags.txt"
  rg -A2 -m1 'build .*scna_exp2.c.obj:' "${dsp_ship_candidates[0]%/ship}/build.ninja" >>"$out/compile_flags.txt"
}

# The first build also produces the shared Android host binaries.
build_and_archive d7_pairret_noinline 3
build_and_archive static_d8_ref 0 --dsp-only

python3 "$project_dir/tools/create_artifact_manifest.py" \
  --artifact-root "$artifact_root" --output "$artifact_root/manifest.json"

echo "Built fresh Stage 1 baselines under $artifact_root"
