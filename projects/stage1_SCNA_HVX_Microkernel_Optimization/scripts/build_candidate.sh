#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
artifact_root="$project_dir/artifacts/variants"

[[ $# -eq 1 ]] || {
  echo "Usage: $0 {d7_pairret_two_acc|d7_pairret_short_const|d7_pairret_two_neuron|d7_pairret_scale_probe}" >&2
  exit 2
}

kernel_impl="$1"
case "$kernel_impl" in
  d7_pairret_two_acc) kernel_impl_id=10 ;;
  d7_pairret_short_const) kernel_impl_id=11 ;;
  d7_pairret_two_neuron) kernel_impl_id=12 ;;
  d7_pairret_scale_probe) kernel_impl_id=13 ;;
  *)
    echo "Usage: $0 {d7_pairret_two_acc|d7_pairret_short_const|d7_pairret_two_neuron|d7_pairret_scale_probe}" >&2
    exit 2 ;;
esac
"$script_dir/build.sh" --variant pair_static_d8 --kernel-impl "$kernel_impl" --dsp-only

dsp_ship_candidates=("$htp_dir"/hexagon_ReleaseG_toolv*_v79/ship)
[[ ${#dsp_ship_candidates[@]} -eq 1 && -f "${dsp_ship_candidates[0]}/libhtp_ops_skel.so" ]] || {
  echo "Missing unique v79 DSP artifact for $kernel_impl" >&2
  exit 1
}

out="$artifact_root/$kernel_impl"
mkdir -p "$out"
cp -f "${dsp_ship_candidates[0]}/libhtp_ops_skel.so" "$out/"
sha256sum "$out/libhtp_ops_skel.so" >"$out/sha256.txt"
printf 'schema_version=1\nruntime_variant=pair_static_d8\nkernel_impl=%s\nkernel_impl_id=%s\n' \
  "$kernel_impl" "$kernel_impl_id" >"$out/build_id.txt"
rg -m1 '^  FLAGS = .* -mv79' "${dsp_ship_candidates[0]%/ship}/build.ninja" >"$out/compile_flags.txt"
rg -A2 -m1 'build .*scna_exp2.c.obj:' "${dsp_ship_candidates[0]%/ship}/build.ninja" >>"$out/compile_flags.txt"

echo "Built candidate $kernel_impl under $out"
