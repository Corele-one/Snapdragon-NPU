#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
candidate="${1:-tail_s_resident}"
fine_timers="${2:-0}"

case "$candidate" in
  fusion_baseline) softmax_impl=baseline ;;
  tail_s_resident) softmax_impl=tail_s_resident ;;
  fused_state_update) softmax_impl=fused_state_update ;;
  fused_state_unroll2) softmax_impl=fused_state_unroll2 ;;
  component_profile_baseline) softmax_impl=baseline; fine_timers=1 ;;
  fused_component_profile) softmax_impl=fused_state_update; fine_timers=1 ;;
  *) echo "Unsupported candidate: $candidate" >&2; exit 2 ;;
esac

"$script_dir/build.sh" --variant pair_static_d8 --kernel-impl d7_pairret_noinline \
  --softmax-impl "$softmax_impl" --fine-timers "$fine_timers" --dsp-only

dsp_ship_candidates=("$htp_dir"/hexagon_ReleaseG_toolv*_v79/ship)
[[ ${#dsp_ship_candidates[@]} -eq 1 && -f "${dsp_ship_candidates[0]}/libhtp_ops_skel.so" ]] || {
  echo "Missing unique v79 DSP artifact for $candidate" >&2
  exit 1
}

out="$project_dir/artifacts/variants/$candidate"
mkdir -p "$out"
cp -f "${dsp_ship_candidates[0]}/libhtp_ops_skel.so" "$out/"
sha256sum "$out/libhtp_ops_skel.so" >"$out/sha256.txt"
printf 'schema_version=1\nruntime_variant=pair_static_d8\nkernel_impl=d7_pairret_noinline\nkernel_impl_id=3\nsoftmax_impl=%s\nfine_timers=%s\n' \
  "$candidate" "$fine_timers" >"$out/build_id.txt"
rg -m1 '^  FLAGS = .* -mv79' "${dsp_ship_candidates[0]%/ship}/build.ninja" >"$out/compile_flags.txt"
rg -A2 -m1 'build .*flash_attn.c.obj:' "${dsp_ship_candidates[0]%/ship}/build.ninja" >>"$out/compile_flags.txt"

echo "Built Stage 2 candidate $candidate under $out"
