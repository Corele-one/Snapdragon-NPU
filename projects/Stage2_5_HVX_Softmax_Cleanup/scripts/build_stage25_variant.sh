#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
label="${1:-}"
fine_timers="${2:-0}"

case "$label" in
  stage1_d7)       softmax_impl=baseline;           flags=0 ;;
  stage2_final)    softmax_impl=fused_state_update; flags=0 ;;
  stage2_diagnostic) softmax_impl=fused_state_update; flags=0; fine_timers=1 ;;
  s25_a_mask_index) softmax_impl=fused_state_update; flags=1 ;;
  s25_b_full_mask_fastpath) softmax_impl=fused_state_update; flags=2 ;;
  s25_c_aligned_mask_load) softmax_impl=fused_state_update; flags=4 ;;
  s25_d_invariant_hoist) softmax_impl=fused_state_update; flags=8 ;;
  s25_e_task_geometry) softmax_impl=fused_state_update; flags=16 ;;
  s25_f_roundtrip_audit) softmax_impl=fused_state_update; flags=32 ;;
  # Formal 5-session gates retained only experiment B.  Keep the final mapping
  # explicit so rejected single-variable probes cannot leak into the artifact.
  stage2_5_final) softmax_impl=fused_state_update; flags=2 ;;
  stage3_diagnostic) softmax_impl=fused_state_update; flags=2; fine_timers=1 ;;
  *)
    echo "Usage: $0 {stage1_d7|stage2_final|stage2_diagnostic|s25_a_mask_index|s25_b_full_mask_fastpath|s25_c_aligned_mask_load|s25_d_invariant_hoist|s25_e_task_geometry|s25_f_roundtrip_audit|stage2_5_final|stage3_diagnostic} [fine_timers]" >&2
    exit 2
    ;;
esac

"$script_dir/build.sh" --variant pair_static_d8 --kernel-impl d7_pairret_noinline \
  --softmax-impl "$softmax_impl" --fine-timers "$fine_timers" \
  --stage25-flags "$flags" --dsp-only

dsp_ship_candidates=("$htp_dir"/hexagon_ReleaseG_toolv*_v79/ship)
[[ ${#dsp_ship_candidates[@]} -eq 1 && -f "${dsp_ship_candidates[0]}/libhtp_ops_skel.so" ]] || {
  echo "Missing unique v79 DSP artifact for $label" >&2
  exit 1
}

out="$project_dir/artifacts/variants/$label"
mkdir -p "$out"
cp -f "${dsp_ship_candidates[0]}/libhtp_ops_skel.so" "$out/"
sha256sum "$out/libhtp_ops_skel.so" >"$out/sha256.txt"
printf 'schema_version=1\nruntime_variant=pair_static_d8\nkernel_impl=d7_pairret_noinline\nkernel_impl_id=3\nsoftmax_impl=%s\nstage25_flags=%s\nfine_timers=%s\n' \
  "$softmax_impl" "$flags" "$fine_timers" >"$out/build_id.txt"
build_dir="${dsp_ship_candidates[0]%/ship}"
rg -m1 '^  FLAGS = .* -mv79' "$build_dir/build.ninja" >"$out/compile_flags.txt"
rg -A2 -m1 'build .*flash_attn.c.obj:' "$build_dir/build.ninja" >>"$out/compile_flags.txt"

objdump=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump
"$objdump" -d --no-show-raw-insn "$out/libhtp_ops_skel.so" >"$out/$label.v79.disasm.txt"

# Capture the exact project-owned source delta used by every compile-time
# variant.  The imported Stage2 tree remains read-only and is only a diff base.
stage2_dir="$project_dir/../Stage2_HVX_Softmax_Dataflow/src/htp-ops-lib-main"
: >"$out/source_diff.patch"
for rel in CMakeLists.txt include/op_reg.h src/dsp/ops/flash_attn.c src/host/test.c; do
  set +e
  diff -u --label "Stage2/$rel" --label "Stage2.5/$rel" \
    "$stage2_dir/$rel" "$htp_dir/$rel" >>"$out/source_diff.patch"
  diff_status=$?
  set -e
  [[ $diff_status -eq 0 || $diff_status -eq 1 ]] || exit "$diff_status"
done
sha256sum "$htp_dir/CMakeLists.txt" "$htp_dir/include/op_reg.h" \
  "$htp_dir/src/dsp/ops/flash_attn.c" "$htp_dir/src/host/test.c" >"$out/source_sha256.txt"

echo "Built $label softmax=$softmax_impl flags=$flags fine_timers=$fine_timers sha256=$(cut -d' ' -f1 "$out/sha256.txt")"
