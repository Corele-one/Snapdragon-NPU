#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
remote_dir="/data/local/tmp/stage2_softmax_v79"
run_id="${1:-20260822_stage2_component_profile_v1}"
out="$project_dir/results/runs/$run_id"
host="$project_dir/src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/htp_ops_test"
mkdir -p "$out/raw" "$out/evidence"

"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" \
  --kernel-impl d7_pairret_noinline --mode ping >"$out/raw/deploy.log" 2>&1
adb push "$host" "$remote_dir/htp_ops_test" >/dev/null

for label in component_profile_baseline fused_component_profile; do
  artifact="$project_dir/artifacts/variants/$label/libhtp_ops_skel.so"
  adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >/dev/null
  adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >/dev/null
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test \
    --figure8-attn --mode scna-fp16 --scna-variant pair_static_d8 --workers 1 --scna-width 8 \
    --mask-mode full --qo-len 32 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 \
    --warmup 2 --iters 3 --seed figure8_fixed --events --compare-reference" \
    >"$out/raw/${label}_q32_kv4096.log" 2>&1
  grep -Eq 'FIG8_ATTENTION_COMPARE .*pass=1' "$out/raw/${label}_q32_kv4096.log"
  grep -q 'rowmax_mem=' "$out/raw/${label}_q32_kv4096.log"
done

sha256sum \
  "$project_dir/artifacts/variants/component_profile_baseline/libhtp_ops_skel.so" \
  "$project_dir/artifacts/variants/fused_component_profile/libhtp_ops_skel.so" \
  >"$out/evidence/artifact_sha256.txt"
cp "$0" "$out/evidence/run_component_profile.sh"
echo "completed: $out"
