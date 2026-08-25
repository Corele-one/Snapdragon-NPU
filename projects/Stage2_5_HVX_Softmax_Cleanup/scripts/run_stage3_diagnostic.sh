#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
htp_dir="$root_dir/src/htp-ops-lib-main"
label="${1:-stage3_diagnostic}"
case "$label" in
    stage2_diagnostic) fast_arg="" ;;
    stage3_diagnostic) fast_arg="--full-mask-fastpath" ;;
    *) echo "usage: $0 [stage2_diagnostic|stage3_diagnostic]" >&2; exit 2 ;;
esac
artifact="$root_dir/artifacts/variants/$label/libhtp_ops_skel.so"
remote_dir=/data/local/tmp/stage25_softmax_v79
out="$root_dir/results/stage3_handoff"
mkdir -p "$out"
test -s "$artifact"
fixture="$out/full_q32_kv4096_d128_seedfigure8_fixed.bin"
python3 "$root_dir/tools/generate_fixture.py" --output "$fixture" --qo-len 32 --kv-len 4096 \
    --n-heads 12 --n-kv-heads 2 --head-dim 128 --mask-mode full --seed figure8_fixed >"$out/fixture_generation.log"
adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >/dev/null
adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >/dev/null
adb push "$fixture" "$remote_dir/fixture.bin" >/dev/null
log="$out/${label}_q32.log"
adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --figure8-attn --mode scna-fp16 --scna-variant pair_static_d8 --workers 1 --scna-width 8 --mask-mode full --qo-len 32 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup 5 --iters 5 --seed figure8_fixed --fixture '$remote_dir/fixture.bin' $fast_arg --events" >"$log" 2>&1
grep 'FIG8_ATTENTION_TIMERS' "$log" >"$out/${label}_timers.txt"
grep 'FIG8_ATTENTION_KERNEL' "$log" >"$out/${label}_kernel.txt" || :
sha256sum "$artifact" "$fixture" >"$out/sha256.txt"
python3 "$root_dir/tools/analyze_stage3_handoff.py" \
    --input "$log" --output "$out/${label}_summary.json"
echo "completed: $out"
