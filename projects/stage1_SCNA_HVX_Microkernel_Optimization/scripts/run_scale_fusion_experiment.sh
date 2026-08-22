#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
remote_dir="/data/local/tmp/stage1_scna_v79"
run_id="scale_fusion_$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --help|-h) echo "Usage: $0 [--run-id ID] [--remote-dir PATH]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

out="$project_dir/results/runs/$run_id"
mkdir -p "$out"/{raw,evidence,static}
artifact="$project_dir/artifacts/variants/d7_pairret_scale_probe/libhtp_ops_skel.so"
[[ -s "$artifact" ]] || { echo "Missing scale probe artifact" >&2; exit 1; }

adb get-state >/dev/null
"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" \
  --kernel-impl d7_pairret_scale_probe --mode ping >"$out/raw/deploy.log" 2>&1

{
  echo "schema_version=1"
  echo "run_id=$run_id"
  echo "captured_at=$(date -u +%FT%TZ)"
  echo "source_git_commit=$(git -C "$project_dir" rev-parse HEAD)"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.board.platform
  adb shell getprop ro.soc.model
  adb shell getprop ro.build.fingerprint
  sha256sum "$artifact"
} >"$out/evidence/device_manifest.txt"
adb shell dumpsys thermalservice >"$out/evidence/thermal_before.txt" 2>&1 || true
cp "$0" "$out/evidence/run_scale_fusion_experiment.sh"
cp "$project_dir/artifacts/variants/d7_pairret_scale_probe/build_id.txt" "$out/evidence/"
cp "$project_dir/artifacts/variants/d7_pairret_scale_probe/compile_flags.txt" "$out/evidence/"

objdump=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump
"$objdump" -d --no-show-raw-insn "$artifact" >"$out/static/d7_pairret_scale_probe.v79.disasm.txt"

run_bench() {
  local log="$1" iters="$2"
  timeout 120s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --scna-exp-bench --scna-variant pair_static_d8 --scna-width 8 --warmup 5 --iters '$iters'" >"$log" 2>&1
  grep -q 'SCNA_EXP_BENCH .*scale_head_dim=128' "$log"
}

pilot="$out/raw/pilot.log"
run_bench "$pilot" 200000
elapsed="$(sed -n 's/.*separate_scale_scna_elapsed_us=\([0-9][0-9]*\).*/\1/p' "$pilot" | tail -1)"
[[ -n "$elapsed" && "$elapsed" -gt 0 ]] || { echo "Scale probe calibration failed" >&2; exit 1; }
iters=$(( (80000 * 200000 + elapsed - 1) / elapsed ))
(( iters < 200000 )) && iters=200000
for sample in $(seq 1 30); do
  run_bench "$out/raw/sample${sample}.log" "$iters"
  echo "done: sample${sample}"
done

adb shell dumpsys thermalservice >"$out/evidence/thermal_after.txt" 2>&1 || true
echo "completed: $out"
