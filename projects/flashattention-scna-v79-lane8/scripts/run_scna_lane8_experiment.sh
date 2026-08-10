#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
timestamp="${SCNA_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
result_dir="${SCNA_RESULT_DIR:-$project_dir/results/v79/scna-lane8/$timestamp}"
remote_dir="${SCNA_REMOTE_DIR:-/data/local/tmp/scna_v79_lane8}"
qos=(4 8 16 32)
sessions=3
warmup=5
iters=30

mkdir -p "$result_dir"/{raw,micro,correctness,trace,provenance,static,analysis}

device_line="$(adb devices -l | awk 'NR>1 && $2=="device" {print; exit}')"
if [[ -z "$device_line" ]]; then
  printf '%s\n' "DEVICE_BLOCKED reason=no_authorized_adb_device timestamp=$timestamp" | tee "$result_dir/provenance/device_blocker.txt"
  exit 3
fi

"$script_dir/build.sh" --dsp-arch v79 2>&1 | tee "$result_dir/provenance/build.log"
"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" --mode ping 2>&1 | tee "$result_dir/provenance/ping.log"

dsp_binary="$htp_dir/hexagon_ReleaseG_toolv19_v79/ship/libhtp_ops_skel.so"
binary_sha256="$(sha256sum "$dsp_binary" | awk '{print $1}')"
printf 'SCNA_PROVENANCE binary_sha256=%s isa=v79 sdk=6.6.0.0 tools=19.0.07 seed=figure8_fixed\n' "$binary_sha256" \
  | tee "$result_dir/provenance/run.txt"
sha256sum "$dsp_binary" "$htp_dir/android_ReleaseG_aarch64/ship/htp_ops_test" \
  "$htp_dir/include/dsp/scna_params.h" > "$result_dir/provenance/sha256sum.txt"
git -C "$project_dir/../.." status --short > "$result_dir/provenance/git_status.txt"
adb shell getprop > "$result_dir/provenance/device_getprop.txt"

run_remote_attention() {
  local mode="$1" layout="$2" qo="$3" phase_iters="$4" event_arg="$5" extra_arg="${6:-}"
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --figure8-attn --mode '$mode' --scna-layout '$layout' --scna-width 8 --mask-mode full --qo-len '$qo' --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup '$warmup' --iters '$phase_iters' $event_arg $extra_arg"
}

run_remote_micro() {
  local layout="$1" phase_iters="$2"
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --scna-exp-bench --scna-layout '$layout' --scna-width 8 --warmup 5 --iters '$phase_iters'"
}

run_remote_correctness() {
  local layout="$1" mask="$2" kv="$3" dim="$4"
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --figure8-attn --mode scna-fp16 --scna-layout '$layout' --scna-width 8 --mask-mode '$mask' --qo-len 4 --kv-len '$kv' --n-heads 12 --n-kv-heads 2 --head-dim '$dim' --warmup 1 --iters 1 --no-events --compare-reference"
}

# Headline matrix: event recording disabled. Serial/lane8 use adjacent ABBA order.
for session in $(seq 1 "$sessions"); do
  session_dir="$result_dir/raw/session_$session"
  mkdir -p "$session_dir"
  adb shell dumpsys thermalservice > "$session_dir/thermal_before.txt" 2>&1 || true
  for qo in "${qos[@]}"; do
    run_remote_attention baseline serial "$qo" "$iters" --no-events 2>&1 | tee "$session_dir/baseline_q${qo}.log"
    run_remote_attention lut-exp serial "$qo" "$iters" --no-events 2>&1 | tee "$session_dir/lut_exp_q${qo}.log"
    for layout in serial lane8 lane8 serial; do
      repetition_file="$session_dir/scna_${layout}_q${qo}_$(find "$session_dir" -name "scna_${layout}_q${qo}_*.log" | wc -l).log"
      run_remote_attention scna-fp16 "$layout" "$qo" "$iters" --no-events 2>&1 | tee "$repetition_file"
    done
  done
  adb shell dumpsys thermalservice > "$session_dir/thermal_after.txt" 2>&1 || true
done

# Calibrate each evaluator to >=50 ms, then collect 30 aggregate samples.
for layout in serial lane8; do
  calibrated_iters=1000
  while :; do
    calibration="$result_dir/micro/${layout}_calibration_${calibrated_iters}.log"
    run_remote_micro "$layout" "$calibrated_iters" 2>&1 | tee "$calibration"
    elapsed="$(sed -nE 's/.* elapsed_us=([0-9]+).*/\1/p' "$calibration" | tail -n 1)"
    [[ -n "$elapsed" ]] || { echo "Unable to parse microbenchmark elapsed_us" >&2; exit 1; }
    (( elapsed >= 50000 )) && break
    calibrated_iters=$((calibrated_iters * 2))
  done
  printf '%s\n' "$calibrated_iters" > "$result_dir/micro/${layout}_iters.txt"
  for sample in $(seq 0 29); do
    run_remote_micro "$layout" "$calibrated_iters" 2>&1 | tee "$result_dir/micro/${layout}_sample_${sample}.log"
  done
done

# Correctness matrix. Device-side microbench covers lane oracle and dense [-256,0].
for layout in serial lane8; do
  for mask in full causal padding; do
    for kv in 4093 4096; do
      for dim in 64 128; do
        run_remote_correctness "$layout" "$mask" "$kv" "$dim" 2>&1 \
          | tee "$result_dir/correctness/${layout}_${mask}_kv${kv}_d${dim}.log"
      done
    done
  done
done

# Independent event replay: exactly 1 warmup + 3 measured, same binary/shape/seed.
for layout in serial lane8; do
  layout_dir="$result_dir/trace/$layout"
  mkdir -p "$layout_dir"
  for qo in "${qos[@]}"; do
    log="$layout_dir/raw_q${qo}.log"
    printf 'SCNA_PROVENANCE binary_sha256=%s isa=v79 layout=%s scna_width=8 qo_len=%s seed=figure8_fixed\n' \
      "$binary_sha256" "$layout" "$qo" | tee "$log"
    warmup=1 run_remote_attention scna-fp16 "$layout" "$qo" 3 "" 2>&1 | tee -a "$log"
  done
  python3 "$project_dir/tools/generate_figure8_perfetto_trace.py" --input-dir "$layout_dir" --out-dir "$layout_dir" \
    --binary-sha256 "$binary_sha256"
  python3 "$project_dir/tools/audit_perfetto_trace.py" --trace "$layout_dir/scna_${layout}_all_q.perfetto.json" \
    --summary "$layout_dir/event_trace_summary.json" --layout "$layout" --width 8 --binary-sha256 "$binary_sha256" \
    | tee "$layout_dir/perfetto_audit.json"
done
python3 "$project_dir/tools/combine_scna_perfetto_traces.py" \
  --serial "$result_dir/trace/serial/scna_serial_all_q.perfetto.json" \
  --lane8 "$result_dir/trace/lane8/scna_lane8_all_q.perfetto.json" \
  --output "$result_dir/trace/scna_serial_vs_lane8_all_q.perfetto.json"

python3 "$project_dir/tools/analyze_scna_disassembly.py" --binary "$dsp_binary" \
  --objdump /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump \
  --out-dir "$result_dir/static"
python3 "$project_dir/tools/analyze_scna_lane8_results.py" --result-dir "$result_dir"

echo "SCNA lane8 experiment complete: $result_dir"
