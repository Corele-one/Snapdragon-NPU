#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
remote_dir="/data/local/tmp/scna_hvx_fp16_d8_pipeline_v79"
run_id="$(date -u +scna_hvx_d8_v79_%Y%m%dT%H%M%SZ)"
quick=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --quick) quick=1; shift ;;
    --help|-h)
      echo "Usage: $0 [--run-id ID] [--remote-dir PATH] [--quick]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

out="$project_dir/results/runs/$run_id"
mkdir -p "$out"/{raw/{micro,accuracy,attention,confirm,diagnostic,scaling,recovery},static,evidence,tables,figures}

impls=(static_d8_ref d7_serial d7_scalar_w d7_pairret_noinline d7_pairret_inline d7_quad_pipeline d7_prebroadcast qf16_tree_control piecewise_control combined_confirm)
screening=(d7_serial d7_scalar_w d7_pairret_noinline d7_quad_pipeline qf16_tree_control piecewise_control)
order_preserving=(d7_serial d7_scalar_w d7_pairret_noinline d7_quad_pipeline)
seeds=(figure8_fixed 20260810 20260811)
qos=(1 4 8 16 32)

micro_samples=30; micro_target_us=80000; sessions=5; warmup=5; iters=20
accuracy_masks=(full causal padding); accuracy_q=(1 4); accuracy_kv=(4093 4096); accuracy_dim=(64 128)
if [[ "$quick" == 1 ]]; then
  micro_samples=3; micro_target_us=10000; sessions=2; warmup=2; iters=3
  accuracy_masks=(full); accuracy_q=(1); accuracy_kv=(4093); accuracy_dim=(64); seeds=(figure8_fixed)
fi

for impl in "${impls[@]}"; do
  [[ -s "$project_dir/artifacts/variants/$impl/libhtp_ops_skel.so" ]] || {
    echo "Missing artifact $impl; run scripts/build_all_variants.sh" >&2; exit 1;
  }
done

complete() { [[ -s "$1" ]] && grep -Eq "$2" "$1"; }
run_remote() {
  local log="$1" marker="$2"; shift 2
  if complete "$log" "$marker"; then echo "resume: ${log#$out/}"; return 0; fi
  local attempt status recovery
  for attempt in 1 2; do
    set +e
    timeout 120s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test $*" >"$log" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]] && complete "$log" "$marker"; then return 0; fi
    recovery="$out/raw/recovery/$(basename "${log%.log}")_attempt${attempt}.log"
    {
      echo "exit_code=$status expected=$marker"
      adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./scna_env_smoke"
    } >"$recovery" 2>&1 || true
  done
  echo "UNAVAILABLE: ${log#$out/}" >&2
  return 1
}

deploy_artifact() {
  local impl="$1" artifact="$project_dir/artifacts/variants/$1/libhtp_ops_skel.so"
  adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >/dev/null
  adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >/dev/null
}

attention_args() {
  local label="$1" q="$2" kv="$3" awarmup="$4" aiters="$5" seed="$6" mask="${7:-full}" dim="${8:-128}" workers="${9:-1}"
  local mode=scna-fp16
  [[ "$label" == baseline ]] && mode=baseline
  [[ "$label" == lut-exp ]] && mode=lut-exp
  printf '%s' "--figure8-attn --mode $mode --scna-variant pair_static_d8 --workers $workers --scna-width 8 --mask-mode $mask --qo-len $q --kv-len $kv --n-heads 12 --n-kv-heads 2 --head-dim $dim --warmup $awarmup --iters $aiters --seed $seed --no-events"
}

carrier_for() {
  case "$1" in baseline|lut-exp|static_d8_ref) echo static_d8_ref ;; *) echo "$1" ;; esac
}

adb get-state >/dev/null
"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" --kernel-impl static_d8_ref --mode ping >"$out/raw/deploy.log" 2>&1

python3 "$project_dir/tools/collect_static_metrics.py" --project "$project_dir" --out-dir "$out/static"
python3 "$project_dir/tools/evaluate_static_gates.py" --metrics "$out/static/static_metrics.json" \
  --json-out "$out/static/static_gates.json" --csv-out "$out/static/static_gates.csv"

{
  echo "schema_version=3"
  echo "run_id=$run_id"
  echo "quick=$quick"
  echo "captured_at=$(date -u +%FT%TZ)"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.product.device
  adb shell getprop ro.board.platform
  adb shell getprop ro.soc.model
  adb shell getprop ro.build.fingerprint
  for impl in "${impls[@]}"; do
    sha256sum "$project_dir/artifacts/variants/$impl/libhtp_ops_skel.so"
  done
} >"$out/evidence/device_manifest.txt"
adb shell dumpsys thermalservice >"$out/evidence/thermal_before.txt" 2>&1 || true
{
  echo "captured_at=$(date -u +%FT%TZ)"
  echo "dsp_devfreq=UNAVAILABLE_ON_ANDROID_USER_BUILD"
  adb shell 'for c in /sys/devices/system/cpu/cpufreq/policy*; do echo -n "$(basename "$c")="; cat "$c/scaling_cur_freq" 2>/dev/null; echo -n "governor="; cat "$c/scaling_governor" 2>/dev/null; done'
} >"$out/evidence/frequency_snapshot.txt" 2>&1 || true
cp "$project_dir/experiment_spec.json" "$out/evidence/experiment_spec.json"
cp "$0" "$out/evidence/run_pipeline_experiment.sh"
{
  echo "hexagon_sdk=6.6.0.0"
  echo "hexagon_llvm_tools=19.0.07"
  echo "dsp_target=v79"
  echo "required_flag=-mv79"
  echo "artifact_compile_flags=artifacts/variants/<kernel_impl>/compile_flags.txt"
} >"$out/evidence/toolchain_manifest.txt"
python3 "$project_dir/tools/create_artifact_manifest.py" \
  --artifact-root "$project_dir/artifacts/variants" \
  --output "$out/evidence/artifact_manifest.json"

for impl in "${impls[@]}"; do
  deploy_artifact "$impl"
  pilot="$out/raw/micro/${impl}_pilot.log"
  run_remote "$pilot" 'SCNA_EXP_BENCH' --scna-exp-bench --scna-variant pair_static_d8 --scna-width 8 --warmup 5 --iters 200000
  elapsed="$(sed -n 's/.*pair_elapsed_us=\([0-9][0-9]*\).*/\1/p' "$pilot" | tail -1)"
  [[ -n "$elapsed" && "$elapsed" -gt 0 ]] || { echo "Micro calibration failed for $impl" >&2; exit 1; }
  calibrated=$(( (micro_target_us * 200000 + elapsed - 1) / elapsed ))
  (( calibrated < 200000 )) && calibrated=200000
  for sample in $(seq 1 "$micro_samples"); do
    run_remote "$out/raw/micro/${impl}_sample${sample}.log" 'SCNA_EXP_BENCH' \
      --scna-exp-bench --scna-variant pair_static_d8 --scna-width 8 --warmup 5 --iters "$calibrated"
  done
done

static_checksum="$(sed -n 's/.*checksum=\([^ ]*\).*/\1/p' "$out/raw/micro/static_d8_ref_pilot.log" | tail -1)"
{
  echo "kernel_impl,expected_checksum,observed_checksum,bitwise_pass"
  for impl in "${order_preserving[@]}"; do
    observed="$(sed -n 's/.*checksum=\([^ ]*\).*/\1/p' "$out/raw/micro/${impl}_pilot.log" | tail -1)"
    pass=0; [[ "$observed" == "$static_checksum" ]] && pass=1
    echo "$impl,$static_checksum,$observed,$pass"
  done
} >"$out/static/micro_bitwise_gate.csv"

accuracy_run() {
  local impl="$1"
  deploy_artifact "$impl"
  for mask in "${accuracy_masks[@]}"; do for q in "${accuracy_q[@]}"; do
    for kv in "${accuracy_kv[@]}"; do for dim in "${accuracy_dim[@]}"; do for seed in "${seeds[@]}"; do
      args="$(attention_args "$impl" "$q" "$kv" 1 1 "$seed" "$mask" "$dim")"
      log="$out/raw/accuracy/${impl}_${mask}_q${q}_kv${kv}_d${dim}_seed${seed}.log"
      run_remote "$log" 'FIG8_ATTENTION_COMPARE ' "$args --compare-reference --numeric-debug" || true
    done; done; done
  done; done
}

accuracy_run static_d8_ref
for impl in "${screening[@]}"; do accuracy_run "$impl"; done

accuracy_pass() {
  local impl="$1" expected
  expected=$((${#accuracy_masks[@]} * ${#accuracy_q[@]} * ${#accuracy_kv[@]} * ${#accuracy_dim[@]} * ${#seeds[@]}))
  local count pass_count=0 log
  count="$(find "$out/raw/accuracy" -maxdepth 1 -name "${impl}_*.log" -type f | wc -l)"
  for log in "$out/raw/accuracy/${impl}"_*.log; do
    [[ -f "$log" ]] && grep -q 'FIG8_ATTENTION_COMPARE .*pass=1' "$log" && pass_count=$((pass_count + 1))
  done
  [[ "$count" -eq "$expected" && "$pass_count" -eq "$expected" ]]
}

comparisons=(baseline lut-exp static_d8_ref)
for impl in "${screening[@]}"; do
  if accuracy_pass "$impl"; then
    if [[ " ${order_preserving[*]} " == *" $impl "* ]]; then
      grep -q "^$impl,.*,[1]$" "$out/static/micro_bitwise_gate.csv" && comparisons+=("$impl")
    else
      comparisons+=("$impl")
    fi
  fi
done
printf '%s\n' "${comparisons[@]}" >"$out/evidence/screening_comparisons.txt"

for session in $(seq 1 "$sessions"); do
  seed="${seeds[$(((session - 1) % ${#seeds[@]}))]}"
  for offset in "${!comparisons[@]}"; do
    index=$(((offset + session - 1) % ${#comparisons[@]}))
    label="${comparisons[$index]}"; deploy_artifact "$(carrier_for "$label")"
    for q in "${qos[@]}"; do
      args="$(attention_args "$label" "$q" 4096 "$warmup" "$iters" "$seed")"
      run_remote "$out/raw/attention/${label}_q${q}_s${session}.log" "phase=measure iteration=$((iters - 1)) .*ret=0" "$args"
    done
  done
  adb shell dumpsys thermalservice >"$out/evidence/thermal_session${session}.txt" 2>&1 || true
done

python3 "$project_dir/tools/analyze_pipeline_experiment.py" --run-dir "$out" --select-only \
  --selection-out "$out/evidence/combined_selection.json"
combined_enabled="$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["combined_enabled"]))' "$out/evidence/combined_selection.json")"
if [[ "$combined_enabled" == 1 ]]; then
  accuracy_run combined_confirm
  if accuracy_pass combined_confirm; then
    confirm=(baseline lut-exp static_d8_ref combined_confirm)
    for session in $(seq 1 "$sessions"); do
      seed="${seeds[$(((session - 1) % ${#seeds[@]}))]}"
      for offset in "${!confirm[@]}"; do
        index=$(((offset + session - 1) % ${#confirm[@]}))
        label="${confirm[$index]}"; deploy_artifact "$(carrier_for "$label")"
        args="$(attention_args "$label" 32 4096 "$warmup" "$iters" "$seed")"
        run_remote "$out/raw/confirm/${label}_q32_s${session}.log" "phase=measure iteration=$((iters - 1)) .*ret=0" "$args"
      done
    done
  fi
fi

for label in "${comparisons[@]}"; do
  deploy_artifact "$(carrier_for "$label")"
  for kv in 64 4096; do
    args="$(attention_args "$label" 32 "$kv" 1 1 figure8_fixed)"
    run_remote "$out/raw/diagnostic/${label}_kv${kv}.log" 'FIG8_ATTENTION_TIMERS' "$args --events" || true
  done
done

python3 "$project_dir/tools/analyze_pipeline_experiment.py" --run-dir "$out"
winner="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); a=[(d["latency"][k]["32"]["median"],k) for k,v in d["decisions"].items() if v["eligible"] and d["latency"][k]["32"]]; print(min(a)[1] if a else "static_d8_ref")' "$out/summary.json")"
for workers in 1 2 3 4 5 6 auto; do
  deploy_artifact "$(carrier_for "$winner")"
  args="$(attention_args "$winner" 32 4096 "$warmup" "$iters" figure8_fixed full 128 "$workers")"
  run_remote "$out/raw/scaling/${winner}_w${workers}.log" "phase=measure iteration=$((iters - 1)) .*ret=0" "$args" || true
done
adb shell dumpsys thermalservice >"$out/evidence/thermal_after.txt" 2>&1 || true
python3 "$project_dir/tools/analyze_pipeline_experiment.py" --run-dir "$out"
echo "completed: $out/SCNA_HVX_D8_PIPELINE_V79_REPORT_ZH.md"
