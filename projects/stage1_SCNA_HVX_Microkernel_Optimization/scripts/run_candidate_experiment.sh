#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
remote_dir="/data/local/tmp/stage1_scna_v79"
candidate="d7_pairret_short_const"
run_id="candidate_$(date -u +%Y%m%dT%H%M%SZ)"
quick=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate) candidate="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --quick) quick=1; shift ;;
    --help|-h)
      echo "Usage: $0 [--candidate IMPL] [--run-id ID] [--remote-dir PATH] [--quick]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$candidate" in
  d7_pairret_short_const) ;;
  *) echo "Candidate is not registered for a dynamic experiment: $candidate" >&2; exit 2 ;;
esac

out="$project_dir/results/runs/$run_id"
mkdir -p "$out"/{raw/{micro,lut_micro,accuracy,attention,diagnostic,recovery},static,evidence}

samples=30
micro_target_us=80000
lut_target_ms=200
sessions=5
warmup=5
iters=20
seeds=(figure8_fixed 20260810 20260811)
accuracy_masks=(full causal padding)
accuracy_q=(1 4)
accuracy_kv=(4093 4096)
accuracy_dim=(64 128)
if [[ "$quick" == 1 ]]; then
  samples=3
  micro_target_us=10000
  lut_target_ms=20
  sessions=2
  warmup=2
  iters=3
  seeds=(figure8_fixed)
  accuracy_masks=(full)
  accuracy_q=(1)
  accuracy_kv=(4093)
  accuracy_dim=(64)
fi

pairret="$project_dir/artifacts/variants/d7_pairret_noinline/libhtp_ops_skel.so"
candidate_artifact="$project_dir/artifacts/variants/$candidate/libhtp_ops_skel.so"
for artifact in "$pairret" "$candidate_artifact"; do
  [[ -s "$artifact" ]] || { echo "Missing artifact: $artifact" >&2; exit 1; }
done

complete() { [[ -s "$1" ]] && grep -Eq "$2" "$1"; }

run_remote() {
  local log="$1" marker="$2"
  shift 2
  if complete "$log" "$marker"; then
    echo "resume: ${log#$out/}"
    return 0
  fi
  local attempt status recovery
  for attempt in 1 2; do
    set +e
    timeout 120s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test $*" >"$log" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]] && complete "$log" "$marker"; then
      echo "done: ${log#$out/}"
      return 0
    fi
    recovery="$out/raw/recovery/$(basename "${log%.log}")_attempt${attempt}.log"
    {
      echo "exit_code=$status"
      echo "expected_marker=$marker"
      adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./scna_env_smoke"
    } >"$recovery" 2>&1 || true
  done
  echo "UNAVAILABLE after retry: ${log#$out/}" >&2
  return 1
}

deploy_artifact() {
  local artifact="$1"
  adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >/dev/null
  adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >/dev/null
}

carrier_for() {
  case "$1" in
    "$candidate") printf '%s' "$candidate_artifact" ;;
    *) printf '%s' "$pairret" ;;
  esac
}

attention_mode() {
  case "$1" in
    exp_lut) echo lut-exp ;;
    *) echo scna-fp16 ;;
  esac
}

attention_args() {
  local label="$1" q="$2" kv="$3" awarmup="$4" aiters="$5" seed="$6"
  local mask="${7:-full}" dim="${8:-128}" events="--no-events"
  [[ $# -ge 9 ]] && events="$9"
  printf '%s' "--figure8-attn --mode $(attention_mode "$label") --scna-variant pair_static_d8 --workers 1 --scna-width 8 --mask-mode $mask --qo-len $q --kv-len $kv --n-heads 12 --n-kv-heads 2 --head-dim $dim --warmup $awarmup --iters $aiters --seed $seed $events"
}

adb get-state >/dev/null
"$script_dir/deploy_and_smoke.sh" --remote-dir "$remote_dir" \
  --kernel-impl "$candidate" --mode ping >"$out/raw/deploy.log" 2>&1

{
  echo "schema_version=1"
  echo "run_id=$run_id"
  echo "candidate=$candidate"
  echo "quick=$quick"
  echo "captured_at=$(date -u +%FT%TZ)"
  echo "source_git_commit=$(git -C "$project_dir" rev-parse HEAD)"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.product.device
  adb shell getprop ro.board.platform
  adb shell getprop ro.soc.model
  adb shell getprop ro.build.fingerprint
  sha256sum "$pairret" "$candidate_artifact"
} >"$out/evidence/device_manifest.txt"
adb shell dumpsys thermalservice >"$out/evidence/thermal_before.txt" 2>&1 || true
cp "$project_dir/experiment_spec.json" "$out/evidence/experiment_spec.json"
cp "$0" "$out/evidence/run_candidate_experiment.sh"
cp "$project_dir/artifacts/variants/$candidate/build_id.txt" "$out/evidence/candidate_build_id.txt"
cp "$project_dir/artifacts/variants/$candidate/compile_flags.txt" "$out/evidence/candidate_compile_flags.txt"

objdump=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump
"$objdump" -d --no-show-raw-insn "$pairret" >"$out/static/d7_pairret_noinline.v79.disasm.txt"
"$objdump" -d --no-show-raw-insn "$candidate_artifact" >"$out/static/$candidate.v79.disasm.txt"

for label in d7_pairret_noinline "$candidate"; do
  deploy_artifact "$(carrier_for "$label")"
  pilot="$out/raw/micro/${label}_pilot.log"
  run_remote "$pilot" 'SCNA_EXP_BENCH' \
    --scna-exp-bench --scna-variant pair_static_d8 --scna-width 8 --warmup 5 --iters 200000
  elapsed="$(sed -n 's/.*pair_elapsed_us=\([0-9][0-9]*\).*/\1/p' "$pilot" | tail -1)"
  [[ -n "$elapsed" && "$elapsed" -gt 0 ]] || { echo "Micro calibration failed for $label" >&2; exit 1; }
  calibrated=$(( (micro_target_us * 200000 + elapsed - 1) / elapsed ))
  (( calibrated < 200000 )) && calibrated=200000
  for sample in $(seq 1 "$samples"); do
    run_remote "$out/raw/micro/${label}_sample${sample}.log" 'SCNA_EXP_BENCH' \
      --scna-exp-bench --scna-variant pair_static_d8 --scna-width 8 --warmup 5 --iters "$calibrated"
  done
done

deploy_artifact "$pairret"
for sample in $(seq 1 "$samples"); do
  run_remote "$out/raw/lut_micro/exp_lut_attention_sample${sample}.log" '^2,lut_exp,' \
    --lut-exp-bench --distribution attention --target-ms "$lut_target_ms" --warmup 4 --iters 1 \
    --qo-len 32 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 --seed figure8_fixed
done

deploy_artifact "$candidate_artifact"
for mask in "${accuracy_masks[@]}"; do
  for q in "${accuracy_q[@]}"; do
    for kv in "${accuracy_kv[@]}"; do
      for dim in "${accuracy_dim[@]}"; do
        for seed in "${seeds[@]}"; do
          args="$(attention_args "$candidate" "$q" "$kv" 1 1 "$seed" "$mask" "$dim")"
          log="$out/raw/accuracy/${candidate}_${mask}_q${q}_kv${kv}_d${dim}_seed${seed}.log"
          run_remote "$log" 'FIG8_ATTENTION_COMPARE .*pass=1' "$args --compare-reference --numeric-debug"
        done
      done
    done
  done
done

labels=(d7_pairret_noinline "$candidate" exp_lut)
qos=(1 4 8 16 32)
for session in $(seq 1 "$sessions"); do
  seed="${seeds[$(((session - 1) % ${#seeds[@]}))]}"
  for offset in "${!labels[@]}"; do
    index=$(((offset + session - 1) % ${#labels[@]}))
    label="${labels[$index]}"
    deploy_artifact "$(carrier_for "$label")"
    for q in "${qos[@]}"; do
      args="$(attention_args "$label" "$q" 4096 "$warmup" "$iters" "$seed")"
      run_remote "$out/raw/attention/${label}_q${q}_s${session}.log" \
        "phase=measure iteration=$((iters - 1)) .*ret=0" "$args"
    done
  done
done

for label in "${labels[@]}"; do
  deploy_artifact "$(carrier_for "$label")"
  args="$(attention_args "$label" 32 4096 1 1 figure8_fixed full 128 '')"
  run_remote "$out/raw/diagnostic/${label}_q32_kv4096.log" 'FIG8_ATTENTION_TIMERS' "$args --events"
done

adb shell dumpsys thermalservice >"$out/evidence/thermal_after.txt" 2>&1 || true
echo "completed: $out"
