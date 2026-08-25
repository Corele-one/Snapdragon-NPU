#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
htp_dir="$project_dir/src/htp-ops-lib-main"
remote_dir="/data/local/tmp/stage25_softmax_v79"
run_id="stage25_$(date -u +%Y%m%dT%H%M%SZ)"
quick=0
candidate_labels=(s25_b_full_mask_fastpath s25_c_aligned_mask_load s25_e_task_geometry)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --quick) quick=1; shift ;;
    --candidates)
      IFS=',' read -r -a candidate_labels <<<"$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--run-id ID] [--remote-dir PATH] [--quick] [--candidates comma,list]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

out="$project_dir/results/runs/$run_id"
mkdir -p "$out"/{fixtures,raw/{accuracy,attention,diagnostic,recovery},static,evidence}

sessions=5
warmup=5
iters=20
seeds=(figure8_fixed 20260810 20260811)
accuracy_masks=(full causal padding)
accuracy_q=(1 4)
accuracy_kv=(4093 4096)
accuracy_dim=(64 128)
qos=(1 4 8 16 32)
if [[ "$quick" == 1 ]]; then
  sessions=2
  warmup=2
  iters=3
  seeds=(figure8_fixed)
  accuracy_masks=(full)
  accuracy_q=(1)
  accuracy_kv=(4093)
  accuracy_dim=(64)
fi

host_ship="$htp_dir/android_ReleaseG_aarch64/ship"
dsp_ship_candidates=("$htp_dir"/hexagon_ReleaseG_toolv*_v79/ship)
dsp_ship="${dsp_ship_candidates[0]:-}"
for path in "$host_ship/htp_ops_test" "$host_ship/libhtp_ops.so" "$host_ship/scna_env_smoke" \
            "$host_ship/libscna_env.so" "$dsp_ship/libscna_env_skel.so"; do
  [[ -s "$path" ]] || { echo "Missing build product: $path" >&2; exit 1; }
done

artifact_for() {
  case "$1" in
    origin_hvx|exp_lut|stage2_final) printf '%s' "$project_dir/artifacts/variants/stage2_final/libhtp_ops_skel.so" ;;
    stage1_d7) printf '%s' "$project_dir/artifacts/variants/stage1_d7/libhtp_ops_skel.so" ;;
    *) printf '%s' "$project_dir/artifacts/variants/$1/libhtp_ops_skel.so" ;;
  esac
}

mode_for() {
  case "$1" in
    origin_hvx) echo baseline ;;
    exp_lut) echo lut-exp ;;
    *) echo scna-fp16 ;;
  esac
}

full_fast_arg() {
  local build_id="$project_dir/artifacts/variants/$1/build_id.txt"
  local flags=0
  [[ -f "$build_id" ]] && flags="$(sed -n 's/^stage25_flags=//p' "$build_id")"
  if (( flags & 2 )); then
    printf '%s' '--full-mask-fastpath'
  fi
}

complete() { [[ -s "$1" ]] && grep -Eq "$2" "$1"; }

recovery_probe() {
  adb get-state
  adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./scna_env_smoke"
}

recover() {
  local path="$1"
  recovery_probe >"$path" 2>&1 || true
}

run_remote() {
  local log="$1" marker="$2"
  shift 2
  if complete "$log" "$marker"; then
    echo "resume: ${log#$out/}"
    return 0
  fi
  local attempt status
  for attempt in 1 2; do
    set +e
    timeout 180s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test $*" >"$log" 2>&1
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
      recovery_probe
    } >"$recovery" 2>&1
  done
  echo "UNAVAILABLE after retry: ${log#$out/}" >&2
  return 1
}

deploy_artifact() {
  local artifact
  artifact="$(artifact_for "$1")"
  [[ -s "$artifact" ]] || { echo "Missing artifact: $artifact" >&2; exit 1; }
  adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >/dev/null
  adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >/dev/null
}

fixture_for() {
  local q="$1" kv="$2" dim="$3" mask="$4" seed="$5"
  local path="$out/fixtures/${mask}_q${q}_kv${kv}_d${dim}_seed${seed}.bin"
  if [[ ! -s "$path" ]]; then
    python3 "$project_dir/tools/generate_fixture.py" --output "$path" --qo-len "$q" --kv-len "$kv" \
      --n-heads 12 --n-kv-heads 2 --head-dim "$dim" --mask-mode "$mask" --seed "$seed" >/dev/null
  fi
  printf '%s' "$path"
}

push_fixture() {
  local local_fixture="$1"
  adb push "$local_fixture" "$remote_dir/fixture.bin" >/dev/null
}

attention_args() {
  local label="$1" q="$2" kv="$3" dim="$4" mask="$5" seed="$6" awarmup="$7" aiters="$8"
  local extra="${9:-}"
  local fast=""
  [[ "$mask" == full ]] && fast="$(full_fast_arg "$label")"
  printf '%s' "--figure8-attn --mode $(mode_for "$label") --scna-variant pair_static_d8 --workers 1 --scna-width 8 --mask-mode $mask --qo-len $q --kv-len $kv --n-heads 12 --n-kv-heads 2 --head-dim $dim --warmup $awarmup --iters $aiters --seed $seed --fixture '$remote_dir/fixture.bin' $fast $extra"
}

adb get-state >/dev/null
adb shell "mkdir -p '$remote_dir/cdsp' '$remote_dir/dsp'"
for artifact in htp_ops_test libhtp_ops.so scna_env_smoke libscna_env.so; do
  adb push "$host_ship/$artifact" "$remote_dir/" >/dev/null
done
adb push "$dsp_ship/libscna_env_skel.so" "$remote_dir/cdsp/" >/dev/null
adb push "$dsp_ship/libscna_env_skel.so" "$remote_dir/dsp/" >/dev/null
adb shell "chmod 755 '$remote_dir/htp_ops_test' '$remote_dir/scna_env_smoke'"
deploy_artifact stage2_final
recover "$out/raw/deploy.log"

{
  echo "schema_version=1"
  echo "run_id=$run_id"
  echo "quick=$quick"
  echo "captured_at=$(date -u +%FT%TZ)"
  echo "candidates=${candidate_labels[*]}"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.product.device
  adb shell getprop ro.board.platform
  adb shell getprop ro.soc.model
  adb shell getprop ro.build.fingerprint
  for label in stage1_d7 stage2_final "${candidate_labels[@]}"; do
    sha256sum "$(artifact_for "$label")"
  done
} >"$out/evidence/device_manifest.txt"
adb shell dumpsys thermalservice >"$out/evidence/thermal_before.txt" 2>&1 || true
cp "$project_dir/experiment_spec.json" "$out/evidence/experiment_spec.json"
cp "$project_dir/source_import_manifest.json" "$out/evidence/source_import_manifest.json"
cp "$project_dir/results/static_audit.json" "$out/evidence/static_audit.json"
cp "$0" "$out/evidence/run_stage25_experiment.sh"

correctness_labels=(stage1_d7 stage2_final "${candidate_labels[@]}")
for label in "${correctness_labels[@]}"; do
  deploy_artifact "$label"
  for mask in "${accuracy_masks[@]}"; do
    for q in "${accuracy_q[@]}"; do
      for kv in "${accuracy_kv[@]}"; do
        for dim in "${accuracy_dim[@]}"; do
          for seed in "${seeds[@]}"; do
            fixture="$(fixture_for "$q" "$kv" "$dim" "$mask" "$seed")"
            push_fixture "$fixture"
            log="$out/raw/accuracy/${label}_${mask}_q${q}_kv${kv}_d${dim}_seed${seed}.log"
            args="$(attention_args "$label" "$q" "$kv" "$dim" "$mask" "$seed" 1 1 '--no-events --compare-reference --numeric-debug')"
            run_remote "$log" 'FIG8_ATTENTION_COMPARE .*pass=1' "$args"
          done
        done
      done
    done
  done
done

perf_labels=(origin_hvx exp_lut stage1_d7 stage2_final "${candidate_labels[@]}")
for session in $(seq 1 "$sessions"); do
  seed="${seeds[$(((session - 1) % ${#seeds[@]}))]}"
  for offset in "${!perf_labels[@]}"; do
    index=$(((offset + session - 1) % ${#perf_labels[@]}))
    label="${perf_labels[$index]}"
    deploy_artifact "$label"
    for q in "${qos[@]}"; do
      fixture="$(fixture_for "$q" 4096 128 full "$seed")"
      push_fixture "$fixture"
      log="$out/raw/attention/${label}_q${q}_s${session}.log"
      args="$(attention_args "$label" "$q" 4096 128 full "$seed" "$warmup" "$iters" '--no-events')"
      run_remote "$log" "phase=measure iteration=$((iters - 1)) .*ret=0" "$args"
    done
  done
done

for label in "${perf_labels[@]}"; do
  deploy_artifact "$label"
  fixture="$(fixture_for 32 4096 128 full figure8_fixed)"
  push_fixture "$fixture"
  log="$out/raw/diagnostic/${label}_q32_kv4096.log"
  args="$(attention_args "$label" 32 4096 128 full figure8_fixed 1 1 '--events')"
  run_remote "$log" 'FIG8_ATTENTION_TIMERS' "$args"
done

find "$out/fixtures" -type f -print0 | sort -z | xargs -0 sha256sum >"$out/evidence/fixture_sha256.txt"
adb shell dumpsys thermalservice >"$out/evidence/thermal_after.txt" 2>&1 || true
python3 "$project_dir/tools/analyze_stage25.py" --run-dir "$out" --output "$out/summary.json"
echo "completed: $out"
