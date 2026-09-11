#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
remote_dir="/data/local/tmp/stage2_75_scheduler_v79"
run_id="scheduler_$(date -u +%Y%m%dT%H%M%SZ)"
rows="4 32"
workers="1 auto"
labels="exp_lut fused_state_update"
qos="32"
kvs="4096"
sessions=5
warmup=5
iters=20
quick=0

usage() {
  cat <<'EOF' >&2
Usage: run_scheduler_matrix.sh [options]
  --run-id ID
  --rows "4 8 16 32"
  --workers "1 2 3 4 5 6 auto"
  --labels "origin_hvx exp_lut d7_pairret_noinline fused_state_update"
  --qos "8 16 32" --kvs "512 2048 4096"
  --remote-dir PATH
  --quick  (2 sessions, 2 warmup, 3 measurements)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --rows) rows="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --labels) labels="$2"; shift 2 ;;
    --qos) qos="$2"; shift 2 ;;
    --kvs) kvs="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --quick) quick=1; sessions=2; warmup=2; iters=3; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

read -r -a row_values <<<"$rows"
read -r -a worker_values <<<"$workers"
read -r -a label_values <<<"$labels"
read -r -a qo_values <<<"$qos"
read -r -a kv_values <<<"$kvs"

for row in "${row_values[@]}"; do
  [[ "$row" == auto || "$row" == 4 || "$row" == 8 || "$row" == 16 || "$row" == 32 ]] || {
    echo "Unsupported q_task_rows: $row" >&2; exit 2;
  }
done
for worker in "${worker_values[@]}"; do
  [[ "$worker" == auto || "$worker" =~ ^[1-6]$ ]] || {
    echo "Unsupported worker count: $worker" >&2; exit 2;
  }
done

out="$project_dir/results/runs/$run_id"
mkdir -p "$out"/{raw/attention,evidence,recovery}

baseline_artifact="$project_dir/artifacts/variants/d7_pairret_noinline/libhtp_ops_skel.so"
fused_artifact="$project_dir/artifacts/variants/fused_state_update/libhtp_ops_skel.so"
[[ -s "$baseline_artifact" && -s "$fused_artifact" ]] || {
  echo "Missing scheduler artifacts; run build_baselines.sh and build_candidate.sh fused_state_update" >&2
  exit 1
}

artifact_for() {
  case "$1" in
    fused_state_update) printf '%s' "$fused_artifact" ;;
    origin_hvx|exp_lut|d7_pairret_noinline) printf '%s' "$baseline_artifact" ;;
    *) echo "Unknown label: $1" >&2; exit 2 ;;
  esac
}

mode_for() {
  case "$1" in
    origin_hvx) echo baseline ;;
    exp_lut) echo lut-exp ;;
    d7_pairret_noinline|fused_state_update) echo scna-fp16 ;;
  esac
}

worker_cli() {
  [[ "$1" == auto ]] && echo auto || echo "$1"
}

deploy_artifact() {
  local artifact="$1"
  adb push "$artifact" "$remote_dir/cdsp/libhtp_ops_skel.so" >/dev/null
  adb push "$artifact" "$remote_dir/dsp/libhtp_ops_skel.so" >/dev/null
}

complete() {
  [[ -s "$1" ]] && grep -q "phase=measure iteration=$((iters - 1)).*ret=0" "$1"
}

run_one() {
  local log="$1" label="$2" row="$3" worker="$4" q="$5" kv="$6"
  if complete "$log"; then
    echo "resume: ${log#$out/}"
    return
  fi
  local attempt status
  for attempt in 1 2; do
    set +e
    timeout 120s adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test \
      --figure8-attn --mode '$(mode_for "$label")' --scna-variant pair_static_d8 \
      --workers '$(worker_cli "$worker")' --q-task-rows '$row' --scna-width 8 \
      --mask-mode full --qo-len '$q' --kv-len '$kv' --n-heads 12 --n-kv-heads 2 \
      --head-dim 128 --warmup '$warmup' --iters '$iters' --seed figure8_fixed --no-events" >"$log" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]] && complete "$log"; then
      echo "done: ${log#$out/}"
      return
    fi
    {
      echo "exit_code=$status"
      adb get-state
      adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./scna_env_smoke"
    } >"$out/recovery/$(basename "${log%.log}")_attempt${attempt}.log" 2>&1 || true
  done
  echo "UNAVAILABLE: ${log#$out/}" >&2
  return 1
}

adb get-state >/dev/null
"$script_dir/deploy_and_smoke.sh" --mode ping --remote-dir "$remote_dir" >"$out/evidence/deploy.log" 2>&1

{
  echo "schema_version=1"
  echo "run_id=$run_id"
  echo "quick=$quick"
  echo "rows=$rows"
  echo "workers=$workers"
  echo "labels=$labels"
  echo "qos=$qos"
  echo "kvs=$kvs"
  echo "sessions=$sessions"
  echo "warmup=$warmup"
  echo "iters=$iters"
  echo "captured_at=$(date -u +%FT%TZ)"
  echo "source_git_commit=$(git -C "$project_dir" rev-parse HEAD)"
  adb get-serialno
  adb shell getprop ro.product.model
  adb shell getprop ro.soc.model
  adb shell getprop ro.build.fingerprint
  sha256sum "$baseline_artifact" "$fused_artifact"
} >"$out/evidence/manifest.txt"
cp "$project_dir/experiment_spec.json" "$out/evidence/experiment_spec.json"
cp "$0" "$out/evidence/run_scheduler_matrix.sh"
adb shell dumpsys thermalservice >"$out/evidence/thermal_before.txt" 2>&1 || true

for session in $(seq 1 "$sessions"); do
  for label in "${label_values[@]}"; do
    deploy_artifact "$(artifact_for "$label")"
    for q in "${qo_values[@]}"; do
      for kv in "${kv_values[@]}"; do
        for row in "${row_values[@]}"; do
          for worker in "${worker_values[@]}"; do
            run_one "$out/raw/attention/${label}_q${q}_kv${kv}_r${row}_w${worker}_s${session}.log" \
              "$label" "$row" "$worker" "$q" "$kv"
          done
        done
      done
    done
  done
done

adb shell dumpsys thermalservice >"$out/evidence/thermal_after.txt" 2>&1 || true
echo "completed: $out"
