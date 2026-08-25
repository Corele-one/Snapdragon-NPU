#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
artifact_dir="$root_dir/artifacts/upstream_0a50d990"
remote_dir=/data/local/tmp/stage25_upstream_v79
run_id="upstream_$(date -u +%Y%m%dT%H%M%SZ)"
quick=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) run_id="$2"; shift 2 ;;
        --remote-dir) remote_dir="$2"; shift 2 ;;
        --quick) quick=1; shift ;;
        *) echo "usage: $0 [--run-id ID] [--remote-dir PATH] [--quick]" >&2; exit 2 ;;
    esac
done

out="$root_dir/results/upstream/$run_id"
mkdir -p "$out"/{fixtures,raw/{correctness,performance,recovery},evidence}
sessions=5; warmups=5; iterations=20
seeds=(figure8_fixed 20260810 20260811)
qos=(1 4 8 16 32)
if [[ "$quick" == 1 ]]; then
    sessions=2; warmups=2; iterations=3; seeds=(figure8_fixed); qos=(1 8)
fi

for path in "$artifact_dir/test-stage25-fa" "$artifact_dir/lib/libggml.so" \
            "$artifact_dir/lib/libggml-base.so" "$artifact_dir/lib/libggml-cpu.so" \
            "$artifact_dir/lib/libggml-hexagon.so" "$artifact_dir/lib/libggml-htp-v79.so"; do
    test -s "$path" || { echo "missing upstream artifact: $path" >&2; exit 1; }
done

adb get-state >/dev/null
adb shell "mkdir -p '$remote_dir/lib'"
adb push "$artifact_dir/test-stage25-fa" "$remote_dir/" >/dev/null
for lib in "$artifact_dir/lib/"*.so; do adb push "$lib" "$remote_dir/lib/" >/dev/null; done
adb shell "chmod 755 '$remote_dir/test-stage25-fa'"

fixture_for() {
    local q="$1" seed="$2"
    local path="$out/fixtures/full_q${q}_kv4096_d128_seed${seed}.bin"
    if [[ ! -s "$path" ]]; then
        python3 "$root_dir/tools/generate_fixture.py" --output "$path" --qo-len "$q" --kv-len 4096 \
            --n-heads 12 --n-kv-heads 2 --head-dim 128 --mask-mode full --seed "$seed" >/dev/null
    fi
    printf '%s' "$path"
}

recover() {
    local target="$1" status="$2"
    {
        echo "exit_code=$status"
        adb get-state
        adb shell getprop ro.soc.model
    } >"$target" 2>&1 || true
}

run_one() {
    local config="$1" selector="$2" nhvx="$3" q="$4" seed="$5" session="$6" check="$7"
    local fixture log status attempt nhvx_env check_arg
    fixture=$(fixture_for "$q" "$seed")
    adb push "$fixture" "$remote_dir/fixture.bin" >/dev/null
    if [[ "$check" == 1 ]]; then
        log="$out/raw/correctness/${config}_q${q}_seed${seed}.log"
        check_arg=--check
    else
        log="$out/raw/performance/${config}_q${q}_s${session}.log"
        check_arg=""
    fi
    [[ -s "$log" ]] && grep -q '"record":"measurement"' "$log" && { echo "resume: ${log#$out/}"; return; }
    if [[ "$nhvx" == default ]]; then nhvx_env="unset GGML_HEXAGON_NHVX;"; else nhvx_env="export GGML_HEXAGON_NHVX=$nhvx;"; fi
    for attempt in 1 2; do
        set +e
        timeout 180s adb shell "cd '$remote_dir' && export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_FA_SELECT=$selector; $nhvx_env ./test-stage25-fa --fixture fixture.bin --warmups $warmups --iterations $iterations $check_arg" >"$log" 2>&1
        status=$?
        set -e
        if [[ $status -eq 0 ]] && grep -q '"record":"measurement"' "$log"; then
            if [[ "$check" == 0 || $(grep -c '"record":"correctness".*"pass":true' "$log") -eq 1 ]]; then
                echo "done: ${log#$out/}"
                return
            fi
        fi
        recover "$out/raw/recovery/${config}_q${q}_s${session}_attempt${attempt}.log" "$status"
    done
    echo "UNAVAILABLE: ${log#$out/}" >&2
    return 1
}

{
    echo schema_version=1
    echo run_id="$run_id"
    echo quick="$quick"
    echo sessions="$sessions"
    echo warmups="$warmups"
    echo iterations="$iterations"
    echo captured_at="$(date -u +%FT%TZ)"
    adb get-serialno
    adb shell getprop ro.product.model
    adb shell getprop ro.board.platform
    adb shell getprop ro.soc.model
    sha256sum "$artifact_dir/test-stage25-fa" "$artifact_dir/lib/"*.so
} >"$out/evidence/device_manifest.txt"
cp "$root_dir/experiment_spec.json" "$out/evidence/experiment_spec.json"
cp "$artifact_dir/upstream_instrumentation.patch" "$out/evidence/"
cp "$artifact_dir/upstream_status.txt" "$out/evidence/"
adb shell dumpsys thermalservice >"$out/evidence/thermal_before.txt" 2>&1 || true

# HMX activation/correctness gate: selector 2, q8/q16/q32, three seeds.
if [[ "$quick" == 0 ]]; then
    for q in 8 16 32; do
        for seed in "${seeds[@]}"; do run_one controlled_fa2 2 1 "$q" "$seed" 0 1; done
    done
else
    run_one controlled_fa2 2 1 8 figure8_fixed 0 1
fi

configs=(controlled_fa1 controlled_fa2 system_default_fa2)
selectors=(1 2 2)
nhvxs=(1 1 default)
for session in $(seq 1 "$sessions"); do
    seed="${seeds[$(((session - 1) % ${#seeds[@]}))]}"
    for offset in "${!configs[@]}"; do
        index=$(((offset + session - 1) % ${#configs[@]}))
        config="${configs[$index]}"
        for q in "${qos[@]}"; do
            run_one "$config" "${selectors[$index]}" "${nhvxs[$index]}" "$q" "$seed" "$session" 0
        done
    done
done
adb shell dumpsys thermalservice >"$out/evidence/thermal_after.txt" 2>&1 || true
sha256sum "$out"/fixtures/*.bin >"$out/evidence/fixture_sha256.txt"
python3 "$root_dir/tools/analyze_upstream.py" --run-dir "$out" --output "$out/summary.json"
echo "completed: $out"

