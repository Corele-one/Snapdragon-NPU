#!/usr/bin/env bash
# Paper-aligned Qwen2.5-1.5B evaluation runner for the isolated SCNA project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/llama-scna-v81-qwen-eval}"
MODEL="${MODEL:-/data/local/tmp/llama-npu-chat/qwen2.5-1.5b-instruct.iq4_nl+q8_0-hmx.gguf}"
OUT_DIR="${OUT_DIR:-$ROOT/results/v81/scna/qwen25-paper-eval-$(date +%Y%m%d-%H%M%S)}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
THREADS="${THREADS:-4}"
N_GEN="${N_GEN:-32}"
PPL_CHUNKS="${PPL_CHUNKS:--1}"
WIKITEXT_FILE="${WIKITEXT_FILE:-}"
MODES="${MODES:-baseline lut-exp scna}"

HOST_BIN="$ROOT/src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/htp_ops_test"
HOST_STUB="$ROOT/src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/libhtp_ops.so"
HOST_SKEL="$ROOT/src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so"

mkdir -p "$OUT_DIR"/{raw,provenance,attention,model,accuracy}

adb shell "mkdir -p $REMOTE_DIR/cdsp"
adb push "$HOST_BIN" "$HOST_STUB" "$REMOTE_DIR/" >/dev/null
adb push "$HOST_SKEL" "$REMOTE_DIR/cdsp/" >/dev/null

device_env() {
  local mode="$1"
  case "$mode" in
    baseline) printf '%s' 'LLAMA_NPU_MODE=baseline LLAMA_NPU_KV_PIPELINE=1' ;;
    lut-exp)  printf '%s' 'LLAMA_NPU_MODE=lut-exp LLAMA_NPU_KV_PIPELINE=1' ;;
    scna)     printf '%s' 'LLAMA_NPU_MODE=scna-fp16 LLAMA_NPU_SCNA_FUNCTION=exp LLAMA_NPU_SCNA_KERNEL=tree LLAMA_NPU_SCNA_WIDTH=8 LLAMA_NPU_KV_PIPELINE=1' ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
  esac
}

run_attention() {
  local mode="$1" qo="$2" kv="$3" log="$OUT_DIR/attention/${mode}_q${qo}_kv${kv}.log"
  if [[ -s "$log" ]] && grep -q 'phase=measure.*ret=0' "$log"; then
    return
  fi
  local kernel_mode="$mode"
  local args=(--figure8-attn --qo-len "$qo" --kv-len "$kv" --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup "$WARMUP" --iters "$ITERS" --no-events --compare-reference --scna-pipeline on)
  if [[ "$mode" == scna ]]; then
    kernel_mode=scna-fp16
    args+=(--scna-function exp --scna-kernel tree --scna-width 8)
  fi
  adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=.\:/system/lib64\:/vendor/lib64 DSP_LIBRARY_PATH='./cdsp;.' ./htp_ops_test --mode $kernel_mode ${args[*]}" >"$log" 2>&1
}

run_prefill() {
  local mode="$1" prompt="$2" log="$OUT_DIR/model/${mode}_pp${prompt}.jsonl"
  if [[ -s "$log" ]] && grep -q '"samples_ts"' "$log"; then
    return
  fi
  local env; env="$(device_env "$mode")"
  # A discardable warm-up preserves the paper's 5 + 20 protocol without
  # mixing warm-up samples into the saved JSONL.
  adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=.\:/system/lib64\:/vendor/lib64 DSP_LIBRARY_PATH=. $env ./llama-bench -m $MODEL -fa 1 -p $prompt -n 0 -b 2048 -ub 64 -t $THREADS -r $WARMUP -o jsonl" >"$OUT_DIR/raw/${mode}_pp${prompt}_warmup.log" 2>&1
  adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=.\:/system/lib64\:/vendor/lib64 DSP_LIBRARY_PATH=. $env ./llama-bench -m $MODEL -fa 1 -p $prompt -n 0 -b 2048 -ub 64 -t $THREADS -r $ITERS -o jsonl" >"$log" 2>&1
}

run_decode() {
  local mode="$1" log="$OUT_DIR/model/${mode}_tg${N_GEN}.jsonl"
  if [[ -s "$log" ]] && grep -q '"samples_ts"' "$log"; then
    return
  fi
  local env; env="$(device_env "$mode")"
  adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=.\:/system/lib64\:/vendor/lib64 DSP_LIBRARY_PATH=. $env ./llama-bench -m $MODEL -fa 1 -p 0 -n $N_GEN -b 2048 -ub 64 -t $THREADS -r $WARMUP -o jsonl" >"$OUT_DIR/raw/${mode}_tg${N_GEN}_warmup.log" 2>&1
  adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=.\:/system/lib64\:/vendor/lib64 DSP_LIBRARY_PATH=. $env ./llama-bench -m $MODEL -fa 1 -p 0 -n $N_GEN -b 2048 -ub 64 -t $THREADS -r $ITERS -o jsonl" >"$log" 2>&1
}

run_ppl() {
  local mode="$1" env; env="$(device_env "$mode")"
  [[ -n "$WIKITEXT_FILE" ]] || return 0
  local log="$OUT_DIR/accuracy/${mode}_wikitext2.log"
  if [[ -s "$log" ]] && grep -qi 'perplexity' "$log"; then
    return
  fi
  adb push "$WIKITEXT_FILE" "$REMOTE_DIR/wikitext-2.test.raw" >/dev/null
  local chunks=()
  if [[ "$PPL_CHUNKS" != "-1" ]]; then
    chunks=(--chunks "$PPL_CHUNKS")
  fi
  adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=.\:/system/lib64\:/vendor/lib64 DSP_LIBRARY_PATH=. $env ./llama-perplexity -m $MODEL -f wikitext-2.test.raw -fa -b 512 -ub 64 -t $THREADS ${chunks[*]}" >"$log" 2>&1
}

# Figure 14-style operator study in the referenced paper.
for mode in $MODES; do
  for qo in 1 4 16; do
    for kv in 1024 4096 16384; do
      run_attention "$mode" "$qo" "$kv"
    done
  done
done

# Figure 13/17-style model throughput and prompt-length sensitivity.
for mode in baseline lut-exp scna; do
  for prompt in 512 1024 2048 4096; do
    run_prefill "$mode" "$prompt"
  done
  run_decode "$mode"
  run_ppl "$mode"
done

adb shell 'getprop; uname -a; sha256sum /vendor/lib64/libcdsprpc.so /system/lib64/libbinder.so /system/lib64/libbinder_ndk.so' >"$OUT_DIR/provenance/device.txt"
adb shell "cd $REMOTE_DIR && sha256sum llama-bench llama-perplexity htp_ops_test libhtp_ops.so cdsp/libhtp_ops_skel.so" >"$OUT_DIR/provenance/device-artifacts.txt"
git -C "$ROOT" rev-parse HEAD >"$OUT_DIR/provenance/git-head.txt"
git -C "$ROOT" diff -- src/llama.cpp-npu-htp-backend/ggml/src/ggml-htp >"$OUT_DIR/provenance/llama-scna-mode.patch"
python3 "$ROOT/scripts/analyze_qwen25_scna_vs_lut.py" --input-dir "$OUT_DIR" --out-dir "$OUT_DIR/summary"
printf 'Qwen2.5-1.5B paper-aligned evaluation complete: %s\n' "$OUT_DIR"
