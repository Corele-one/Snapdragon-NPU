#!/system/bin/sh
set -eu

# LPBQ deploy-v1 AppDir isolation fix: switch scripts already `cd` into the
# requested runtime directory before invoking this runner, so default to that
# directory when APP_DIR is not explicitly set. The old fixed default is kept
# here as a rollback note:
# APP_DIR="${APP_DIR:-/data/local/tmp/llama-npu-chat}"
APP_DIR="${APP_DIR:-$(pwd)}"
cd "$APP_DIR"

export LD_LIBRARY_PATH="$APP_DIR:/vendor/lib64:/system/lib64"
export DSP_LIBRARY_PATH="$APP_DIR"
export ADSP_LIBRARY_PATH="$APP_DIR/;$APP_DIR;${ADSP_LIBRARY_PATH:-/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp/;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp}"

DEFAULT_MODEL="qwen2.5-1.5b-instruct.f16-hmx.gguf"
API_KEY_FILE="${LLAMA_API_KEY_FILE:-$APP_DIR/api_key.txt}"

ensure_api_key() {
  if [ -n "${LLAMA_API_KEY:-}" ]; then
    AUTH_ARGS="--api-key $LLAMA_API_KEY"
    AUTH_SOURCE="LLAMA_API_KEY"
    return
  fi

  if [ ! -f "$API_KEY_FILE" ]; then
    umask 077
    printf '%s\n' "${LLAMA_DEFAULT_API_KEY:-llama-npu-local-key}" > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE" 2>/dev/null || true
  fi

  AUTH_ARGS="--api-key-file $API_KEY_FILE"
  AUTH_SOURCE="$API_KEY_FILE"
}

show_api_key() {
  if [ -n "${LLAMA_API_KEY:-}" ]; then
    printf '%s\n' "$LLAMA_API_KEY"
  elif [ -f "$API_KEY_FILE" ]; then
    cat "$API_KEY_FILE"
  else
    printf '%s\n' "${LLAMA_DEFAULT_API_KEY:-llama-npu-local-key}"
  fi
}

set_api_key() {
  if [ -z "${1:-}" ]; then
    echo "usage: $0 set-key <api-key>" >&2
    exit 2
  fi
  umask 077
  printf '%s\n' "$1" > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE" 2>/dev/null || true
  echo "API key written to $API_KEY_FILE"
}

start_server() {
  MODEL="${1:-${LLAMA_MODEL:-$DEFAULT_MODEL}}"
  export LLAMA_NPU_MODE="${2:-${LLAMA_NPU_MODE:-pure_fp16}}"
  export LLAMA_NPU_TRACE="${3:-${LLAMA_NPU_TRACE:-0}}"
  export LLAMA_NPU_DETAILED_TRACE="${4:-${LLAMA_NPU_DETAILED_TRACE:-0}}"
  PORT="${5:-${LLAMA_PORT:-8080}}"
  case "$LLAMA_NPU_MODE" in
    lpbq_int8|lpbq-int8|lpbq-w4a8-int8|w4a8_lpbq)
      export LLAMA_NPU_LPBQ_SIDECAR_DIR="${LLAMA_NPU_LPBQ_SIDECAR_DIR:-$APP_DIR/sidecars/lpbq_g16_a8w8}"
      # LPBQ deploy-v1 FP16-base retune: DSP K32-safe grouping is compiled on
      # again, so let the host generate K32-safe flags from packed sidecars for
      # the R4/nibble fallback path. The old disabled default is left here as a
      # rollback note: LLAMA_NPU_LPBQ_DISABLE_K32_SAFE_SIDECAR="${...:-1}".
      export LLAMA_NPU_LPBQ_DISABLE_K32_SAFE_SIDECAR="${LLAMA_NPU_LPBQ_DISABLE_K32_SAFE_SIDECAR:-0}"
      # LPBQ deploy-v1 R4 retune: fold inverse input-scale into the host-loaded
      # R4 sidecar by default so the DSP rotate+quant stage does not multiply
      # activation by input_scale online. The old behavior is kept as an env
      # rollback: LLAMA_NPU_LPBQ_ENABLE_R4_SCALE_FOLD=0.
      export LLAMA_NPU_LPBQ_ENABLE_R4_SCALE_FOLD="${LLAMA_NPU_LPBQ_ENABLE_R4_SCALE_FOLD:-1}"
      # LPBQ deploy-v1 FWHT diagnostic selector: default 4 preserves the
      # accepted small-M-only behavior; callers can set -1 for explicit
      # no-quality all-M FWHT speed probes.
      export LLAMA_NPU_LPBQ_R4_STRUCTURED_FWHT_SMALL_M_MAX="${LLAMA_NPU_LPBQ_R4_STRUCTURED_FWHT_SMALL_M_MAX:-4}"
      ;;
  esac

  export LLAMA_NPU_DETAILED_TRACE_MAX_EVENTS="${LLAMA_NPU_DETAILED_TRACE_MAX_EVENTS:-65536}"
  # Keep SKIP_HTP_OPS unset unless the caller explicitly requests a truthy value.
  # The pure_fp16 GGUF stores HTP matrices in HMX layout; CPU fallback on those
  # tensors is only a debug mode and will not produce valid text.
  case "${SKIP_HTP_OPS:-}" in
    1|true|TRUE|yes|YES|on|ON) export SKIP_HTP_OPS=1 ;;
    *) unset SKIP_HTP_OPS ;;
  esac

  # Positional args intentionally override env vars so adb shell commands can
  # carry the exact benchmark shape without a prefix full of temporary exports.
  LLAMA_N_BATCH="${6:-${LLAMA_N_BATCH:-512}}"
  LLAMA_N_UBATCH="${7:-${LLAMA_N_UBATCH:-512}}"
  LLAMA_N_THREADS="${8:-${LLAMA_N_THREADS:-4}}"
  LLAMA_N_CTX="${9:-${LLAMA_N_CTX:-4096}}"
  LLAMA_N_PREDICT="${10:-${LLAMA_N_PREDICT:-2048}}"
  LLAMA_NO_WARMUP_FLAG="${11:-${LLAMA_NO_WARMUP:-0}}"
  NO_WARMUP_ARGS=""
  case "$LLAMA_NO_WARMUP_FLAG" in
    1|true|TRUE|yes|YES|on|ON) NO_WARMUP_ARGS="--no-warmup" ;;
  esac
  # Pure FP16 is weight-bandwidth heavy. The current v73 tuning keeps logical
  # batch and ubatch aligned at 512: this avoids the small-prompt split penalty
  # while staying stable for the long-prompt PD sweep after the parallel publish
  # and output-store changes.
  SERVER_ARGS="-t $LLAMA_N_THREADS -fa -m $MODEL --host 0.0.0.0 --port $PORT -c $LLAMA_N_CTX -b $LLAMA_N_BATCH -ub $LLAMA_N_UBATCH -n $LLAMA_N_PREDICT $NO_WARMUP_ARGS"

  ensure_api_key
  pkill -f llama-server 2>/dev/null || true
  rm -f server.log server.pid
  echo "starting llama-server model=$MODEL LLAMA_NPU_MODE=$LLAMA_NPU_MODE sidecar=${LLAMA_NPU_LPBQ_SIDECAR_DIR:-unset} disable_k32_safe=${LLAMA_NPU_LPBQ_DISABLE_K32_SAFE_SIDECAR:-unset} r4_enable_path=${LLAMA_NPU_LPBQ_ENABLE_R4_PATH:-unset} r4_scale_fold=${LLAMA_NPU_LPBQ_ENABLE_R4_SCALE_FOLD:-unset} r4_structured_fwht_small_m_max=${LLAMA_NPU_LPBQ_R4_STRUCTURED_FWHT_SMALL_M_MAX:-unset} force_r4_full_u8_safe=${LLAMA_NPU_LPBQ_FORCE_R4_FULL_U8_SAFE:-unset} r4_compact_full_u8_safe_ab=${LLAMA_NPU_LPBQ_R4_COMPACT_FULL_U8_SAFE_AB:-unset} r4_use_full_v6_weight_fd=${LLAMA_NPU_LPBQ_R4_USE_FULL_V6_WEIGHT_FD:-unset} trace=$LLAMA_NPU_TRACE detailed=$LLAMA_NPU_DETAILED_TRACE SKIP_HTP_OPS=${SKIP_HTP_OPS:-unset} no_warmup=$LLAMA_NO_WARMUP_FLAG port=$PORT n_batch=$LLAMA_N_BATCH n_ubatch=$LLAMA_N_UBATCH threads=$LLAMA_N_THREADS ctx=$LLAMA_N_CTX predict=$LLAMA_N_PREDICT" > server.log
  nohup ./llama-server $SERVER_ARGS $AUTH_ARGS >> server.log 2>&1 < /dev/null &
  echo $! > server.pid
  echo "llama-server started with API auth from $AUTH_SOURCE"
}

case "${1:-start}" in
  start)
    shift || true
    start_server "$@"
    ;;
  stop)
    pkill -f llama-server 2>/dev/null || true
    ;;
  status)
    ps -A | grep llama-server || true
    if [ -n "${LLAMA_API_KEY:-}" ]; then
      echo "API auth source: LLAMA_API_KEY"
    elif [ -f "$API_KEY_FILE" ]; then
      echo "API auth source: $API_KEY_FILE"
    else
      echo "API auth source: will create $API_KEY_FILE on start"
    fi
    tail -160 server.log 2>/dev/null || true
    ;;
  foreground)
    shift || true
    MODEL="${1:-${LLAMA_MODEL:-$DEFAULT_MODEL}}"
    export LLAMA_NPU_MODE="${2:-${LLAMA_NPU_MODE:-pure_fp16}}"
    export LLAMA_NPU_TRACE="${3:-${LLAMA_NPU_TRACE:-0}}"
    export LLAMA_NPU_DETAILED_TRACE="${4:-${LLAMA_NPU_DETAILED_TRACE:-0}}"
    PORT="${5:-${LLAMA_PORT:-8080}}"
    case "$LLAMA_NPU_MODE" in
      lpbq_int8|lpbq-int8|lpbq-w4a8-int8|w4a8_lpbq)
        export LLAMA_NPU_LPBQ_SIDECAR_DIR="${LLAMA_NPU_LPBQ_SIDECAR_DIR:-$APP_DIR/sidecars/lpbq_g16_a8w8}"
        # See start_server: K32-safe flags are generated by host and consumed by
        # the DSP R4/nibble fallback. Old disabled default was "${...:-1}".
        export LLAMA_NPU_LPBQ_DISABLE_K32_SAFE_SIDECAR="${LLAMA_NPU_LPBQ_DISABLE_K32_SAFE_SIDECAR:-0}"
        # See start_server: default R4 input-scale folding removes one online
        # multiply from the dense R4 rotate path; set env to 0 for rollback.
        export LLAMA_NPU_LPBQ_ENABLE_R4_SCALE_FOLD="${LLAMA_NPU_LPBQ_ENABLE_R4_SCALE_FOLD:-1}"
        # See start_server: default 4 preserves accepted small-M-only FWHT;
        # set -1 only for explicit no-quality all-M speed probes.
        export LLAMA_NPU_LPBQ_R4_STRUCTURED_FWHT_SMALL_M_MAX="${LLAMA_NPU_LPBQ_R4_STRUCTURED_FWHT_SMALL_M_MAX:-4}"
        ;;
    esac
    LLAMA_N_BATCH="${6:-${LLAMA_N_BATCH:-512}}"
    LLAMA_N_UBATCH="${7:-${LLAMA_N_UBATCH:-512}}"
    LLAMA_N_THREADS="${8:-${LLAMA_N_THREADS:-4}}"
    LLAMA_N_CTX="${9:-${LLAMA_N_CTX:-4096}}"
    LLAMA_N_PREDICT="${10:-${LLAMA_N_PREDICT:-2048}}"
    LLAMA_NO_WARMUP_FLAG="${11:-${LLAMA_NO_WARMUP:-0}}"
    NO_WARMUP_ARGS=""
    case "$LLAMA_NO_WARMUP_FLAG" in
      1|true|TRUE|yes|YES|on|ON) NO_WARMUP_ARGS="--no-warmup" ;;
    esac
    ensure_api_key
    # Pure-FP16 prefill uses the tuned 512/512 batch pair; keep foreground aligned with start for reproducible benchmarks.
    exec ./llama-server -t "$LLAMA_N_THREADS" -fa -m "$MODEL" --host 0.0.0.0 --port "$PORT" -c "$LLAMA_N_CTX" -b "$LLAMA_N_BATCH" -ub "$LLAMA_N_UBATCH" -n "$LLAMA_N_PREDICT" $NO_WARMUP_ARGS $AUTH_ARGS
    ;;
  show-key)
    show_api_key
    ;;
  set-key)
    set_api_key "${2:-}"
    ;;
  *)
    echo "usage: $0 {start [model mode trace detailed port [n_batch n_ubatch threads n_ctx n_predict no_warmup]]|stop|status|foreground [model mode trace detailed port [n_batch n_ubatch threads n_ctx n_predict no_warmup]]|show-key|set-key <api-key>}" >&2
    exit 2
    ;;
esac
