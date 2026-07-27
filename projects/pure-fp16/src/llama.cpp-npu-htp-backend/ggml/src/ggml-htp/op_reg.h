#pragma once

#include <stdint.h>

enum HtpOpsIndex {
  HTP_OPS_RMS_NORM_F32,
  HTP_OPS_MAT_MUL_PERMUTED_W16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
  HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT,
  HTP_OPS_MAT_MUL_LPBQ_A8W8,
  HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
  HTP_OPS_FLASH_ATTN_PROFILE_QO_F32_KV_F16,
  HTP_OPS_HMX_INT8_GATE,
  HTP_OPS_COUNT,
};

enum LlmNpuModeFlags {
  LLM_NPU_MODE_LUT_EXP          = 1 << 0,
  LLM_NPU_MODE_HMX_AWARE_TILE   = 1 << 1,
  // Pure-FP16 experiment marker. This does not select a different math path by
  // itself; GGML_TYPE_F16 weights already route to HTP_OPS_MAT_MUL_PERMUTED_W16A32.
  // The flag keeps trace artifacts distinguishable from the legacy baseline label.
  LLM_NPU_MODE_PURE_FP16        = 1 << 2,
  // LPBQ deploy-v1 route: static signed A8 activation, offline-expanded int8
  // LPBQ weight, and unsigned activation HMX with zero-point correction.
  LLM_NPU_MODE_LPBQ_INT8        = 1 << 3,
  LLM_NPU_MODE_LPBQ_R4          = 1 << 4,
  LLM_NPU_MODE_LPBQ_R4_FOLDED_INPUT_SCALE = 1 << 5,
  LLM_NPU_MODE_TRACE            = 1 << 8,
  LLM_NPU_MODE_DETAILED_TRACE   = 1 << 9,
  // Off by default: LPBQ detailed trace normally emits aggregate stage rows
  // only. Enable this for bring-up when per-tile DSP profile rows are needed.
  LLM_NPU_MODE_LPBQ_TILE_TRACE  = 1 << 10,
  // Host-side packed-weight loader has transposed the compact sidecar from
  // N-major [n_tile][k_tile] to K-major [k_tile][n_tile] for sequential DSP
  // streaming across all output tiles of the current K tile.
  LLM_NPU_MODE_LPBQ_PACKED_K_MAJOR = 1 << 11,
  // Experimental LPBQ deploy-v1 sidecar: packed_weight points at grouped V6
  // full-stride HMX tiles, so the DSP hot loop can skip compact-to-full weight
  // expansion. This is guarded and not enabled by the default LLM route.
  LLM_NPU_MODE_LPBQ_PACKED_V6_FULL = 1 << 12,
  // Bring-up only: emit a compact LPBQ path diagnostic event into the profile
  // buffer so real-layer gates do not depend on device logcat/FARF delivery.
  LLM_NPU_MODE_LPBQ_PATH_DIAG = 1 << 13,
  // Runtime safety gate for R4 full-U8 math. The compile-time full-U8 path can
  // stay built in, but trained ffn_down layers only enter it after a real-layer
  // safe list proves the per-layer recover law is accurate.
  LLM_NPU_MODE_LPBQ_R4_FULL_U8_SAFE = 1 << 14,
  // Per-layer R4 grouped-V6 recover override: use fp16 scale 1/16 with <<5
  // recovery for layers whose real-layer gate rejects the default 1/8 + <<4
  // law. This remains safe-list driven so other ffn_down layers keep the
  // stricter default path.
  LLM_NPU_MODE_LPBQ_R4_V6_SCALE_1_16 = 1 << 15,
  // Correctness-first Q/K route: non-R4 grouped-V6 has row-varying after.uh
  // recover error on attn_q/attn_k, so only explicitly flagged layers enter
  // the exact unscaled K64/fallback path.
  LLM_NPU_MODE_LPBQ_EXACT_NON_R4 = 1 << 16,
  // Structured R4 sidecar: r4 fd carries a compact D2/H/D1 FWHT payload instead
  // of dense column-major coefficients. The dense R4 path remains the fallback.
  LLM_NPU_MODE_LPBQ_R4_STRUCTURED_FWHT = 1 << 17,
  // No-quality/performance-first Stage-A sidecar: r4_hmx_dense_fp16 carries
  // prepacked FP16 HMX dense-R4 tiles so the DSP HFSCALE probe can skip runtime
  // FP32 R4 -> FP16 HMX weight conversion. This is not deploy-quality.
  LLM_NPU_MODE_LPBQ_R4_HMX_DENSE_FP16_SIDECAR = 1 << 18,
};

struct RpcmemBufAddr {
  int32_t fd;
  int32_t offset;
} __attribute__((packed));

struct RmsNormF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  int32_t       ne0;
  int32_t       ne1;
  int64_t       trace_id;
  int32_t       mode_flags;
  int32_t       max_profile_events;
  struct RpcmemBufAddr profile;
} __attribute__((packed));

struct MatMulParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr activation; // m * k
  struct RpcmemBufAddr weight; // k * n
  int32_t m;
  int32_t k;
  int32_t n;
  int64_t trace_id;
  int32_t mode_flags;
  int32_t max_profile_events;
  struct RpcmemBufAddr profile;
} __attribute__((packed));

struct LpbqA8W8MatMulParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr activation; // m * k FP32
  struct RpcmemBufAddr weight;     // GGUF Q8_0 container carrying HMX tile-major int8 payload
  struct RpcmemBufAddr packed_weight; // optional compact K4 HMX weight chunks, fd < 0 means repack GGUF weight online
  struct RpcmemBufAddr sum_w;      // optional n INT32 zero-point column sums for packed_weight
  struct RpcmemBufAddr k32_safe;   // optional uint8 [Ktile,Ntile] or [Ntile,Ktile] flags for safe K32 HMX grouping
  struct RpcmemBufAddr k64_safe;   // optional uint8 [K64tile,Ntile] or [Ntile,K64tile] flags for exact K64 grouping
  struct RpcmemBufAddr scale2;     // n FP32, includes LPBQ second-level scale
  struct RpcmemBufAddr bias;       // optional n FP32, fd < 0 means no bias
  struct RpcmemBufAddr r4;         // optional R4 block matrix, fd < 0 means no online rotation
  struct RpcmemBufAddr r4_hmx_dense_fp16; // optional prepacked FP16 HMX dense-R4 tiles, experiment-only
  struct RpcmemBufAddr input_scale; // optional k FP32 inverse LET scale fused into activation quant
  struct RpcmemBufAddr out_scale;  // optional n FP32 = act_scale * scale2
  struct RpcmemBufAddr bias_eff;   // optional n FP32 = bias - 128 * sum_w * out_scale
  float act_scale;                 // static signed A8 scale for this Linear input
  int32_t r4_block;                // R4 block dimension, normally 128 for OST down_proj
  int32_t m;
  int32_t k;
  int32_t n;
  int64_t trace_id;
  int32_t mode_flags;
  int32_t max_profile_events;
  struct RpcmemBufAddr profile;
} __attribute__((packed));

struct HmxInt8GateParams {
  struct RpcmemBufAddr output;
  int32_t max_results;
  int32_t reserved;
} __attribute__((packed));

struct HmxInt8GateResult {
  int32_t selector;
  int32_t nan_count;
  int32_t variant;
  int32_t output_kind;
  int32_t tile_bytes;
  int32_t reserved;
  float expected;
  float first8[8];
  float min_value;
  float max_value;
  float mean_value;
  float rmse;
} __attribute__((packed));

struct FlashAttnParams {
  struct RpcmemBufAddr o;
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr mask;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  int64_t trace_id;
  int32_t mode_flags;
  int32_t max_profile_events;
  struct RpcmemBufAddr profile;
} __attribute__((packed));

#define LLM_TRACE_PROFILE_MAGIC 0x4c545250

enum LlmTraceStageComponent {
  LLM_TRACE_STAGE_VALIDATE_IN = 1,
  LLM_TRACE_STAGE_VALIDATE_OUT,
  LLM_TRACE_STAGE_ACTIVATION_HVX_LOAD,
  LLM_TRACE_STAGE_ACTIVATION_DMA_INFLIGHT,
  LLM_TRACE_STAGE_ACTIVATION_DMA_WAIT,
  LLM_TRACE_STAGE_WEIGHT_DMA_INFLIGHT,
  LLM_TRACE_STAGE_WEIGHT_DMA_WAIT,
  LLM_TRACE_STAGE_WEIGHT_HVX_DEQUANT,
  LLM_TRACE_STAGE_WEIGHT_HVX_LOAD,
  LLM_TRACE_STAGE_HMX_MMA,
  LLM_TRACE_STAGE_HVX_COMPUTE,
  LLM_TRACE_STAGE_OUTPUT_STORE,
  LLM_TRACE_STAGE_ACTIVATION_QUANTIZE,
  LLM_TRACE_STAGE_ACTIVATION_PACK,
  LLM_TRACE_STAGE_FLASH_Q_LOAD,
  LLM_TRACE_STAGE_FLASH_K_LOAD,
  LLM_TRACE_STAGE_FLASH_V_LOAD,
  LLM_TRACE_STAGE_FLASH_QK_DOT,
  LLM_TRACE_STAGE_FLASH_SAFE_SM,
  LLM_TRACE_STAGE_FLASH_CORE_ACC,
  LLM_TRACE_STAGE_FLASH_O_SCALE,
  LLM_TRACE_STAGE_FLASH_O_STORE,
  LLM_TRACE_STAGE_ZERO_POINT_CORRECTION,
  LLM_TRACE_STAGE_DEQUANT_STORE,
  LLM_TRACE_STAGE_R4_ROTATE,
  LLM_TRACE_STAGE_R4_SCALE_CACHE,
  LLM_TRACE_STAGE_R4_DOT_PACK,
  LLM_TRACE_STAGE_R4_DENSE_DOT,
  LLM_TRACE_STAGE_R4_QUANT_SCALE,
  LLM_TRACE_STAGE_R4_PACK_UB_LAYOUT,
  LLM_TRACE_STAGE_R4_UNACCOUNTED,
  LLM_TRACE_STAGE_HMX_BEGIN,
  LLM_TRACE_STAGE_HMX_WEIGHT_EXPAND,
  LLM_TRACE_STAGE_HMX_ISSUE,
  LLM_TRACE_STAGE_HMX_FINISH,
  LLM_TRACE_STAGE_HMX_ACCUMULATE,
  LLM_TRACE_STAGE_HMX_CORE,
  LLM_TRACE_STAGE_HMX_UNACCOUNTED,
  LLM_TRACE_STAGE_HMX_ADAPTIVE_PROBE_ISSUE,
  LLM_TRACE_STAGE_HMX_ADAPTIVE_PROBE_FINISH,
  LLM_TRACE_STAGE_HMX_ADAPTIVE_SCALE_DERIVE,
  LLM_TRACE_STAGE_HMX_ADAPTIVE_FINAL_ISSUE,
  LLM_TRACE_STAGE_HMX_ADAPTIVE_FINAL_FINISH,
  LLM_TRACE_STAGE_HMX_GROUPED_V6_CONTROL_GAP,
  LLM_TRACE_STAGE_LPBQ_PATH_DIAG,
  LLM_TRACE_STAGE_WEIGHT_RPCMEM_READ,
  LLM_TRACE_STAGE_WEIGHT_L2FETCH_OR_DMA,
  LLM_TRACE_STAGE_WEIGHT_COMPACT_DECODE,
  LLM_TRACE_STAGE_WEIGHT_GROUP_TILE_COPY,
  LLM_TRACE_STAGE_WEIGHT_ROWBLOCK4_PUBLISH,
  LLM_TRACE_STAGE_WEIGHT_G32_STAGING,
  LLM_TRACE_STAGE_WEIGHT_DMA_ISSUE,
  LLM_TRACE_STAGE_WEIGHT_DMA_WAIT_LPBQ,
  LLM_TRACE_STAGE_WEIGHT_VISIBILITY_SYNC,
  LLM_TRACE_STAGE_WEIGHT_FALLBACK_HVX_PUBLISH,
  LLM_TRACE_STAGE_WEIGHT_VTCM_COMMIT,
  LLM_TRACE_STAGE_WEIGHT_HVX_LOAD_ACTUAL,
  LLM_TRACE_STAGE_WEIGHT_CACHE_LOOKUP,
  LLM_TRACE_STAGE_WEIGHT_CACHE_FILL,
  LLM_TRACE_STAGE_WEIGHT_UNATTRIBUTED,
  LLM_TRACE_STAGE_R4_FWHT_LOAD,
  LLM_TRACE_STAGE_R4_FWHT_BUTTERFLY,
  LLM_TRACE_STAGE_R4_FWHT_SCALE,
  LLM_TRACE_STAGE_R4_FWHT_QUANT,
  LLM_TRACE_STAGE_R4_FWHT_V6_STORE,
  LLM_TRACE_STAGE_ACT_REDUCE_MAX,
  LLM_TRACE_STAGE_ACT_RECIP_SCALE,
  LLM_TRACE_STAGE_ACT_QUANT,
  LLM_TRACE_STAGE_ACT_V6_STORE,
  LLM_TRACE_STAGE_HMX_ACQUIRE,
  LLM_TRACE_STAGE_HMX_SCALE_PAYLOAD_LOAD,
  LLM_TRACE_STAGE_HMX_ACC_CLEAR,
  LLM_TRACE_STAGE_HMX_LOAD_ISSUE,
  LLM_TRACE_STAGE_HMX_ACCUMULATE_WAIT,
  LLM_TRACE_STAGE_HMX_CONVERT_ISSUE,
  LLM_TRACE_STAGE_HMX_CONVERT_WAIT,
  LLM_TRACE_STAGE_HMX_EPILOGUE_HVX,
};

enum LlmTraceStageUnit {
  LLM_TRACE_UNIT_OTHER = 0,
  LLM_TRACE_UNIT_DMA = 1,
  LLM_TRACE_UNIT_HVX = 2,
  LLM_TRACE_UNIT_HMX = 3,
  LLM_TRACE_UNIT_STORE = 4,
  LLM_TRACE_UNIT_MEMORY = 5,
  LLM_TRACE_UNIT_SCALAR = 6,
};

struct LlmTraceProfileHeader {
  int32_t magic;
  int32_t max_events;
  int32_t event_count;
  int32_t event_overflow;
  int32_t reserved0;
  int32_t reserved1;
};

struct LlmTraceProfileEvent {
  int64_t trace_id;
  int32_t op_index;
  int32_t stage;
  int32_t unit;
  int32_t worker;
  int32_t m;
  int32_t k;
  int32_t n;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  int32_t mr;
  int32_t nc;
  int32_t kk;
  int32_t chunk_m;
  int32_t chunk_n;
  int32_t chunk_k;
  int32_t flags;
  int64_t bytes;
  int64_t t0_us;
  int64_t t1_us;
  int64_t dur_us;
};

static inline struct LlmTraceProfileEvent *llm_trace_profile_events(struct LlmTraceProfileHeader *header) {
  return (struct LlmTraceProfileEvent *) (header + 1);
}

static inline const struct LlmTraceProfileEvent *llm_trace_profile_events_const(
  const struct LlmTraceProfileHeader *header) {
  return (const struct LlmTraceProfileEvent *) (header + 1);
}
