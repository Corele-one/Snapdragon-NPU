#pragma once

#include <stdint.h>

enum HtpOpsIndex {
  HTP_OPS_RMS_NORM_F32,
  HTP_OPS_MAT_MUL_PERMUTED_W16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
  HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT,
  HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
  HTP_OPS_FLASH_ATTN_PROFILE_QO_F32_KV_F16,
  HTP_OPS_HMX_INT8_GATE,
  HTP_OPS_COUNT,
};

enum LlmNpuModeFlags {
  LLM_NPU_MODE_LUT_EXP          = 1 << 0,
  LLM_NPU_MODE_HMX_AWARE_TILE   = 1 << 1,
  LLM_NPU_MODE_SCNA_FP16        = 1 << 2,
  LLM_NPU_MODE_SCNA_INT8        = 1 << 3,
  LLM_NPU_MODE_SCNA_D8          = 1 << 4,
  LLM_NPU_MODE_SCNA_D32         = 1 << 5,
  LLM_NPU_MODE_NUMERIC_DEBUG    = 1 << 6,
  LLM_NPU_MODE_SCNA_FUNCTION_EXP = 1 << 7,
  LLM_NPU_MODE_TRACE            = 1 << 8,
  LLM_NPU_MODE_DETAILED_TRACE   = 1 << 9,
  LLM_NPU_MODE_SCNA_TREE        = 1 << 10,
  LLM_NPU_MODE_SCNA_KV_PIPELINE = 1 << 11,
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
