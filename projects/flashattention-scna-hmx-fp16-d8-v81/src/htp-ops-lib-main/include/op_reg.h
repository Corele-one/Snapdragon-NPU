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
  HTP_OPS_ROOFLINE_BENCH,
  HTP_OPS_SCNA_EXP2_BENCH,
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
  LLM_NPU_MODE_SCNA_HMX_HYBRID  = 1 << 12,
  LLM_NPU_MODE_SCNA_HMX_TWO_PASS = 1 << 13,
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

enum RooflineBenchMode {
  ROOFLINE_BENCH_MODE_HMX_FP16 = 1,
  ROOFLINE_BENCH_MODE_DDR_BW = 2,
  ROOFLINE_BENCH_MODE_VTCM_BW = 3,
  ROOFLINE_BENCH_MODE_HMX_DMA_BW = 4,
  ROOFLINE_BENCH_MODE_HVX_FP16 = 5,
  ROOFLINE_BENCH_MODE_MIX_PRECISION = 6,
  ROOFLINE_BENCH_MODE_HMX_INT8_SHAPE_SWEEP = 7,
  ROOFLINE_BENCH_MODE_SIGNED_INT8_ZERO_OVERHEAD = 8,
};

enum RooflineBenchKind {
  ROOFLINE_BENCH_KIND_HMX_FP16_GEMM = 1,
  ROOFLINE_BENCH_KIND_DDR_READ = 2,
  ROOFLINE_BENCH_KIND_DDR_WRITE = 3,
  ROOFLINE_BENCH_KIND_DDR_COPY = 4,
  ROOFLINE_BENCH_KIND_VTCM_READ = 5,
  ROOFLINE_BENCH_KIND_VTCM_WRITE = 6,
  ROOFLINE_BENCH_KIND_VTCM_COPY = 7,
  ROOFLINE_BENCH_KIND_HMX_DMA_READ = 8,
  ROOFLINE_BENCH_KIND_HVX_FP16_GEMM = 9,
  ROOFLINE_BENCH_KIND_HVX_FP32_GEMM = 10,
  ROOFLINE_BENCH_KIND_HVX_INT16_GEMM = 11,
  // Reserved legacy ids from an earlier semantic-only prototype.  The mixed-precision
  // roofline does not emit these as peaks because sign-extending INT8/INT4 into INT16
  // is not a native INT8/INT4 hardware MAC measurement.
  ROOFLINE_BENCH_KIND_HVX_INT8_AS_INT16_GEMM = 12,
  ROOFLINE_BENCH_KIND_HVX_INT4_AS_INT16_GEMM = 13,
  ROOFLINE_BENCH_KIND_HVX_INT4_INT8_AS_INT16_GEMM = 14,
  ROOFLINE_BENCH_KIND_HVX_INT4_INT16_GEMM = 15,
  ROOFLINE_BENCH_KIND_HVX_INT8_INT16_GEMM = 16,
  ROOFLINE_BENCH_KIND_HMX_INT8_RAW_UB_B_GEMM = 17,
  ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_K2_GEMM = 18,
  ROOFLINE_BENCH_KIND_FORMAT_Q4_0_DECODE_FP16_GEMM = 19,
  ROOFLINE_BENCH_KIND_FORMAT_IQ4_NL_DECODE_FP16_GEMM = 20,
  ROOFLINE_BENCH_KIND_NOT_AVAILABLE = 21,
  ROOFLINE_BENCH_KIND_HMX_INT8_INT4_WEIGHT_N_GEMM = 22,
  ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_ZP_CORRECTED_GEMM = 23,
  ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_COPY = 24,
  ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_COPY_XOR = 25,
  ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_XOR_INPLACE = 26,
  ROOFLINE_BENCH_KIND_SIGNED_A8_COLSUM_PRECOMPUTE = 27,
  ROOFLINE_BENCH_KIND_SIGNED_A8_HVX_BIAS_SCALE_STORE = 28,
  ROOFLINE_BENCH_KIND_SIGNED_A8_REQUANT_STORE_ZP0 = 29,
  ROOFLINE_BENCH_KIND_SIGNED_A8_REQUANT_STORE_ZP128 = 30,
  ROOFLINE_BENCH_KIND_HMX_FP16_INT8_WEIGHT_B_GEMM = 31,
  ROOFLINE_BENCH_KIND_HMX_FP8_GEMM = 32,
  // V81 HVX register-resident throughput probes. These are benchmark-only
  // instruction roofs and remain separate from the software GEMM rows above.
  ROOFLINE_BENCH_KIND_HVX_FP32_MULADD_PEAK = 33,
  ROOFLINE_BENCH_KIND_HVX_FP16_MAC_PEAK = 34,
  ROOFLINE_BENCH_KIND_HVX_BF16_MAC_PEAK = 35,
  ROOFLINE_BENCH_KIND_HVX_FP8_MAC_PEAK = 36,
  ROOFLINE_BENCH_KIND_HVX_S16_MAC_PEAK = 37,
  ROOFLINE_BENCH_KIND_HVX_U16_MAC_PEAK = 38,
  ROOFLINE_BENCH_KIND_HVX_S16_U16_MAC_PEAK = 39,
  ROOFLINE_BENCH_KIND_HVX_S8_MAC_PEAK = 40,
  ROOFLINE_BENCH_KIND_HVX_U8_MAC_PEAK = 41,
  ROOFLINE_BENCH_KIND_HVX_U8_S8_MAC_PEAK = 42,
  ROOFLINE_BENCH_KIND_HMX_FP16_INT4_WEIGHT_N_GEMM = 43,
  // V81 manual-only smoke rows. These validate control/data movement
  // instructions and intentionally do not publish a compute peak.
  ROOFLINE_BENCH_KIND_V81_HMX_MANUAL_SMOKE = 44,
};

enum RooflineBenchEngine {
  ROOFLINE_BENCH_ENGINE_UNKNOWN = 0,
  ROOFLINE_BENCH_ENGINE_HMX = 1,
  ROOFLINE_BENCH_ENGINE_HVX = 2,
  ROOFLINE_BENCH_ENGINE_FORMAT_EFFECTIVE = 3,
  ROOFLINE_BENCH_ENGINE_SCALAR = 4,
};

enum RooflineBenchDType {
  ROOFLINE_BENCH_DTYPE_UNKNOWN = 0,
  ROOFLINE_BENCH_DTYPE_FP32 = 1,
  ROOFLINE_BENCH_DTYPE_FP16 = 2,
  ROOFLINE_BENCH_DTYPE_INT16 = 3,
  ROOFLINE_BENCH_DTYPE_INT8 = 4,
  ROOFLINE_BENCH_DTYPE_INT4_LINEAR = 5,
  ROOFLINE_BENCH_DTYPE_Q4_0 = 6,
  ROOFLINE_BENCH_DTYPE_IQ4_NL = 7,
  ROOFLINE_BENCH_DTYPE_INT32 = 8,
  ROOFLINE_BENCH_DTYPE_FP8 = 9,
  ROOFLINE_BENCH_DTYPE_BF16 = 10,
  ROOFLINE_BENCH_DTYPE_UINT16 = 11,
  ROOFLINE_BENCH_DTYPE_UINT8 = 12,
};

enum RooflineBenchPath {
  ROOFLINE_BENCH_PATH_UNKNOWN = 0,
  ROOFLINE_BENCH_PATH_HMX_FP16_TILE = 1,
  ROOFLINE_BENCH_PATH_HMX_RAW_UB_B_DEEP_GEMM = 2,
  ROOFLINE_BENCH_PATH_HMX_SIGNED_K2 = 3,
  ROOFLINE_BENCH_PATH_HVX_NATIVE = 4,
  ROOFLINE_BENCH_PATH_HVX_SIGNEXT_I16 = 5,
  ROOFLINE_BENCH_PATH_FORMAT_DECODE_TO_FP16 = 6,
  ROOFLINE_BENCH_PATH_NOT_AVAILABLE = 7,
  ROOFLINE_BENCH_PATH_HMX_RAW_UB_N_DEEP_GEMM = 8,
  ROOFLINE_BENCH_PATH_HMX_SIGNED_A8_VIA_UB_COLSUM = 9,
  ROOFLINE_BENCH_PATH_HVX_COPY = 10,
  ROOFLINE_BENCH_PATH_HVX_COPY_XOR_0X80 = 11,
  ROOFLINE_BENCH_PATH_HVX_XOR_0X80_INPLACE = 12,
  ROOFLINE_BENCH_PATH_OFFLINE_COLSUM = 13,
  ROOFLINE_BENCH_PATH_HVX_BIAS_SCALE_FLOAT_STORE = 14,
  ROOFLINE_BENCH_PATH_HVX_REQUANT_STORE_ZP = 15,
  ROOFLINE_BENCH_PATH_HMX_HF_B_GEMM = 16,
  ROOFLINE_BENCH_PATH_HMX_F8_F8_GEMM = 17,
  ROOFLINE_BENCH_PATH_HVX_FP32_QF32_MUL_ADD = 18,
  ROOFLINE_BENCH_PATH_HVX_FP16_VDMPYACC = 19,
  ROOFLINE_BENCH_PATH_HVX_BF16_VMPYACC = 20,
  ROOFLINE_BENCH_PATH_HVX_FP8_VMPYACC = 21,
  ROOFLINE_BENCH_PATH_HVX_S16_VMPYACC = 22,
  ROOFLINE_BENCH_PATH_HVX_U16_VMPYACC = 23,
  ROOFLINE_BENCH_PATH_HVX_S16_U16_VMPYACC = 24,
  ROOFLINE_BENCH_PATH_HVX_S8_VRMPYACC = 25,
  ROOFLINE_BENCH_PATH_HVX_U8_VRMPYACC = 26,
  ROOFLINE_BENCH_PATH_HVX_U8_S8_VRMPYACC = 27,
  ROOFLINE_BENCH_PATH_HMX_HF_N_GEMM = 28,
};

struct RooflineBenchParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr src;
  struct RpcmemBufAddr dst;
  int32_t max_results;
  int32_t mode;
  int32_t warmup;
  int32_t iters;
  int32_t bytes;
} __attribute__((packed));

struct RooflineBenchResult {
  int32_t mode;
  int32_t kind;
  int32_t variant;
  int32_t size;
  int32_t iters;
  int64_t elapsed_us;
  int64_t work_items;
  int64_t metric_x10000;
  int32_t engine;
  int32_t lhs_dtype;
  int32_t rhs_dtype;
  int32_t acc_dtype;
  int32_t path;
  int32_t correctness;
  int32_t m;
  int32_t k;
  int32_t n;
  int32_t mt;
  int32_t kt;
  int32_t nt;
  int32_t tile_bytes;
  int32_t reserved0;
  int64_t a_bytes;
  int64_t b_bytes;
  int64_t c_bytes;
  int64_t scales_bytes;
  int64_t allocated_bytes;
  int64_t total_vtcm_bytes;
} __attribute__((packed));

enum ScnaPrecision {
  SCNA_PRECISION_FP16 = 1,
  SCNA_PRECISION_INT8 = 2,
};

enum ScnaFunction {
  SCNA_FUNCTION_EXP2 = 0,
  SCNA_FUNCTION_EXP  = 1,
};

enum ScnaKernel {
  SCNA_KERNEL_DIRECT = 0,
  SCNA_KERNEL_TREE   = 1,
};

enum ScnaEngine {
  SCNA_ENGINE_HVX          = 1,
  SCNA_ENGINE_HMX_HYBRID   = 2,
  SCNA_ENGINE_HMX_TWO_PASS = 3,
};

struct ScnaExp2BenchParams {
  struct RpcmemBufAddr output;
  int32_t width;
  int32_t mode_flags;
  int32_t warmup;
  int32_t iters;
} __attribute__((packed));

struct ScnaExp2BenchResult {
  int32_t width;
  int32_t precision;
  int32_t function;
  int32_t kernel;
  int32_t lanes;
  int32_t iters;
  int64_t elapsed_us;
  int64_t pair_elapsed_us;
  float rmse;
  float max_abs_error;
  float dense_rmse;
  float dense_max_abs_error;
  float pair_max_abs_diff;
  float direct_tree_max_abs_diff;
  float output_at_min;
  float output_near_minus12;
  float output_near_minus4;
  float output_at_zero;
  int32_t dense_samples;
  int32_t monotonic_violations;
  int32_t negative_count;
  int32_t nan_count;
  int32_t int8_s32_pack_mismatches;
  int32_t int8_s32_pack_max_abs_diff;
  int16_t int8_s32_pack_probe_input[8];
  int16_t int8_s32_pack_probe_output[8];
  float convert_zero;
  float convert_one;
  float convert_quarter;
  float convert_neg_half;
  int32_t engine;
  int32_t profile_version;
  int64_t pack_us;
  int64_t hmx_affine_relu_us;
  int64_t reduction_us;
  int64_t unpack_us;
  int64_t kernel_total_us;
  float implementation_rmse;
  float implementation_max_abs_error;
  int32_t random_samples;
  float random_implementation_rmse;
  float random_implementation_max_abs_error;
  float tail_implementation_max_abs_error;
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

#define FIGURE8_PROFILE_MAGIC 0x46494738

enum Figure8ProfileComponent {
  FIGURE8_COMP_Q_LOAD = 1,
  FIGURE8_COMP_K_LOAD,
  FIGURE8_COMP_V_LOAD,
  FIGURE8_COMP_QK_DOT,
  FIGURE8_COMP_SAFE_SM,
  FIGURE8_COMP_CORE_ACC,
  FIGURE8_COMP_O_SCALE,
  FIGURE8_COMP_O_STORE,
  FIGURE8_COMP_SCNA_EXP2,
  FIGURE8_COMP_KV_DMA_ISSUE,
  FIGURE8_COMP_KV_DMA_WAIT,
  FIGURE8_COMP_KV_TRANSFORM,
};

struct Figure8ProfileHeader {
  int32_t magic;
  int32_t max_records;
  int32_t record_count;
  int32_t max_events;
  int32_t event_count;
  int32_t event_overflow;
  int32_t reserved0;
  int32_t reserved1;
};

struct Figure8ProfileRecord {
  int32_t lut_exp;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  int32_t kv_head;
  int32_t worker;
  int64_t profiled_total;
  int64_t q_load;
  int64_t k_load;
  int64_t v_load;
  int64_t qk_dot;
  int64_t safe_sm;
  int64_t core_acc;
  int64_t o_scale;
  int64_t o_store;
  int64_t scna_exp2;
  int32_t nonlinear_mode;
  int32_t scna_width;
  int32_t scna_function;
  int32_t scna_kernel;
  int32_t scna_pipeline;
  int64_t kv_dma_issue;
  int64_t kv_dma_wait;
  int64_t kv_transform;
  int32_t debug_rowsum0_bits;
  int32_t debug_l0_bits;
  int32_t debug_core_o0_bits;
  int32_t debug_inv_l0_bits;
  int32_t debug_scaled_o0_bits;
  int32_t debug_p0_first_bits;
  int32_t debug_p0_last_bits;
  int32_t debug_sum0_first_bits;
  int32_t debug_sum0_last_bits;
  int32_t profile_version;
  int32_t scna_engine;
  int64_t scna_pack;
  int64_t scna_hmx_affine_relu;
  int64_t scna_reduction;
  int64_t scna_unpack;
};

struct Figure8ProfileEvent {
  int32_t component;
  int32_t lut_exp;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  int32_t kv_head;
  int32_t worker;
  int32_t block_r;
  int32_t block_c;
  int32_t reserved;
  int64_t t0_us;
  int64_t t1_us;
  int64_t dur_us;
};

static inline struct Figure8ProfileRecord *figure8_profile_records(struct Figure8ProfileHeader *header) {
  return (struct Figure8ProfileRecord *) (header + 1);
}

static inline const struct Figure8ProfileRecord *figure8_profile_records_const(
  const struct Figure8ProfileHeader *header) {
  return (const struct Figure8ProfileRecord *) (header + 1);
}

static inline struct Figure8ProfileEvent *figure8_profile_events(struct Figure8ProfileHeader *header) {
  return (struct Figure8ProfileEvent *) (figure8_profile_records(header) + header->max_records);
}

static inline const struct Figure8ProfileEvent *figure8_profile_events_const(
  const struct Figure8ProfileHeader *header) {
  return (const struct Figure8ProfileEvent *) (figure8_profile_records_const(header) + header->max_records);
}

struct FlashAttnProfileParams {
  struct FlashAttnParams attn;
  struct RpcmemBufAddr   profile;
  int32_t                max_records;
  int32_t                max_events;
} __attribute__((packed));
