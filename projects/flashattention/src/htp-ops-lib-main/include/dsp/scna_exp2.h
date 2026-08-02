#pragma once

#include <math.h>
#include <stdbool.h>

#include "dsp/hvx_internal.h"
#include "dsp/scna_params.h"
#include "dsp/utils.h"
#include "op_reg.h"

static inline bool scna_exp_enabled(int mode_flags) {
  return (mode_flags & (LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_INT8)) != 0;
}

static inline bool scna_exp2_enabled(int mode_flags) {
  return scna_exp_enabled(mode_flags);
}

static inline int scna_width_from_mode(int mode_flags) {
  if ((mode_flags & LLM_NPU_MODE_SCNA_D8) != 0) return 8;
  if ((mode_flags & LLM_NPU_MODE_SCNA_D32) != 0) return 32;
  return 16;
}

static inline int scna_exp2_width_from_mode(int mode_flags) {
  return scna_width_from_mode(mode_flags);
}

static inline int scna_precision_from_mode(int mode_flags) {
  if ((mode_flags & LLM_NPU_MODE_SCNA_INT8) != 0) return SCNA_PRECISION_INT8;
  if ((mode_flags & LLM_NPU_MODE_SCNA_FP16) != 0) return SCNA_PRECISION_FP16;
  return 0;
}

static inline int scna_exp2_profile_mode(int mode_flags) {
  return scna_precision_from_mode(mode_flags);
}

static inline int scna_function_from_mode(int mode_flags) {
  return (mode_flags & LLM_NPU_MODE_SCNA_FUNCTION_EXP) != 0 ? SCNA_FUNCTION_EXP : SCNA_FUNCTION_EXP2;
}

static inline int scna_kernel_from_mode(int mode_flags) {
  return (mode_flags & LLM_NPU_MODE_SCNA_TREE) != 0 ? SCNA_KERNEL_TREE : SCNA_KERNEL_DIRECT;
}

typedef struct {
  int width;
  int precision;
  int function;
  int kernel;
  int tree_depth;
  int tree_leaves;
  uint32_t coeff_bits[SCNA_MAX_WIDTH];
  uint32_t int8_coeff_bits[SCNA_MAX_WIDTH];
  uint16_t int8_output_scale_bits;
  uint16_t tree_threshold_bits[SCNA_MAX_TREE_LEAVES];
  uint16_t tree_slope_bits[SCNA_MAX_TREE_LEAVES];
  uint16_t tree_bias_bits[SCNA_MAX_TREE_LEAVES];
  int16_t tree_threshold_int16[SCNA_MAX_TREE_LEAVES];
  int16_t tree_slope_int16[SCNA_MAX_TREE_LEAVES];
  int16_t tree_bias_int16[SCNA_MAX_TREE_LEAVES];
} scna_hvx_params_t;

typedef scna_hvx_params_t scna_exp2_hvx_params_t;

static inline void scna_prepare_hvx_params(scna_hvx_params_t *hvx_params, int mode_flags) {
  const int function = scna_function_from_mode(mode_flags);
  const scna_params_t *params = scna_get_params(function, scna_width_from_mode(mode_flags));
  hvx_params->width = params->width;
  hvx_params->precision = scna_precision_from_mode(mode_flags);
  hvx_params->function = function;
  hvx_params->kernel = scna_kernel_from_mode(mode_flags);
  hvx_params->tree_depth = params->tree_depth;
  hvx_params->tree_leaves = params->tree_leaves;

  __fp16 output_scale_hf = (__fp16) params->output_scale;
  hvx_params->int8_output_scale_bits = fp16_to_bits(&output_scale_hf);

  for (int i = 0; i < params->width; ++i) {
    __fp16 wk_hf = params->wk_fp16[i];
    __fp16 bk_hf = params->bk_fp16[i];
    hvx_params->coeff_bits[i] = (uint32_t) fp16_to_bits(&wk_hf) | ((uint32_t) fp16_to_bits(&bk_hf) << 16);
    hvx_params->int8_coeff_bits[i] = (uint8_t) params->wk_int8[i] |
                                      ((uint32_t) (uint16_t) params->bk_int16[i] << 16);
  }
  for (int i = 0; i < SCNA_MAX_TREE_LEAVES; ++i) {
    hvx_params->tree_threshold_bits[i] = 0;
    hvx_params->tree_slope_bits[i] = 0;
    hvx_params->tree_bias_bits[i] = 0;
    hvx_params->tree_threshold_int16[i] = 0;
    hvx_params->tree_slope_int16[i] = 0;
    hvx_params->tree_bias_int16[i] = 0;
  }
  for (int i = 0; i < params->tree_leaves - 1; ++i) {
    const int physical = 2 * (i & 31) + (i >> 5);
    __fp16 threshold = params->tree_threshold_fp16[i];
    hvx_params->tree_threshold_bits[physical] = fp16_to_bits(&threshold);
    hvx_params->tree_threshold_int16[physical] = params->tree_threshold_int16[i];
  }
  for (int i = 0; i < params->tree_leaves; ++i) {
    const int physical = 2 * (i & 31) + (i >> 5);
    __fp16 slope = params->tree_slope_fp16[i];
    __fp16 bias = params->tree_bias_fp16[i];
    hvx_params->tree_slope_bits[physical] = fp16_to_bits(&slope);
    hvx_params->tree_bias_bits[physical] = fp16_to_bits(&bias);
    hvx_params->tree_slope_int16[physical] = params->tree_slope_int16[i];
    hvx_params->tree_bias_int16[physical] = params->tree_bias_int16[i];
  }
}

static inline void scna_exp2_prepare_hvx_params(scna_exp2_hvx_params_t *params, int mode_flags) {
  scna_prepare_hvx_params(params, mode_flags);
}

static inline float scna_scalar(float x, int mode_flags) {
  const scna_params_t *params = scna_get_params(scna_function_from_mode(mode_flags),
                                                scna_width_from_mode(mode_flags));
  x = fmaxf(SCNA_MIN_INPUT, fminf(SCNA_MAX_INPUT, x));
  if ((mode_flags & LLM_NPU_MODE_SCNA_INT8) != 0) {
    const float quant_x = fmaxf(-128.0f, fminf(0.0f, truncf(fmaxf(SCNA_INT8_MIN_INPUT, x) *
                                                             SCNA_INT8_INPUT_MULTIPLIER)));
    int32_t sum = 0;
    for (int i = 0; i < params->width; ++i) {
      const int32_t affine = (int32_t) quant_x * params->wk_int8[i] + params->bk_int16[i];
      sum += affine > 0 ? affine : 0;
    }
    return sum * params->output_scale;
  }
  float sum = 0.0f;
  for (int i = 0; i < params->width; ++i) {
    sum += fmaxf(x * (float) params->wk_fp16[i] + (float) params->bk_fp16[i], 0.0f);
  }
  return sum;
}

static inline float scna_exp2_scalar(float x, int mode_flags) {
  return scna_scalar(x, mode_flags);
}

HVX_Vector hvx_scna_exp_vhf(HVX_Vector x, const scna_hvx_params_t *params);
void hvx_scna_exp_pair_vhf(HVX_Vector x0, HVX_Vector x1, const scna_hvx_params_t *params,
                           HVX_Vector *out0, HVX_Vector *out1);

static inline HVX_Vector hvx_scna_exp2_vhf(HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  return hvx_scna_exp_vhf(x, params);
}

static inline void hvx_scna_exp2_pair_vhf(HVX_Vector x0, HVX_Vector x1,
                                          const scna_exp2_hvx_params_t *params,
                                          HVX_Vector *out0, HVX_Vector *out1) {
  hvx_scna_exp_pair_vhf(x0, x1, params, out0, out1);
}
