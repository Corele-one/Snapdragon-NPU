#pragma once

#include <math.h>
#include <stdbool.h>

#include "dsp/hvx_internal.h"
#include "dsp/scna_params.h"
#include "dsp/utils.h"
#include "op_reg.h"

static inline bool scna_exp2_enabled(int mode_flags) {
  return (mode_flags & (LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_INT8)) != 0;
}

static inline int scna_exp2_width_from_mode(int mode_flags) {
  if ((mode_flags & LLM_NPU_MODE_SCNA_D8) != 0) return 8;
  if ((mode_flags & LLM_NPU_MODE_SCNA_D32) != 0) return 32;
  return 16;
}

static inline int scna_exp2_profile_mode(int mode_flags) {
  if ((mode_flags & LLM_NPU_MODE_SCNA_INT8) != 0) return SCNA_PRECISION_INT8;
  if ((mode_flags & LLM_NPU_MODE_SCNA_FP16) != 0) return SCNA_PRECISION_FP16;
  return 0;
}

typedef struct {
  int width;
  int precision;
  uint32_t coeff_bits[SCNA_EXP2_MAX_WIDTH];
  uint32_t int8_coeff_bits[SCNA_EXP2_MAX_WIDTH];
  uint16_t int8_output_scale_bits;
} scna_exp2_hvx_params_t;

#define SCNA_EXP2_INT8_INPUT_MULTIPLIER 8.0f
#define SCNA_EXP2_INT8_MIN_INPUT (-16.0f)

// Convert deployment parameters once per attention invocation. Keeping packed
// FP16 bit patterns out of the block loop avoids repeated scalar FP conversion.
static inline void scna_exp2_prepare_hvx_params(scna_exp2_hvx_params_t *hvx_params, int mode_flags) {
  const scna_exp2_params_t *params = scna_exp2_get_params(scna_exp2_width_from_mode(mode_flags));
  hvx_params->width = params->width;
  hvx_params->precision = scna_exp2_profile_mode(mode_flags);

  // qx=trunc(clamp(x,-16,0)*8), qw=round(w/weight_scale), and
  // qb=round(b/output_scale).
  // The S8 product and S16 bias then share output_scale=weight_scale/8.
  // Bias needs S16: forcing it into S8 erases the small positive slopes that
  // model the long exp2 tail.
  __fp16 output_scale_hf = (__fp16) params->output_scale;
  hvx_params->int8_output_scale_bits = fp16_to_bits(&output_scale_hf);

  for (int i = 0; i < params->width; ++i) {
    __fp16 wk_hf = params->wk_fp16[i];
    __fp16 bk_hf = params->bk_fp16[i];
    hvx_params->coeff_bits[i] = (uint32_t) fp16_to_bits(&wk_hf) | ((uint32_t) fp16_to_bits(&bk_hf) << 16);

    const int qw = params->wk_int8[i];
    const int qb = params->bk_int16[i];
    hvx_params->int8_coeff_bits[i] = (uint8_t) (int8_t) qw | ((uint32_t) (uint16_t) (int16_t) qb << 16);
  }
}

static inline float scna_exp2_scalar(float x, int mode_flags) {
  const scna_exp2_params_t *params = scna_exp2_get_params(scna_exp2_width_from_mode(mode_flags));
  x = fmaxf(SCNA_EXP2_MIN_INPUT, fminf(SCNA_EXP2_MAX_INPUT, x));
  if ((mode_flags & LLM_NPU_MODE_SCNA_INT8) != 0) {
    const float output_scale = params->output_scale;
    const float quant_x = fmaxf(-128.0f, fminf(0.0f, roundf(x * SCNA_EXP2_INT8_INPUT_MULTIPLIER)));
    int sum = 0;
    for (int i = 0; i < params->width; ++i) {
      const int qw = params->wk_int8[i];
      const int qb = params->bk_int16[i];
      const int affine = (int) quant_x * qw + qb;
      sum += affine > 0 ? affine : 0;
    }
    return sum * output_scale;
  }
  float sum = 0.0f;
  for (int i = 0; i < params->width; ++i) {
    sum += fmaxf(x * (float) params->wk_fp16[i] + (float) params->bk_fp16[i], 0.0f);
  }
  return sum;
}

HVX_Vector hvx_scna_exp2_vhf(HVX_Vector x, const scna_exp2_hvx_params_t *params);

void hvx_scna_exp2_pair_vhf(HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,
                            HVX_Vector *out0, HVX_Vector *out1);
