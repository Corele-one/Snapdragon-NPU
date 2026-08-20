#pragma once

#include <stdbool.h>

#include "dsp/scna_params.h"
#include "op_reg.h"

typedef struct {
  int width;
  int layout;
  int variant;
  int build_variant;
  int optimized_impl;
  int64_t prepare_us;
  uint32_t coeff_bits[SCNA_D8_WIDTH];
} scna_exp2_hvx_params_t;

static inline bool scna_exp2_enabled(int mode_flags) {
  return (mode_flags & LLM_NPU_MODE_SCNA_FP16) != 0;
}

static inline int scna_exp2_layout_from_mode(int mode_flags) {
  return (mode_flags & LLM_NPU_MODE_SCNA_LANE8) != 0 ? SCNA_LAYOUT_LANE8 : SCNA_LAYOUT_SERIAL;
}

static inline int scna_exp2_width_from_mode(int mode_flags) {
  if ((mode_flags & LLM_NPU_MODE_SCNA_D8) != 0) return 8;
  if ((mode_flags & LLM_NPU_MODE_SCNA_D32) != 0) return 32;
  return 16;
}

int scna_exp2_prepare_hvx_params(scna_exp2_hvx_params_t *params, int mode_flags);
int scna_exp2_build_variant(void);
int scna_exp2_build_optimized_inline(void);
int scna_exp2_build_optimized_impl(void);

HVX_Vector hvx_scna_exp2_vhf(HVX_Vector input, const scna_exp2_hvx_params_t *params);
void hvx_scna_exp2_pair_vhf(HVX_Vector input0, HVX_Vector input1,
                            const scna_exp2_hvx_params_t *params,
                            HVX_Vector *output0, HVX_Vector *output1);
int scna_exp2_bench_run(struct ScnaExp2BenchResult *result, int width, int layout, int variant,
                        int warmup, int iters);
