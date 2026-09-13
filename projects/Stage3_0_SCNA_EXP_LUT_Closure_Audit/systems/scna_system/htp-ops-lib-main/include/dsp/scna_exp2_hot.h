#pragma once

#include <stdint.h>

#include "dsp/hvx_convert.h"
#include "dsp/scna_exp2.h"

static HVX_INLINE_ALWAYS HVX_Vector scna_hot_clamp_vhf(HVX_Vector x) {
  const HVX_Vector zero = Q6_V_vzero();
  __fp16 min_hf = (__fp16) SCNA_MIN_INPUT;
  const HVX_Vector minimum = Q6_Vh_vsplat_R(fp16_to_bits(&min_hf));
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(minimum, x), minimum, x);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, zero), zero, x);
}

static HVX_INLINE_ALWAYS HVX_Vector scna_hot_affine_scalar_w(
    HVX_Vector sum, HVX_Vector x, int weight_hf_bits, HVX_Vector b) {
  /* Rt.hf selects by lane parity, so duplicate the FP16 weight. */
  const int duplicated_weight = weight_hf_bits | (weight_hf_bits << 16);
  HVX_Vector affine_qf = Q6_Vqf16_vmpy_VhfRhf(x, duplicated_weight);
  affine_qf = Q6_Vqf16_vadd_Vqf16Vhf(affine_qf, b);
  HVX_Vector affine = Q6_Vhf_equals_Vqf16(affine_qf);
  affine = Q6_Vhf_vmax_VhfVhf(affine, Q6_V_vzero());
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(sum, affine));
}

static HVX_INLINE_ALWAYS HVX_Vector scna_d7_scalar_single_inline(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum = Q6_V_vzero();
  x = scna_hot_clamp_vhf(x);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = params->coeff_bits[i];
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    sum = scna_hot_affine_scalar_w(sum, x, (int) (packed & 0xffff), b);
  }
  return sum;
}

static HVX_INLINE_ALWAYS HVX_VectorPair scna_d7_scalar_pair_inline(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum0 = Q6_V_vzero(), sum1 = Q6_V_vzero();
  x0 = scna_hot_clamp_vhf(x0);
  x1 = scna_hot_clamp_vhf(x1);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = params->coeff_bits[i];
    const int w = (int) (packed & 0xffff);
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    sum0 = scna_hot_affine_scalar_w(sum0, x0, w, b);
    sum1 = scna_hot_affine_scalar_w(sum1, x1, w, b);
  }
  return Q6_W_vcombine_VV(sum1, sum0);
}
