#pragma once

#include <stdint.h>

#include "dsp/hvx_convert.h"
#include "dsp/scna_exp2.h"

#ifndef SCNA_KERNEL_IMPL
#define SCNA_KERNEL_IMPL SCNA_KERNEL_IMPL_STATIC_D8_REF
#endif

#ifndef SCNA_D8_ACTIVE_WIDTH
#define SCNA_D8_ACTIVE_WIDTH 7
#endif

typedef struct {
  HVX_Vector w[SCNA_D8_ACTIVE_WIDTH];
  HVX_Vector b[SCNA_D8_ACTIVE_WIDTH];
} scna_exp2_prebroadcast_t;

static HVX_INLINE_ALWAYS HVX_Vector scna_hot_clamp_vhf(HVX_Vector x) {
  const HVX_Vector zero = Q6_V_vzero();
  __fp16 min_hf = (__fp16) SCNA_MIN_INPUT;
  const HVX_Vector minimum = Q6_Vh_vsplat_R(fp16_to_bits(&min_hf));
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(minimum, x), minimum, x);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, zero), zero, x);
}

static HVX_INLINE_ALWAYS HVX_Vector scna_hot_affine_vector(
    HVX_Vector sum, HVX_Vector x, HVX_Vector w, HVX_Vector b) {
  HVX_Vector affine_qf = Q6_Vqf16_vmpy_VhfVhf(x, w);
  affine_qf = Q6_Vqf16_vadd_Vqf16Vhf(affine_qf, b);
  HVX_Vector affine = Q6_Vhf_equals_Vqf16(affine_qf);
  affine = Q6_Vhf_vmax_VhfVhf(affine, Q6_V_vzero());
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(sum, affine));
}

static HVX_INLINE_ALWAYS HVX_Vector scna_hot_affine_scalar_w(
    HVX_Vector sum, HVX_Vector x, int weight_hf_bits, HVX_Vector b) {
  /* Rt.hf contains two FP16 scalars selected by lane parity.  Duplicate the
   * weight into both halfwords so every vector lane receives the same value. */
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

static HVX_INLINE_ALWAYS HVX_Vector scna_d7_serial_single_inline(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum = Q6_V_vzero();
  x = scna_hot_clamp_vhf(x);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = params->coeff_bits[i];
    const HVX_Vector w = Q6_Vh_vsplat_R((int) (packed & 0xffff));
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    sum = scna_hot_affine_vector(sum, x, w, b);
  }
  return sum;
}

static HVX_INLINE_ALWAYS HVX_VectorPair scna_d7_serial_pair_inline(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum0 = Q6_V_vzero(), sum1 = Q6_V_vzero();
  x0 = scna_hot_clamp_vhf(x0);
  x1 = scna_hot_clamp_vhf(x1);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = params->coeff_bits[i];
    const HVX_Vector w = Q6_Vh_vsplat_R((int) (packed & 0xffff));
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    sum0 = scna_hot_affine_vector(sum0, x0, w, b);
    sum1 = scna_hot_affine_vector(sum1, x1, w, b);
  }
  return Q6_W_vcombine_VV(sum1, sum0);
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

static HVX_INLINE_ALWAYS HVX_Vector scna_hot_clamp_min_vhf(HVX_Vector x, int min_hf_bits) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector minimum = Q6_Vh_vsplat_R(min_hf_bits);
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(minimum, x), minimum, x);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, zero), zero, x);
}

/* Experiment I benchmark helper.  The caller supplies weights pre-scaled by
 * a positive input scale and a correspondingly expanded raw-input clamp. */
static HVX_INLINE_ALWAYS HVX_VectorPair scna_d7_scalar_pair_fused_scale_inline(
    HVX_Vector raw0, HVX_Vector raw1, const scna_exp2_hvx_params_t *fused_params,
    int raw_min_hf_bits) {
  HVX_Vector sum0 = Q6_V_vzero(), sum1 = Q6_V_vzero();
  raw0 = scna_hot_clamp_min_vhf(raw0, raw_min_hf_bits);
  raw1 = scna_hot_clamp_min_vhf(raw1, raw_min_hf_bits);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = fused_params->coeff_bits[i];
    const int w = (int) (packed & 0xffff);
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    sum0 = scna_hot_affine_scalar_w(sum0, raw0, w, b);
    sum1 = scna_hot_affine_scalar_w(sum1, raw1, w, b);
  }
  return Q6_W_vcombine_VV(sum1, sum0);
}

/* Experiment D1: retain the pair-return algorithm while forcing coefficient
 * words to be observed in neuron order.  The volatile view is deliberately
 * local to this candidate: it prevents the compiler from collecting all
 * coefficient loads up front, so the static gate can test whether shorter
 * constant live ranges improve packet scheduling without global prebroadcast. */
static HVX_INLINE_ALWAYS HVX_Vector scna_d7_scalar_single_short_const_inline(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum = Q6_V_vzero();
  const volatile uint32_t *coeff_bits = params->coeff_bits;
  x = scna_hot_clamp_vhf(x);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = coeff_bits[i];
    const int w = (int) (packed & 0xffff);
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    sum = scna_hot_affine_scalar_w(sum, x, w, b);
  }
  return sum;
}

static HVX_INLINE_ALWAYS HVX_VectorPair scna_d7_scalar_pair_short_const_inline(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum0 = Q6_V_vzero(), sum1 = Q6_V_vzero();
  const volatile uint32_t *coeff_bits = params->coeff_bits;
  x0 = scna_hot_clamp_vhf(x0);
  x1 = scna_hot_clamp_vhf(x1);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = coeff_bits[i];
    const int w = (int) (packed & 0xffff);
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    sum0 = scna_hot_affine_scalar_w(sum0, x0, w, b);
    sum1 = scna_hot_affine_scalar_w(sum1, x1, w, b);
  }
  return Q6_W_vcombine_VV(sum1, sum0);
}

static HVX_INLINE_ALWAYS void scna_d7_two_neuron_group_pair_inline(
    HVX_Vector x0, HVX_Vector x1, uint32_t packed0, uint32_t packed1,
    HVX_Vector *sum0, HVX_Vector *sum1) {
  const int w0 = (int) (packed0 & 0xffff);
  const int w1 = (int) (packed1 & 0xffff);
  const HVX_Vector b0 = Q6_Vh_vsplat_R((int) (packed0 >> 16));
  const HVX_Vector b1 = Q6_Vh_vsplat_R((int) (packed1 >> 16));

  HVX_Vector a00 = Q6_Vqf16_vmpy_VhfRhf(x0, w0 | (w0 << 16));
  HVX_Vector a10 = Q6_Vqf16_vmpy_VhfRhf(x1, w0 | (w0 << 16));
  HVX_Vector a01 = Q6_Vqf16_vmpy_VhfRhf(x0, w1 | (w1 << 16));
  HVX_Vector a11 = Q6_Vqf16_vmpy_VhfRhf(x1, w1 | (w1 << 16));
  a00 = Q6_Vqf16_vadd_Vqf16Vhf(a00, b0);
  a10 = Q6_Vqf16_vadd_Vqf16Vhf(a10, b0);
  a01 = Q6_Vqf16_vadd_Vqf16Vhf(a01, b1);
  a11 = Q6_Vqf16_vadd_Vqf16Vhf(a11, b1);

  HVX_Vector h00 = Q6_Vhf_vmax_VhfVhf(Q6_Vhf_equals_Vqf16(a00), Q6_V_vzero());
  HVX_Vector h10 = Q6_Vhf_vmax_VhfVhf(Q6_Vhf_equals_Vqf16(a10), Q6_V_vzero());
  HVX_Vector h01 = Q6_Vhf_vmax_VhfVhf(Q6_Vhf_equals_Vqf16(a01), Q6_V_vzero());
  HVX_Vector h11 = Q6_Vhf_vmax_VhfVhf(Q6_Vhf_equals_Vqf16(a11), Q6_V_vzero());
  *sum0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*sum0, h00));
  *sum1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*sum1, h10));
  *sum0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*sum0, h01));
  *sum1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*sum1, h11));
}

/* Experiment E1: expose ILP only within adjacent-neuron groups.  The empty
 * asm makes each group's two accumulated rows observable before the next
 * group, bounding live ranges without emitting an instruction. */
static HVX_INLINE_ALWAYS HVX_VectorPair scna_d7_scalar_pair_two_neuron_inline(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum0 = Q6_V_vzero(), sum1 = Q6_V_vzero();
  x0 = scna_hot_clamp_vhf(x0);
  x1 = scna_hot_clamp_vhf(x1);
  scna_d7_two_neuron_group_pair_inline(x0, x1, params->coeff_bits[0], params->coeff_bits[1],
                                        &sum0, &sum1);
  __asm__ volatile("" : "+v"(sum0), "+v"(sum1));
  scna_d7_two_neuron_group_pair_inline(x0, x1, params->coeff_bits[2], params->coeff_bits[3],
                                        &sum0, &sum1);
  __asm__ volatile("" : "+v"(sum0), "+v"(sum1));
  scna_d7_two_neuron_group_pair_inline(x0, x1, params->coeff_bits[4], params->coeff_bits[5],
                                        &sum0, &sum1);
  __asm__ volatile("" : "+v"(sum0), "+v"(sum1));
  const uint32_t packed6 = params->coeff_bits[6];
  const HVX_Vector b6 = Q6_Vh_vsplat_R((int) (packed6 >> 16));
  sum0 = scna_hot_affine_scalar_w(sum0, x0, (int) (packed6 & 0xffff), b6);
  sum1 = scna_hot_affine_scalar_w(sum1, x1, (int) (packed6 & 0xffff), b6);
  return Q6_W_vcombine_VV(sum1, sum0);
}

static HVX_INLINE_ALWAYS HVX_Vector scna_d7_scalar_single_two_acc_inline(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum_even = Q6_V_vzero(), sum_odd = Q6_V_vzero();
  x = scna_hot_clamp_vhf(x);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = params->coeff_bits[i];
    const int w = (int) (packed & 0xffff);
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    if ((i & 1) == 0) {
      sum_even = scna_hot_affine_scalar_w(sum_even, x, w, b);
    } else {
      sum_odd = scna_hot_affine_scalar_w(sum_odd, x, w, b);
    }
  }
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(sum_even, sum_odd));
}

static HVX_INLINE_ALWAYS HVX_VectorPair scna_d7_scalar_pair_two_acc_inline(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum0_even = Q6_V_vzero(), sum0_odd = Q6_V_vzero();
  HVX_Vector sum1_even = Q6_V_vzero(), sum1_odd = Q6_V_vzero();
  x0 = scna_hot_clamp_vhf(x0);
  x1 = scna_hot_clamp_vhf(x1);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = params->coeff_bits[i];
    const int w = (int) (packed & 0xffff);
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    if ((i & 1) == 0) {
      sum0_even = scna_hot_affine_scalar_w(sum0_even, x0, w, b);
      sum1_even = scna_hot_affine_scalar_w(sum1_even, x1, w, b);
    } else {
      sum0_odd = scna_hot_affine_scalar_w(sum0_odd, x0, w, b);
      sum1_odd = scna_hot_affine_scalar_w(sum1_odd, x1, w, b);
    }
  }
  const HVX_Vector out0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(sum0_even, sum0_odd));
  const HVX_Vector out1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(sum1_even, sum1_odd));
  return Q6_W_vcombine_VV(out1, out0);
}

static HVX_INLINE_ALWAYS void scna_d7_scalar_quad_inline(
    HVX_Vector x0, HVX_Vector x1, HVX_Vector x2, HVX_Vector x3,
    const scna_exp2_hvx_params_t *params, HVX_Vector *out0,
    HVX_Vector *out1, HVX_Vector *out2, HVX_Vector *out3) {
  HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
  HVX_Vector s2 = Q6_V_vzero(), s3 = Q6_V_vzero();
  x0 = scna_hot_clamp_vhf(x0);
  x1 = scna_hot_clamp_vhf(x1);
  x2 = scna_hot_clamp_vhf(x2);
  x3 = scna_hot_clamp_vhf(x3);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = params->coeff_bits[i];
    const int w = (int) (packed & 0xffff);
    const HVX_Vector b = Q6_Vh_vsplat_R((int) (packed >> 16));
    s0 = scna_hot_affine_scalar_w(s0, x0, w, b);
    s1 = scna_hot_affine_scalar_w(s1, x1, w, b);
    s2 = scna_hot_affine_scalar_w(s2, x2, w, b);
    s3 = scna_hot_affine_scalar_w(s3, x3, w, b);
  }
  *out0 = s0; *out1 = s1; *out2 = s2; *out3 = s3;
}

static HVX_INLINE_ALWAYS scna_exp2_prebroadcast_t scna_prebroadcast_prepare_inline(
    const scna_exp2_hvx_params_t *params) {
  scna_exp2_prebroadcast_t prepared;
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    const uint32_t packed = params->coeff_bits[i];
    prepared.w[i] = Q6_Vh_vsplat_R((int) (packed & 0xffff));
    prepared.b[i] = Q6_Vh_vsplat_R((int) (packed >> 16));
  }
  return prepared;
}

static HVX_INLINE_ALWAYS HVX_VectorPair scna_prebroadcast_pair_inline(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_prebroadcast_t *prepared) {
  HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
  x0 = scna_hot_clamp_vhf(x0);
  x1 = scna_hot_clamp_vhf(x1);
#pragma unroll
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    s0 = scna_hot_affine_vector(s0, x0, prepared->w[i], prepared->b[i]);
    s1 = scna_hot_affine_vector(s1, x1, prepared->w[i], prepared->b[i]);
  }
  return Q6_W_vcombine_VV(s1, s0);
}
