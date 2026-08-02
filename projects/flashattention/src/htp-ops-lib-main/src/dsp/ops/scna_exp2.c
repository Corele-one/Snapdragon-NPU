#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <HAP_perf.h>

#include "dsp/hvx_convert.h"
#include "dsp/scna_exp2.h"
#include "dsp/utils.h"
#include "op_reg.h"

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_exp2_clamp_vhf(HVX_Vector x) {
  const HVX_Vector v_zero = Q6_V_vzero();
  __fp16 min_hf = (__fp16) SCNA_EXP2_MIN_INPUT;
  const HVX_Vector v_min = Q6_Vh_vsplat_R(fp16_to_bits(&min_hf));
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(v_min, x), v_min, x);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, v_zero), v_zero, x);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_exp2_quantize_s8_vhf(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  (void) params;
  const HVX_Vector v_zero = Q6_V_vzero();
  __fp16 min_hf = (__fp16) SCNA_EXP2_INT8_MIN_INPUT;
  const HVX_Vector v_min = Q6_Vh_vsplat_R(fp16_to_bits(&min_hf));
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(v_min, x), v_min, x);
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, v_zero), v_zero, x);

  // For clamped x in [-16, 0], qx=trunc(x*8) can be derived directly
  // from IEEE FP16 fields. Adding three to the unbiased exponent is the
  // exact power-of-two scale; the result is already bounded to S8.
  const HVX_Vector abs_bits = Q6_V_vand_VV(x, Q6_Vh_vsplat_R(0x7fffu));
  HVX_Vector exponent = Q6_Vuh_vlsr_VuhR(abs_bits, 10);
  exponent = Q6_V_vand_VV(exponent, Q6_Vh_vsplat_R(0x1fu));
  exponent = Q6_Vh_vsub_VhVh(exponent, Q6_Vh_vsplat_R(12));
  const HVX_VectorPred q_less_than_one = Q6_Q_vcmp_gt_VhVh(v_zero, exponent);
  exponent = Q6_Vh_vmax_VhVh(exponent, v_zero);

  const HVX_Vector shift = Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(10), exponent);
  HVX_Vector mantissa = Q6_V_vand_VV(abs_bits, Q6_Vh_vsplat_R(0x03ffu));
  mantissa = Q6_Vh_vadd_VhVh(mantissa, Q6_Vh_vsplat_R(0x0400u));
  HVX_Vector magnitude = Q6_Vh_vlsr_VhVh(mantissa, shift);
  magnitude = Q6_V_vmux_QVV(q_less_than_one, v_zero, magnitude);
  return Q6_Vh_vsub_VhVh(v_zero, magnitude);
}

static HVX_INLINE_ALWAYS HVX_VectorPair hvx_scna_exp2_i8_product_ordered(
    HVX_Vector activations, HVX_Vector weights) {
  const HVX_VectorPair raw = Q6_Wh_vmpy_VbVb(activations, weights);
  return Q6_W_vshuff_VVR(Q6_V_hi_W(raw), Q6_V_lo_W(raw), -2);
}

#define SCNA_DEFINE_HVX_KERNELS(SUFFIX, WIDTH)                                                               \
  static __attribute__((noinline)) HVX_Vector hvx_scna_exp2_##SUFFIX##_vhf(                                  \
      HVX_Vector x, const scna_exp2_hvx_params_t *params) {                                                   \
    const HVX_Vector v_zero = Q6_V_vzero();                                                                  \
    HVX_Vector sum = v_zero;                                                                                  \
    x = hvx_scna_exp2_clamp_vhf(x);                                                                           \
    _Pragma("clang loop unroll(full)")                                                                        \
    for (int i = 0; i < WIDTH; ++i) {                                                                         \
      const uint32_t coeff = params->coeff_bits[i];                                                           \
      const HVX_Vector w = Q6_Vh_vsplat_R(coeff & 0xffffu);                                                   \
      HVX_Vector affine = Q6_Vh_vsplat_R(coeff >> 16);                                                        \
      affine = Q6_Vhf_vmpyacc_VhfVhfVhf(affine, x, w);                                                       \
      affine = Q6_Vhf_vmax_VhfVhf(affine, v_zero);                                                           \
      sum = Q6_Vhf_vadd_VhfVhf(sum, affine);                                                                  \
    }                                                                                                         \
    return sum;                                                                                                \
  }                                                                                                           \
  static __attribute__((noinline)) void hvx_scna_exp2_pair_##SUFFIX##_vhf(                                   \
      HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,                                    \
      HVX_Vector *out0, HVX_Vector *out1) {                                                                   \
    const HVX_Vector v_zero = Q6_V_vzero();                                                                  \
    HVX_Vector sum0 = v_zero;                                                                                 \
    HVX_Vector sum1 = v_zero;                                                                                 \
    x0 = hvx_scna_exp2_clamp_vhf(x0);                                                                         \
    x1 = hvx_scna_exp2_clamp_vhf(x1);                                                                         \
    _Pragma("clang loop unroll(full)")                                                                        \
    for (int i = 0; i < WIDTH; ++i) {                                                                         \
      const uint32_t coeff = params->coeff_bits[i];                                                           \
      const HVX_Vector w = Q6_Vh_vsplat_R(coeff & 0xffffu);                                                   \
      const HVX_Vector b = Q6_Vh_vsplat_R(coeff >> 16);                                                       \
      HVX_Vector affine0 = Q6_Vhf_vmpyacc_VhfVhfVhf(b, x0, w);                                               \
      HVX_Vector affine1 = Q6_Vhf_vmpyacc_VhfVhfVhf(b, x1, w);                                               \
      affine0 = Q6_Vhf_vmax_VhfVhf(affine0, v_zero);                                                         \
      affine1 = Q6_Vhf_vmax_VhfVhf(affine1, v_zero);                                                         \
      sum0 = Q6_Vhf_vadd_VhfVhf(sum0, affine0);                                                               \
      sum1 = Q6_Vhf_vadd_VhfVhf(sum1, affine1);                                                               \
    }                                                                                                         \
    *out0 = sum0;                                                                                              \
    *out1 = sum1;                                                                                              \
  }

SCNA_DEFINE_HVX_KERNELS(d8, 8)
SCNA_DEFINE_HVX_KERNELS(d16, 16)
SCNA_DEFINE_HVX_KERNELS(d32, 32)

#undef SCNA_DEFINE_HVX_KERNELS

#define SCNA_DEFINE_HVX_INT8_KERNELS(SUFFIX, WIDTH)                                                          \
  static __attribute__((noinline)) HVX_Vector hvx_scna_exp2_int8_##SUFFIX##_vhf(                             \
      HVX_Vector x, const scna_exp2_hvx_params_t *params) {                                                   \
    const HVX_Vector v_zero = Q6_V_vzero();                                                                   \
    const HVX_Vector qx_h = hvx_scna_exp2_quantize_s8_vhf(x, params);                                        \
    const HVX_Vector qx_b = Q6_Vb_vpack_VhVh_sat(qx_h, qx_h);                                                \
    HVX_Vector sum_h = v_zero;                                                                                 \
    _Pragma("clang loop unroll(full)")                                                                        \
    for (int i = 0; i < WIDTH; ++i) {                                                                          \
      const uint32_t coeff = params->int8_coeff_bits[i];                                                       \
      const HVX_Vector qw = Q6_Vb_vsplat_R(coeff & 0xffu);                                                     \
      const HVX_Vector qb = Q6_Vh_vsplat_R(coeff >> 16);                                                       \
      const HVX_VectorPair product = hvx_scna_exp2_i8_product_ordered(qx_b, qw);                              \
      HVX_Vector affine = Q6_Vh_vadd_VhVh(Q6_V_lo_W(product), qb);                                            \
      affine = Q6_Vh_vmax_VhVh(affine, v_zero);                                                               \
      sum_h = Q6_Vh_vadd_VhVh(sum_h, affine);                                                                 \
    }                                                                                                          \
    const HVX_Vector output_scale = Q6_Vh_vsplat_R(params->int8_output_scale_bits);                            \
    return Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vcvt_Vh(sum_h), output_scale);                                           \
  }                                                                                                            \
  static __attribute__((noinline)) void hvx_scna_exp2_pair_int8_##SUFFIX##_vhf(                               \
      HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,                                     \
      HVX_Vector *out0, HVX_Vector *out1) {                                                                    \
    const HVX_Vector v_zero = Q6_V_vzero();                                                                   \
    const HVX_Vector qx0_h = hvx_scna_exp2_quantize_s8_vhf(x0, params);                                       \
    const HVX_Vector qx1_h = hvx_scna_exp2_quantize_s8_vhf(x1, params);                                       \
    const HVX_Vector qx_b = Q6_Vb_vpack_VhVh_sat(qx1_h, qx0_h);                                               \
    HVX_Vector sum0_h = v_zero;                                                                                \
    HVX_Vector sum1_h = v_zero;                                                                                \
    _Pragma("clang loop unroll(full)")                                                                        \
    for (int i = 0; i < WIDTH; ++i) {                                                                          \
      const uint32_t coeff = params->int8_coeff_bits[i];                                                       \
      const HVX_Vector qw = Q6_Vb_vsplat_R(coeff & 0xffu);                                                     \
      const HVX_Vector qb = Q6_Vh_vsplat_R(coeff >> 16);                                                       \
      const HVX_VectorPair product = hvx_scna_exp2_i8_product_ordered(qx_b, qw);                              \
      HVX_Vector affine0 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(product), qb);                                           \
      HVX_Vector affine1 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(product), qb);                                           \
      affine0 = Q6_Vh_vmax_VhVh(affine0, v_zero);                                                             \
      affine1 = Q6_Vh_vmax_VhVh(affine1, v_zero);                                                             \
      sum0_h = Q6_Vh_vadd_VhVh(sum0_h, affine0);                                                              \
      sum1_h = Q6_Vh_vadd_VhVh(sum1_h, affine1);                                                              \
    }                                                                                                          \
    const HVX_Vector output_scale = Q6_Vh_vsplat_R(params->int8_output_scale_bits);                            \
    *out0 = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vcvt_Vh(sum0_h), output_scale);                                         \
    *out1 = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vcvt_Vh(sum1_h), output_scale);                                         \
  }

SCNA_DEFINE_HVX_INT8_KERNELS(d8, 8)
SCNA_DEFINE_HVX_INT8_KERNELS(d16, 16)
SCNA_DEFINE_HVX_INT8_KERNELS(d32, 32)

#undef SCNA_DEFINE_HVX_INT8_KERNELS

__attribute__((noinline)) HVX_Vector hvx_scna_exp2_vhf(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  if (params->precision == SCNA_PRECISION_INT8) {
    switch (params->width) {
      case 8: return hvx_scna_exp2_int8_d8_vhf(x, params);
      case 16: return hvx_scna_exp2_int8_d16_vhf(x, params);
      default: return hvx_scna_exp2_int8_d32_vhf(x, params);
    }
  }
  switch (params->width) {
    case 8: return hvx_scna_exp2_d8_vhf(x, params);
    case 16: return hvx_scna_exp2_d16_vhf(x, params);
    default: return hvx_scna_exp2_d32_vhf(x, params);
  }
}

__attribute__((noinline)) void hvx_scna_exp2_pair_vhf(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,
    HVX_Vector *out0, HVX_Vector *out1) {
  if (params->precision == SCNA_PRECISION_INT8) {
    switch (params->width) {
      case 8: hvx_scna_exp2_pair_int8_d8_vhf(x0, x1, params, out0, out1); return;
      case 16: hvx_scna_exp2_pair_int8_d16_vhf(x0, x1, params, out0, out1); return;
      default: hvx_scna_exp2_pair_int8_d32_vhf(x0, x1, params, out0, out1); return;
    }
  }
  switch (params->width) {
    case 8: hvx_scna_exp2_pair_d8_vhf(x0, x1, params, out0, out1); return;
    case 16: hvx_scna_exp2_pair_d16_vhf(x0, x1, params, out0, out1); return;
    default: hvx_scna_exp2_pair_d32_vhf(x0, x1, params, out0, out1); return;
  }
}

static __attribute__((noinline)) HVX_Vector scna_exp2_bench_eval(
    HVX_Vector input, const scna_exp2_hvx_params_t *params) {
  __asm__ volatile("" ::: "memory");
  return hvx_scna_exp2_vhf(input, params);
}

static __attribute__((noinline)) void scna_exp2_bench_eval_pair(
    HVX_Vector input0, HVX_Vector input1, const scna_exp2_hvx_params_t *params,
    HVX_Vector *output0, HVX_Vector *output1) {
  __asm__ volatile("" ::: "memory");
  hvx_scna_exp2_pair_vhf(input0, input1, params, output0, output1);
}

int scna_exp2_bench_run(struct ScnaExp2BenchResult *result, int width, int mode_flags, int warmup, int iters) {
  if (result == NULL || (width != 8 && width != 16 && width != 32) || warmup < 0 || iters <= 0) return -1;

  _Alignas(VLEN) __fp16 input[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) __fp16 output[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) __fp16 pair_input1[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) __fp16 pair_output0[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) __fp16 pair_output1[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) __fp16 single_output0[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) __fp16 single_output1[VLEN / sizeof(__fp16)];
  const int lanes = VLEN / (int) sizeof(__fp16);
  double sq_error = 0.0;
  float max_abs = 0.0f;
  int nan_count = 0;

  _Alignas(VLEN) __fp16 convert_input[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) float convert_output0[VLEN / sizeof(float)];
  _Alignas(VLEN) float convert_output1[VLEN / sizeof(float)];
  for (int lane = 0; lane < lanes / 2; ++lane) {
    convert_input[2 * lane] = (__fp16) 0.0f;
    convert_input[2 * lane + 1] = (__fp16) 1.0f;
  }
  HVX_VectorPair convert_pair = hvx_my_vhf_to_wsf(vmem(convert_input));
  vmem(convert_output0) = Q6_V_lo_W(convert_pair);
  vmem(convert_output1) = Q6_V_hi_W(convert_pair);
  const float convert_zero = convert_output0[0];
  const float convert_one = convert_output1[0];

  for (int lane = 0; lane < lanes / 2; ++lane) {
    convert_input[2 * lane] = (__fp16) 0.25f;
    convert_input[2 * lane + 1] = (__fp16) -0.5f;
  }
  convert_pair = hvx_my_vhf_to_wsf(vmem(convert_input));
  vmem(convert_output0) = Q6_V_lo_W(convert_pair);
  vmem(convert_output1) = Q6_V_hi_W(convert_pair);
  const float convert_quarter = convert_output0[0];
  const float convert_neg_half = convert_output1[0];

  for (int lane = 0; lane < lanes; ++lane) {
    input[lane] = (__fp16) (-256.0f + 256.0f * lane / (lanes - 1));
    pair_input1[lane] = (__fp16) (-16.0f + 16.0f * lane / (lanes - 1));
  }
  const int selected_mode = (mode_flags & ~LLM_NPU_MODE_SCNA_D32 & ~LLM_NPU_MODE_SCNA_D8) |
                            (width == 8 ? LLM_NPU_MODE_SCNA_D8 : width == 32 ? LLM_NPU_MODE_SCNA_D32 : 0);
  scna_exp2_hvx_params_t hvx_params;
  scna_exp2_prepare_hvx_params(&hvx_params, selected_mode);

  for (int i = 0; i < warmup; ++i) {
    vmem(output) = scna_exp2_bench_eval(vmem(input), &hvx_params);
  }
  const int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    vmem(output) = scna_exp2_bench_eval(vmem(input), &hvx_params);
  }
  const int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);

  for (int i = 0; i < warmup; ++i) {
    scna_exp2_bench_eval_pair(vmem(input), vmem(pair_input1), &hvx_params,
                              (HVX_Vector *) pair_output0, (HVX_Vector *) pair_output1);
  }
  const int64_t pair_t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    scna_exp2_bench_eval_pair(vmem(input), vmem(pair_input1), &hvx_params,
                              (HVX_Vector *) pair_output0, (HVX_Vector *) pair_output1);
  }
  const int64_t pair_elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - pair_t0);

  vmem(single_output0) = scna_exp2_bench_eval(vmem(input), &hvx_params);
  vmem(single_output1) = scna_exp2_bench_eval(vmem(pair_input1), &hvx_params);
  float pair_max_abs_diff = 0.0f;
  for (int lane = 0; lane < lanes; ++lane) {
    const float diff0 = fabsf((float) pair_output0[lane] - (float) single_output0[lane]);
    const float diff1 = fabsf((float) pair_output1[lane] - (float) single_output1[lane]);
    if (diff0 > pair_max_abs_diff) pair_max_abs_diff = diff0;
    if (diff1 > pair_max_abs_diff) pair_max_abs_diff = diff1;
  }

  for (int lane = 0; lane < lanes; ++lane) {
    const float expected = exp2f(input[lane]);
    const float actual = output[lane];
    const float error = actual - expected;
    sq_error += error * error;
    if (fabsf(error) > max_abs) max_abs = fabsf(error);
    if (!isfinite(actual)) ++nan_count;
  }

  const int dense_blocks = 16;
  const int dense_samples = dense_blocks * lanes;
  double dense_sq_error = 0.0;
  float dense_max_abs = 0.0f;
  float previous = -INFINITY;
  int monotonic_violations = 0;
  int negative_count = 0;
  for (int block = 0; block < dense_blocks; ++block) {
    for (int lane = 0; lane < lanes; ++lane) {
      const int sample = block * lanes + lane;
      input[lane] = (__fp16) (-16.0f + 16.0f * sample / (dense_samples - 1));
    }
    vmem(output) = scna_exp2_bench_eval(vmem(input), &hvx_params);
    for (int lane = 0; lane < lanes; ++lane) {
      const float actual = output[lane];
      const float error = actual - exp2f(input[lane]);
      dense_sq_error += error * error;
      if (fabsf(error) > dense_max_abs) dense_max_abs = fabsf(error);
      if (actual < previous) ++monotonic_violations;
      if (actual < 0.0f) ++negative_count;
      if (!isfinite(actual)) ++nan_count;
      previous = actual;
    }
  }
  *result = (struct ScnaExp2BenchResult) {
    .width = width,
    .precision = (selected_mode & LLM_NPU_MODE_SCNA_INT8) ? SCNA_PRECISION_INT8 : SCNA_PRECISION_FP16,
    .lanes = lanes,
    .iters = iters,
    .elapsed_us = elapsed_us,
    .pair_elapsed_us = pair_elapsed_us,
    .rmse = (float) sqrt(sq_error / lanes),
    .max_abs_error = max_abs,
    .dense_rmse = (float) sqrt(dense_sq_error / dense_samples),
    .dense_max_abs_error = dense_max_abs,
    .pair_max_abs_diff = pair_max_abs_diff,
    .output_at_min = single_output0[0],
    .output_near_minus12 = single_output0[60],
    .output_near_minus4 = single_output0[62],
    .output_at_zero = single_output0[63],
    .dense_samples = dense_samples,
    .monotonic_violations = monotonic_violations,
    .negative_count = negative_count,
    .nan_count = nan_count,
    .convert_zero = convert_zero,
    .convert_one = convert_one,
    .convert_quarter = convert_quarter,
    .convert_neg_half = convert_neg_half,
  };
  return 0;
}
