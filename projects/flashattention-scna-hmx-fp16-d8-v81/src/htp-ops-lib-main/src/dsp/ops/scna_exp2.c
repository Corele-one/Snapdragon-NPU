#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <HAP_perf.h>

#include "dsp/hvx_convert.h"
#include "dsp/hmx_mgr.h"
#include "dsp/scna_exp2.h"
#include "dsp/scna_hmx.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "op_reg.h"

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_exp2_clamp_vhf(HVX_Vector x) {
  const HVX_Vector v_zero = Q6_V_vzero();
  __fp16 min_hf = (__fp16) SCNA_MIN_INPUT;
  const HVX_Vector v_min = Q6_Vh_vsplat_R(fp16_to_bits(&min_hf));
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(v_min, x), v_min, x);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, v_zero), v_zero, x);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_exp2_quantize_s8_vhf(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  (void) params;
  const HVX_Vector v_zero = Q6_V_vzero();
  __fp16 min_hf = (__fp16) SCNA_INT8_MIN_INPUT;
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

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_pack_ordered_s32_to_s16(HVX_VectorPair values) {
  const HVX_VectorPair ordered = Q6_W_vshuff_VVR(Q6_V_hi_W(values), Q6_V_lo_W(values), -4);
  return Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(ordered), Q6_V_lo_W(ordered));
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

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_vlut16_select(
    HVX_Vector indexes, HVX_Vector table, int selector) {
  return Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(indexes, table, selector));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_vlut16_up_to(
    HVX_Vector indexes, HVX_Vector table, int selectors) {
  HVX_Vector result = hvx_scna_vlut16_select(indexes, table, 0);
  if (selectors > 1) result = Q6_V_vor_VV(result, hvx_scna_vlut16_select(indexes, table, 1));
  if (selectors > 2) result = Q6_V_vor_VV(result, hvx_scna_vlut16_select(indexes, table, 2));
  if (selectors > 3) result = Q6_V_vor_VV(result, hvx_scna_vlut16_select(indexes, table, 3));
  return result;
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_tree_threshold_lookup(
    HVX_Vector node, HVX_Vector table, int level) {
  HVX_Vector result = hvx_scna_vlut16_select(node, table, 0);
  if (level >= 4) result = Q6_V_vor_VV(result, hvx_scna_vlut16_select(node, table, 1));
  if (level >= 5) {
    result = Q6_V_vor_VV(result, hvx_scna_vlut16_select(node, table, 2));
    result = Q6_V_vor_VV(result, hvx_scna_vlut16_select(node, table, 3));
  }
  return result;
}

#define SCNA_TREE_TRAVERSE(X, TABLE, DEPTH, LEAVES, CMP, LEAF)                                              \
  do {                                                                                                       \
    const HVX_Vector v_zero = Q6_V_vzero();                                                                  \
    const HVX_Vector v_one = Q6_Vh_vsplat_R(1);                                                              \
    HVX_Vector node = v_zero;                                                                                \
    _Pragma("clang loop unroll(full)")                                                                      \
    for (int level = 0; level < (DEPTH); ++level) {                                                          \
      const HVX_Vector threshold = hvx_scna_tree_threshold_lookup(node, (TABLE), level);                     \
      const HVX_Vector right = Q6_V_vmux_QVV(CMP((X), threshold), v_one, v_zero);                            \
      node = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(node, node), v_one);                                           \
      node = Q6_Vh_vadd_VhVh(node, right);                                                                   \
    }                                                                                                        \
    (LEAF) = Q6_Vh_vsub_VhVh(node, Q6_Vh_vsplat_R((LEAVES) - 1));                                           \
  } while (0)

#define SCNA_DEFINE_HVX_TREE_KERNELS(SUFFIX, DEPTH, LEAVES, SELECTORS)                                      \
  static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_tree_eval_##SUFFIX##_vhf(                                    \
      HVX_Vector x, HVX_Vector thresholds, HVX_Vector slopes, HVX_Vector biases) {                           \
    x = hvx_scna_exp2_clamp_vhf(x);                                                                           \
    HVX_Vector leaf;                                                                                          \
    SCNA_TREE_TRAVERSE(x, thresholds, DEPTH, LEAVES, Q6_Q_vcmp_gt_VhfVhf, leaf);                            \
    const HVX_Vector slope = hvx_scna_vlut16_up_to(leaf, slopes, SELECTORS);                                 \
    const HVX_Vector bias = hvx_scna_vlut16_up_to(leaf, biases, SELECTORS);                                  \
    const HVX_Vector output = Q6_Vhf_vmpyacc_VhfVhfVhf(bias, x, slope);                                     \
    return Q6_Vhf_vmax_VhfVhf(output, Q6_V_vzero());                                                         \
  }                                                                                                          \
  static __attribute__((noinline)) HVX_Vector hvx_scna_tree_##SUFFIX##_vhf(                                 \
      HVX_Vector x, const scna_hvx_params_t *params) {                                                       \
    const HVX_Vector thresholds = vmemu(params->tree_threshold_bits);                                        \
    const HVX_Vector slopes = vmemu(params->tree_slope_bits);                                                \
    const HVX_Vector biases = vmemu(params->tree_bias_bits);                                                 \
    return hvx_scna_tree_eval_##SUFFIX##_vhf(x, thresholds, slopes, biases);                                 \
  }                                                                                                          \
  static __attribute__((noinline)) void hvx_scna_tree_pair_##SUFFIX##_vhf(                                 \
      HVX_Vector x0, HVX_Vector x1, const scna_hvx_params_t *params, HVX_Vector *out0, HVX_Vector *out1) {  \
    const HVX_Vector thresholds = vmemu(params->tree_threshold_bits);                                        \
    const HVX_Vector slopes = vmemu(params->tree_slope_bits);                                                \
    const HVX_Vector biases = vmemu(params->tree_bias_bits);                                                 \
    *out0 = hvx_scna_tree_eval_##SUFFIX##_vhf(x0, thresholds, slopes, biases);                               \
    *out1 = hvx_scna_tree_eval_##SUFFIX##_vhf(x1, thresholds, slopes, biases);                               \
  }

#define SCNA_DEFINE_HVX_INT8_TREE_KERNELS(SUFFIX, DEPTH, LEAVES, SELECTORS)                                 \
  static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_tree_int8_eval_##SUFFIX##_vhf(                               \
      HVX_Vector x, HVX_Vector thresholds, HVX_Vector slopes, HVX_Vector biases,                            \
      const scna_hvx_params_t *params) {                                                                     \
    const HVX_Vector v_zero = Q6_V_vzero();                                                                  \
    const HVX_Vector qx_h = hvx_scna_exp2_quantize_s8_vhf(x, params);                                       \
    HVX_Vector leaf;                                                                                          \
    SCNA_TREE_TRAVERSE(qx_h, thresholds, DEPTH, LEAVES, Q6_Q_vcmp_gt_VhVh, leaf);                           \
    const HVX_Vector slope = hvx_scna_vlut16_up_to(leaf, slopes, SELECTORS);                                 \
    const HVX_Vector bias = hvx_scna_vlut16_up_to(leaf, biases, SELECTORS);                                  \
    HVX_VectorPair sum_w = Q6_Ww_vmpyacc_WwVhVh(Q6_W_vzero(), qx_h, slope);                                 \
    sum_w = Q6_Ww_vadd_WwWw(sum_w, Q6_Ww_vsxt_Vh(bias));                                                     \
    HVX_Vector sum_h = hvx_scna_pack_ordered_s32_to_s16(sum_w);                                             \
    sum_h = Q6_Vh_vmax_VhVh(sum_h, v_zero);                                                                  \
    const HVX_Vector output_scale = Q6_Vh_vsplat_R(params->int8_output_scale_bits);                          \
    return Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vcvt_Vh(sum_h), output_scale);                                         \
  }                                                                                                          \
  static __attribute__((noinline)) HVX_Vector hvx_scna_tree_int8_##SUFFIX##_vhf(                            \
      HVX_Vector x, const scna_hvx_params_t *params) {                                                       \
    const HVX_Vector thresholds = vmemu(params->tree_threshold_int16);                                       \
    const HVX_Vector slopes = vmemu(params->tree_slope_int16);                                               \
    const HVX_Vector biases = vmemu(params->tree_bias_int16);                                                \
    return hvx_scna_tree_int8_eval_##SUFFIX##_vhf(x, thresholds, slopes, biases, params);                    \
  }                                                                                                          \
  static __attribute__((noinline)) void hvx_scna_tree_pair_int8_##SUFFIX##_vhf(                            \
      HVX_Vector x0, HVX_Vector x1, const scna_hvx_params_t *params, HVX_Vector *out0, HVX_Vector *out1) {  \
    const HVX_Vector thresholds = vmemu(params->tree_threshold_int16);                                       \
    const HVX_Vector slopes = vmemu(params->tree_slope_int16);                                               \
    const HVX_Vector biases = vmemu(params->tree_bias_int16);                                                \
    *out0 = hvx_scna_tree_int8_eval_##SUFFIX##_vhf(x0, thresholds, slopes, biases, params);                  \
    *out1 = hvx_scna_tree_int8_eval_##SUFFIX##_vhf(x1, thresholds, slopes, biases, params);                  \
  }

SCNA_DEFINE_HVX_TREE_KERNELS(d8, 4, 16, 1)
SCNA_DEFINE_HVX_TREE_KERNELS(d16, 5, 32, 2)
SCNA_DEFINE_HVX_TREE_KERNELS(d32, 6, 64, 4)
SCNA_DEFINE_HVX_INT8_TREE_KERNELS(d8, 4, 16, 1)
SCNA_DEFINE_HVX_INT8_TREE_KERNELS(d16, 5, 32, 2)
SCNA_DEFINE_HVX_INT8_TREE_KERNELS(d32, 6, 64, 4)

#undef SCNA_DEFINE_HVX_INT8_TREE_KERNELS
#undef SCNA_DEFINE_HVX_TREE_KERNELS
#undef SCNA_TREE_TRAVERSE

__attribute__((noinline)) HVX_Vector hvx_scna_exp_vhf(
    HVX_Vector x, const scna_hvx_params_t *params) {
  if (params->kernel == SCNA_KERNEL_TREE) {
    if (params->precision == SCNA_PRECISION_INT8) {
      switch (params->width) {
        case 8: return hvx_scna_tree_int8_d8_vhf(x, params);
        case 16: return hvx_scna_tree_int8_d16_vhf(x, params);
        default: return hvx_scna_tree_int8_d32_vhf(x, params);
      }
    }
    switch (params->width) {
      case 8: return hvx_scna_tree_d8_vhf(x, params);
      case 16: return hvx_scna_tree_d16_vhf(x, params);
      default: return hvx_scna_tree_d32_vhf(x, params);
    }
  }
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

__attribute__((noinline)) void hvx_scna_exp_pair_vhf(
    HVX_Vector x0, HVX_Vector x1, const scna_hvx_params_t *params,
    HVX_Vector *out0, HVX_Vector *out1) {
  if (params->kernel == SCNA_KERNEL_TREE) {
    if (params->precision == SCNA_PRECISION_INT8) {
      switch (params->width) {
        case 8: hvx_scna_tree_pair_int8_d8_vhf(x0, x1, params, out0, out1); return;
        case 16: hvx_scna_tree_pair_int8_d16_vhf(x0, x1, params, out0, out1); return;
        default: hvx_scna_tree_pair_int8_d32_vhf(x0, x1, params, out0, out1); return;
      }
    }
    switch (params->width) {
      case 8: hvx_scna_tree_pair_d8_vhf(x0, x1, params, out0, out1); return;
      case 16: hvx_scna_tree_pair_d16_vhf(x0, x1, params, out0, out1); return;
      default: hvx_scna_tree_pair_d32_vhf(x0, x1, params, out0, out1); return;
    }
  }
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

static __attribute__((noinline)) HVX_Vector scna_exp2_bench_eval_dispatch(
    HVX_Vector input, const scna_exp2_hvx_params_t *params, scna_hmx_context_t *hmx_ctx) {
  if (hmx_ctx != NULL) return scna_hmx_fp16_d8_vhf(input, hmx_ctx);
  return scna_exp2_bench_eval(input, params);
}

static __attribute__((noinline)) void scna_exp2_bench_eval_pair_dispatch(
    HVX_Vector input0, HVX_Vector input1, const scna_exp2_hvx_params_t *params,
    scna_hmx_context_t *hmx_ctx, HVX_Vector *output0, HVX_Vector *output1) {
  if (hmx_ctx != NULL) {
    scna_hmx_fp16_d8_pair_vhf(input0, input1, hmx_ctx, output0, output1);
    return;
  }
  scna_exp2_bench_eval_pair(input0, input1, params, output0, output1);
}

static inline float scna_exp_reference(float x, int function) {
  return function == SCNA_FUNCTION_EXP ? expf(x) : exp2f(x);
}

int scna_exp2_bench_run(struct ScnaExp2BenchResult *result, int width, int mode_flags, int warmup, int iters) {
  if (result == NULL || (width != 8 && width != 16 && width != 32) || warmup < 0 || iters <= 0) return -1;

  const int engine = scna_engine_from_mode(mode_flags);
  if (engine != SCNA_ENGINE_HVX &&
      (width != 8 || (mode_flags & LLM_NPU_MODE_SCNA_INT8) != 0 ||
       scna_function_from_mode(mode_flags) != SCNA_FUNCTION_EXP2 ||
       scna_kernel_from_mode(mode_flags) != SCNA_KERNEL_DIRECT)) {
    return -1;
  }

  _Alignas(VLEN) __fp16 input[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) __fp16 output[VLEN / sizeof(__fp16)];
  _Alignas(VLEN) __fp16 alternate_output[VLEN / sizeof(__fp16)];
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
  scna_exp2_hvx_params_t alternate_params = hvx_params;
  alternate_params.kernel = hvx_params.kernel == SCNA_KERNEL_TREE ? SCNA_KERNEL_DIRECT : SCNA_KERNEL_TREE;
  scna_hmx_context_t hmx_context;
  scna_hmx_context_t *hmx_ctx = NULL;
  int layout_mismatches = 0;
  int overlap_mismatches = 0;
  int64_t overlap_serial_ticks = 0;
  int64_t overlap_ticks = 0;
  if (engine != SCNA_ENGINE_HVX) {
    uintptr_t vtcm = (uintptr_t) vtcm_manager_get_vtcm_base();
    const uintptr_t vtcm_aligned = (vtcm + 2047u) & ~(uintptr_t) 2047u;
    const size_t total = vtcm_manager_get_total_size();
    if (vtcm == 0 || vtcm_aligned < vtcm || vtcm_aligned - vtcm + SCNA_HMX_CONTEXT_BYTES > total ||
        scna_hmx_context_init(&hmx_context, (void *) vtcm_aligned, SCNA_HMX_CONTEXT_BYTES,
                              engine, mode_flags) != 0) {
      return -1;
    }
    hmx_manager_enable_execution();
    hmx_ctx = &hmx_context;
    if ((mode_flags & LLM_NPU_MODE_SCNA_HMX_VTRANSPOSE) != 0) {
      layout_mismatches = scna_hmx_vtranspose_layout_gate(hmx_ctx);
      /* Keep the overlap probe long enough to amortize command/setup noise.  On
       * SM8750P 500k iterations is slightly above the experiment's 50 ms
       * minimum-sample gate. */
      const int probe_iters = iters > 500000 ? iters : 500000;
      if (scna_hmx_overlap_probe(hmx_ctx, warmup < 5 ? warmup : 5, probe_iters,
                                 &overlap_serial_ticks, &overlap_ticks,
                                 &overlap_mismatches) != 0) {
        return -1;
      }
    }
  }

  _Alignas(VLEN) int16_t int8_pack_input[VLEN / sizeof(int16_t)];
  _Alignas(VLEN) int16_t int8_pack_output[VLEN / sizeof(int16_t)];
  const HVX_Vector int8_qx = hvx_scna_exp2_quantize_s8_vhf(vmem(pair_input1), &hvx_params);
  const HVX_VectorPair int8_qx_w = Q6_Ww_vsxt_Vh(int8_qx);
  vmem(int8_pack_input) = int8_qx;
  vmem(int8_pack_output) = hvx_scna_pack_ordered_s32_to_s16(int8_qx_w);
  int int8_s32_pack_mismatches = 0;
  int int8_s32_pack_max_abs_diff = 0;
  for (int lane = 0; lane < lanes; ++lane) {
    int diff = int8_pack_output[lane] - int8_pack_input[lane];
    if (diff < 0) diff = -diff;
    if (diff != 0) ++int8_s32_pack_mismatches;
    if (diff > int8_s32_pack_max_abs_diff) int8_s32_pack_max_abs_diff = diff;
  }
  static const int int8_probe_lanes[8] = { 0, 1, 2, 30, 31, 32, 62, 63 };

  for (int i = 0; i < warmup; ++i) {
    vmem(output) = scna_exp2_bench_eval_dispatch(vmem(input), &hvx_params, hmx_ctx);
  }
  if (hmx_ctx != NULL) scna_hmx_reset_profile(hmx_ctx);
  const int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    vmem(output) = scna_exp2_bench_eval_dispatch(vmem(input), &hvx_params, hmx_ctx);
  }
  const int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  const int64_t pack_us = hmx_ctx != NULL ? HAP_perf_qtimer_count_to_us(hmx_ctx->pack_ticks) : 0;
  const int64_t affine_relu_us = hmx_ctx != NULL ? HAP_perf_qtimer_count_to_us(hmx_ctx->affine_relu_ticks) : 0;
  const int64_t reduction_us = hmx_ctx != NULL ? HAP_perf_qtimer_count_to_us(hmx_ctx->reduction_ticks) : 0;
  const int64_t unpack_us = hmx_ctx != NULL ? HAP_perf_qtimer_count_to_us(hmx_ctx->unpack_ticks) : 0;
  const int64_t transpose_us = hmx_ctx != NULL ? HAP_perf_qtimer_count_to_us(hmx_ctx->transpose_ticks) : 0;
  const int64_t p_store_us = hmx_ctx != NULL ? HAP_perf_qtimer_count_to_us(hmx_ctx->p_store_ticks) : 0;
  const int64_t lock_us = hmx_ctx != NULL ? HAP_perf_qtimer_count_to_us(hmx_ctx->lock_ticks) : 0;
  const int64_t completion_fence_us = hmx_ctx != NULL
      ? HAP_perf_qtimer_count_to_us(hmx_ctx->completion_fence_ticks) : 0;
  const int64_t pipeline_overlap_us = overlap_ticks != 0
      ? HAP_perf_qtimer_count_to_us(overlap_ticks) : 0;
  const int64_t kernel_total_us = hmx_ctx != NULL ? HAP_perf_qtimer_count_to_us(hmx_ctx->total_ticks) : elapsed_us;

  for (int i = 0; i < warmup; ++i) {
    scna_exp2_bench_eval_pair_dispatch(vmem(input), vmem(pair_input1), &hvx_params, hmx_ctx,
                                       (HVX_Vector *) pair_output0, (HVX_Vector *) pair_output1);
  }
  const int64_t pair_t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    scna_exp2_bench_eval_pair_dispatch(vmem(input), vmem(pair_input1), &hvx_params, hmx_ctx,
                                       (HVX_Vector *) pair_output0, (HVX_Vector *) pair_output1);
  }
  const int64_t pair_elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - pair_t0);

  vmem(single_output0) = scna_exp2_bench_eval_dispatch(vmem(input), &hvx_params, hmx_ctx);
  vmem(single_output1) = scna_exp2_bench_eval_dispatch(vmem(pair_input1), &hvx_params, hmx_ctx);
  float pair_max_abs_diff = 0.0f;
  for (int lane = 0; lane < lanes; ++lane) {
    const float diff0 = fabsf((float) pair_output0[lane] - (float) single_output0[lane]);
    const float diff1 = fabsf((float) pair_output1[lane] - (float) single_output1[lane]);
    if (diff0 > pair_max_abs_diff) pair_max_abs_diff = diff0;
    if (diff1 > pair_max_abs_diff) pair_max_abs_diff = diff1;
  }

  for (int lane = 0; lane < lanes; ++lane) {
    const float expected = scna_exp_reference(input[lane], hvx_params.function);
    const float actual = output[lane];
    const float error = actual - expected;
    sq_error += error * error;
    if (fabsf(error) > max_abs) max_abs = fabsf(error);
    if (!isfinite(actual)) ++nan_count;
  }

  const int dense_blocks = 64;
  const int dense_samples = dense_blocks * lanes;
  double dense_sq_error = 0.0;
  float dense_max_abs = 0.0f;
  float previous = -INFINITY;
  int monotonic_violations = 0;
  int negative_count = 0;
  float direct_tree_max_abs_diff = 0.0f;
  double implementation_sq_error = 0.0;
  float implementation_max_abs_error = 0.0f;
  for (int block = 0; block < dense_blocks; ++block) {
    for (int lane = 0; lane < lanes; ++lane) {
      const int sample = block * lanes + lane;
      input[lane] = (__fp16) (SCNA_MIN_INPUT + (SCNA_MAX_INPUT - SCNA_MIN_INPUT) *
                                                sample / (dense_samples - 1));
    }
    vmem(output) = scna_exp2_bench_eval_dispatch(vmem(input), &hvx_params, hmx_ctx);
    if (hmx_ctx != NULL) {
      vmem(alternate_output) = scna_exp2_bench_eval(vmem(input), &hvx_params);
    } else {
      vmem(alternate_output) = scna_exp2_bench_eval(vmem(input), &alternate_params);
    }
    for (int lane = 0; lane < lanes; ++lane) {
      const float actual = output[lane];
      const float error = actual - scna_exp_reference(input[lane], hvx_params.function);
      const float direct_tree_diff = fabsf(actual - (float) alternate_output[lane]);
      if (hmx_ctx != NULL) implementation_sq_error += direct_tree_diff * direct_tree_diff;
      dense_sq_error += error * error;
      if (fabsf(error) > dense_max_abs) dense_max_abs = fabsf(error);
      if (direct_tree_diff > direct_tree_max_abs_diff) direct_tree_max_abs_diff = direct_tree_diff;
      if (hmx_ctx != NULL && direct_tree_diff > implementation_max_abs_error) {
        implementation_max_abs_error = direct_tree_diff;
      }
      if (actual < previous) ++monotonic_violations;
      if (actual < 0.0f) ++negative_count;
      if (!isfinite(actual)) ++nan_count;
      previous = actual;
    }
  }

  // Fixed-seed random migration gate.  This compares the HMX mapping against
  // the independent HVX FP16 SCNA implementation, not against true exp2.
  const int random_blocks = 64;
  const int random_samples = random_blocks * lanes;
  uint32_t random_state = 8108u;
  double random_implementation_sq_error = 0.0;
  float random_implementation_max_abs_error = 0.0f;
  if (hmx_ctx != NULL) {
    for (int block = 0; block < random_blocks; ++block) {
      for (int lane = 0; lane < lanes; ++lane) {
        random_state = random_state * 1664525u + 1013904223u;
        const float unit = (float) (random_state >> 8) * (1.0f / 16777215.0f);
        input[lane] = (__fp16) (SCNA_MIN_INPUT + (SCNA_MAX_INPUT - SCNA_MIN_INPUT) * unit);
      }
      vmem(output) = scna_exp2_bench_eval_dispatch(vmem(input), &hvx_params, hmx_ctx);
      vmem(alternate_output) = scna_exp2_bench_eval(vmem(input), &hvx_params);
      for (int lane = 0; lane < lanes; ++lane) {
        const float diff = fabsf((float) output[lane] - (float) alternate_output[lane]);
        random_implementation_sq_error += diff * diff;
        if (diff > random_implementation_max_abs_error) random_implementation_max_abs_error = diff;
      }
    }
  }

  // Tail isolation gate: invalid suffix lanes contain alternating extrema;
  // valid prefix lanes must still match HVX for lengths around the 32-lane HMX batch boundary.
  static const int tail_lengths[] = { 1, 7, 31, 32, 33, 63 };
  float tail_implementation_max_abs_error = 0.0f;
  if (hmx_ctx != NULL) {
    for (int tail_index = 0; tail_index < (int) (sizeof(tail_lengths) / sizeof(tail_lengths[0])); ++tail_index) {
      const int valid = tail_lengths[tail_index];
      for (int lane = 0; lane < lanes; ++lane) {
        input[lane] = lane < valid ? (__fp16) (-16.0f * lane / (valid > 1 ? valid - 1 : 1))
                                   : (__fp16) ((lane & 1) ? SCNA_MIN_INPUT : SCNA_MAX_INPUT);
      }
      vmem(output) = scna_exp2_bench_eval_dispatch(vmem(input), &hvx_params, hmx_ctx);
      vmem(alternate_output) = scna_exp2_bench_eval(vmem(input), &hvx_params);
      for (int lane = 0; lane < valid; ++lane) {
        const float diff = fabsf((float) output[lane] - (float) alternate_output[lane]);
        if (diff > tail_implementation_max_abs_error) tail_implementation_max_abs_error = diff;
      }
    }
  }
  *result = (struct ScnaExp2BenchResult) {
    .width = width,
    .precision = (selected_mode & LLM_NPU_MODE_SCNA_INT8) ? SCNA_PRECISION_INT8 : SCNA_PRECISION_FP16,
    .function = hvx_params.function,
    .kernel = hvx_params.kernel,
    .lanes = lanes,
    .iters = iters,
    .elapsed_us = elapsed_us,
    .pair_elapsed_us = pair_elapsed_us,
    .rmse = (float) sqrt(sq_error / lanes),
    .max_abs_error = max_abs,
    .dense_rmse = (float) sqrt(dense_sq_error / dense_samples),
    .dense_max_abs_error = dense_max_abs,
    .pair_max_abs_diff = pair_max_abs_diff,
    .direct_tree_max_abs_diff = direct_tree_max_abs_diff,
    .output_at_min = single_output0[0],
    .output_near_minus12 = single_output0[60],
    .output_near_minus4 = single_output0[62],
    .output_at_zero = single_output0[63],
    .dense_samples = dense_samples,
    .monotonic_violations = monotonic_violations,
    .negative_count = negative_count,
    .nan_count = nan_count,
    .int8_s32_pack_mismatches = int8_s32_pack_mismatches,
    .int8_s32_pack_max_abs_diff = int8_s32_pack_max_abs_diff,
    .convert_zero = convert_zero,
    .convert_one = convert_one,
    .convert_quarter = convert_quarter,
    .convert_neg_half = convert_neg_half,
    .engine = engine,
    .profile_version = SCNA_HMX_PROFILE_VERSION,
    .pack_us = pack_us,
    .hmx_affine_relu_us = affine_relu_us,
    .reduction_us = reduction_us,
    .unpack_us = unpack_us,
    .kernel_total_us = kernel_total_us,
    .implementation_rmse = hmx_ctx != NULL ? (float) sqrt(implementation_sq_error / dense_samples) : 0.0f,
    .implementation_max_abs_error = implementation_max_abs_error,
    .random_samples = hmx_ctx != NULL ? random_samples : 0,
    .random_implementation_rmse = hmx_ctx != NULL
        ? (float) sqrt(random_implementation_sq_error / random_samples) : 0.0f,
    .random_implementation_max_abs_error = random_implementation_max_abs_error,
    .tail_implementation_max_abs_error = tail_implementation_max_abs_error,
    .transpose_us = transpose_us,
    .p_store_us = p_store_us,
    .lock_us = lock_us,
    .completion_fence_us = completion_fence_us,
    .pipeline_overlap_us = pipeline_overlap_us,
    .hmx_command_count = hmx_ctx != NULL ? hmx_ctx->hmx_command_count : 0,
    .physical_macs = hmx_ctx != NULL ? hmx_ctx->physical_macs : 0,
    .useful_macs = hmx_ctx != NULL ? hmx_ctx->useful_macs : 0,
    .layout_mismatches = layout_mismatches,
    .overlap_mismatches = overlap_mismatches,
    .overlap_speedup = overlap_ticks > 0 ? (float) overlap_serial_ticks / (float) overlap_ticks : 0.0f,
  };
  for (int i = 0; i < 8; ++i) {
    const int lane = int8_probe_lanes[i];
    result->int8_s32_pack_probe_input[i] = int8_pack_input[lane];
    result->int8_s32_pack_probe_output[i] = int8_pack_output[lane];
  }
  if (hmx_ctx != NULL) hmx_manager_disable_execution();
  return 0;
}
