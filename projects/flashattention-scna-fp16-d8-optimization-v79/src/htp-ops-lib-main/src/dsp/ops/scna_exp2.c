#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <HAP_perf.h>

#include "dsp/hvx_convert.h"
#include "dsp/hvx_math.h"
#include "dsp/scna_exp2.h"
#include "dsp/utils.h"

#ifndef SCNA_BUILD_VARIANT
#define SCNA_BUILD_VARIANT SCNA_VARIANT_STAGE1_DYNAMIC_ROW
#endif
#ifndef SCNA_OPTIMIZED_INLINE
#define SCNA_OPTIMIZED_INLINE 0
#endif

#if SCNA_BUILD_VARIANT < 0 || SCNA_BUILD_VARIANT >= 7
#error "SCNA_BUILD_VARIANT must be in [0,6]"
#endif

int scna_exp2_build_variant(void) { return SCNA_BUILD_VARIANT; }
int scna_exp2_build_optimized_inline(void) { return SCNA_OPTIMIZED_INLINE ? 1 : 0; }

/* Stage one deliberately performs a table lookup and fp16->float->fp16
 * conversion in the row hot path.  Later builds call the same helper once at
 * Attention invocation setup and carry only packed halfword bits thereafter. */
static __attribute__((noinline, unused)) uint32_t scna_lookup_convert_coeff(int neuron) {
  float wf = (float) scna_exp2_d8_wk[neuron];
  float bf = (float) scna_exp2_d8_bk[neuron];
  __fp16 wh = (__fp16) wf;
  __fp16 bh = (__fp16) bf;
  return (uint32_t) fp16_to_bits(&wh) | ((uint32_t) fp16_to_bits(&bh) << 16);
}

int scna_exp2_prepare_hvx_params(scna_exp2_hvx_params_t *params, int mode_flags) {
  if (params == NULL) return -1;
  params->width = scna_exp2_width_from_mode(mode_flags);
  params->layout = scna_exp2_layout_from_mode(mode_flags);
  params->variant = (mode_flags >> 10) & 7;
  params->build_variant = SCNA_BUILD_VARIANT;
  params->prepare_us = 0;
  if (params->variant != SCNA_BUILD_VARIANT || params->layout != SCNA_LAYOUT_SERIAL || params->width != 8) {
    return -2;
  }
  if (SCNA_BUILD_VARIANT != SCNA_VARIANT_STAGE1_DYNAMIC_ROW) {
    for (int i = 0; i < params->width; ++i) params->coeff_bits[i] = scna_lookup_convert_coeff(i);
  }
  return 0;
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_exp2_clamp_vhf(HVX_Vector x) {
  const HVX_Vector zero = Q6_V_vzero();
  __fp16 min_hf = (__fp16) SCNA_MIN_INPUT;
  const HVX_Vector minimum = Q6_Vh_vsplat_R(fp16_to_bits(&min_hf));
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(minimum, x), minimum, x);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, zero), zero, x);
}

static HVX_INLINE_ALWAYS void unpack_coeff(uint32_t packed, HVX_Vector *w, HVX_Vector *b) {
  *w = Q6_Vh_vsplat_R((int) (packed & 0xffff));
  *b = Q6_Vh_vsplat_R((int) (packed >> 16));
}

static HVX_INLINE_ALWAYS HVX_Vector qf16_affine_relu_sum(
    HVX_Vector sum, HVX_Vector x, HVX_Vector w, HVX_Vector b) {
  HVX_Vector affine_qf = Q6_Vqf16_vmpy_VhfVhf(x, w);
  affine_qf = Q6_Vqf16_vadd_Vqf16Vhf(affine_qf, b);
  HVX_Vector affine = Q6_Vhf_equals_Vqf16(affine_qf);
  affine = Q6_Vhf_vmax_VhfVhf(affine, Q6_V_vzero());
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(sum, affine));
}

static __attribute__((noinline, unused)) HVX_Vector stage1_dynamic_row(HVX_Vector x, int width) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector sum = zero;
  x = hvx_scna_exp2_clamp_vhf(x);
  for (int i = 0; i < width; ++i) {
    HVX_Vector w, b;
    unpack_coeff(scna_lookup_convert_coeff(i), &w, &b);
    sum = qf16_affine_relu_sum(sum, x, w, b);
  }
  return sum;
}

static __attribute__((noinline, unused)) HVX_Vector prepared_dynamic_row(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector sum = zero;
  x = hvx_scna_exp2_clamp_vhf(x);
  for (int i = 0; i < params->width; ++i) {
    HVX_Vector w, b;
    unpack_coeff(params->coeff_bits[i], &w, &b);
    sum = qf16_affine_relu_sum(sum, x, w, b);
  }
  return sum;
}

static __attribute__((noinline, unused)) void pair_shared_dynamic_qf16(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,
    HVX_Vector *out0, HVX_Vector *out1) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector sum0 = zero, sum1 = zero;
  x0 = hvx_scna_exp2_clamp_vhf(x0);
  x1 = hvx_scna_exp2_clamp_vhf(x1);
  for (int i = 0; i < params->width; ++i) {
    HVX_Vector w, b;
    unpack_coeff(params->coeff_bits[i], &w, &b); /* one w/b broadcast for two rows */
    sum0 = qf16_affine_relu_sum(sum0, x0, w, b);
    sum1 = qf16_affine_relu_sum(sum1, x1, w, b);
  }
  *out0 = sum0;
  *out1 = sum1;
}

static __attribute__((noinline, unused)) void pair_static_d8_qf16(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,
    HVX_Vector *out0, HVX_Vector *out1) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector sum0 = zero, sum1 = zero;
  x0 = hvx_scna_exp2_clamp_vhf(x0);
  x1 = hvx_scna_exp2_clamp_vhf(x1);
#pragma unroll
  for (int i = 0; i < SCNA_D8_WIDTH; ++i) {
    HVX_Vector w, b;
    unpack_coeff(params->coeff_bits[i], &w, &b);
    sum0 = qf16_affine_relu_sum(sum0, x0, w, b);
    sum1 = qf16_affine_relu_sum(sum1, x1, w, b);
  }
  *out0 = sum0;
  *out1 = sum1;
}

static HVX_INLINE_ALWAYS HVX_Vector fma_affine_relu_sum(
    HVX_Vector sum, HVX_Vector x, HVX_Vector w, HVX_Vector b) {
  HVX_Vector affine = Q6_Vhf_vmpyacc_VhfVhfVhf(b, x, w);
  affine = Q6_Vhf_vmax_VhfVhf(affine, Q6_V_vzero());
  return Q6_Vhf_vadd_VhfVhf(sum, affine);
}

static HVX_INLINE_ALWAYS void pair_static_d8_fma_body(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,
    HVX_Vector *out0, HVX_Vector *out1) {
  HVX_Vector sum0 = Q6_V_vzero(), sum1 = Q6_V_vzero();
  x0 = hvx_scna_exp2_clamp_vhf(x0);
  x1 = hvx_scna_exp2_clamp_vhf(x1);
#pragma unroll
  for (int i = 0; i < SCNA_D8_WIDTH; ++i) {
    HVX_Vector w, b;
    unpack_coeff(params->coeff_bits[i], &w, &b);
    sum0 = fma_affine_relu_sum(sum0, x0, w, b);
    sum1 = fma_affine_relu_sum(sum1, x1, w, b);
  }
  *out0 = sum0;
  *out1 = sum1;
}

static __attribute__((noinline, unused)) void pair_static_d8_fma_noinline(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,
    HVX_Vector *out0, HVX_Vector *out1) {
  pair_static_d8_fma_body(x0, x1, params, out0, out1);
}

static HVX_INLINE_ALWAYS void pair_static_d8_fma_inline(
    HVX_Vector x0, HVX_Vector x1, const scna_exp2_hvx_params_t *params,
    HVX_Vector *out0, HVX_Vector *out1) {
  pair_static_d8_fma_body(x0, x1, params, out0, out1);
}

static __attribute__((noinline, unused)) HVX_Vector static_d8_fma_row(
    HVX_Vector x, const scna_exp2_hvx_params_t *params) {
  HVX_Vector sum = Q6_V_vzero();
  x = hvx_scna_exp2_clamp_vhf(x);
#pragma unroll
  for (int i = 0; i < SCNA_D8_WIDTH; ++i) {
    HVX_Vector w, b;
    unpack_coeff(params->coeff_bits[i], &w, &b);
    sum = fma_affine_relu_sum(sum, x, w, b);
  }
  return sum;
}

HVX_Vector hvx_scna_exp2_vhf(HVX_Vector input, const scna_exp2_hvx_params_t *params) {
  if (params == NULL || params->variant != SCNA_BUILD_VARIANT || params->width != 8) return Q6_V_vzero();
#if SCNA_BUILD_VARIANT == 0
  return stage1_dynamic_row(input, params->width);
#elif SCNA_BUILD_VARIANT <= 3
  return prepared_dynamic_row(input, params);
#else
  return static_d8_fma_row(input, params);
#endif
}

void hvx_scna_exp2_pair_vhf(HVX_Vector input0, HVX_Vector input1,
                            const scna_exp2_hvx_params_t *params,
                            HVX_Vector *output0, HVX_Vector *output1) {
  if (params == NULL || params->variant != SCNA_BUILD_VARIANT || params->width != 8) {
    *output0 = Q6_V_vzero();
    *output1 = Q6_V_vzero();
    return;
  }
#if SCNA_BUILD_VARIANT == 0
  *output0 = stage1_dynamic_row(input0, params->width);
  *output1 = stage1_dynamic_row(input1, params->width);
#elif SCNA_BUILD_VARIANT == 1
  *output0 = prepared_dynamic_row(input0, params);
  *output1 = prepared_dynamic_row(input1, params);
#elif SCNA_BUILD_VARIANT == 2
  pair_shared_dynamic_qf16(input0, input1, params, output0, output1);
#elif SCNA_BUILD_VARIANT == 3
  pair_static_d8_qf16(input0, input1, params, output0, output1);
#elif SCNA_BUILD_VARIANT == 5
  pair_static_d8_fma_inline(input0, input1, params, output0, output1);
#elif SCNA_BUILD_VARIANT == 6 && SCNA_OPTIMIZED_INLINE
  pair_static_d8_fma_inline(input0, input1, params, output0, output1);
#else
  pair_static_d8_fma_noinline(input0, input1, params, output0, output1);
  /* Keep the noinline experiment as a real call/return boundary.  Without a
   * post-call observable barrier LLVM may legally turn this into a tail jump,
   * which would no longer measure the preregistered call-policy difference. */
  __asm__ volatile("" : : "m"(*output0), "m"(*output1) : "memory");
#endif
}

static void scna_scalar_fp16_oracle(const __fp16 *input, __fp16 *output) {
  __fp16 x = *input;
  if ((float) x < SCNA_MIN_INPUT) x = (__fp16) SCNA_MIN_INPUT;
  if ((float) x > 0.0f) x = (__fp16) 0.0f;
  __fp16 sum = (__fp16) 0.0f;
  for (int i = 0; i < SCNA_D8_WIDTH; ++i) {
    __fp16 affine = (__fp16) (scna_exp2_d8_bk[i] + x * scna_exp2_d8_wk[i]);
    if ((float) affine < 0.0f) affine = (__fp16) 0.0f;
    sum = (__fp16) (sum + affine);
  }
  *output = sum;
}

int scna_exp2_bench_run(struct ScnaExp2BenchResult *result, int width, int layout, int variant,
                        int warmup, int iters) {
  if (result == NULL || width != 8 || layout != SCNA_LAYOUT_SERIAL ||
      variant != SCNA_BUILD_VARIANT || warmup < 0 || iters <= 0) return -1;
  _Alignas(VLEN) __fp16 input0[64], input1[64], output0[64], output1[64];
  _Alignas(VLEN) __fp16 single0[64], single1[64];
  scna_exp2_hvx_params_t params;
  const int mode_flags = LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_D8 | (variant << 10);
  if (scna_exp2_prepare_hvx_params(&params, mode_flags) != 0) return -2;
  volatile uint32_t prepare_guard = 0;
  const int64_t prepare_t0 = HAP_perf_get_qtimer_count();
  for (int sample = 0; sample < iters; ++sample) {
#if SCNA_BUILD_VARIANT == 0
    for (int neuron = 0; neuron < width; ++neuron) prepare_guard ^= scna_lookup_convert_coeff(neuron);
#else
    scna_exp2_hvx_params_t prepared_sample;
    (void) scna_exp2_prepare_hvx_params(&prepared_sample, mode_flags);
    prepare_guard ^= prepared_sample.coeff_bits[sample & 7];
#endif
  }
  const int64_t prepare_elapsed = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - prepare_t0);
  for (int lane = 0; lane < 64; ++lane) {
    input0[lane] = (__fp16) (-256.0f + 256.0f * lane / 63.0f);
    input1[lane] = (__fp16) (-16.0f + 16.0f * lane / 63.0f);
  }
  volatile uint16_t timing_nonce = 0;
  for (int i = 0; i < warmup; ++i) {
    timing_nonce ^= 1u;
    const HVX_Vector timed_input = Q6_V_vxor_VV(
        vmem(input0), Q6_Vh_vsplat_R((int) timing_nonce));
    vmem(output0) = hvx_scna_exp2_vhf(timed_input, &params);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0) : "memory");
  }
  const int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    timing_nonce ^= 1u;
    const HVX_Vector timed_input = Q6_V_vxor_VV(
        vmem(input0), Q6_Vh_vsplat_R((int) timing_nonce));
    vmem(output0) = hvx_scna_exp2_vhf(timed_input, &params);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0) : "memory");
  }
  const int64_t elapsed = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  vmem(single0) = hvx_scna_exp2_vhf(vmem(input0), &params);
  vmem(single1) = hvx_scna_exp2_vhf(vmem(input1), &params);
  for (int i = 0; i < warmup; ++i) {
    timing_nonce ^= 1u;
    const HVX_Vector perturb = Q6_Vh_vsplat_R((int) timing_nonce);
    hvx_scna_exp2_pair_vhf(Q6_V_vxor_VV(vmem(input0), perturb),
                           Q6_V_vxor_VV(vmem(input1), perturb), &params,
                           (HVX_Vector *) output0, (HVX_Vector *) output1);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0),
                                  "m"(*(const __fp16 (*)[64]) output1) : "memory");
  }
  const int64_t pt0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    timing_nonce ^= 1u;
    const HVX_Vector perturb = Q6_Vh_vsplat_R((int) timing_nonce);
    hvx_scna_exp2_pair_vhf(Q6_V_vxor_VV(vmem(input0), perturb),
                           Q6_V_vxor_VV(vmem(input1), perturb), &params,
                           (HVX_Vector *) output0, (HVX_Vector *) output1);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0),
                                  "m"(*(const __fp16 (*)[64]) output1) : "memory");
  }
  const int64_t pair_elapsed = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - pt0);

  /* Timed inputs deliberately vary to defeat loop-invariant code motion.
   * Restore the canonical outputs before correctness and checksum gates. */
  hvx_scna_exp2_pair_vhf(vmem(input0), vmem(input1), &params,
                         (HVX_Vector *) output0, (HVX_Vector *) output1);

  int64_t expand_elapsed = 0, affine_relu_elapsed = 0, reduce_elapsed = 0, pack_elapsed = 0;

  int paired_single_mismatches = 0;
  float pair_max_abs_diff = 0.0f;
  for (int lane = 0; lane < 64; ++lane) {
    if (memcmp(&output0[lane], &single0[lane], sizeof(__fp16)) != 0) ++paired_single_mismatches;
    if (memcmp(&output1[lane], &single1[lane], sizeof(__fp16)) != 0) ++paired_single_mismatches;
    const float diff0 = fabsf((float) output0[lane] - (float) single0[lane]);
    const float diff1 = fabsf((float) output1[lane] - (float) single1[lane]);
    if (diff0 > pair_max_abs_diff) pair_max_abs_diff = diff0;
    if (diff1 > pair_max_abs_diff) pair_max_abs_diff = diff1;
  }
  int canonical_oracle_mismatches = 0;
  for (int lane = 0; lane < 64; ++lane) {
    const __fp16 canonical_input = (__fp16) (-256.0f + 256.0f * lane / 63.0f);
    __fp16 expected;
    scna_scalar_fp16_oracle(&canonical_input, &expected);
    if (memcmp(&single0[lane], &expected, sizeof(__fp16)) != 0) ++canonical_oracle_mismatches;
  }

  double sq = 0.0;
  float max_abs = 0.0f;
  int nan_count = 0;
  uint32_t checksum = 0;
  for (int lane = 0; lane < 64; ++lane) {
    const float error = (float) output0[lane] - exp2f((float) input0[lane]);
    sq += (double) error * error;
    if (fabsf(error) > max_abs) max_abs = fabsf(error);
    if (!isfinite((float) output0[lane])) ++nan_count;
    uint16_t bits;
    memcpy(&bits, &output0[lane], sizeof(bits));
    checksum = (checksum << 5) ^ (checksum >> 2) ^ bits;
  }

  const int dense_blocks = 64;
  double dense_sq = 0.0;
  float dense_max_abs = 0.0f;
  float previous = -INFINITY;
  int monotonic = 0, negative = 0;
  for (int block = 0; block < dense_blocks; ++block) {
    for (int lane = 0; lane < 64; ++lane) {
      const int sample = block * 64 + lane;
      input0[lane] = (__fp16) (-256.0f + 256.0f * sample / (dense_blocks * 64 - 1));
    }
    vmem(output0) = hvx_scna_exp2_vhf(vmem(input0), &params);
    for (int lane = 0; lane < 64; ++lane) {
      const float actual = (float) output0[lane];
      const float error = actual - exp2f((float) input0[lane]);
      dense_sq += (double) error * error;
      if (fabsf(error) > dense_max_abs) dense_max_abs = fabsf(error);
      if (actual < previous) ++monotonic;
      if (actual < 0.0f) ++negative;
      previous = actual;
    }
  }

  const int random_blocks = 16;
  uint32_t random_state = UINT32_C(0x5c4e4138);
  double random_sq = 0.0;
  float random_max_abs = 0.0f;
  int random_nonfinite = 0;
  for (int block = 0; block < random_blocks; ++block) {
    for (int lane = 0; lane < 64; ++lane) {
      random_state = random_state * UINT32_C(1664525) + UINT32_C(1013904223);
      input0[lane] = (__fp16) (-256.0f * (float) (random_state & 0xffff) / 65535.0f);
    }
    vmem(output0) = hvx_scna_exp2_vhf(vmem(input0), &params);
    for (int lane = 0; lane < 64; ++lane) {
      const float actual = (float) output0[lane];
      const float error = actual - exp2f((float) input0[lane]);
      random_sq += (double) error * error;
      if (fabsf(error) > random_max_abs) random_max_abs = fabsf(error);
      if (!isfinite(actual)) ++random_nonfinite;
    }
  }

  double native_sq = 0.0;
  float native_max_abs = 0.0f;
  for (int lane = 0; lane < 64; ++lane) {
    input0[lane] = (__fp16) (-16.0f + 16.0f * lane / 63.0f);
  }
  vmem(output0) = hvx_my_exp2_xqf_vhf(vmem(input0));
  for (int lane = 0; lane < 64; ++lane) {
    const float error = (float) output0[lane] - exp2f((float) input0[lane]);
    native_sq += (double) error * error;
    if (fabsf(error) > native_max_abs) native_max_abs = fabsf(error);
  }
  double native_qf16_sq = 0.0;
  float native_qf16_max_abs = 0.0f;
  const HVX_Vector native_qf16_input = Q6_Vqf16_vsub_VhfVhf(vmem(input0), Q6_V_vzero());
  vmem(output0) = hvx_my_exp2_vhf_vqf16(native_qf16_input);
  for (int lane = 0; lane < 64; ++lane) {
    const float error = (float) output0[lane] - exp2f((float) input0[lane]);
    native_qf16_sq += (double) error * error;
    if (fabsf(error) > native_qf16_max_abs) native_qf16_max_abs = fabsf(error);
  }

  for (int lane = 0; lane < 64; ++lane) input0[lane] = (__fp16) 1.0f;
  HVX_VectorPair ones_pair = hvx_my_vhf_to_wqf32(vmem(input0));
  HVX_Vector ones_sum = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(ones_pair), Q6_V_hi_W(ones_pair));
  for (int shift = 64; shift >= 4; shift >>= 1) {
    ones_sum = Q6_Vqf32_vadd_Vqf32Vqf32(ones_sum, Q6_V_vror_VR(ones_sum, shift));
  }
  _Alignas(VLEN) float rowsum_probe_lanes[32];
  vmem(rowsum_probe_lanes) = Q6_Vsf_equals_Vqf32(ones_sum);
  const float rowsum_ones_probe = rowsum_probe_lanes[31];

  /* Exercise the reciprocal used by the common Attention output-scale path.
   * The largest FP16 subnormal has a finite FP16 reciprocal; zero is the only
   * input in this probe for which infinity is the required result. */
  for (int lane = 0; lane < 64; ++lane) input0[lane] = (__fp16) 1.0f;
  {
    const uint16_t largest_subnormal_bits = 0x03ff;
    memcpy(&input0[0], &largest_subnormal_bits, sizeof(largest_subnormal_bits));
  }
  input0[1] = (__fp16) 0.0f;
  for (int exponent = -10; exponent <= 10; ++exponent) {
    input0[exponent + 12] = (__fp16) ldexpf(1.0f, exponent);
  }
  input0[33] = (__fp16) 1.5f;
  input0[34] = (__fp16) 7.75f;
  input0[35] = (__fp16) 64.0f;
  input0[36] = (__fp16) 511.5f;
  input0[37] = (__fp16) 4096.0f;
  vmem(output0) = hvx_my_inv_vhf(vmem(input0));
  float reciprocal_max_relative_error = 0.0f;
  int reciprocal_nonfinite_count = 0;
  for (int lane = 0; lane < 64; ++lane) {
    const float x = (float) input0[lane];
    const float actual = (float) output0[lane];
    if (x == 0.0f) continue;
    if (!isfinite(actual)) {
      ++reciprocal_nonfinite_count;
      continue;
    }
    const float expected = 1.0f / x;
    const float relative_error = fabsf(actual - expected) / expected;
    if (relative_error > reciprocal_max_relative_error) {
      reciprocal_max_relative_error = relative_error;
    }
  }
  const int reciprocal_zero_inf_pass = isinf((float) output0[1]) ? 1 : 0;

  *result = (struct ScnaExp2BenchResult) {
    .width = width, .layout = layout, .variant = variant, .lanes = 64, .iters = iters,
    .build_variant = SCNA_BUILD_VARIANT, .build_optimized_inline = SCNA_OPTIMIZED_INLINE ? 1 : 0,
    .elapsed_us = elapsed, .pair_elapsed_us = pair_elapsed,
    .prepare_elapsed_us = prepare_elapsed,
    .expand_elapsed_us = expand_elapsed, .affine_relu_elapsed_us = affine_relu_elapsed,
    .reduce_elapsed_us = reduce_elapsed, .pack_elapsed_us = pack_elapsed,
    .rmse = (float) sqrt(sq / 64.0), .max_abs_error = max_abs,
    .dense_rmse = (float) sqrt(dense_sq / (dense_blocks * 64)),
    .dense_max_abs_error = dense_max_abs,
    .random_rmse = (float) sqrt(random_sq / (random_blocks * 64)),
    .random_max_abs_error = random_max_abs, .pair_max_abs_diff = pair_max_abs_diff,
    .dense_samples = dense_blocks * 64, .random_samples = random_blocks * 64,
    .random_nonfinite_count = random_nonfinite, .monotonic_violations = monotonic,
    .negative_count = negative, .nan_count = nan_count,
    .lane_oracle_mismatches = 0,
    .canonical_oracle_mismatches = canonical_oracle_mismatches,
    .paired_single_mismatches = paired_single_mismatches,
    .native_exp2_rmse = (float) sqrt(native_sq / 64.0),
    .native_exp2_max_abs_error = native_max_abs,
    .native_qf16_exp2_rmse = (float) sqrt(native_qf16_sq / 64.0),
    .native_qf16_exp2_max_abs_error = native_qf16_max_abs,
    .rowsum_ones_probe = rowsum_ones_probe,
    .reciprocal_max_relative_error = reciprocal_max_relative_error,
    .reciprocal_nonfinite_count = reciprocal_nonfinite_count,
    .reciprocal_zero_inf_pass = reciprocal_zero_inf_pass,
    .checksum_bits = checksum ^ prepare_guard,
  };
  return 0;
}
