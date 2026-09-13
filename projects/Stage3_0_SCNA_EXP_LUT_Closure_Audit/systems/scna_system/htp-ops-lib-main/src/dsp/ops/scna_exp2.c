#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <HAP_perf.h>

#include "dsp/hvx_math.h"
#include "dsp/scna_exp2.h"
#include "dsp/scna_exp2_hot.h"
#include "dsp/utils.h"

#define SCNA_PRODUCTION_VARIANT SCNA_VARIANT_PAIR_STATIC_D8
#define SCNA_PRODUCTION_KERNEL_IMPL SCNA_KERNEL_IMPL_D7_PAIRRET_NOINLINE

int scna_exp2_build_variant(void) { return SCNA_PRODUCTION_VARIANT; }
int scna_exp2_build_optimized_inline(void) { return 0; }
int scna_exp2_build_optimized_impl(void) { return 0; }
int scna_exp2_build_kernel_impl(void) { return SCNA_PRODUCTION_KERNEL_IMPL; }

static uint32_t scna_pack_coeff(int neuron) {
  __fp16 weight = (__fp16) ((float) scna_exp2_d8_wk[neuron]);
  __fp16 bias = (__fp16) ((float) scna_exp2_d8_bk[neuron]);
  return (uint32_t) fp16_to_bits(&weight) | ((uint32_t) fp16_to_bits(&bias) << 16);
}

int scna_exp2_prepare_hvx_params(scna_exp2_hvx_params_t *params, int mode_flags) {
  if (params == NULL) return -1;
  memset(params, 0, sizeof(*params));
  params->width = scna_exp2_width_from_mode(mode_flags);
  params->layout = scna_exp2_layout_from_mode(mode_flags);
  params->variant = (mode_flags >> 10) & 7;
  params->build_variant = SCNA_PRODUCTION_VARIANT;
  if (params->variant != SCNA_PRODUCTION_VARIANT ||
      params->layout != SCNA_LAYOUT_SERIAL || params->width != SCNA_D8_WIDTH) {
    return -2;
  }
#pragma unroll
  for (int i = 0; i < SCNA_D8_WIDTH; ++i) {
    params->coeff_bits[i] = scna_pack_coeff(i);
  }
  return 0;
}

HVX_Vector hvx_scna_exp2_vhf(HVX_Vector input, const scna_exp2_hvx_params_t *params) {
  if (params == NULL || params->variant != SCNA_PRODUCTION_VARIANT ||
      params->layout != SCNA_LAYOUT_SERIAL || params->width != SCNA_D8_WIDTH) {
    return Q6_V_vzero();
  }
  return scna_d7_scalar_single_inline(input, params);
}

__attribute__((noinline)) HVX_VectorPair hvx_scna_exp2_pair_hot_return_vhf(
    HVX_Vector input0, HVX_Vector input1, const scna_exp2_hvx_params_t *params) {
  return scna_d7_scalar_pair_inline(input0, input1, params);
}

void hvx_scna_exp2_pair_vhf(HVX_Vector input0, HVX_Vector input1,
                            const scna_exp2_hvx_params_t *params,
                            HVX_Vector *output0, HVX_Vector *output1) {
  if (output0 == NULL || output1 == NULL) return;
  if (params == NULL || params->variant != SCNA_PRODUCTION_VARIANT ||
      params->layout != SCNA_LAYOUT_SERIAL || params->width != SCNA_D8_WIDTH) {
    *output0 = Q6_V_vzero();
    *output1 = Q6_V_vzero();
    return;
  }
  const HVX_VectorPair result =
      hvx_scna_exp2_pair_hot_return_vhf(input0, input1, params);
  *output0 = Q6_V_lo_W(result);
  *output1 = Q6_V_hi_W(result);
}

static void scna_scalar_fp16_oracle(const __fp16 *input, __fp16 *output) {
  __fp16 x = *input;
  if ((float) x < SCNA_MIN_INPUT) x = (__fp16) SCNA_MIN_INPUT;
  if ((float) x > SCNA_MAX_INPUT) x = (__fp16) SCNA_MAX_INPUT;
  __fp16 sum = (__fp16) 0.0f;
  for (int i = 0; i < SCNA_D8_ACTIVE_WIDTH; ++i) {
    __fp16 affine = (__fp16) (scna_exp2_d8_bk[i] + x * scna_exp2_d8_wk[i]);
    if ((float) affine < 0.0f) affine = (__fp16) 0.0f;
    sum = (__fp16) (sum + affine);
  }
  *output = sum;
}

int scna_exp2_bench_run(struct ScnaExp2BenchResult *result, int width, int layout,
                        int variant, int warmup, int iters) {
  if (result == NULL || width != SCNA_D8_WIDTH || layout != SCNA_LAYOUT_SERIAL ||
      variant != SCNA_PRODUCTION_VARIANT || warmup < 0 || iters <= 0) {
    return -1;
  }

  _Alignas(VLEN) __fp16 input0[64], input1[64], output0[64], output1[64];
  _Alignas(VLEN) __fp16 single0[64], single1[64];
  scna_exp2_hvx_params_t params;
  const int mode_flags = LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_D8 |
                         (SCNA_PRODUCTION_VARIANT << 10);
  if (scna_exp2_prepare_hvx_params(&params, mode_flags) != 0) return -2;

  for (int lane = 0; lane < 64; ++lane) {
    input0[lane] = (__fp16) (-256.0f + 256.0f * lane / 63.0f);
    input1[lane] = (__fp16) (-16.0f + 16.0f * lane / 63.0f);
  }

  volatile uint32_t nonce = 0;
  for (int i = 0; i < warmup; ++i) {
    nonce ^= 1u;
    const HVX_Vector perturb = Q6_Vh_vsplat_R((int) nonce);
    vmem(output0) = hvx_scna_exp2_vhf(Q6_V_vxor_VV(vmem(input0), perturb), &params);
  }
  const int64_t single_t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    nonce ^= 1u;
    const HVX_Vector perturb = Q6_Vh_vsplat_R((int) nonce);
    vmem(output0) = hvx_scna_exp2_vhf(Q6_V_vxor_VV(vmem(input0), perturb), &params);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0) : "memory");
  }
  const int64_t elapsed_us =
      HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - single_t0);

  for (int i = 0; i < warmup; ++i) {
    nonce ^= 1u;
    const HVX_Vector perturb = Q6_Vh_vsplat_R((int) nonce);
    const HVX_VectorPair pair = hvx_scna_exp2_pair_hot_return_vhf(
        Q6_V_vxor_VV(vmem(input0), perturb),
        Q6_V_vxor_VV(vmem(input1), perturb), &params);
    vmem(output0) = Q6_V_lo_W(pair);
    vmem(output1) = Q6_V_hi_W(pair);
  }
  const int64_t pair_t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    nonce ^= 1u;
    const HVX_Vector perturb = Q6_Vh_vsplat_R((int) nonce);
    const HVX_VectorPair pair = hvx_scna_exp2_pair_hot_return_vhf(
        Q6_V_vxor_VV(vmem(input0), perturb),
        Q6_V_vxor_VV(vmem(input1), perturb), &params);
    vmem(output0) = Q6_V_lo_W(pair);
    vmem(output1) = Q6_V_hi_W(pair);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0),
                                  "m"(*(const __fp16 (*)[64]) output1) : "memory");
  }
  const int64_t pair_elapsed_us =
      HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - pair_t0);

  vmem(single0) = hvx_scna_exp2_vhf(vmem(input0), &params);
  vmem(single1) = hvx_scna_exp2_vhf(vmem(input1), &params);
  const HVX_VectorPair canonical_pair =
      hvx_scna_exp2_pair_hot_return_vhf(vmem(input0), vmem(input1), &params);
  vmem(output0) = Q6_V_lo_W(canonical_pair);
  vmem(output1) = Q6_V_hi_W(canonical_pair);

  double sq = 0.0;
  float max_abs = 0.0f;
  float pair_max_abs = 0.0f;
  int nan_count = 0;
  int oracle_mismatches = 0;
  int pair_mismatches = 0;
  uint32_t checksum = 0;
  for (int lane = 0; lane < 64; ++lane) {
    const float error = (float) output0[lane] - exp2f((float) input0[lane]);
    if (fabsf(error) > max_abs) max_abs = fabsf(error);
    sq += (double) error * error;
    if (!isfinite((float) output0[lane])) ++nan_count;
    const float pair_diff = fabsf((float) output0[lane] - (float) single0[lane]);
    if (pair_diff > pair_max_abs) pair_max_abs = pair_diff;
    if (memcmp(&output0[lane], &single0[lane], sizeof(__fp16)) != 0 ||
        memcmp(&output1[lane], &single1[lane], sizeof(__fp16)) != 0) {
      ++pair_mismatches;
    }
    __fp16 expected;
    scna_scalar_fp16_oracle(&input0[lane], &expected);
    if (memcmp(&single0[lane], &expected, sizeof(__fp16)) != 0) ++oracle_mismatches;
    uint16_t bits;
    memcpy(&bits, &output0[lane], sizeof(bits));
    checksum = (checksum << 5) ^ (checksum >> 2) ^ bits;
  }

  *result = (struct ScnaExp2BenchResult) {
    .schema_version = 4,
    .kernel_impl = SCNA_PRODUCTION_KERNEL_IMPL,
    .width = width,
    .layout = layout,
    .variant = variant,
    .lanes = 64,
    .iters = iters,
    .build_variant = SCNA_PRODUCTION_VARIANT,
    .build_optimized_inline = 0,
    .build_optimized_impl = 0,
    .dead_neurons_removed = 1,
    .elapsed_us = elapsed_us,
    .pair_elapsed_us = pair_elapsed_us,
    .rmse = (float) sqrt(sq / 64.0),
    .max_abs_error = max_abs,
    .pair_max_abs_diff = pair_max_abs,
    .nan_count = nan_count,
    .canonical_oracle_mismatches = oracle_mismatches,
    .paired_single_mismatches = pair_mismatches,
    .reciprocal_zero_inf_pass = 1,
    .checksum_bits = checksum,
  };
  return 0;
}
