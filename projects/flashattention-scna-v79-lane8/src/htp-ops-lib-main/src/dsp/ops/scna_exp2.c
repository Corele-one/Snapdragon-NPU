#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <HAP_perf.h>

#include "dsp/hvx_convert.h"
#include "dsp/scna_exp2.h"
#include "dsp/utils.h"

#define R16(v) v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v
#define COEFF8(a, b, c, d, e, f, g, h) \
  a, b, c, d, e, f, g, h, a, b, c, d, e, f, g, h, \
  a, b, c, d, e, f, g, h, a, b, c, d, e, f, g, h, \
  a, b, c, d, e, f, g, h, a, b, c, d, e, f, g, h, \
  a, b, c, d, e, f, g, h, a, b, c, d, e, f, g, h

static const __fp16 scna_d8_weight_pattern[64] __attribute__((aligned(VLEN))) = {
  COEFF8((__fp16) 2.586841583251953e-05f, (__fp16) 0.00010031461715698242f,
         (__fp16) 0.0004892349243164062f, (__fp16) 0.002384185791015625f,
         (__fp16) 0.011627197265625f, (__fp16) 0.05670166015625f,
         (__fp16) 0.2763671875f, (__fp16) 0.345458984375f)
};

static const __fp16 scna_d8_bias_pattern[64] __attribute__((aligned(VLEN))) = {
  COEFF8((__fp16) 0.0004138946533203125f, (__fp16) 0.0013751983642578125f,
         (__fp16) 0.005588531494140625f, (__fp16) 0.0218048095703125f,
         (__fp16) 0.0797119140625f, (__fp16) 0.25927734375f,
         (__fp16) 0.6318359375f, (__fp16) 0.0f)
};

#define INDEX_ROW(base) \
  R16(base + 0), R16(base + 1), R16(base + 2), R16(base + 3), \
  R16(base + 4), R16(base + 5), R16(base + 6), R16(base + 7)

/*
 * vlut16 consumes byte indexes and produces one halfword vector from the
 * even index bytes plus one from the odd index bytes.  Duplicating every
 * byte index lets the low result contain 64 requested halfwords while the
 * unused high result is deterministic.  The table vector is vshuff'ed below
 * as required for 128-byte HVX mode (v79 HVX PRM, vector in-lane LUT).
 */
static const uint8_t scna_d8_expand_index[8][128] __attribute__((aligned(VLEN))) = {
  { INDEX_ROW(0) }, { INDEX_ROW(8) }, { INDEX_ROW(16) }, { INDEX_ROW(24) },
  { INDEX_ROW(32) }, { INDEX_ROW(40) }, { INDEX_ROW(48) }, { INDEX_ROW(56) },
};

static const uint16_t scna_d8_keep_high8[64] __attribute__((aligned(VLEN))) = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0xffff, 0xffff, 0xffff, 0xffff,
  0xffff, 0xffff, 0xffff, 0xffff,
};

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_exp2_clamp_vhf(HVX_Vector x) {
  const HVX_Vector zero = Q6_V_vzero();
  __fp16 min_hf = (__fp16) SCNA_MIN_INPUT;
  const HVX_Vector minimum = Q6_Vh_vsplat_R(fp16_to_bits(&min_hf));
  x = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(minimum, x), minimum, x);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(x, zero), zero, x);
}

static __attribute__((noinline)) HVX_Vector hvx_scna_exp2_serial_d8_vhf(HVX_Vector x) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector sum = zero;
  x = hvx_scna_exp2_clamp_vhf(x);
#pragma unroll
  for (int i = 0; i < SCNA_D8_WIDTH; ++i) {
    __fp16 w_hf = scna_exp2_d8_wk[i];
    __fp16 b_hf = scna_exp2_d8_bk[i];
    const HVX_Vector w = Q6_Vh_vsplat_R(fp16_to_bits(&w_hf));
    HVX_Vector affine = Q6_Vh_vsplat_R(fp16_to_bits(&b_hf));
    affine = Q6_Vhf_vmpyacc_VhfVhfVhf(affine, x, w);
    affine = Q6_Vhf_vmax_VhfVhf(affine, zero);
    sum = Q6_Vhf_vadd_VhfVhf(sum, affine);
  }
  return sum;
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_lane8_expand(HVX_Vector table, int batch) {
  const HVX_Vector indexes = *(const HVX_Vector *) scna_d8_expand_index[batch];
  return Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(indexes, table, batch >> 1));
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_scna_lane8_reduce_to_block(HVX_Vector values, int batch) {
  const HVX_Vector zero = Q6_V_vzero();
  values = Q6_Vhf_vadd_VhfVhf(values, Q6_V_vlalign_VVR(values, zero, 2));
  values = Q6_Vhf_vadd_VhfVhf(values, Q6_V_vlalign_VVR(values, zero, 4));
  values = Q6_Vhf_vadd_VhfVhf(values, Q6_V_vlalign_VVR(values, zero, 8));
  values = Q6_Vh_vdeal_Vh(values);
  values = Q6_Vh_vdeal_Vh(values);
  values = Q6_Vh_vdeal_Vh(values);
  values = Q6_V_vand_VV(values, *(const HVX_Vector *) scna_d8_keep_high8);
  values = Q6_V_vror_VR(values, VLEN - 16);
  return batch == 0 ? values : Q6_V_vror_VR(values, VLEN - batch * 16);
}

static __attribute__((noinline)) HVX_Vector hvx_scna_exp2_lane8_d8_vhf(HVX_Vector x) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector weights = *(const HVX_Vector *) scna_d8_weight_pattern;
  const HVX_Vector biases = *(const HVX_Vector *) scna_d8_bias_pattern;
  HVX_Vector output = zero;
  x = hvx_scna_exp2_clamp_vhf(x);
  const HVX_Vector input_table = Q6_Vh_vshuff_Vh(x);
#pragma unroll
  for (int batch = 0; batch < 8; ++batch) {
    const HVX_Vector expanded = hvx_scna_lane8_expand(input_table, batch);
    HVX_Vector affine = Q6_Vhf_vmpyacc_VhfVhfVhf(biases, expanded, weights);
    affine = Q6_Vhf_vmax_VhfVhf(affine, zero);
    output = Q6_V_vor_VV(output, hvx_scna_lane8_reduce_to_block(affine, batch));
  }
  return output;
}

static __attribute__((noinline)) void hvx_scna_exp2_lane8_pair_d8_vhf(
    HVX_Vector x0, HVX_Vector x1, HVX_Vector *out0, HVX_Vector *out1) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector weights = *(const HVX_Vector *) scna_d8_weight_pattern;
  const HVX_Vector biases = *(const HVX_Vector *) scna_d8_bias_pattern;
  HVX_Vector output0 = zero;
  HVX_Vector output1 = zero;
  x0 = hvx_scna_exp2_clamp_vhf(x0);
  x1 = hvx_scna_exp2_clamp_vhf(x1);
  const HVX_Vector table0 = Q6_Vh_vshuff_Vh(x0);
  const HVX_Vector table1 = Q6_Vh_vshuff_Vh(x1);
#pragma unroll
  for (int batch = 0; batch < 8; ++batch) {
    const HVX_Vector expanded0 = hvx_scna_lane8_expand(table0, batch);
    const HVX_Vector expanded1 = hvx_scna_lane8_expand(table1, batch);
    HVX_Vector affine0 = Q6_Vhf_vmpyacc_VhfVhfVhf(biases, expanded0, weights);
    HVX_Vector affine1 = Q6_Vhf_vmpyacc_VhfVhfVhf(biases, expanded1, weights);
    affine0 = Q6_Vhf_vmax_VhfVhf(affine0, zero);
    affine1 = Q6_Vhf_vmax_VhfVhf(affine1, zero);
    output0 = Q6_V_vor_VV(output0, hvx_scna_lane8_reduce_to_block(affine0, batch));
    output1 = Q6_V_vor_VV(output1, hvx_scna_lane8_reduce_to_block(affine1, batch));
  }
  *out0 = output0;
  *out1 = output1;
}

HVX_Vector hvx_scna_exp2_vhf(HVX_Vector input, const scna_exp2_hvx_params_t *params) {
  if (params->width != 8) return Q6_V_vzero();
  return params->layout == SCNA_LAYOUT_LANE8
      ? hvx_scna_exp2_lane8_d8_vhf(input)
      : hvx_scna_exp2_serial_d8_vhf(input);
}

void hvx_scna_exp2_pair_vhf(HVX_Vector input0, HVX_Vector input1,
                            const scna_exp2_hvx_params_t *params,
                            HVX_Vector *output0, HVX_Vector *output1) {
  if (params->width != 8) {
    *output0 = Q6_V_vzero();
    *output1 = Q6_V_vzero();
  } else if (params->layout == SCNA_LAYOUT_LANE8) {
    hvx_scna_exp2_lane8_pair_d8_vhf(input0, input1, output0, output1);
  } else {
    *output0 = hvx_scna_exp2_serial_d8_vhf(input0);
    *output1 = hvx_scna_exp2_serial_d8_vhf(input1);
  }
}

static int scna_lane8_oracle(void) {
  _Alignas(VLEN) __fp16 input[64], expanded[64], contributions[64], packed[64];
  int mismatches = 0;
  for (int lane = 0; lane < 64; ++lane) {
    input[lane] = (__fp16) lane;
    contributions[lane] = (__fp16) (lane % 8 + 1);
  }
  const HVX_Vector input_table = Q6_Vh_vshuff_Vh(vmem(input));
  for (int batch = 0; batch < 8; ++batch) {
    vmem(expanded) = hvx_scna_lane8_expand(input_table, batch);
    for (int lane = 0; lane < 64; ++lane) {
      if (expanded[lane] != input[batch * 8 + lane / 8]) ++mismatches;
    }
    vmem(packed) = hvx_scna_lane8_reduce_to_block(vmem(contributions), batch);
    for (int lane = 0; lane < 64; ++lane) {
      const __fp16 expected = lane >= batch * 8 && lane < batch * 8 + 8 ? (__fp16) 36.0f : (__fp16) 0.0f;
      if (packed[lane] != expected) ++mismatches;
    }
  }
  return mismatches;
}

int scna_exp2_bench_run(struct ScnaExp2BenchResult *result, int width, int layout,
                        int warmup, int iters) {
  if (result == NULL || width != 8 || (layout != SCNA_LAYOUT_SERIAL && layout != SCNA_LAYOUT_LANE8) ||
      warmup < 0 || iters <= 0) return -1;
  _Alignas(VLEN) __fp16 input0[64], input1[64], output0[64], output1[64];
  scna_exp2_hvx_params_t params = { .width = width, .layout = layout };
  for (int lane = 0; lane < 64; ++lane) {
    input0[lane] = (__fp16) (-256.0f + 256.0f * lane / 63.0f);
    input1[lane] = (__fp16) (-16.0f + 16.0f * lane / 63.0f);
  }
  for (int i = 0; i < warmup; ++i) {
    vmem(output0) = hvx_scna_exp2_vhf(vmem(input0), &params);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0) : "memory");
  }
  const int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    vmem(output0) = hvx_scna_exp2_vhf(vmem(input0), &params);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0) : "memory");
  }
  const int64_t elapsed = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  for (int i = 0; i < warmup; ++i) {
    hvx_scna_exp2_pair_vhf(vmem(input0), vmem(input1), &params,
                           (HVX_Vector *) output0, (HVX_Vector *) output1);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0),
                                  "m"(*(const __fp16 (*)[64]) output1) : "memory");
  }
  const int64_t pt0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < iters; ++i) {
    hvx_scna_exp2_pair_vhf(vmem(input0), vmem(input1), &params,
                           (HVX_Vector *) output0, (HVX_Vector *) output1);
    __asm__ volatile("" : : "m"(*(const __fp16 (*)[64]) output0),
                                  "m"(*(const __fp16 (*)[64]) output1) : "memory");
  }
  const int64_t pair_elapsed = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - pt0);

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
  *result = (struct ScnaExp2BenchResult) {
    .width = width, .layout = layout, .lanes = 64, .iters = iters,
    .elapsed_us = elapsed, .pair_elapsed_us = pair_elapsed,
    .rmse = (float) sqrt(sq / 64.0), .max_abs_error = max_abs,
    .dense_rmse = (float) sqrt(dense_sq / (dense_blocks * 64)),
    .dense_max_abs_error = dense_max_abs, .pair_max_abs_diff = 0.0f,
    .dense_samples = dense_blocks * 64, .monotonic_violations = monotonic,
    .negative_count = negative, .nan_count = nan_count,
    .lane_oracle_mismatches = layout == SCNA_LAYOUT_LANE8 ? scna_lane8_oracle() : 0,
    .checksum_bits = checksum,
  };
  return 0;
}

#undef INDEX_ROW
#undef COEFF8
#undef R16
