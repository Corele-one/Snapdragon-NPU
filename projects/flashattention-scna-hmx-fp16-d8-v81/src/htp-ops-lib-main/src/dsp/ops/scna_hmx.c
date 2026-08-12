#include "dsp/scna_hmx.h"

#include <string.h>

#include <HAP_perf.h>
#include "dsp/hmx_mgr.h"
#include "dsp/hvx_internal.h"
#include "dsp/scna_params.h"
#include "dsp/utils.h"

#define SCNA_HMX_TILE_BYTES 2048
#define SCNA_HMX_PARTIAL_WEIGHT_BYTES 512
#define SCNA_HMX_HALF_WEIGHT_BYTES 1024
#define SCNA_HMX_FULL_WEIGHT_BYTES 2048
#define SCNA_HMX_SPATIAL_MASK 0x1c

static inline int64_t scna_hmx_now(void) {
  return HAP_perf_get_qtimer_count();
}

static inline uint32_t scna_hmx_activation_rt(int stop_channel) {
  return (((SCNA_HMX_SPATIAL_MASK >> 1) << 7) | (stop_channel << 2) |
          ((SCNA_HMX_SPATIAL_MASK & 1) << 1));
}

static inline uint16_t scna_hmx_fp16_bits(const __fp16 *value) {
  uint16_t bits;
  memcpy(&bits, value, sizeof(bits));
  return bits;
}

static void scna_hmx_init_bias(uint32_t *bias, const __fp16 *input_bias,
                               int relu, int repeated_groups) {
  for (int output = 0; output < 32; ++output) {
    bias[output] = 0x00003c00u;
    int neuron = repeated_groups ? (output & 7) : output;
    const uint16_t input_bits = input_bias != NULL &&
        (repeated_groups || output < 8) ? scna_hmx_fp16_bits(&input_bias[neuron]) : 0;
    bias[32 + output] = ((uint32_t) input_bits << 16) | (relu ? (2u << 8) : 0u);
  }
}

static void scna_hmx_init_weights(scna_hmx_context_t *ctx) {
  const int batch4 = (ctx->variant_flags & LLM_NPU_MODE_SCNA_HMX_BATCH4) != 0;

  memset(ctx->weights_affine, 0, SCNA_HMX_PARTIAL_WEIGHT_BYTES);
  for (int input_pair = 0; input_pair < 4; ++input_pair) {
    for (int output = 0; output < 32; ++output) {
      for (int in_pair = 0; in_pair < 2; ++in_pair) {
        const int input_channel = input_pair * 2 + in_pair;
        const int index = input_pair * 64 + output * 2 + in_pair;
        if ((!batch4 && output < 8 && input_channel == 0) ||
            (batch4 && input_channel == output / 8)) {
          ctx->weights_affine[index] = scna_exp2_d8_wk[output & 7];
        }
      }
    }
  }

  memset(ctx->weights_reduce, 0, SCNA_HMX_FULL_WEIGHT_BYTES);
  const int input_pairs = batch4 ? 16 : 4;
  for (int input_pair = 0; input_pair < input_pairs; ++input_pair) {
    for (int output = 0; output < 32; ++output) {
      for (int in_pair = 0; in_pair < 2; ++in_pair) {
        const int input_channel = input_pair * 2 + in_pair;
        const int index = input_pair * 64 + output * 2 + in_pair;
        if ((!batch4 && output == 0 && input_channel < 8) ||
            (batch4 && output < 4 && input_channel >= output * 8 &&
             input_channel < output * 8 + 8)) {
          ctx->weights_reduce[index] = (__fp16) 1.0f;
        }
      }
    }
  }
}

int scna_hmx_context_init(scna_hmx_context_t *ctx, void *aligned_vtcm, size_t bytes,
                          int engine, int variant_flags) {
  if (ctx == NULL || aligned_vtcm == NULL || bytes < SCNA_HMX_CONTEXT_BYTES ||
      (((uintptr_t) aligned_vtcm) & (SCNA_HMX_TILE_BYTES - 1)) != 0 ||
      (engine != SCNA_ENGINE_HMX_HYBRID && engine != SCNA_ENGINE_HMX_TWO_PASS)) {
    return -1;
  }

  memset(ctx, 0, sizeof(*ctx));
  memset(aligned_vtcm, 0, SCNA_HMX_CONTEXT_BYTES);
  ctx->base = (uint8_t *) aligned_vtcm;
  ctx->activation = (__fp16 *) (ctx->base + 0);
  ctx->weights_affine = (__fp16 *) (ctx->base + 2048);
  ctx->bias_relu = (uint32_t *) (ctx->base + 4096);
  ctx->intermediate = (__fp16 *) (ctx->base + 6144);
  ctx->weights_reduce = (__fp16 *) (ctx->base + 8192);
  ctx->bias_identity = (uint32_t *) (ctx->base + 10240);
  ctx->reduced = (__fp16 *) (ctx->base + 12288);
  ctx->channel_vectors = (__fp16 *) (ctx->base + 14336);
  ctx->scatter_offsets = (uint16_t *) (ctx->base + 15360);
  ctx->activation_alt = (__fp16 *) (ctx->base + 16384);
  ctx->intermediate_alt = (__fp16 *) (ctx->base + 18432);
  ctx->reduced_alt = (__fp16 *) (ctx->base + 20480);
  ctx->engine = engine;
  ctx->variant_flags = variant_flags;

  scna_hmx_init_weights(ctx);
  scna_hmx_init_bias(ctx->bias_relu, scna_exp2_d8_bk, 1,
                     (variant_flags & LLM_NPU_MODE_SCNA_HMX_BATCH4) != 0);
  scna_hmx_init_bias(ctx->bias_identity, NULL, 0, 0);
  for (int spatial = 0; spatial < 64; ++spatial) {
    ctx->scatter_offsets[spatial] = spatial < 32
        ? (uint16_t) ((spatial >> 1) * 128 + (spatial & 1) * 2) : 0;
  }
  return 0;
}

void scna_hmx_reset_profile(scna_hmx_context_t *ctx) {
  if (ctx == NULL) return;
  ctx->pack_ticks = 0;
  ctx->affine_relu_ticks = 0;
  ctx->reduction_ticks = 0;
  ctx->unpack_ticks = 0;
  ctx->transpose_ticks = 0;
  ctx->p_store_ticks = 0;
  ctx->lock_ticks = 0;
  ctx->completion_fence_ticks = 0;
  ctx->pipeline_overlap_ticks = 0;
  ctx->hmx_command_count = 0;
  ctx->physical_macs = 0;
  ctx->useful_macs = 0;
  ctx->total_ticks = 0;
}

static __attribute__((noinline)) void scna_hmx_fp16_d8_affine_relu_kernel(
    const __fp16 *activation, const __fp16 *weights, const uint32_t *bias, __fp16 *output) {
  const uintptr_t act_rs = (uintptr_t) activation;
  const uint32_t act_rt = scna_hmx_activation_rt(7);
  const uintptr_t weight_rs = (uintptr_t) weights;
  const uint32_t weight_rt = SCNA_HMX_PARTIAL_WEIGHT_BYTES - 1;
  asm volatile("mxclracc.hf" ::: "memory");
  asm volatile(
    "{ activation.hf = mxmem(%0, %1)\n"
    "  weight.hf = mxmem(%2, %3) }\n" :: "r"(act_rs), "r"(act_rt),
    "r"(weight_rs), "r"(weight_rt) : "memory");
  asm volatile("bias = mxmem2(%0)" :: "r"(bias) : "memory");
  asm volatile("cvt.hf = acc(%0)" :: "r"(0) : "memory");
  asm volatile("mxmem(%0, %1) = cvt" :: "r"(output), "r"(0) : "memory");
}

static __attribute__((noinline)) void scna_hmx_fp16_d8_reduce_kernel(
    const __fp16 *activation, const __fp16 *weights, const uint32_t *bias,
    __fp16 *output, int stop_channel) {
  const uintptr_t act_rs = (uintptr_t) activation;
  const uint32_t act_rt = scna_hmx_activation_rt(stop_channel);
  const uintptr_t weight_rs = (uintptr_t) weights;
  const uint32_t weight_rt = (stop_channel == 31 ? SCNA_HMX_FULL_WEIGHT_BYTES :
                              stop_channel == 15 ? SCNA_HMX_HALF_WEIGHT_BYTES :
                                                   SCNA_HMX_PARTIAL_WEIGHT_BYTES) - 1;
  asm volatile("mxclracc.hf" ::: "memory");
  asm volatile(
    "{ activation.hf = mxmem(%0, %1)\n"
    "  weight.hf = mxmem(%2, %3) }\n" :: "r"(act_rs), "r"(act_rt),
    "r"(weight_rs), "r"(weight_rt) : "memory");
  asm volatile("bias = mxmem2(%0)" :: "r"(bias) : "memory");
  asm volatile("cvt.hf = acc(%0)" :: "r"(0) : "memory");
  asm volatile("mxmem(%0, %1) = cvt" :: "r"(output), "r"(0) : "memory");
}

static inline void scna_hmx_completion_fence(scna_hmx_context_t *ctx,
                                              const __fp16 *output) {
  const int64_t t0 = scna_hmx_now();
  const volatile uint16_t sync = ((volatile const uint16_t *) output)[0];
  asm volatile("" :: "r"(sync) : "memory");
  ctx->completion_fence_ticks += scna_hmx_now() - t0;
}

#define SCNA_TRANSPOSE_PAIR(A, B, SHIFT) do {                    \
  const HVX_VectorPair pair_ = Q6_W_vshuff_VVR((B), (A), (SHIFT)); \
  (A) = Q6_V_lo_W(pair_);                                        \
  (B) = Q6_V_hi_W(pair_);                                        \
} while (0)

/*
 * Transpose the first eight channels of a 32x32 FP16 crouton.  Each input
 * vector already contains one spatial pair interleaved by halfword, which is
 * stage 1 of a 32x32 transpose.  The four vshuff stages below finish the
 * transpose.  Each output holds two channels: low 32 lanes then high 32 lanes.
 */
static HVX_INLINE_ALWAYS void scna_hmx_transpose8_pairs(
    const __fp16 *tile, int channel_base, HVX_Vector *pair01,
    HVX_Vector *pair23, HVX_Vector *pair45, HVX_Vector *pair67) {
  const HVX_Vector *src = (const HVX_Vector *) tile;
  const int rotate = channel_base * 4;
#define SCNA_LOAD_ROT(I) Q6_V_vror_VR(src[(I)], rotate)
  HVX_Vector a0 = SCNA_LOAD_ROT(0),  a1 = SCNA_LOAD_ROT(1);
  HVX_Vector a2 = SCNA_LOAD_ROT(2),  a3 = SCNA_LOAD_ROT(3);
  HVX_Vector a4 = SCNA_LOAD_ROT(4),  a5 = SCNA_LOAD_ROT(5);
  HVX_Vector a6 = SCNA_LOAD_ROT(6),  a7 = SCNA_LOAD_ROT(7);
  HVX_Vector a8 = SCNA_LOAD_ROT(8),  a9 = SCNA_LOAD_ROT(9);
  HVX_Vector a10 = SCNA_LOAD_ROT(10), a11 = SCNA_LOAD_ROT(11);
  HVX_Vector a12 = SCNA_LOAD_ROT(12), a13 = SCNA_LOAD_ROT(13);
  HVX_Vector a14 = SCNA_LOAD_ROT(14), a15 = SCNA_LOAD_ROT(15);
#undef SCNA_LOAD_ROT

  SCNA_TRANSPOSE_PAIR(a0, a1, -4);
  SCNA_TRANSPOSE_PAIR(a2, a3, -4);
  SCNA_TRANSPOSE_PAIR(a4, a5, -4);
  SCNA_TRANSPOSE_PAIR(a6, a7, -4);
  SCNA_TRANSPOSE_PAIR(a8, a9, -4);
  SCNA_TRANSPOSE_PAIR(a10, a11, -4);
  SCNA_TRANSPOSE_PAIR(a12, a13, -4);
  SCNA_TRANSPOSE_PAIR(a14, a15, -4);

  /* Only the low output of each stage-2 pair contains channels 0..15. */
  SCNA_TRANSPOSE_PAIR(a0, a2, -8);
  SCNA_TRANSPOSE_PAIR(a4, a6, -8);
  SCNA_TRANSPOSE_PAIR(a8, a10, -8);
  SCNA_TRANSPOSE_PAIR(a12, a14, -8);

  /* Only a0/a4/a8/a12 contain the requested first eight channels. */
  SCNA_TRANSPOSE_PAIR(a0, a4, -16);
  SCNA_TRANSPOSE_PAIR(a8, a12, -16);
  SCNA_TRANSPOSE_PAIR(a0, a8, -32);
  SCNA_TRANSPOSE_PAIR(a4, a12, -32);

  *pair01 = a0;
  *pair23 = a8;
  *pair45 = a4;
  *pair67 = a12;
}

static HVX_INLINE_ALWAYS HVX_Vector scna_hmx_sum_transposed8(
    const __fp16 *tile, int channel_base) {
  HVX_Vector p01, p23, p45, p67;
  scna_hmx_transpose8_pairs(tile, channel_base, &p01, &p23, &p45, &p67);
  p01 = Q6_Vhf_vadd_VhfVhf(p01, Q6_V_vror_VR(p01, 64));
  p23 = Q6_Vhf_vadd_VhfVhf(p23, Q6_V_vror_VR(p23, 64));
  p45 = Q6_Vhf_vadd_VhfVhf(p45, Q6_V_vror_VR(p45, 64));
  p67 = Q6_Vhf_vadd_VhfVhf(p67, Q6_V_vror_VR(p67, 64));
  return Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vadd_VhfVhf(p01, p23),
                             Q6_Vhf_vadd_VhfVhf(p45, p67));
}

static HVX_INLINE_ALWAYS HVX_Vector scna_hmx_combine_low_halves(
    HVX_Vector high, HVX_Vector low) {
  return Q6_V_lo_W(Q6_W_vshuff_VVR(high, low, -64));
}

static HVX_INLINE_ALWAYS HVX_Vector scna_hmx_clamp_vhf(HVX_Vector input) {
  const HVX_Vector zero = Q6_V_vzero();
  __fp16 min_value = (__fp16) SCNA_MIN_INPUT;
  const HVX_Vector minimum = Q6_Vh_vsplat_R(fp16_to_bits(&min_value));
  input = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(minimum, input), minimum, input);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(input, zero), zero, input);
}

static void scna_hmx_pack_legacy(scna_hmx_context_t *ctx, __fp16 *activation,
                                 HVX_Vector input) {
  const HVX_Vector offsets = vmem(ctx->scatter_offsets);
  const HVX_VectorPred first_32 = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  Q6_vscatter_QRMVhV(first_32, (size_t) activation, SCNA_HMX_TILE_BYTES - 1,
                     offsets, input);
}

static void scna_hmx_pack_batch2(scna_hmx_context_t *ctx, __fp16 *activation,
                                 HVX_Vector input) {
  const HVX_Vector offsets = vmem(ctx->scatter_offsets);
  const HVX_VectorPred first_32 = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  Q6_vscatter_QRMVhV(first_32, (size_t) (activation + 0), SCNA_HMX_TILE_BYTES - 1,
                     offsets, input);
  Q6_vscatter_QRMVhV(first_32, (size_t) (activation + 2), SCNA_HMX_TILE_BYTES - 1,
                     offsets, Q6_V_vror_VR(input, 64));
}

static void scna_hmx_pack_batch4(scna_hmx_context_t *ctx, __fp16 *activation,
                                 HVX_Vector input0, HVX_Vector input1) {
  const HVX_Vector offsets = vmem(ctx->scatter_offsets);
  const HVX_VectorPred first_32 = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  Q6_vscatter_QRMVhV(first_32, (size_t) (activation + 0), SCNA_HMX_TILE_BYTES - 1,
                     offsets, input0);
  Q6_vscatter_QRMVhV(first_32, (size_t) (activation + 2), SCNA_HMX_TILE_BYTES - 1,
                     offsets, Q6_V_vror_VR(input0, 64));
  Q6_vscatter_QRMVhV(first_32, (size_t) (activation + 4), SCNA_HMX_TILE_BYTES - 1,
                     offsets, input1);
  Q6_vscatter_QRMVhV(first_32, (size_t) (activation + 6), SCNA_HMX_TILE_BYTES - 1,
                     offsets, Q6_V_vror_VR(input1, 64));
}

static inline uint16_t scna_hmx_tile_bits(const __fp16 *tile, int spatial, int channel) {
  const int spatial_pair = spatial >> 1;
  const int element_in_pair = spatial & 1;
  return ((const uint16_t *) tile)[spatial_pair * 64 + channel * 2 + element_in_pair];
}

static void scna_hmx_gather_hybrid_scalar(scna_hmx_context_t *ctx, int batch) {
  for (int channel = 0; channel < 8; ++channel) {
    for (int spatial = 0; spatial < 32; ++spatial) {
      ((uint16_t *) ctx->channel_vectors)[channel * 64 + batch * 32 + spatial] =
          scna_hmx_tile_bits(ctx->intermediate, spatial, channel);
    }
  }
}

static void scna_hmx_unpack_reduced_scalar(scna_hmx_context_t *ctx, int batch) {
  for (int spatial = 0; spatial < 32; ++spatial) {
    ((uint16_t *) ctx->channel_vectors)[batch * 32 + spatial] =
        scna_hmx_tile_bits(ctx->reduced, spatial, 0);
  }
}

static __attribute__((noinline)) HVX_Vector scna_hmx_hybrid_reduce_hvx(
    const __fp16 *channel_vectors) {
  HVX_Vector sum = vmem(channel_vectors);
#pragma unroll
  for (int channel = 1; channel < 8; ++channel) {
    sum = Q6_Vhf_vadd_VhfVhf(sum, vmem(channel_vectors + channel * 64));
  }
  return sum;
}

static void scna_hmx_lock_profiled(scna_hmx_context_t *ctx) {
  const int64_t t0 = scna_hmx_now();
  hmx_unit_acquire();
  ctx->lock_ticks += scna_hmx_now() - t0;
}

static void scna_hmx_record_affine(scna_hmx_context_t *ctx, int useful_macs) {
  ++ctx->hmx_command_count;
  ctx->physical_macs += 32LL * 8 * 32;
  ctx->useful_macs += useful_macs;
}

static void scna_hmx_record_reduce(scna_hmx_context_t *ctx, int stop_channel,
                                   int useful_macs) {
  ++ctx->hmx_command_count;
  ctx->physical_macs += 32LL * (stop_channel + 1) * 32;
  ctx->useful_macs += useful_macs;
}

static __attribute__((noinline)) HVX_Vector scna_hmx_fp16_d8_legacy_vhf(
    HVX_Vector input, scna_hmx_context_t *ctx) {
  HVX_Vector batch_sum0 = Q6_V_vzero();
  const int64_t total_t0 = scna_hmx_now();
  const int vector_transpose = (ctx->variant_flags & LLM_NPU_MODE_SCNA_HMX_VTRANSPOSE) != 0;

  for (int batch = 0; batch < 2; ++batch) {
    int64_t t0 = scna_hmx_now();
    const HVX_Vector batch_input = batch == 0 ? input : Q6_V_vror_VR(input, 64);
    scna_hmx_pack_legacy(ctx, ctx->activation, batch_input);
    ctx->pack_ticks += scna_hmx_now() - t0;

    const int64_t lock_t0 = scna_hmx_now();
    hmx_unit_acquire();
    const int64_t lock_delta = scna_hmx_now() - lock_t0;
    t0 = scna_hmx_now();
    scna_hmx_fp16_d8_affine_relu_kernel(ctx->activation, ctx->weights_affine,
                                        ctx->bias_relu, ctx->intermediate);
    scna_hmx_completion_fence(ctx, ctx->intermediate);
    scna_hmx_record_affine(ctx, 32 * 8);
    ctx->lock_ticks += lock_delta;
    ctx->affine_relu_ticks += scna_hmx_now() - t0;
    if (ctx->engine == SCNA_ENGINE_HMX_TWO_PASS) {
      t0 = scna_hmx_now();
      scna_hmx_fp16_d8_reduce_kernel(ctx->intermediate, ctx->weights_reduce,
                                     ctx->bias_identity, ctx->reduced, 7);
      scna_hmx_completion_fence(ctx, ctx->reduced);
      scna_hmx_record_reduce(ctx, 7, 32 * 8);
      ctx->reduction_ticks += scna_hmx_now() - t0;
    }
    hmx_unit_release();

    t0 = scna_hmx_now();
    if (vector_transpose) {
      HVX_Vector current;
      if (ctx->engine == SCNA_ENGINE_HMX_HYBRID) {
        current = scna_hmx_sum_transposed8(ctx->intermediate, 0);
      } else {
        HVX_Vector p01, p23, p45, p67;
        scna_hmx_transpose8_pairs(ctx->reduced, 0, &p01, &p23, &p45, &p67);
        current = p01;
      }
      if (batch == 0) {
        batch_sum0 = current;
      } else {
        batch_sum0 = scna_hmx_combine_low_halves(current, batch_sum0);
      }
      ctx->transpose_ticks += scna_hmx_now() - t0;
    } else {
      if (ctx->engine == SCNA_ENGINE_HMX_HYBRID) {
        scna_hmx_gather_hybrid_scalar(ctx, batch);
      } else {
        scna_hmx_unpack_reduced_scalar(ctx, batch);
      }
      ctx->unpack_ticks += scna_hmx_now() - t0;
    }
  }

  HVX_Vector output;
  if (vector_transpose) {
    output = batch_sum0;
  } else if (ctx->engine == SCNA_ENGINE_HMX_HYBRID) {
    const int64_t t0 = scna_hmx_now();
    output = scna_hmx_hybrid_reduce_hvx(ctx->channel_vectors);
    ctx->reduction_ticks += scna_hmx_now() - t0;
  } else {
    output = vmem(ctx->channel_vectors);
  }
  ctx->total_ticks += scna_hmx_now() - total_t0;
  return output;
}

static void scna_hmx_fp16_d8_batch4_pair(HVX_Vector input0, HVX_Vector input1,
                                          scna_hmx_context_t *ctx,
                                          HVX_Vector *output0, HVX_Vector *output1) {
  const int64_t total_t0 = scna_hmx_now();
  const int slot = ctx->next_pair_slot++ & 1;
  __fp16 *activation = slot ? ctx->activation_alt : ctx->activation;
  __fp16 *intermediate = slot ? ctx->intermediate_alt : ctx->intermediate;
  __fp16 *reduced = slot ? ctx->reduced_alt : ctx->reduced;
  int64_t t0 = scna_hmx_now();
  scna_hmx_pack_batch4(ctx, activation, input0, input1);
  ctx->pack_ticks += scna_hmx_now() - t0;

  const int64_t lock_t0 = scna_hmx_now();
  hmx_unit_acquire();
  const int64_t lock_delta = scna_hmx_now() - lock_t0;
  t0 = scna_hmx_now();
  scna_hmx_fp16_d8_affine_relu_kernel(activation, ctx->weights_affine,
                                      ctx->bias_relu, intermediate);
  scna_hmx_completion_fence(ctx, intermediate);
  scna_hmx_record_affine(ctx, 32 * 4 * 8);
  ctx->lock_ticks += lock_delta;
  ctx->affine_relu_ticks += scna_hmx_now() - t0;
  if (ctx->engine == SCNA_ENGINE_HMX_TWO_PASS) {
    t0 = scna_hmx_now();
    scna_hmx_fp16_d8_reduce_kernel(intermediate, ctx->weights_reduce,
                                   ctx->bias_identity, reduced, 31);
    scna_hmx_completion_fence(ctx, reduced);
    scna_hmx_record_reduce(ctx, 31, 32 * 4 * 8);
    ctx->reduction_ticks += scna_hmx_now() - t0;
  }
  hmx_unit_release();

  t0 = scna_hmx_now();
  if (ctx->engine == SCNA_ENGINE_HMX_HYBRID) {
    const HVX_Vector g0 = scna_hmx_sum_transposed8(intermediate, 0);
    const HVX_Vector g1 = scna_hmx_sum_transposed8(intermediate, 8);
    const HVX_Vector g2 = scna_hmx_sum_transposed8(intermediate, 16);
    const HVX_Vector g3 = scna_hmx_sum_transposed8(intermediate, 24);
    *output0 = scna_hmx_combine_low_halves(g1, g0);
    *output1 = scna_hmx_combine_low_halves(g3, g2);
  } else {
    HVX_Vector p01, p23, p45, p67;
    scna_hmx_transpose8_pairs(reduced, 0, &p01, &p23, &p45, &p67);
    *output0 = p01;
    *output1 = p23;
  }
      ctx->transpose_ticks += scna_hmx_now() - t0;
  if ((ctx->variant_flags & LLM_NPU_MODE_SCNA_HMX_DIRECT_P) == 0) {
    t0 = scna_hmx_now();
    vmem((HVX_Vector *) ctx->channel_vectors + 0) = *output0;
    vmem((HVX_Vector *) ctx->channel_vectors + 1) = *output1;
    *output0 = vmem((HVX_Vector *) ctx->channel_vectors + 0);
    *output1 = vmem((HVX_Vector *) ctx->channel_vectors + 1);
    ctx->unpack_ticks += scna_hmx_now() - t0;
  }
  ctx->total_ticks += scna_hmx_now() - total_t0;
}

static __attribute__((unused)) HVX_Vector scna_hmx_fp16_d8_batch2_vhf(
    HVX_Vector input, scna_hmx_context_t *ctx) {
  const int64_t total_t0 = scna_hmx_now();
  /* The online-softmax m-diff SCNA follows a batch4 score SCNA in the same
   * command stream.  Keep it on the alternate croutons so HVX does not
   * overwrite the activation/intermediate buffers still retiring from the
   * preceding HMX store. */
  __fp16 *activation = ctx->activation_alt;
  __fp16 *intermediate = ctx->intermediate_alt;
  __fp16 *reduced = ctx->reduced_alt;
  int64_t t0 = scna_hmx_now();
  scna_hmx_pack_batch2(ctx, activation, input);
  ctx->pack_ticks += scna_hmx_now() - t0;

  const int64_t lock_t0 = scna_hmx_now();
  hmx_unit_acquire();
  const int64_t lock_delta = scna_hmx_now() - lock_t0;
  t0 = scna_hmx_now();
  scna_hmx_fp16_d8_affine_relu_kernel(activation, ctx->weights_affine,
                                      ctx->bias_relu, intermediate);
  scna_hmx_completion_fence(ctx, intermediate);
  scna_hmx_record_affine(ctx, 32 * 2 * 8);
  ctx->lock_ticks += lock_delta;
  ctx->affine_relu_ticks += scna_hmx_now() - t0;
  if (ctx->engine == SCNA_ENGINE_HMX_TWO_PASS) {
    t0 = scna_hmx_now();
    scna_hmx_fp16_d8_reduce_kernel(intermediate, ctx->weights_reduce,
                                   ctx->bias_identity, reduced, 15);
    scna_hmx_completion_fence(ctx, reduced);
    scna_hmx_record_reduce(ctx, 15, 32 * 2 * 8);
    ctx->reduction_ticks += scna_hmx_now() - t0;
  }
  hmx_unit_release();

  t0 = scna_hmx_now();
  HVX_Vector output;
  if (ctx->engine == SCNA_ENGINE_HMX_HYBRID) {
    const HVX_Vector g0 = scna_hmx_sum_transposed8(intermediate, 0);
    const HVX_Vector g1 = scna_hmx_sum_transposed8(intermediate, 8);
    output = scna_hmx_combine_low_halves(g1, g0);
  } else {
    HVX_Vector p01, p23, p45, p67;
    scna_hmx_transpose8_pairs(reduced, 0, &p01, &p23, &p45, &p67);
    output = p01;
  }
  ctx->transpose_ticks += scna_hmx_now() - t0;
  if ((ctx->variant_flags & LLM_NPU_MODE_SCNA_HMX_DIRECT_P) == 0) {
    t0 = scna_hmx_now();
    vmem((HVX_Vector *) ctx->channel_vectors) = output;
    output = vmem((HVX_Vector *) ctx->channel_vectors);
    ctx->unpack_ticks += scna_hmx_now() - t0;
  }
  ctx->total_ticks += scna_hmx_now() - total_t0;
  return output;
}

HVX_Vector scna_hmx_fp16_d8_vhf(HVX_Vector input, scna_hmx_context_t *ctx) {
  input = scna_hmx_clamp_vhf(input);
  if ((ctx->variant_flags & LLM_NPU_MODE_SCNA_HMX_BATCH4) != 0) {
    HVX_Vector output, discarded;
    scna_hmx_fp16_d8_batch4_pair(input, Q6_V_vzero(), ctx, &output, &discarded);
    return output;
  }
  return scna_hmx_fp16_d8_legacy_vhf(input, ctx);
}

void scna_hmx_fp16_d8_pair_vhf(HVX_Vector input0, HVX_Vector input1,
                               scna_hmx_context_t *ctx,
                               HVX_Vector *output0, HVX_Vector *output1) {
  input0 = scna_hmx_clamp_vhf(input0);
  input1 = scna_hmx_clamp_vhf(input1);
  if ((ctx->variant_flags & LLM_NPU_MODE_SCNA_HMX_BATCH4) != 0) {
    scna_hmx_fp16_d8_batch4_pair(input0, input1, ctx, output0, output1);
    return;
  }
  *output0 = scna_hmx_fp16_d8_legacy_vhf(input0, ctx);
  *output1 = scna_hmx_fp16_d8_legacy_vhf(input1, ctx);
}

static inline __fp16 *scna_hmx_slot_activation(scna_hmx_context_t *ctx, int slot) {
  return slot ? ctx->activation_alt : ctx->activation;
}

static inline __fp16 *scna_hmx_slot_intermediate(scna_hmx_context_t *ctx, int slot) {
  return slot ? ctx->intermediate_alt : ctx->intermediate;
}

static inline __fp16 *scna_hmx_slot_reduced(scna_hmx_context_t *ctx, int slot) {
  return slot ? ctx->reduced_alt : ctx->reduced;
}

void scna_hmx_fp16_d8_pair_issue(HVX_Vector input0, HVX_Vector input1,
                                 scna_hmx_context_t *ctx, int slot) {
  input0 = scna_hmx_clamp_vhf(input0);
  input1 = scna_hmx_clamp_vhf(input1);
  slot &= 1;
  __fp16 *activation = scna_hmx_slot_activation(ctx, slot);
  __fp16 *intermediate = scna_hmx_slot_intermediate(ctx, slot);
  __fp16 *reduced = scna_hmx_slot_reduced(ctx, slot);

  int64_t t0 = scna_hmx_now();
  scna_hmx_pack_batch4(ctx, activation, input0, input1);
  ctx->pack_ticks += scna_hmx_now() - t0;

  scna_hmx_lock_profiled(ctx);
  t0 = scna_hmx_now();
  scna_hmx_fp16_d8_affine_relu_kernel(activation, ctx->weights_affine,
                                      ctx->bias_relu, intermediate);
  scna_hmx_record_affine(ctx, 32 * 4 * 8);
  ctx->affine_relu_ticks += scna_hmx_now() - t0;
  if (ctx->engine == SCNA_ENGINE_HMX_TWO_PASS) {
    /* The second pass consumes the first store, so it is the required first
     * pass completion point. The final store remains unfenced until consume. */
    scna_hmx_completion_fence(ctx, intermediate);
    t0 = scna_hmx_now();
    scna_hmx_fp16_d8_reduce_kernel(intermediate, ctx->weights_reduce,
                                   ctx->bias_identity, reduced, 31);
    scna_hmx_record_reduce(ctx, 31, 32 * 4 * 8);
    ctx->reduction_ticks += scna_hmx_now() - t0;
  }
  ctx->slot_issue_tick[slot] = scna_hmx_now();
  hmx_unit_release();
}

void scna_hmx_fp16_d8_pair_consume(scna_hmx_context_t *ctx, int slot,
                                   HVX_Vector *output0, HVX_Vector *output1) {
  const int64_t total_t0 = scna_hmx_now();
  slot &= 1;
  const __fp16 *intermediate = scna_hmx_slot_intermediate(ctx, slot);
  const __fp16 *reduced = scna_hmx_slot_reduced(ctx, slot);
  const __fp16 *source = ctx->engine == SCNA_ENGINE_HMX_HYBRID ? intermediate : reduced;
  const int64_t consume_start = scna_hmx_now();
  if (ctx->slot_issue_tick[slot] > 0 && consume_start > ctx->slot_issue_tick[slot]) {
    ctx->pipeline_overlap_ticks += consume_start - ctx->slot_issue_tick[slot];
  }
  scna_hmx_completion_fence(ctx, source);

  const int64_t t0 = scna_hmx_now();
  if (ctx->engine == SCNA_ENGINE_HMX_HYBRID) {
    const HVX_Vector g0 = scna_hmx_sum_transposed8(intermediate, 0);
    const HVX_Vector g1 = scna_hmx_sum_transposed8(intermediate, 8);
    const HVX_Vector g2 = scna_hmx_sum_transposed8(intermediate, 16);
    const HVX_Vector g3 = scna_hmx_sum_transposed8(intermediate, 24);
    *output0 = scna_hmx_combine_low_halves(g1, g0);
    *output1 = scna_hmx_combine_low_halves(g3, g2);
  } else {
    HVX_Vector p01, p23, p45, p67;
    scna_hmx_transpose8_pairs(reduced, 0, &p01, &p23, &p45, &p67);
    *output0 = p01;
    *output1 = p23;
  }
  ctx->transpose_ticks += scna_hmx_now() - t0;
  ctx->total_ticks += scna_hmx_now() - total_t0;
}

int scna_hmx_vtranspose_layout_gate(scna_hmx_context_t *ctx) {
  if (ctx == NULL) return -1;
  uint16_t *tile = (uint16_t *) ctx->intermediate_alt;
  for (int spatial = 0; spatial < 32; ++spatial) {
    for (int channel = 0; channel < 32; ++channel) {
      const int pair = spatial >> 1;
      const int lane = pair * 64 + channel * 2 + (spatial & 1);
      tile[lane] = (uint16_t) (channel * 32 + spatial);
    }
  }

  int mismatches = 0;
  for (int group = 0; group < 4; ++group) {
    HVX_Vector p01, p23, p45, p67;
    scna_hmx_transpose8_pairs(ctx->intermediate_alt, group * 8,
                              &p01, &p23, &p45, &p67);
    HVX_Vector pairs[4] = { p01, p23, p45, p67 };
    for (int pair = 0; pair < 4; ++pair) {
      vmem((HVX_Vector *) ctx->channel_vectors) = pairs[pair];
      const uint16_t *got = (const uint16_t *) ctx->channel_vectors;
      for (int half = 0; half < 2; ++half) {
        const int channel = group * 8 + pair * 2 + half;
        for (int spatial = 0; spatial < 32; ++spatial) {
          const uint16_t expected = (uint16_t) (channel * 32 + spatial);
          if (got[half * 32 + spatial] != expected) ++mismatches;
        }
      }
    }
  }
  return mismatches;
}

int scna_hmx_overlap_probe(scna_hmx_context_t *ctx, int warmup, int iters,
                           int64_t *serial_ticks, int64_t *overlap_ticks,
                           int *mismatches) {
  if (ctx == NULL || warmup < 0 || iters <= 0 || serial_ticks == NULL ||
      overlap_ticks == NULL || mismatches == NULL) return -1;

  scna_hmx_pack_batch2(ctx, ctx->activation_alt, Q6_V_vzero());
  hmx_unit_acquire();
  scna_hmx_fp16_d8_affine_relu_kernel(ctx->activation_alt, ctx->weights_affine,
                                      ctx->bias_relu, ctx->intermediate);
  scna_hmx_completion_fence(ctx, ctx->intermediate);
  scna_hmx_fp16_d8_affine_relu_kernel(ctx->activation_alt, ctx->weights_affine,
                                      ctx->bias_relu, ctx->intermediate_alt);
  scna_hmx_completion_fence(ctx, ctx->intermediate_alt);

  volatile HVX_Vector sink = Q6_V_vzero();
  for (int i = 0; i < warmup; ++i) {
    __fp16 *out = (i & 1) ? ctx->intermediate : ctx->intermediate_alt;
    scna_hmx_fp16_d8_affine_relu_kernel(ctx->activation_alt, ctx->weights_affine,
                                        ctx->bias_relu, out);
    scna_hmx_completion_fence(ctx, out);
    sink = Q6_V_vxor_VV(sink, scna_hmx_sum_transposed8(out, 0));
  }

  int64_t t0 = scna_hmx_now();
  for (int i = 0; i < iters; ++i) {
    __fp16 *out = (i & 1) ? ctx->intermediate : ctx->intermediate_alt;
    scna_hmx_fp16_d8_affine_relu_kernel(ctx->activation_alt, ctx->weights_affine,
                                        ctx->bias_relu, out);
    scna_hmx_completion_fence(ctx, out);
    sink = Q6_V_vxor_VV(sink, scna_hmx_sum_transposed8(out, 0));
  }
  *serial_ticks = scna_hmx_now() - t0;

  t0 = scna_hmx_now();
  for (int i = 0; i < iters; ++i) {
    __fp16 *out = (i & 1) ? ctx->intermediate : ctx->intermediate_alt;
    const __fp16 *previous = (i & 1) ? ctx->intermediate_alt : ctx->intermediate;
    scna_hmx_fp16_d8_affine_relu_kernel(ctx->activation_alt, ctx->weights_affine,
                                        ctx->bias_relu, out);
    sink = Q6_V_vxor_VV(sink, scna_hmx_sum_transposed8(previous, 0));
    scna_hmx_completion_fence(ctx, out);
  }
  *overlap_ticks = scna_hmx_now() - t0;

  *mismatches = 0;
  for (int spatial = 0; spatial < 32; ++spatial) {
    for (int channel = 0; channel < 32; ++channel) {
      if (scna_hmx_tile_bits(ctx->intermediate, spatial, channel) !=
          scna_hmx_tile_bits(ctx->intermediate_alt, spatial, channel)) {
        ++*mismatches;
      }
    }
  }
  (void) sink;
  hmx_unit_release();
  return 0;
}
