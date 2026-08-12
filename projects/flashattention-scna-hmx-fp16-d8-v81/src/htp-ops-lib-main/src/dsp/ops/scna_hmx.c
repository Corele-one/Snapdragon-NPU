#include "dsp/scna_hmx.h"

#include <string.h>

#include <HAP_perf.h>

#include "dsp/hmx_mgr.h"
#include "dsp/hvx_internal.h"
#include "dsp/scna_params.h"
#include "dsp/utils.h"

#define SCNA_HMX_TILE_BYTES 2048
#define SCNA_HMX_PARTIAL_WEIGHT_BYTES 512
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

static void scna_hmx_init_bias(uint32_t *bias, const __fp16 *input_bias, int relu) {
  for (int output = 0; output < 32; ++output) {
    bias[output] = 0x00003c00u;
    const uint16_t input_bits = input_bias != NULL && output < 8 ? scna_hmx_fp16_bits(&input_bias[output]) : 0;
    bias[32 + output] = ((uint32_t) input_bits << 16) | (relu ? (2u << 8) : 0u);
  }
}

int scna_hmx_context_init(scna_hmx_context_t *ctx, void *aligned_vtcm, size_t bytes, int engine) {
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
  ctx->engine = engine;

  for (int input_pair = 0; input_pair < 4; ++input_pair) {
    for (int output = 0; output < 32; ++output) {
      for (int in_pair = 0; in_pair < 2; ++in_pair) {
        const int input_channel = input_pair * 2 + in_pair;
        const int index = input_pair * 64 + output * 2 + in_pair;
        ctx->weights_affine[index] = output < 8 && input_channel == 0
            ? scna_exp2_d8_wk[output] : (__fp16) 0.0f;
        ctx->weights_reduce[index] = output == 0 && input_channel < 8
            ? (__fp16) 1.0f : (__fp16) 0.0f;
      }
    }
  }
  scna_hmx_init_bias(ctx->bias_relu, scna_exp2_d8_bk, 1);
  scna_hmx_init_bias(ctx->bias_identity, NULL, 0);
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
    "  weight.hf = mxmem(%2, %3) }\n" :: "r"(act_rs), "r"(act_rt), "r"(weight_rs), "r"(weight_rt) : "memory");
  asm volatile("bias = mxmem2(%0)" :: "r"(bias) : "memory");
  asm volatile("cvt.hf = acc(%0)" :: "r"(0) : "memory");
  asm volatile("mxmem(%0, %1) = cvt" :: "r"(output), "r"(0) : "memory");
}

static __attribute__((noinline)) void scna_hmx_fp16_d8_reduce_kernel(
    const __fp16 *activation, const __fp16 *weights, const uint32_t *bias, __fp16 *output) {
  const uintptr_t act_rs = (uintptr_t) activation;
  const uint32_t act_rt = scna_hmx_activation_rt(7);
  const uintptr_t weight_rs = (uintptr_t) weights;
  const uint32_t weight_rt = SCNA_HMX_PARTIAL_WEIGHT_BYTES - 1;
  asm volatile("mxclracc.hf" ::: "memory");
  asm volatile(
    "{ activation.hf = mxmem(%0, %1)\n"
    "  weight.hf = mxmem(%2, %3) }\n" :: "r"(act_rs), "r"(act_rt), "r"(weight_rs), "r"(weight_rt) : "memory");
  asm volatile("bias = mxmem2(%0)" :: "r"(bias) : "memory");
  asm volatile("cvt.hf = acc(%0)" :: "r"(0) : "memory");
  asm volatile("mxmem(%0, %1) = cvt" :: "r"(output), "r"(0) : "memory");
}

static __attribute__((noinline)) HVX_Vector scna_hmx_fp16_d8_hybrid_reduce_hvx(
    const __fp16 *channel_vectors) {
  HVX_Vector sum = vmem(channel_vectors);
#pragma unroll
  for (int channel = 1; channel < 8; ++channel) {
    sum = Q6_Vhf_vadd_VhfVhf(sum, vmem(channel_vectors + channel * 64));
  }
  return sum;
}

static HVX_INLINE_ALWAYS HVX_Vector scna_hmx_clamp_vhf(HVX_Vector input) {
  const HVX_Vector zero = Q6_V_vzero();
  __fp16 min_value = (__fp16) SCNA_MIN_INPUT;
  const HVX_Vector minimum = Q6_Vh_vsplat_R(fp16_to_bits(&min_value));
  input = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(minimum, input), minimum, input);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhfVhf(input, zero), zero, input);
}

static void scna_hmx_pack_batch(scna_hmx_context_t *ctx, HVX_Vector input) {
  const HVX_Vector offsets = vmem(ctx->scatter_offsets);
  const HVX_VectorPred first_32 = Q6_Q_vsetq_R(32 * sizeof(__fp16));
  Q6_vscatter_QRMVhV(first_32, (size_t) ctx->activation, SCNA_HMX_TILE_BYTES - 1, offsets, input);
}

static inline uint16_t scna_hmx_tile_bits(const __fp16 *tile, int spatial, int channel) {
  const int spatial_pair = spatial >> 1;
  const int element_in_pair = spatial & 1;
  return ((const uint16_t *) tile)[spatial_pair * 64 + channel * 2 + element_in_pair];
}

static void scna_hmx_gather_hybrid(scna_hmx_context_t *ctx, int batch) {
  for (int channel = 0; channel < 8; ++channel) {
    for (int spatial = 0; spatial < 32; ++spatial) {
      ((uint16_t *) ctx->channel_vectors)[channel * 64 + batch * 32 + spatial] =
          scna_hmx_tile_bits(ctx->intermediate, spatial, channel);
    }
  }
}

static void scna_hmx_unpack_reduced(scna_hmx_context_t *ctx, int batch) {
  for (int spatial = 0; spatial < 32; ++spatial) {
    ((uint16_t *) ctx->channel_vectors)[batch * 32 + spatial] =
        scna_hmx_tile_bits(ctx->reduced, spatial, 0);
  }
}

HVX_Vector scna_hmx_fp16_d8_vhf(HVX_Vector input, scna_hmx_context_t *ctx) {
  input = scna_hmx_clamp_vhf(input);
  HVX_Vector output_vector = Q6_V_vzero();
  const int64_t total_t0 = scna_hmx_now();

  for (int batch = 0; batch < 2; ++batch) {
    int64_t t0 = scna_hmx_now();
    const HVX_Vector batch_input = batch == 0 ? input : Q6_V_vror_VR(input, 64);
    scna_hmx_pack_batch(ctx, batch_input);
    ctx->pack_ticks += scna_hmx_now() - t0;

    hmx_unit_acquire();
    t0 = scna_hmx_now();
    scna_hmx_fp16_d8_affine_relu_kernel(ctx->activation, ctx->weights_affine,
                                        ctx->bias_relu, ctx->intermediate);
    const volatile uint16_t affine_sync = ((volatile const uint16_t *) ctx->intermediate)[0];
    asm volatile("" :: "r"(affine_sync) : "memory");
    ctx->affine_relu_ticks += scna_hmx_now() - t0;
    if (ctx->engine == SCNA_ENGINE_HMX_TWO_PASS) {
      t0 = scna_hmx_now();
      scna_hmx_fp16_d8_reduce_kernel(ctx->intermediate, ctx->weights_reduce,
                                     ctx->bias_identity, ctx->reduced);
      const volatile uint16_t reduce_sync = ((volatile const uint16_t *) ctx->reduced)[0];
      asm volatile("" :: "r"(reduce_sync) : "memory");
      ctx->reduction_ticks += scna_hmx_now() - t0;
    }
    hmx_unit_release();

    t0 = scna_hmx_now();
    if (ctx->engine == SCNA_ENGINE_HMX_HYBRID) {
      scna_hmx_gather_hybrid(ctx, batch);
    } else {
      scna_hmx_unpack_reduced(ctx, batch);
    }
    ctx->unpack_ticks += scna_hmx_now() - t0;
  }

  if (ctx->engine == SCNA_ENGINE_HMX_HYBRID) {
    const int64_t t0 = scna_hmx_now();
    output_vector = scna_hmx_fp16_d8_hybrid_reduce_hvx(ctx->channel_vectors);
    ctx->reduction_ticks += scna_hmx_now() - t0;
  } else {
    output_vector = vmem(ctx->channel_vectors);
  }
  ctx->total_ticks += scna_hmx_now() - total_t0;
  return output_vector;
}

void scna_hmx_fp16_d8_pair_vhf(HVX_Vector input0, HVX_Vector input1, scna_hmx_context_t *ctx,
                               HVX_Vector *output0, HVX_Vector *output1) {
  *output0 = scna_hmx_fp16_d8_vhf(input0, ctx);
  *output1 = scna_hmx_fp16_d8_vhf(input1, ctx);
}
