#pragma once

#include <stddef.h>
#include <stdint.h>

#include <hexagon_types.h>

#include "op_reg.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SCNA_HMX_CONTEXT_BYTES 16384
#define SCNA_HMX_PROFILE_VERSION 1

typedef struct {
  uint8_t *base;
  __fp16 *activation;
  __fp16 *weights_affine;
  uint32_t *bias_relu;
  __fp16 *intermediate;
  __fp16 *weights_reduce;
  uint32_t *bias_identity;
  __fp16 *reduced;
  __fp16 *channel_vectors;
  uint16_t *scatter_offsets;
  int engine;
  int64_t pack_ticks;
  int64_t affine_relu_ticks;
  int64_t reduction_ticks;
  int64_t unpack_ticks;
  int64_t total_ticks;
} scna_hmx_context_t;

int scna_hmx_context_init(scna_hmx_context_t *ctx, void *aligned_vtcm, size_t bytes, int engine);
void scna_hmx_reset_profile(scna_hmx_context_t *ctx);

HVX_Vector scna_hmx_fp16_d8_vhf(HVX_Vector input, scna_hmx_context_t *ctx);
void scna_hmx_fp16_d8_pair_vhf(HVX_Vector input0, HVX_Vector input1, scna_hmx_context_t *ctx,
                               HVX_Vector *output0, HVX_Vector *output1);

#ifdef __cplusplus
}
#endif
