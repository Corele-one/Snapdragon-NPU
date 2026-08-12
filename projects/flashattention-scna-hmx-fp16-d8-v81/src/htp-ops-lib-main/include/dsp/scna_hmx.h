#pragma once

#include <stddef.h>
#include <stdint.h>

#include <hexagon_types.h>

#include "op_reg.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SCNA_HMX_CONTEXT_BYTES 24576
#define SCNA_HMX_PROFILE_VERSION 2

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
  __fp16 *activation_alt;
  __fp16 *intermediate_alt;
  __fp16 *reduced_alt;
  int engine;
  int variant_flags;
  int64_t pack_ticks;
  int64_t affine_relu_ticks;
  int64_t reduction_ticks;
  int64_t unpack_ticks;
  int64_t transpose_ticks;
  int64_t p_store_ticks;
  int64_t lock_ticks;
  int64_t completion_fence_ticks;
  int64_t pipeline_overlap_ticks;
  int64_t hmx_command_count;
  int64_t physical_macs;
  int64_t useful_macs;
  int64_t total_ticks;
  int64_t slot_issue_tick[2];
  int next_pair_slot;
} scna_hmx_context_t;

int scna_hmx_context_init(scna_hmx_context_t *ctx, void *aligned_vtcm, size_t bytes,
                          int engine, int variant_flags);
void scna_hmx_reset_profile(scna_hmx_context_t *ctx);

HVX_Vector scna_hmx_fp16_d8_vhf(HVX_Vector input, scna_hmx_context_t *ctx);
void scna_hmx_fp16_d8_pair_vhf(HVX_Vector input0, HVX_Vector input1, scna_hmx_context_t *ctx,
                               HVX_Vector *output0, HVX_Vector *output1);

/* Double-buffered Attention path: issue HMX to slot, then consume after
 * independent work or the following slot has been issued. */
void scna_hmx_fp16_d8_pair_issue(HVX_Vector input0, HVX_Vector input1,
                                 scna_hmx_context_t *ctx, int slot);
void scna_hmx_fp16_d8_pair_consume(scna_hmx_context_t *ctx, int slot,
                                   HVX_Vector *output0, HVX_Vector *output1);

/* Device-side deterministic lane-ID gate for the register crouton transpose. */
int scna_hmx_vtranspose_layout_gate(scna_hmx_context_t *ctx);

/* Measures whether an unfenced HMX store overlaps independent HVX transpose work. */
int scna_hmx_overlap_probe(scna_hmx_context_t *ctx, int warmup, int iters,
                           int64_t *serial_ticks, int64_t *overlap_ticks,
                           int *mismatches);

#ifdef __cplusplus
}
#endif
