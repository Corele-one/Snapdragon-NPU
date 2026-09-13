#ifndef FLASH_ATTN_ROWSUM_H
#define FLASH_ATTN_ROWSUM_H

/*
 * Evaluator-independent qf32 rowsum packing used by the Stage 3.0 audit.
 *
 * This must be a zero-filling reduction.  A circular rotate duplicates a
 * sparse lane into the vacated positions, so its result depends on mask
 * density (the Stage 2.9 Q=32/KV=32 causal failure).  Callers first perform a
 * zero-filling vlalign reduction; its final qf32 lane is packed to FP16 here
 * without an empirical scale factor.
 */
static inline float flash_attn_sum_vhf(HVX_Vector values) {
  _Alignas(VLEN) __fp16 lanes[64];
  vmem(lanes) = values;
  float total = 0.0f;
#pragma unroll
  for (int lane = 0; lane < 64; ++lane) {
    total += (float) lanes[lane];
  }
  return total;
}

static inline HVX_Vector flash_attn_pack_rowsums_f32(float total0, float total1) {
  __fp16 sum0 = (__fp16) total0;
  __fp16 sum1 = (__fp16) total1;
  const uint32_t pair = (uint32_t) fp16_to_bits(&sum0) | ((uint32_t) fp16_to_bits(&sum1) << 16);
  return Q6_V_vsplat_R(pair);
}

#endif
