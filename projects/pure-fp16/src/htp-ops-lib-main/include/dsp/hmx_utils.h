#pragma once

#include <hexagon_types.h>
#include <stddef.h>
#include <stdint.h>

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

static HMX_INLINE_ALWAYS void hmx_set_output_scales(const void *scales) {
  asm volatile("bias = mxmem2(%0)" ::"r"(scales));
}

// LPBQ deploy-v1 drain-fusion gate: keep the manual bias-load contract
// explicit.  The source must already be a 256B-aligned HMX-visible VTCM slot;
// the low two address bits select the destination HMX bias register set.
static HMX_INLINE_ALWAYS void hmx_load_bias_set_from_vtcm(const void *bias256_vtcm, int bias_idx) {
  uintptr_t rs = ((uintptr_t) bias256_vtcm & ~(uintptr_t) 0xffu) | (uintptr_t) (bias_idx & 3);
  asm volatile("bias = mxmem2(%0)" ::"r"((void *) rs) : "memory");
}

static HMX_INLINE_ALWAYS void hmx_store_bias_set_to_vtcm(void *bias256_vtcm, int bias_idx) {
  uintptr_t rs = ((uintptr_t) bias256_vtcm & ~(uintptr_t) 0xffu) | (uintptr_t) (bias_idx & 3);
  asm volatile("mxmem2(%0) = bias" ::"r"((void *) rs) : "memory");
}

static HMX_INLINE_ALWAYS void hmx_cvt_hf_acc_select_bias(int bias_idx, int retain_acc) {
  uint32_t rs = ((uint32_t) (bias_idx & 3) << 12) | (retain_acc ? 1u : 0u);
  asm volatile("cvt.hf = acc(%0)" ::"r"(rs) : "memory");
}

static HMX_INLINE_ALWAYS void hmx_store_cvt_to_vtcm(void *out_2kb_vtcm, uint32_t spatial_rs) {
  asm volatile("mxmem(%0, %1) = cvt" ::"r"(out_2kb_vtcm), "r"(spatial_rs) : "memory");
}

// set aligned 256 bytes area
static HMX_INLINE_ALWAYS void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
  HVX_Vector *pv = (HVX_Vector *) out_scales;

  *pv++ = v_scale;
  *pv   = Q6_V_vzero();
}

static HMX_INLINE_ALWAYS void hmx_load_tiles_fp16(const __fp16 *row_tiles, const __fp16 *col_tiles, size_t n_tiles) {
  size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  asm volatile(
    "{ activation.hf = mxmem(%0, %1):deep\n"
    "weight.hf = mxmem(%2, %3) }\n" ::"r"(row_tiles),
    "r"(limit), "r"(col_tiles), "r"(limit)
    : "memory");
}

static HMX_INLINE_ALWAYS void hmx_consume_accumulator_fp16(__fp16 *out) {
  asm volatile(
    "cvt.hf = acc(%0)\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(2),
    "r"(out), "r"(0)
    : "memory");
}

// compute inner product of two vectors of tiles
static HMX_INLINE_ALWAYS void hmx_dot_fp16(__fp16 *out, const __fp16 *row_tiles, const __fp16 *col_tiles,
                                           size_t n_tiles) {
  hmx_load_tiles_fp16(row_tiles, col_tiles, n_tiles);
  hmx_consume_accumulator_fp16(out);
}
