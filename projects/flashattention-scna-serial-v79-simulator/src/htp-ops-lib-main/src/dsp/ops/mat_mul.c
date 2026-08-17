#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/quants.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "op_reg.h"

// debug & profile
#include "HAP_farf.h"
#include "HAP_perf.h"

#ifndef HTP_MATMUL_WEIGHT_AREA_KB
#define HTP_MATMUL_WEIGHT_AREA_KB 1024
#endif

#ifndef HTP_MATMUL_ACTIVATION_AREA_KB
#define HTP_MATMUL_ACTIVATION_AREA_KB 2048
#endif

#ifndef HTP_MATMUL_OUTPUT_AREA_KB
#define HTP_MATMUL_OUTPUT_AREA_KB 1536
#endif

#ifndef HTP_MATMUL_SCRATCH_AREA_KB
#define HTP_MATMUL_SCRATCH_AREA_KB 1024
#endif

// These VTCM areas control the HMX matmul tile shapes. Keep them compile-time
// configurable so batch/tile capacity sweeps can explore safe combinations
// without manually editing this file.
#define WEIGHT_AREA_SIZE     (HTP_MATMUL_WEIGHT_AREA_KB * 1024)
#define ACTIVATION_AREA_SIZE (HTP_MATMUL_ACTIVATION_AREA_KB * 1024)
#define OUTPUT_AREA_SIZE     (HTP_MATMUL_OUTPUT_AREA_KB * 1024)
#define SCRATCH_AREA_SIZE    (HTP_MATMUL_SCRATCH_AREA_KB * 1024)

#ifndef HTP_ENABLE_OUT_STATIONARY_PREFILL
#define HTP_ENABLE_OUT_STATIONARY_PREFILL 0
#endif

#ifndef HTP_MATMUL_PIPELINE_MODE
#define HTP_MATMUL_PIPELINE_MODE 0
#endif

#ifndef HTP_OS_DEBUG_COMPARE
#define HTP_OS_DEBUG_COMPARE 0
#endif

static const __fp16 q4_0_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
  -8, 0, -7, 0, -6, 0, -5, 0, -4, 0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
};

static const __fp16 iq4_nl_to_fp16_lut[64] __attribute__((aligned(VLEN))) = {
  -127, 0, -104, 0, -83, 0, -65, 0, -49, 0, -35, 0, -22, 0, -10, 0,
  1,    0, 13,   0, 25,  0, 38,  0, 53,  0, 69,  0, 89,  0, 113, 0,
};

static const uint32_t common_layout_vscatter_offsets_base[32] __attribute__((aligned(VLEN))) = {
  0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
};

static inline void swap_ptr(void **p1, void **p2) {
  void *t = *p1;
  *p1     = *p2;
  *p2     = t;
}

static inline size_t get_super_block_size(enum ggml_type weight_type) {
  switch (weight_type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_IQ4_NL:
      return sizeof(my_block_q4_0);
    case GGML_TYPE_Q8_0:
      return sizeof(my_block_q8_0);
    default:
      return 0;
  }
}

static inline int dma_issue_load_from_ddr(dma_desc_1d_t *desc, void *vtcm_dst, const void *src, size_t size) {
  dma_wait_for_idle();

  desc->next       = 0;
  desc->length     = size;
  desc->type       = DMA_DESC_TYPE_1D;
  desc->src_bypass = 1;
  desc->dst_bypass = 0;
  desc->ordered    = 1;
  desc->dstate     = DMA_DESC_DSTATE_PENDING;
  desc->src        = (uint32_t) src;
  desc->dst        = (uint32_t) vtcm_dst;

  return dma_submit_one(desc);
}

static void find_chunk_size(size_t x_max, size_t y_max, size_t xy_max, size_t x_unit, size_t y_unit, size_t *x_out,
                            size_t *y_out) {
  int64_t best_xy = 0;
  size_t  best_x = 0, best_y = 0;

  for (size_t x = x_max; x > 0; x -= x_unit) {
    size_t  y  = smin(align_down(xy_max / x, y_unit), y_max);
    int64_t xy = x * y;
    if (best_xy < xy) {
      best_xy = xy;
      best_x = x, best_y = y;
    }
  }
  *x_out = best_x, *y_out = best_y;
}

static inline bool qk_matmul_use_pipeline(int m, int k, int n) {
#if HTP_MATMUL_PIPELINE_MODE == 1
  (void) m;
  (void) k;
  (void) n;
  // Experiment mode: force the conservative sequential safe path so capacity
  // sweeps can compare it against the original auto-selected 4-stage pipeline.
  return false;
#else
  // Original policy: use the 4-stage pipeline only when M is large enough to
  // amortize scheduling overhead and K<=N gives enough N chunks to overlap.
  return (m >= 128) && (k <= n);
#endif
}

static int matmul_check_vtcm_capacity(const char *path, size_t required_bytes) {
  const size_t capacity = vtcm_manager_get_seq_capacity();
  if (!vtcm_manager_get_vtcm_base() || capacity == 0 || required_bytes > capacity) {
    FARF(ALWAYS, "%s: VTCM request too large, required=%u KiB, capacity=%u KiB, total=%u KiB", path,
         (unsigned) (required_bytes / 1024), (unsigned) (capacity / 1024),
         (unsigned) (vtcm_manager_get_total_size() / 1024));
    return -1;
  }
  return 0;
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_activation_chunk_fp32_to_fp16(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
                                                   int k_block, int k_stride) {
  assert(k_block % HMX_FP16_TILE_N_COLS == 0 && k_stride % HMX_FP16_TILE_N_COLS == 0);
  assert(VLEN == 32 * sizeof(float));

  for (int r = 0; r < n_rows; r += 2) {
    int prefetch_row_idx = r + 2;
    if (prefetch_row_idx < n_rows) {
      const float *prefetch_addr = src + prefetch_row_idx * k_stride;
      // NOTE(hzx): prefetch 2 rows at a time
      l2fetch(prefetch_addr, k_stride * sizeof(float), k_block * sizeof(float), 2, 0);
    }

    int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
    int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

    const bool next_row_valid = (r + 1) < n_rows;

    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * k_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * k_stride);
    for (int c = 0; c < k_block; c += 32) {
      HVX_Vector v0 = *pv_in0++;
      HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

      HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

      // compute output position
      int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
      int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2]     = v_out;
    }
  }
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  int           k;
  __fp16       *dst;
  const __fp16 *src;
} permuted_weight_transfer_fp16_task_state_t;

static void transfer_permuted_weight_fp16_task(__fp16 *restrict vtcm_dst, const __fp16 *restrict src, int k,
                                               int n_col_tiles) {
  // transfer logical K*(32*n_col_tiles) weight block, direct copy, no extra computation
  size_t size   = k * n_col_tiles * HMX_FP16_TILE_N_COLS * sizeof(__fp16);
  int    n_vecs = size / VLEN;

  const size_t PREFETCH_SIZE   = 4096;
  const int    PREFETCH_N_VECS = PREFETCH_SIZE / VLEN;

  const HVX_Vector *pv_in  = (const HVX_Vector *) src;
  HVX_Vector       *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_vecs; ++i) {
    if (i % PREFETCH_N_VECS == 0) {
      int prefetch_idx = i + PREFETCH_N_VECS;
      if (prefetch_idx < n_vecs) {
        size_t prefetch_n_vecs = smin(n_vecs - prefetch_idx, PREFETCH_N_VECS);
        l2fetch(pv_in + PREFETCH_N_VECS, VLEN, VLEN, prefetch_n_vecs, 0);
      }
    }

    *pv_out++ = *pv_in++;
  }
}

static void transfer_permuted_weight_fp16_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  permuted_weight_transfer_fp16_task_state_t *state = (permuted_weight_transfer_fp16_task_state_t *) data;

  int k = state->k;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }

    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    int           c        = chunk_idx * HMX_FP16_TILE_N_COLS;
    __fp16       *vtcm_dst = state->dst + c * k;
    const __fp16 *src      = state->src + c * k;
    transfer_permuted_weight_fp16_task(vtcm_dst, src, k, chunk_size);
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void transfer_permuted_weight_chunk_fp16(__fp16 *vtcm_dst, const __fp16 *src, int n_cols, int k) {
  // NOTE(hzx): weight matrix is already transposed. n_cols actually turns into n_rows
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  const bool use_dma = true;

  if (use_dma) {
    size_t size = n_cols * k * sizeof(__fp16);

    dma_desc_1d_t desc;
    dma_issue_load_from_ddr(&desc, vtcm_dst, src, size);
    dma_wait_for_idle();

    return;
  }

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = n_cols / HMX_FP16_TILE_N_COLS;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);
  // size_t n_chunks_per_task = 1;

  permuted_weight_transfer_fp16_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.k   = k;
  state.dst = vtcm_dst;
  state.src = src;

  worker_pool_job_t job;
  job.fptr = transfer_permuted_weight_fp16_worker_loop;
  job.dptr = &state;

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  // NOTE: n_tot_chunks = number of total super-blocks
  __fp16        *dst;
  const void    *src;
  enum ggml_type quant_type;
  bool           src_in_vtcm;
  int            k;  // NOTE(hzx): only used in non-pre-permuted (common) weight case
} permuted_weight_dequantize_qk_0_hvx_task_state_t;

#define EXPAND_QK_0_VEC_SCALES_COMPUTATION(blk, vs0_c, vs1_c, vs2_c, vs3_c) \
  do {                                                                      \
    __fp16 s0 = blk.scales[0];                                              \
    __fp16 s1 = blk.scales[1];                                              \
    __fp16 s2 = blk.scales[2];                                              \
    __fp16 s3 = blk.scales[3];                                              \
    __fp16 s4 = blk.scales[4];                                              \
    __fp16 s5 = blk.scales[5];                                              \
    __fp16 s6 = blk.scales[6];                                              \
    __fp16 s7 = blk.scales[7];                                              \
                                                                            \
    HVX_Vector vs0 = Q6_Vh_vsplat_R(fp16_to_bits(&s0));                     \
    HVX_Vector vs1 = Q6_Vh_vsplat_R(fp16_to_bits(&s1));                     \
    HVX_Vector vs2 = Q6_Vh_vsplat_R(fp16_to_bits(&s2));                     \
    HVX_Vector vs3 = Q6_Vh_vsplat_R(fp16_to_bits(&s3));                     \
    HVX_Vector vs4 = Q6_Vh_vsplat_R(fp16_to_bits(&s4));                     \
    HVX_Vector vs5 = Q6_Vh_vsplat_R(fp16_to_bits(&s5));                     \
    HVX_Vector vs6 = Q6_Vh_vsplat_R(fp16_to_bits(&s6));                     \
    HVX_Vector vs7 = Q6_Vh_vsplat_R(fp16_to_bits(&s7));                     \
                                                                            \
    vs0_c = Q6_V_valign_VVR(vs1, vs0, 64);                                  \
    vs1_c = Q6_V_valign_VVR(vs3, vs2, 64);                                  \
    vs2_c = Q6_V_valign_VVR(vs5, vs4, 64);                                  \
    vs3_c = Q6_V_valign_VVR(vs7, vs6, 64);                                  \
  } while (0)

static inline HVX_Vector dequantize_single_q4_0_group(const block_q4_0 *group, const HVX_Vector vlut_cvt) {
  HVX_Vector vq = vmemu(&(group->quants));
  HVX_Vector vs = vmemu(&(group->scale));

  HVX_Vector v_scales = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), vs, 0));

  HVX_Vector v_qs_lo = vq;
  HVX_Vector v_qs_hi = Q6_Vub_vlsr_VubR(vq, 4);

  // concat lo & hi --> 32 elements in a group
  HVX_Vector v_lo_rot = Q6_V_vror_VR(v_qs_lo, 16);
  HVX_Vector v_quants = Q6_V_vlalign_VVR(v_qs_hi, v_lo_rot, 16);

  // convert INT4 -> FP16
  HVX_VectorPair vp = Q6_Wh_vlut16_VbVhR_nomatch(v_quants, vlut_cvt, 0);

  HVX_Vector v_group_hf = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp), Q6_V_lo_W(vp), -2));

  // // convert INT4 -> FP16
  // HVX_VectorPair vp_q0 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_lo, vlut_cvt, 0);
  // HVX_VectorPair vp_q1 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_hi, vlut_cvt, 0);

  // HVX_Vector v0 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp_q0), Q6_V_lo_W(vp_q0), -2));  // 16 valid elements
  // HVX_Vector v1 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp_q1), Q6_V_hi_W(vp_q1), -2));  // 16 valid elements

  // // concat v0 & v1 --> 32 elements in a group
  // HVX_Vector v0_rot     = Q6_V_vror_VR(v0, 32);
  // HVX_Vector v_group_hf = Q6_V_vlalign_VVR(v1, v0_rot, 32);

  // dequantize: quants(FP16) * values(FP16)
  v_group_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_group_hf, v_scales));
  return v_group_hf;
}

static inline HVX_Vector dequantize_single_q8_0_group(const block_q8_0 *group) {
  HVX_Vector vq = vmemu(&(group->quants));
  HVX_Vector vs = vmemu(&(group->scale));

  HVX_Vector v_scales = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), vs, 0));

  HVX_Vector v0         = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vq));
  HVX_Vector v_group_hf = Q6_Vhf_equals_Vh(v0);

  // dequantize: quants(FP16) * values(FP16)
  v_group_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v_group_hf, v_scales));
  return v_group_hf;
}

void dequantize_permuted_weight_q4_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst, const my_block_q4_0 *restrict src,
                                                      int n_blocks, bool src_in_vtcm, bool is_iq4_nl) {
  const int L2_PREFETCH_N_BLOCKS = 32;  // ~ 4K
  const int DC_PREFETCH_N_BLOCKS = 4;

  const bool no_group_coalesce = false;
  const bool no_dequantization = false;

  const HVX_Vector vlut_cvt = is_iq4_nl ? vmem(iq4_nl_to_fp16_lut) : vmem(q4_0_to_fp16_lut);

  static const uint8_t vlut_scales_idx_data[128] __attribute__((aligned(VLEN))) = {
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
  };
  const HVX_Vector vlut_scales_idx0 = vmem(vlut_scales_idx_data);
  const HVX_Vector vlut_scales_idx1 = Q6_Vb_vadd_VbVb(vlut_scales_idx0, Q6_Vb_vsplat_R(4));

  HVX_Vector *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_blocks; ++i) {
    if (!src_in_vtcm && false) {
      if (i % L2_PREFETCH_N_BLOCKS == 0) {
        int prefetch_idx = i + L2_PREFETCH_N_BLOCKS;
        if (prefetch_idx < n_blocks) {
          size_t prefetch_n_blocks = smin(n_blocks - prefetch_idx, L2_PREFETCH_N_BLOCKS);
          l2fetch(src + prefetch_idx, sizeof(my_block_q4_0), sizeof(my_block_q4_0), prefetch_n_blocks, 0);
        }
      }

      if (i + DC_PREFETCH_N_BLOCKS < n_blocks) {
        Q6_dcfetch_A((void *) &(src[i + DC_PREFETCH_N_BLOCKS].scales));
      }
    }

    if (no_group_coalesce) {
      const block_q4_0 *groups = (const block_q4_0 *) (src + i);

      HVX_Vector v_g0 = dequantize_single_q4_0_group(groups + 0, vlut_cvt);
      HVX_Vector v_g1 = dequantize_single_q4_0_group(groups + 1, vlut_cvt);
      HVX_Vector v_g2 = dequantize_single_q4_0_group(groups + 2, vlut_cvt);
      HVX_Vector v_g3 = dequantize_single_q4_0_group(groups + 3, vlut_cvt);
      HVX_Vector v_g4 = dequantize_single_q4_0_group(groups + 4, vlut_cvt);
      HVX_Vector v_g5 = dequantize_single_q4_0_group(groups + 5, vlut_cvt);
      HVX_Vector v_g6 = dequantize_single_q4_0_group(groups + 6, vlut_cvt);
      HVX_Vector v_g7 = dequantize_single_q4_0_group(groups + 7, vlut_cvt);

      HVX_Vector v_g0_rot = Q6_V_vror_VR(v_g0, 64);
      HVX_Vector v_g2_rot = Q6_V_vror_VR(v_g2, 64);
      HVX_Vector v_g4_rot = Q6_V_vror_VR(v_g4, 64);
      HVX_Vector v_g6_rot = Q6_V_vror_VR(v_g6, 64);

      HVX_Vector v0 = Q6_V_vlalign_VVR(v_g1, v_g0_rot, 64);
      HVX_Vector v1 = Q6_V_vlalign_VVR(v_g3, v_g2_rot, 64);
      HVX_Vector v2 = Q6_V_vlalign_VVR(v_g5, v_g4_rot, 64);
      HVX_Vector v3 = Q6_V_vlalign_VVR(v_g7, v_g6_rot, 64);

      *pv_out++ = v0;
      *pv_out++ = v1;
      *pv_out++ = v2;
      *pv_out++ = v3;
      continue;
    }

    HVX_Vector qs = vmemu(src[i].quants);

    if (no_dequantization) {
      *pv_out++ = qs;
      *pv_out++ = qs;
      *pv_out++ = qs;
      *pv_out++ = qs;
      continue;
    }

    HVX_Vector v_qs_lo = qs;  // no need to mask out high 4 bits in each byte since vlut will do that for us
    HVX_Vector v_qs_hi = Q6_Vub_vlsr_VubR(qs, 4);

    HVX_VectorPair vp_q0 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_lo, vlut_cvt, 0);
    HVX_VectorPair vp_q1 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_hi, vlut_cvt, 0);

    // NOTE(hzx): the previous scalar->vector scales implementation is faster when src resides in DDR memory
    // HVX_Vector vs0_c, vs1_c, vs2_c, vs3_c;
    // EXPAND_QK_0_VEC_SCALES_COMPUTATION(src[i], vs0_c, vs1_c, vs2_c, vs3_c);

    HVX_Vector v_packed_scales = vmemu(src[i].scales);
    HVX_Vector vlut_scales     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_packed_scales));

    HVX_VectorPair vp_s0 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx0, vlut_scales, 0);
    HVX_VectorPair vp_s1 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx1, vlut_scales, 0);

    HVX_Vector vs0_c = Q6_V_lo_W(vp_s0), vs1_c = Q6_V_hi_W(vp_s0);
    HVX_Vector vs2_c = Q6_V_lo_W(vp_s1), vs3_c = Q6_V_hi_W(vp_s1);

    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_q0), vs0_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_q0), vs1_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_q1), vs2_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(vp_q1), vs3_c));
  }
}

void dequantize_permuted_weight_q8_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst, const my_block_q8_0 *restrict src,
                                                      int n_blocks, bool src_in_vtcm) {
  const int L2_PREFETCH_N_BLOCKS = 16;  // ~ 4K
  const int DC_PREFETCH_N_BLOCKS = 4;

  const bool no_group_coalesce = false;
  const bool no_dequantization = false;

  static const uint8_t vlut_scales_idx_data[128] __attribute__((aligned(VLEN))) = {
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
  };
  const HVX_Vector vlut_scales_idx0 = vmem(vlut_scales_idx_data);
  const HVX_Vector vlut_scales_idx1 = Q6_Vb_vadd_VbVb(vlut_scales_idx0, Q6_Vb_vsplat_R(4));

  HVX_Vector *pv_out = (HVX_Vector *) vtcm_dst;

  for (int i = 0; i < n_blocks; ++i) {
    if (!src_in_vtcm && false) {
      if (i % L2_PREFETCH_N_BLOCKS == 0) {
        int prefetch_idx = i + L2_PREFETCH_N_BLOCKS;
        if (prefetch_idx < n_blocks) {
          size_t prefetch_n_blocks = smin(n_blocks - prefetch_idx, L2_PREFETCH_N_BLOCKS);
          l2fetch(src + prefetch_idx, sizeof(my_block_q8_0), sizeof(my_block_q8_0), prefetch_n_blocks, 0);
        }
      }

      if (i + DC_PREFETCH_N_BLOCKS < n_blocks) {
        Q6_dcfetch_A((void *) &(src[i + DC_PREFETCH_N_BLOCKS].scales));
      }
    }

    if (no_group_coalesce) {
      const block_q8_0 *groups = (const block_q8_0 *) (src + i);

      HVX_Vector v_g0 = dequantize_single_q8_0_group(groups + 0);
      HVX_Vector v_g1 = dequantize_single_q8_0_group(groups + 1);
      HVX_Vector v_g2 = dequantize_single_q8_0_group(groups + 2);
      HVX_Vector v_g3 = dequantize_single_q8_0_group(groups + 3);
      HVX_Vector v_g4 = dequantize_single_q8_0_group(groups + 4);
      HVX_Vector v_g5 = dequantize_single_q8_0_group(groups + 5);
      HVX_Vector v_g6 = dequantize_single_q8_0_group(groups + 6);
      HVX_Vector v_g7 = dequantize_single_q8_0_group(groups + 7);

      HVX_Vector v_g0_rot = Q6_V_vror_VR(v_g0, 64);
      HVX_Vector v_g2_rot = Q6_V_vror_VR(v_g2, 64);
      HVX_Vector v_g4_rot = Q6_V_vror_VR(v_g4, 64);
      HVX_Vector v_g6_rot = Q6_V_vror_VR(v_g6, 64);

      HVX_Vector v0 = Q6_V_vlalign_VVR(v_g1, v_g0_rot, 64);
      HVX_Vector v1 = Q6_V_vlalign_VVR(v_g3, v_g2_rot, 64);
      HVX_Vector v2 = Q6_V_vlalign_VVR(v_g5, v_g4_rot, 64);
      HVX_Vector v3 = Q6_V_vlalign_VVR(v_g7, v_g6_rot, 64);

      *pv_out++ = v0;
      *pv_out++ = v1;
      *pv_out++ = v2;
      *pv_out++ = v3;
      continue;
    }

    HVX_Vector vq0 = vmemu(src[i].quants);
    HVX_Vector vq1 = vmemu(src[i].quants + VLEN);

    if (no_dequantization) {
      *pv_out++ = vq0;
      *pv_out++ = vq0;
      *pv_out++ = vq1;
      *pv_out++ = vq1;
      continue;
    }

    HVX_VectorPair vp0 = Q6_Wh_vunpack_Vb(vq0);
    HVX_VectorPair vp1 = Q6_Wh_vunpack_Vb(vq1);

    HVX_Vector v0 = Q6_Vhf_equals_Vh(Q6_V_lo_W(vp0));
    HVX_Vector v1 = Q6_Vhf_equals_Vh(Q6_V_hi_W(vp0));
    HVX_Vector v2 = Q6_Vhf_equals_Vh(Q6_V_lo_W(vp1));
    HVX_Vector v3 = Q6_Vhf_equals_Vh(Q6_V_hi_W(vp1));

    // HVX_Vector vs0_c, vs1_c, vs2_c, vs3_c;
    // EXPAND_QK_0_VEC_SCALES_COMPUTATION(src[i], vs0_c, vs1_c, vs2_c, vs3_c);

    HVX_Vector v_packed_scales = vmemu(src[i].scales);
    HVX_Vector vlut_scales     = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_packed_scales));

    HVX_VectorPair vp_s0 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx0, vlut_scales, 0);
    HVX_VectorPair vp_s1 = Q6_Wh_vlut16_VbVhR_nomatch(vlut_scales_idx1, vlut_scales, 0);

    HVX_Vector vs0_c = Q6_V_lo_W(vp_s0), vs1_c = Q6_V_hi_W(vp_s0);
    HVX_Vector vs2_c = Q6_V_lo_W(vp_s1), vs3_c = Q6_V_hi_W(vp_s1);

    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v0, vs0_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v1, vs1_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v2, vs2_c));
    *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v3, vs3_c));
  }
}

static void dequantize_permuted_weight_qk_0_to_fp16_hvx_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  permuted_weight_dequantize_qk_0_hvx_task_state_t *state = (permuted_weight_dequantize_qk_0_hvx_task_state_t *) data;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }

    int    chunk_idx  = task_id * state->n_chunks_per_task;
    size_t chunk_size = smin(state->n_tot_chunks - chunk_idx, state->n_chunks_per_task);

    __fp16 *vtcm_dst = state->dst + chunk_idx * QK_K;

    if (state->quant_type == GGML_TYPE_Q4_0 || state->quant_type == GGML_TYPE_IQ4_NL) {
      const my_block_q4_0 *src = ((const my_block_q4_0 *) state->src) + chunk_idx;
      dequantize_permuted_weight_q4_0_to_fp16_hvx_task(vtcm_dst, src, chunk_size, state->src_in_vtcm,
                                                       state->quant_type == GGML_TYPE_IQ4_NL);
    } else if (state->quant_type == GGML_TYPE_Q8_0) {
      const my_block_q8_0 *src = ((const my_block_q8_0 *) state->src) + chunk_idx;
      dequantize_permuted_weight_q8_0_to_fp16_hvx_task(vtcm_dst, src, chunk_size, state->src_in_vtcm);
    }
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

void dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(__fp16 *vtcm_dst, const void *src, int ne, int k,
                                                       enum ggml_type type, void *vtcm_scratch) {
  assert(ne % QK_K == 0);
  (void) k;

  const bool src_in_vtcm = true;

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = ne / QK_K;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);

  permuted_weight_dequantize_qk_0_hvx_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.dst         = vtcm_dst;
  state.src         = src_in_vtcm ? vtcm_scratch : src;
  state.quant_type  = type;
  state.src_in_vtcm = src_in_vtcm;

  worker_pool_job_t job;
  job.fptr = dequantize_permuted_weight_qk_0_to_fp16_hvx_worker_loop;
  job.dptr = &state;

  // int64_t t0 = HAP_perf_get_qtimer_count();

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  // int64_t e = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  // FARF(ALWAYS, "QK_0 dequantize: ne: %d time: %lld us", ne, e);
}

void dequantize_common_weight_q4_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst, const block_q4_0 *restrict src,
                                                    int start_group_idx, int end_group_idx, int k, bool src_in_vtcm,
                                                    bool is_iq4_nl) {
  assert(src_in_vtcm);

  const size_t GROUP_SIZE = QK_0;
  assert(GROUP_SIZE == HMX_FP16_TILE_N_ROWS);

  const size_t N_GROUPS_PER_SCALAR_COLUMN = k / GROUP_SIZE;
  const size_t N_GROUPS_PER_TILE_COLUMN   = N_GROUPS_PER_SCALAR_COLUMN * HMX_FP16_TILE_N_COLS;

  const HVX_Vector vlut_cvt       = is_iq4_nl ? vmem(iq4_nl_to_fp16_lut) : vmem(q4_0_to_fp16_lut);
  const HVX_Vector v_offsets_base = vmem(common_layout_vscatter_offsets_base);

  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

  for (int g_idx = start_group_idx; g_idx < end_group_idx; ++g_idx) {
    const block_q4_0 *group = src + g_idx;

    HVX_Vector v_group_hf = dequantize_single_q4_0_group(group, vlut_cvt);

    // prepare for scatter
    int i0 = g_idx / N_GROUPS_PER_TILE_COLUMN;
    int i1 = g_idx % N_GROUPS_PER_TILE_COLUMN;

    int gr = i1 % N_GROUPS_PER_SCALAR_COLUMN;
    int gc = i1 / N_GROUPS_PER_SCALAR_COLUMN;

    int     tile_idx  = i0 * (k / HMX_FP16_TILE_N_ROWS) + gr;
    __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

    HVX_Vector v_offsets_delta = Q6_V_vsplat_R(gc * 4);
    HVX_Vector v_offsets       = Q6_Vw_vadd_VwVw(v_offsets_base, v_offsets_delta);
    Q6_vscatter_QRMVwV(q_32_elems_mask, (size_t) tile_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_group_hf);
  }
}

void dequantize_common_weight_q8_0_to_fp16_hvx_task(__fp16 *restrict vtcm_dst, const block_q8_0 *restrict src,
                                                    int start_group_idx, int end_group_idx, int k, bool src_in_vtcm) {
  const size_t GROUP_SIZE = QK_0;
  assert(GROUP_SIZE == HMX_FP16_TILE_N_ROWS);

  const size_t N_GROUPS_PER_SCALAR_COLUMN = k / GROUP_SIZE;
  const size_t N_GROUPS_PER_TILE_COLUMN   = N_GROUPS_PER_SCALAR_COLUMN * HMX_FP16_TILE_N_COLS;

  const HVX_Vector v_offsets_base = vmem(common_layout_vscatter_offsets_base);

  const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

  for (int g_idx = start_group_idx; g_idx < end_group_idx; ++g_idx) {
    const block_q8_0 *group = src + g_idx;

    HVX_Vector v_group_hf = dequantize_single_q8_0_group(group);

    // prepare for scatter
    int i0 = g_idx / N_GROUPS_PER_TILE_COLUMN;
    int i1 = g_idx % N_GROUPS_PER_TILE_COLUMN;

    int gr = i1 % N_GROUPS_PER_SCALAR_COLUMN;
    int gc = i1 / N_GROUPS_PER_SCALAR_COLUMN;

    int     tile_idx  = i0 * (k / HMX_FP16_TILE_N_ROWS) + gr;
    __fp16 *tile_base = vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS;

    HVX_Vector v_offsets_delta = Q6_V_vsplat_R(gc * 4);
    HVX_Vector v_offsets       = Q6_Vw_vadd_VwVw(v_offsets_base, v_offsets_delta);
    Q6_vscatter_QRMVwV(q_32_elems_mask, (size_t) tile_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_group_hf);
  }
}

static void dequantize_common_weight_chunk_qk_0_to_fp16_hvx_worker_loop(void *data, int _worker_index) {
  (void) _worker_index;
  permuted_weight_dequantize_qk_0_hvx_task_state_t *state = (permuted_weight_dequantize_qk_0_hvx_task_state_t *) data;

  const int     k        = state->k;
  __fp16 *const vtcm_dst = state->dst;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
    if (task_id >= state->n_tasks) {
      break;
    }

    size_t start_idx = task_id * state->n_chunks_per_task;
    size_t end_idx   = smin(start_idx + state->n_chunks_per_task, state->n_tot_chunks);

    if (state->quant_type == GGML_TYPE_Q4_0 || state->quant_type == GGML_TYPE_IQ4_NL) {
      const block_q4_0 *src = (const block_q4_0 *) state->src;
      dequantize_common_weight_q4_0_to_fp16_hvx_task(vtcm_dst, src, start_idx, end_idx, k, state->src_in_vtcm,
                                                     state->quant_type == GGML_TYPE_IQ4_NL);
    } else if (state->quant_type == GGML_TYPE_Q8_0) {
      const block_q8_0 *src = (const block_q8_0 *) state->src;
      dequantize_common_weight_q8_0_to_fp16_hvx_task(vtcm_dst, src, start_idx, end_idx, k, state->src_in_vtcm);
    }
  }

  worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

void dequantize_common_weight_chunk_qk_0_to_fp16_hvx(__fp16 *vtcm_dst, const void *src, int ne, int k,
                                                     enum ggml_type type, void *vtcm_scratch) {
  assert(ne % QK_0 == 0);
  assert(k % QK_0 == 0);

  const bool src_in_vtcm = true;

  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = ne / QK_0;
  size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);

  // NOTE: reuse the task state type for now
  permuted_weight_dequantize_qk_0_hvx_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.dst         = vtcm_dst;
  state.src         = src_in_vtcm ? vtcm_scratch : src;
  state.quant_type  = type;
  state.src_in_vtcm = src_in_vtcm;
  state.k           = k;

  worker_pool_job_t job;
  job.fptr = dequantize_common_weight_chunk_qk_0_to_fp16_hvx_worker_loop;
  job.dptr = &state;

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));
}

static void core_dot_chunk_fp16(__fp16 *output, const __fp16 *activation, const __fp16 *weight, const __fp16 *scales,
                                int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
  hmx_unit_acquire();

  asm volatile("mxclracc.hf");
  hmx_set_output_scales(scales);

  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < n_col_tiles; ++c) {
      const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
      const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < n_dot_tiles; k += 32) {
        int    offset  = k * HMX_FP16_TILE_N_ELMS;
        size_t n_tiles = smin(n_dot_tiles - k, 32);
        hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
      }

      __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
      hmx_consume_accumulator_fp16(out_tile);
    }
  }

  hmx_unit_release();
}

// TODO(hzx): current implementation only use one thread. Use multiple threads to improve prefill performance
static void transfer_output_chunk_fp16_to_fp32(float *restrict dst, const __fp16 *restrict vtcm_src, int n_rows,
                                               int n_cols, int n) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

  for (int r = 0; r < n_rows; r += 2) {
    int r0 = r / HMX_FP16_TILE_N_ROWS;
    int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      int c0 = c / HMX_FP16_TILE_N_COLS;

      const __fp16 *tile = vtcm_src + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;

      HVX_Vector v_src = ((const HVX_Vector *) tile)[r1 / 2];

      HVX_VectorPair vp = hvx_my_vhf_to_wsf(v_src);

      HVX_Vector *pv_out0 = (HVX_Vector *) (dst + (r * n + c + 0));
      HVX_Vector *pv_out1 = (HVX_Vector *) (dst + (r * n + c + n));  // next row in global memory

      *pv_out0 = Q6_V_lo_W(vp);
      if (r + 1 < n_rows) {
        *pv_out1 = Q6_V_hi_W(vp);
      }
    }
  }
}

static inline int hmx_i8_tile_idx_32(int row, int col) {
  return (row & ~1) * 32 + col * 2 + (row & 1);
}

static inline int hmx_i8_decode_uh_signed(uint16_t raw) {
  return raw < 32768 ? (int) raw : (int) raw - 65536;
}

static inline int8_t quantize_f32_to_i8_symmetric(float value, float scale) {
  if (scale == 0.0f) {
    return 0;
  }
  float scaled = value / scale;
  int   q      = (int) (scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
  if (q > 127) {
    q = 127;
  } else if (q < -127) {
    q = -127;
  }
  return (int8_t) q;
}

static inline int64_t matmul_trace_now_us(void) {
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
}

static inline bool matmul_trace_enabled(int mode_flags) {
  return (mode_flags & LLM_NPU_MODE_TRACE) != 0;
}

static inline bool matmul_detailed_trace_enabled(int mode_flags, struct LlmTraceProfileHeader *profile) {
  return (mode_flags & LLM_NPU_MODE_DETAILED_TRACE) != 0 && profile != NULL &&
         profile->magic == LLM_TRACE_PROFILE_MAGIC && profile->max_events > 0;
}

static void record_matmul_stage(struct LlmTraceProfileHeader *profile, int64_t trace_id, int mode_flags, int op_index,
                                int stage, int unit, int worker, int m, int k, int n, int mr, int nc, int kk,
                                int chunk_m, int chunk_n, int chunk_k, int64_t bytes, int64_t t0_us, int64_t t1_us) {
  if (!matmul_detailed_trace_enabled(mode_flags, profile)) {
    return;
  }
  if (t1_us < t0_us) {
    t1_us = t0_us;
  }

  const int idx = __sync_fetch_and_add(&(profile->event_count), 1);
  if (idx < 0 || idx >= profile->max_events) {
    __sync_fetch_and_add(&(profile->event_overflow), 1);
    return;
  }

  struct LlmTraceProfileEvent *events = llm_trace_profile_events(profile);
  events[idx]                        = (struct LlmTraceProfileEvent) {
                           .trace_id   = trace_id,
                           .op_index   = op_index,
                           .stage      = stage,
                           .unit       = unit,
                           .worker     = worker,
                           .m          = m,
                           .k          = k,
                           .n          = n,
                           .qo_len     = 0,
                           .kv_len     = 0,
                           .n_heads    = 0,
                           .n_kv_heads = 0,
                           .head_dim   = 0,
                           .mr         = mr,
                           .nc         = nc,
                           .kk         = kk,
                           .chunk_m    = chunk_m,
                           .chunk_n    = chunk_n,
                           .chunk_k    = chunk_k,
                           .flags      = mode_flags,
                           .bytes      = bytes,
                           .t0_us      = t0_us,
                           .t1_us      = t1_us,
                           .dur_us     = t1_us - t0_us,
  };
}

static void log_w8pc_a8pt_event(int64_t trace_id, int mode_flags, const char *phase, int m, int k, int n,
                                size_t input_bytes, size_t output_bytes, int64_t t0_us, int64_t t1_us) {
  if (!matmul_trace_enabled(mode_flags)) {
    return;
  }
  FARF(ALWAYS,
       "LLMTRACE_DSP_EVENT trace_id=%lld flags=%d op=matmul_w8pc_a8pt op_index=%u phase=%s m=%d k=%d n=%d "
       "qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d input_bytes=%llu output_bytes=%llu t0_us=%lld "
       "t1_us=%lld dur_us=%lld",
       (long long) trace_id, mode_flags, HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT, phase, m, k, n, 0, 0, 0, 0, 0,
       (unsigned long long) input_bytes, (unsigned long long) output_bytes, (long long) t0_us, (long long) t1_us,
       (long long) (t1_us - t0_us));
}

static inline void hmx_i8_load_tiles_ub_b(const uint8_t *row_tiles, const int8_t *col_tiles) {
  asm volatile(
    "{ activation.ub = mxmem(%0, %1):above\n"
    "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
    "r"(HMX_FP16_TILE_SIZE - 1), "r"(col_tiles), "r"(HMX_FP16_TILE_SIZE - 1)
    : "memory");
}

static inline void hmx_i8_store_accumulator_uh_after_2x1(uint16_t *out) {
  asm volatile("mxmem(%0, %1):after.uh = acc:2x1" ::"r"(out), "r"(0) : "memory");
}

static void hmx_i8_raw_dot_tile_b(const uint8_t *a, const int8_t *b, uint16_t *out, const __fp16 *scales) {
  memset(out, 0, HMX_FP16_TILE_SIZE);
  asm volatile("mxclracc" ::: "memory");
  hmx_set_output_scales(scales);
  hmx_i8_load_tiles_ub_b(a, b);
  hmx_i8_store_accumulator_uh_after_2x1(out);
}

static void hmx_i8_pack_activation_mag_k2(uint8_t *dst, const int8_t *src, int k_base, bool negative) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < 32; ++r) {
    for (int kk = 0; kk < 2; ++kk) {
      const int    k = k_base + kk;
      const int8_t v = src[r * 32 + k];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag  = v < 0 ? -v : v;
      const int base = 128 * (r / 2) + 4 * kk + 2 * (r & 1) + 1;
      dst[base]      = (uint8_t) (mag << 1);
    }
  }
}

static void hmx_i8_pack_weight_b_full_k2(int8_t *dst, const int8_t *src, int k_base) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int kk = 0; kk < 2; ++kk) {
    const int k = k_base + kk;
    for (int c = 0; c < 32; ++c) {
      dst[4 * c + kk] = src[k * 32 + c];
    }
  }
}

static void hmx_i8_accumulate_uh_tile_unshifted(int32_t *acc, const uint16_t *out, int sign) {
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      const int v = hmx_i8_decode_uh_signed(out[hmx_i8_tile_idx_32(r, c)]);
      acc[r * 32 + c] += sign * v;
    }
  }
}

static void __attribute__((unused)) hmx_i8_tile_full_weight_k2_accumulate(uint8_t *a_pos, uint8_t *a_neg,
                                                                          int8_t *w_full, uint16_t *out,
                                                                          const __fp16 *scales, int32_t *acc,
                                                                          const int8_t *a_ref, const int8_t *w_ref) {
  for (int k_base = 0; k_base < 32; k_base += 2) {
    hmx_i8_pack_activation_mag_k2(a_pos, a_ref, k_base, false);
    hmx_i8_pack_activation_mag_k2(a_neg, a_ref, k_base, true);
    hmx_i8_pack_weight_b_full_k2(w_full, w_ref, k_base);

    hmx_i8_raw_dot_tile_b(a_pos, w_full, out, scales);
    hmx_i8_accumulate_uh_tile_unshifted(acc, out, 1);

    hmx_i8_raw_dot_tile_b(a_neg, w_full, out, scales);
    hmx_i8_accumulate_uh_tile_unshifted(acc, out, -1);
  }
}

static void fill_a8pt_tile(int8_t *a_ref, const float *activation, int m, int k, int mr, int kt,
                           const float *scale_a) {
  for (int r = 0; r < 32; ++r) {
    const int global_r = mr + r;
    for (int kk = 0; kk < 32; ++kk) {
      const int global_k = kt + kk;
      int8_t    q        = 0;
      if (global_r < m && global_k < k) {
        q = quantize_f32_to_i8_symmetric(activation[global_r * k + global_k], scale_a[r]);
      }
      a_ref[r * 32 + kk] = q;
    }
  }
}

static void __attribute__((unused)) fill_w8pc_tile(int8_t *w_ref, float *scale_w, const uint8_t *permuted_weight, int k,
                                                   int n, int nc, int kt) {
  const int k_tiles = k / 32;
  const int n_tile  = nc / 32;
  const int k_tile  = kt / 32;

  const block_q8_0 *blocks = (const block_q8_0 *) permuted_weight;
  const size_t      base   = ((size_t) n_tile * (size_t) k_tiles + (size_t) k_tile) * 32u;

  (void) n;
  for (int c = 0; c < 32; ++c) {
    const block_q8_0 *blk = blocks + base + (size_t) c;
    scale_w[c]            = (float) blk->scale;
    for (int kk = 0; kk < 32; ++kk) {
      w_ref[kk * 32 + c] = blk->quants[kk];
    }
  }
}

static inline void __attribute__((unused)) hvx_accumulate_i16_pair_to_i32(HVX_Vector *acc4,
                                                                          HVX_VectorPair partial) {
  HVX_VectorPair lo_words = Q6_Ww_vunpack_Vh(Q6_V_lo_W(partial));
  HVX_VectorPair hi_words = Q6_Ww_vunpack_Vh(Q6_V_hi_W(partial));
  acc4[0] = Q6_Vw_vadd_VwVw(acc4[0], Q6_V_lo_W(lo_words));
  acc4[1] = Q6_Vw_vadd_VwVw(acc4[1], Q6_V_hi_W(lo_words));
  acc4[2] = Q6_Vw_vadd_VwVw(acc4[2], Q6_V_lo_W(hi_words));
  acc4[3] = Q6_Vw_vadd_VwVw(acc4[3], Q6_V_hi_W(hi_words));
}

static inline HVX_VectorPair __attribute__((unused)) hvx_i8_product_ordered(HVX_Vector weights, int8_t activation) {
  HVX_VectorPair raw = Q6_Wh_vmpy_VbVb(weights, Q6_Vb_vsplat_R(activation));
  return Q6_W_vshuff_VVR(Q6_V_hi_W(raw), Q6_V_lo_W(raw), -2);
}

static void __attribute__((unused)) fill_w8pc_scales_128(float *scale_w, const uint8_t *permuted_weight, int k, int n,
                                                         int nc) {
  const int       k_tiles = k / 32;
  const block_q8_0 *blocks  = (const block_q8_0 *) permuted_weight;

  for (int c = 0; c < 128; ++c) {
    const int global_c = nc + c;
    if (global_c >= n) {
      scale_w[c] = 0.0f;
      continue;
    }
    const int n_tile = global_c / 32;
    const int c_tile = global_c & 31;
    scale_w[c] = (float) blocks[((size_t) n_tile * (size_t) k_tiles) * 32u + (size_t) c_tile].scale;
  }
}

static void fill_w8pc_tile_32(int8_t *w_ref, float *scale_w, const uint8_t *permuted_weight, int k, int n, int nc,
                              int kt) {
  const int       k_tiles = k / 32;
  const int       n_tile  = nc / 32;
  const int       k_tile  = kt / 32;
  const block_q8_0 *blocks  = (const block_q8_0 *) permuted_weight;

  (void) n;
  for (int c = 0; c < 32; ++c) {
    const block_q8_0 *blk = blocks + ((size_t) n_tile * (size_t) k_tiles + (size_t) k_tile) * 32u + (size_t) c;
    if (kt == 0) {
      scale_w[c] = (float) blk->scale;
    }
    for (int kk = 0; kk < 32; ++kk) {
      w_ref[kk * 32 + c] = (int8_t) blk->quants[kk];
    }
  }
}

static void __attribute__((unused)) fill_w8pc_line_128(int8_t *line, const uint8_t *permuted_weight, int k, int n,
                                                       int nc, int global_k) {
  const int       k_tiles = k / 32;
  const int       k_tile  = global_k / 32;
  const int       kk      = global_k & 31;
  const block_q8_0 *blocks  = (const block_q8_0 *) permuted_weight;

  for (int c = 0; c < 128; ++c) {
    const int global_c = nc + c;
    if (global_c >= n) {
      line[c] = 0;
      continue;
    }
    const int n_tile = global_c / 32;
    const int c_tile = global_c & 31;
    const size_t block_idx = ((size_t) n_tile * (size_t) k_tiles + (size_t) k_tile) * 32u + (size_t) c_tile;
    line[c] = blocks[block_idx].quants[kk];
  }
}

static inline int8_t __attribute__((unused)) a8pt_tile_value(const int8_t *a_tiles, int row, int global_k) {
  const int k_tile = global_k / 32;
  const int kk     = global_k & 31;
  return a_tiles[(size_t) k_tile * 32u * 32u + (size_t) row * 32u + (size_t) kk];
}

static void hvx_dequant_store_w8pc_a8pt_tile(float *restrict dst, const int32_t *restrict acc,
                                             const float *restrict scale_a, const float *restrict scale_w, int m,
                                             int n, int mr, int nc) {
  const HVX_Vector v_scale_w = vmem((const HVX_Vector *) scale_w);
  for (int r = 0; r < 32; ++r) {
    const int global_r = mr + r;
    if (global_r >= m) {
      continue;
    }

    const float      sa   = scale_a[r];
    const HVX_Vector v_sa = Q6_V_vsplat_R(*(const int32_t *) &sa);
    const HVX_Vector v_acc =
      vmem((const HVX_Vector *) (acc + (size_t) r * 32u));
    const HVX_Vector v_acc_f = Q6_Vsf_equals_Vw(v_acc);
    const HVX_Vector v_scaled_a =
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_acc_f, v_sa));
    const HVX_Vector v_out =
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_scaled_a, v_scale_w));

    vmem((HVX_Vector *) (dst + (size_t) global_r * (size_t) n + (size_t) nc)) = v_out;
  }
}

int hmx_mat_mul_permuted_w8pc_a8pt(float *restrict dst, const float *restrict activation,
                                   const uint8_t *restrict permuted_weight, int m, int k, int n, int64_t trace_id,
                                   int mode_flags, int op_index, struct LlmTraceProfileHeader *profile) {
  static int debug_log_count = 0;

  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) || !is_aligned(permuted_weight, VLEN)) {
    return -1;
  }

  const size_t input_bytes  = (size_t) m * (size_t) k * sizeof(float) + (size_t) n * (size_t) k / QK_0 * sizeof(block_q8_0);
  const size_t output_bytes = (size_t) m * (size_t) n * sizeof(float);

  uint8_t    *vtcm_ptr = (uint8_t *) vtcm_manager_get_vtcm_base();
  int8_t     *a_tile   = (int8_t *) vtcm_seq_alloc(&vtcm_ptr, 32u * 32u);
  int8_t     *w_tile   = (int8_t *) vtcm_seq_alloc(&vtcm_ptr, 32u * 32u);
  uint8_t    *a_pos    = (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  uint8_t    *a_neg    = (uint8_t *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  int8_t     *w_full   = (int8_t *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  uint16_t   *hmx_out  = (uint16_t *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  int32_t    *acc      = (int32_t *) vtcm_seq_alloc(&vtcm_ptr, 32u * 32u * sizeof(int32_t));
  float      *scale_a  = (float *) vtcm_seq_alloc(&vtcm_ptr, 32 * sizeof(float));
  float      *scale_w  = (float *) vtcm_seq_alloc(&vtcm_ptr, 32 * sizeof(float));
  __fp16     *hmx_scales = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(hmx_scales, Q6_V_vsplat_R(0x3c00));

  int64_t activation_quantize_us = 0;
  int64_t activation_pack_us     = 0;
  int64_t weight_load_us         = 0;
  int64_t hmx_i8_dot_us          = 0;
  int64_t dequant_store_us       = 0;
  const int64_t trace_t0         = matmul_trace_now_us();

  hmx_unit_acquire();

  for (int mr = 0; mr < m; mr += 32) {
    int64_t tq0 = matmul_trace_now_us();
    for (int r = 0; r < 32; ++r) {
      const int global_r = mr + r;
      float     max_abs  = 0.0f;
      if (global_r < m) {
        const float *row = activation + (size_t) global_r * (size_t) k;
        for (int kk = 0; kk < k; ++kk) {
          const float av = fabsf(row[kk]);
          if (av > max_abs) {
            max_abs = av;
          }
        }
      }
      scale_a[r] = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
    }
    int64_t tq1 = matmul_trace_now_us();
    activation_quantize_us += tq1 - tq0;
    record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_ACTIVATION_QUANTIZE,
                        LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, -1, -1, smin(m - mr, 32), 0, k,
                        (int64_t) smin(m - mr, 32) * k * (int64_t) sizeof(float), tq0, tq1);

    for (int nc = 0; nc < n; nc += 32) {
      for (int i = 0; i < 32 * 32; ++i) {
        acc[i] = 0;
      }

      for (int kt = 0; kt < k; kt += 32) {
        int64_t ta0 = matmul_trace_now_us();
        fill_a8pt_tile(a_tile, activation, m, k, mr, kt, scale_a);
        int64_t ta1 = matmul_trace_now_us();
        activation_pack_us += ta1 - ta0;
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_ACTIVATION_PACK,
                            LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, nc, kt, smin(m - mr, 32), 32, 32,
                            32LL * 32LL, ta0, ta1);

        int64_t tw0 = matmul_trace_now_us();
        fill_w8pc_tile_32(w_tile, scale_w, permuted_weight, k, n, nc, kt);
        int64_t tw1 = matmul_trace_now_us();
        weight_load_us += tw1 - tw0;
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_HVX_LOAD,
                            LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, nc, kt, smin(m - mr, 32), 32, 32,
                            32LL * 32LL, tw0, tw1);

        int64_t th0 = matmul_trace_now_us();
        hmx_i8_tile_full_weight_k2_accumulate(a_pos, a_neg, w_full, hmx_out, hmx_scales, acc, a_tile, w_tile);
        int64_t th1 = matmul_trace_now_us();
        hmx_i8_dot_us += th1 - th0;
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_HMX_MMA, LLM_TRACE_UNIT_HMX,
                            -1, m, k, n, mr, nc, kt, smin(m - mr, 32), 32, 32, 32LL * 32LL * 2LL, th0, th1);
      }

      int64_t td0 = matmul_trace_now_us();
      hvx_dequant_store_w8pc_a8pt_tile(dst, acc, scale_a, scale_w, m, n, mr, nc);
      int64_t td1 = matmul_trace_now_us();
      dequant_store_us += td1 - td0;
      record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_OUTPUT_STORE, LLM_TRACE_UNIT_STORE,
                          -1, m, k, n, mr, nc, -1, smin(m - mr, 32), 32, 0,
                          (int64_t) smin(m - mr, 32) * 32LL * (int64_t) sizeof(float), td0, td1);
    }
  }

  hmx_unit_release();

  if (matmul_trace_enabled(mode_flags) && debug_log_count < 16) {
    float min_v = dst[0];
    float max_v = dst[0];
    int   nan_count = 0;
    const int total = m * n;
    for (int i = 0; i < total; ++i) {
      const float v = dst[i];
      if (isnan(v) || isinf(v)) {
        ++nan_count;
        continue;
      }
      if (v < min_v) {
        min_v = v;
      }
      if (v > max_v) {
        max_v = v;
      }
    }
    FARF(ALWAYS,
         "W8PC_A8PT_DEBUG trace_id=%lld idx=%d m=%d k=%d n=%d first8=%g,%g,%g,%g,%g,%g,%g,%g min=%g max=%g "
         "nan=%d",
         (long long) trace_id, debug_log_count, m, k, n, dst[0], total > 1 ? dst[1] : 0.0f,
         total > 2 ? dst[2] : 0.0f, total > 3 ? dst[3] : 0.0f, total > 4 ? dst[4] : 0.0f,
         total > 5 ? dst[5] : 0.0f, total > 6 ? dst[6] : 0.0f, total > 7 ? dst[7] : 0.0f, min_v, max_v,
         nan_count);
    ++debug_log_count;
  }

  int64_t phase_t0 = trace_t0;
  log_w8pc_a8pt_event(trace_id, mode_flags, "activation_quantize", m, k, n, input_bytes, output_bytes, phase_t0,
                      phase_t0 + activation_quantize_us);
  phase_t0 += activation_quantize_us;
  log_w8pc_a8pt_event(trace_id, mode_flags, "activation_tile_pack", m, k, n, input_bytes, output_bytes, phase_t0,
                      phase_t0 + activation_pack_us);
  phase_t0 += activation_pack_us;
  log_w8pc_a8pt_event(trace_id, mode_flags, "weight_load", m, k, n, input_bytes, output_bytes, phase_t0,
                      phase_t0 + weight_load_us);
  phase_t0 += weight_load_us;
  log_w8pc_a8pt_event(trace_id, mode_flags, "hmx_i8_dot_k2", m, k, n, input_bytes, output_bytes, phase_t0,
                      phase_t0 + hmx_i8_dot_us);
  phase_t0 += hmx_i8_dot_us;
  log_w8pc_a8pt_event(trace_id, mode_flags, "dequant_store", m, k, n, input_bytes, output_bytes, phase_t0,
                      phase_t0 + dequant_store_us);
  return 0;
}

int hmx_mat_mul_permuted_w16a32(float *restrict dst, const float *restrict activation,
                                const __fp16 *restrict permuted_weight, int m, int k, int n, int64_t trace_id,
                                int mode_flags, int op_index, struct LlmTraceProfileHeader *profile) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    // TODO(hzx): can we remove this restriction?
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) || !is_aligned(permuted_weight, VLEN)) {
    return -1;
  }

  const size_t weight_area_size     = WEIGHT_AREA_SIZE;
  const size_t activation_area_size = ACTIVATION_AREA_SIZE;
  const size_t output_area_size     = OUTPUT_AREA_SIZE;
  const size_t required_vtcm_size   = weight_area_size + activation_area_size + output_area_size + 256;
  if (matmul_check_vtcm_capacity(__func__, required_vtcm_size) != 0) {
    return -1;
  }

  // VTCM layout: weight | activation | output | scales
  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  size_t vec_dot_size       = k * sizeof(__fp16);
  size_t m_chunk_max_n_rows = align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  size_t n_chunk_max_n_cols = align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  // FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
  if (m_chunk_n_rows == 0 || n_chunk_n_cols == 0) {
    FARF(ALWAYS, "%s: invalid chunk size, m_max=%u, n_max=%u, out=%u KiB", __func__,
         (unsigned) m_chunk_max_n_rows, (unsigned) n_chunk_max_n_cols, (unsigned) (output_area_size / 1024));
    return -1;
  }

  // int64_t activation_load_time, weight_load_time, hmx_core_time, output_store_time;
  // activation_load_time = weight_load_time = hmx_core_time = output_store_time = 0;

  for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
    // transfer activation matrix chunk into VTCM
    size_t n_rows = smin(m - mr, m_chunk_n_rows);

    // int64_t act_t0 = HAP_perf_get_qtimer_count();
    int64_t act_t0_us = matmul_trace_now_us();
    {
      const float *activation_chunk = activation + mr * k;
      transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
    }
    int64_t act_t1_us = matmul_trace_now_us();
    record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_ACTIVATION_HVX_LOAD,
                        LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, -1, -1, n_rows, 0, k,
                        (int64_t) n_rows * k * (int64_t) sizeof(float), act_t0_us, act_t1_us);
    // activation_load_time += HAP_perf_get_qtimer_count() - act_t0;

    // FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);

    for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
      size_t n_cols = smin(n - nc, n_chunk_n_cols);

      // int64_t wei_t0 = HAP_perf_get_qtimer_count();
      int64_t wei_t0_us = matmul_trace_now_us();
      {
        const __fp16 *permuted_weight_chunk = permuted_weight + nc * k;
        transfer_permuted_weight_chunk_fp16(vtcm_weight, permuted_weight_chunk, n_cols, k);
      }
      int64_t wei_t1_us = matmul_trace_now_us();
      record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_HVX_LOAD,
                          LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, nc, -1, n_rows, n_cols, k,
                          (int64_t) n_cols * k * (int64_t) sizeof(__fp16), wei_t0_us, wei_t1_us);
      // weight_load_time += HAP_perf_get_qtimer_count() - wei_t0;

      // FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

      // int64_t core_t0 = HAP_perf_get_qtimer_count();
      int64_t core_t0_us = matmul_trace_now_us();
      {
        const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
        const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
        core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32);
      }
      int64_t core_t1_us = matmul_trace_now_us();
      record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_HMX_MMA, LLM_TRACE_UNIT_HMX, -1,
                          m, k, n, mr, nc, -1, n_rows, n_cols, k, 2LL * n_rows * n_cols * k, core_t0_us,
                          core_t1_us);
      // hmx_core_time += HAP_perf_get_qtimer_count() - core_t0;

      // FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

      // int64_t out_t0 = HAP_perf_get_qtimer_count();
      int64_t out_t0_us = matmul_trace_now_us();
      {
        float *output = dst + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
      }
      int64_t out_t1_us = matmul_trace_now_us();
      record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_OUTPUT_STORE,
                          LLM_TRACE_UNIT_STORE, -1, m, k, n, mr, nc, -1, n_rows, n_cols, 0,
                          (int64_t) n_rows * n_cols * (int64_t) sizeof(float), out_t0_us, out_t1_us);
      // output_store_time += HAP_perf_get_qtimer_count() - out_t0;

      // FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
    }
  }

  // FARF(ALWAYS, "%s: m = %d, k = %d, n = %d", __func__, m, k, n);
  // FARF(ALWAYS, "    activation load: %lld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %lld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %lld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %lld us", HAP_perf_qtimer_count_to_us(output_store_time));

  // size_t weight_size = k * n * sizeof(__fp16);
  // float  bandwidth   = 1e-3 * weight_size / HAP_perf_qtimer_count_to_us(weight_load_time);
  // FARF(ALWAYS, "    weight load bandwidth: %.2f GB/s", bandwidth);

  return 0;
}

extern worker_pool_context_t hmx_worker_pool_ctx;

typedef struct {
  __fp16            *c;
  const __fp16      *a, *b, *s;
  int                n_row_tiles, n_col_tiles, n_dot_tiles;
  worker_synctoken_t sync_ctx;
  struct LlmTraceProfileHeader *profile;
  int64_t            trace_id;
  int                mode_flags;
  int                op_index;
  int                m, k, n;
  int                mr, nc, kk;
  int                chunk_m, chunk_n, chunk_k;
  int64_t            bytes;
} core_dot_fp16_task_state_t;

static void core_dot_fp16_hmx_worker_fn(void *data, int _worker_index) {
  (void) _worker_index;
  core_dot_fp16_task_state_t *st = (core_dot_fp16_task_state_t *) data;

  const int64_t t0_us = matmul_trace_now_us();
  core_dot_chunk_fp16(st->c, st->a, st->b, st->s, st->n_row_tiles, st->n_col_tiles, st->n_dot_tiles);
  const int64_t t1_us = matmul_trace_now_us();
  record_matmul_stage(st->profile, st->trace_id, st->mode_flags, st->op_index, LLM_TRACE_STAGE_HMX_MMA,
                      LLM_TRACE_UNIT_HMX, _worker_index, st->m, st->k, st->n, st->mr, st->nc, st->kk, st->chunk_m,
                      st->chunk_n, st->chunk_k, st->bytes, t0_us, t1_us);

  worker_pool_synctoken_jobdone(&st->sync_ctx);
}

int mat_mul_qk_0_d16a32_out_stationary(float *restrict out, const float *restrict x, const uint8_t *restrict w, int m,
                                       int k, int n, enum ggml_type w_type, int64_t trace_id, int mode_flags,
                                       int op_index, struct LlmTraceProfileHeader *profile);

int hmx_mat_mul_permuted_qk_0_d16a32(float *restrict dst, const float *restrict activation,
                                     const uint8_t *restrict permuted_weight, int m, int k, int n,
                                     enum ggml_type weight_type, int64_t trace_id, int mode_flags, int op_index,
                                     struct LlmTraceProfileHeader *profile) {
  if (!dst || !activation || !permuted_weight || !m || !n || !k) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    // TODO(hzx): can we remove this restriction?
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) || !is_aligned(permuted_weight, VLEN)) {
    return -1;
  }

  // The repaired output-stationary FFN-down path is compile-time opt-in. Its
  // old long-prompt bug was a qweight K-slice addressing error; keep the OS
  // selector tied to the validated FFN-down shape and VTCM capacity sweeps
  // rather than silently changing the generic m/n chunking path below.
  if (HTP_ENABLE_OUT_STATIONARY_PREFILL && m >= 128 && k > n && n > 1024) {
    return mat_mul_qk_0_d16a32_out_stationary(dst, activation, permuted_weight, m, k, n, weight_type, trace_id,
                                              mode_flags, op_index, profile);
  }

  size_t super_block_size = get_super_block_size(weight_type);
  if (super_block_size == 0) {
    return -1;
  }

  const size_t weight_area_size     = WEIGHT_AREA_SIZE;
  const size_t activation_area_size = ACTIVATION_AREA_SIZE;
  const size_t output_area_size     = OUTPUT_AREA_SIZE;
  const size_t scratch_area_size    = SCRATCH_AREA_SIZE;
  const size_t required_vtcm_size =
    weight_area_size + activation_area_size + output_area_size + 3 * scratch_area_size + 256;
  if (matmul_check_vtcm_capacity(__func__, required_vtcm_size) != 0) {
    return -1;
  }

  // VTCM layout: weight | activation | output | scales
  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  void    *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch1   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  void    *vtcm_scratch2   = vtcm_seq_alloc(&vtcm_ptr, scratch_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  size_t vec_dot_size       = k * sizeof(__fp16);
  size_t m_chunk_max_n_rows = align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  size_t n_chunk_max_n_cols = align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
  find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16), HMX_FP16_TILE_N_ROWS,
                  HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  // FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
  if (m_chunk_n_rows == 0 || n_chunk_n_cols == 0) {
    FARF(ALWAYS, "%s: invalid chunk size, m_max=%u, n_max=%u, out=%u KiB", __func__,
         (unsigned) m_chunk_max_n_rows, (unsigned) n_chunk_max_n_cols, (unsigned) (output_area_size / 1024));
    return -1;
  }

  // int64_t activation_load_time, weight_load_time, hmx_core_time, output_store_time;
  // activation_load_time = weight_load_time = hmx_core_time = output_store_time = 0;

  const bool use_pipeline = qk_matmul_use_pipeline(m, k, n);

  if (!use_pipeline) {
    // NOTE(hzx): In this simple implementation, load-matmul-store are executed sequentially
    // only DMA load and dequantization process are overlapped during the load stage

    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
      // transfer activation matrix chunk into VTCM
      size_t n_rows = smin(m - mr, m_chunk_n_rows);

      // int64_t act_t0 = HAP_perf_get_qtimer_count();
      int64_t act_t0_us = matmul_trace_now_us();
      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
      }
      int64_t act_t1_us = matmul_trace_now_us();
      record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_ACTIVATION_HVX_LOAD,
                          LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, -1, -1, n_rows, 0, k,
                          (int64_t) n_rows * k * (int64_t) sizeof(float), act_t0_us, act_t1_us);
      // activation_load_time += HAP_perf_get_qtimer_count() - act_t0;

      // FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);

      void *buf_curr = vtcm_scratch0;
      void *buf_next = vtcm_scratch1;

      static dma_desc_1d_t desc
        __attribute__((aligned(64)));  // NOTE(hzx): make sure the DMA descriptor's lifetime is long enough

      // issue async DDR data transfer for the first weight chunk
      int64_t dma_curr_t0_us = 0;
      {
        const size_t n_cols_first            = smin(n, n_chunk_n_cols);
        const size_t first_weight_chunk_size = n_cols_first * k / QK_K * super_block_size;

        dma_curr_t0_us = matmul_trace_now_us();
        dma_issue_load_from_ddr(&desc, buf_curr, permuted_weight, first_weight_chunk_size);
      }

      for (size_t nc = 0; nc < n; nc += n_chunk_n_cols) {
        size_t n_cols = smin(n - nc, n_chunk_n_cols);
        const size_t weight_chunk_size = n_cols * k / QK_K * super_block_size;

        // int64_t wei_t0 = HAP_perf_get_qtimer_count();
        {
          int64_t dma_wait_t0_us = matmul_trace_now_us();
          dma_wait_for_idle();  // wait until current weight chunk become ready
          int64_t dma_wait_t1_us = matmul_trace_now_us();
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_DMA_INFLIGHT,
                              LLM_TRACE_UNIT_DMA, -1, m, k, n, mr, nc, -1, n_rows, n_cols, k, weight_chunk_size,
                              dma_curr_t0_us, dma_wait_t1_us);
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_DMA_WAIT,
                              LLM_TRACE_UNIT_DMA, -1, m, k, n, mr, nc, -1, n_rows, n_cols, k, weight_chunk_size,
                              dma_wait_t0_us, dma_wait_t1_us);

          const size_t nc_next = nc + n_chunk_n_cols;
          int64_t      dma_next_t0_us = 0;
          if (nc_next < n) {
            const size_t n_cols_next = smin(n - nc_next, n_chunk_n_cols);

            const size_t   next_weight_chunk_size = n_cols_next * k / QK_K * super_block_size;
            const uint8_t *next_weight_chunk      = permuted_weight + nc_next * k / QK_K * super_block_size;

            dma_next_t0_us = matmul_trace_now_us();
            dma_issue_load_from_ddr(&desc, buf_next, next_weight_chunk, next_weight_chunk_size);
          }

          const uint8_t *permuted_weight_chunk = permuted_weight + (nc * k / QK_K) * super_block_size;
          int64_t        deq_t0_us = matmul_trace_now_us();
          dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight, permuted_weight_chunk, n_cols * k, k,
                                                            weight_type, buf_curr);
          int64_t deq_t1_us = matmul_trace_now_us();
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_HVX_DEQUANT,
                              LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, nc, -1, n_rows, n_cols, k, weight_chunk_size,
                              deq_t0_us, deq_t1_us);

          swap_ptr(&buf_curr, &buf_next);
          dma_curr_t0_us = dma_next_t0_us;
        }
        // weight_load_time += HAP_perf_get_qtimer_count() - wei_t0;

        // FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

        // int64_t core_t0 = HAP_perf_get_qtimer_count();
        int64_t core_t0_us = matmul_trace_now_us();
        {
          const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
          core_dot_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, n_row_tiles, n_col_tiles, k / 32);
        }
        int64_t core_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_HMX_MMA, LLM_TRACE_UNIT_HMX, -1,
                            m, k, n, mr, nc, -1, n_rows, n_cols, k, 2LL * n_rows * n_cols * k, core_t0_us,
                            core_t1_us);
        // hmx_core_time += HAP_perf_get_qtimer_count() - core_t0;

        // FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

        // int64_t out_t0 = HAP_perf_get_qtimer_count();
        int64_t out_t0_us = matmul_trace_now_us();
        {
          float *output = dst + (mr * n + nc);
          transfer_output_chunk_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
        }
        int64_t out_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_OUTPUT_STORE,
                            LLM_TRACE_UNIT_STORE, -1, m, k, n, mr, nc, -1, n_rows, n_cols, 0,
                            (int64_t) n_rows * n_cols * (int64_t) sizeof(float), out_t0_us, out_t1_us);
        // output_store_time += HAP_perf_get_qtimer_count() - out_t0;

        // FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
      }
    }
  } else {
    // 4-stage pipeline: DMA load (A), dequantize (B), HMX matmul (C), store (D)
    // stage B and D (dequantize and store) are expected to be on the critical path

    // A --> B: vtcm_qweight, 1 buffer
    // B --> C: vtcm_weight0/vtcm_weight1, 2 buffers
    // C --> D: vtcm_output0/vtcm_output1, 2 buffers

    //
    // LD ||A3|  | B3 ||
    // MM ||    C2    ||
    // ST || D1 |     ||

    static dma_desc_1d_t _Alignas(64) dma_desc;
    static core_dot_fp16_task_state_t mm_task_state;
    static worker_pool_job_t          mm_task_job;

    mm_task_job.dptr = &mm_task_state;
    mm_task_job.fptr = &core_dot_fp16_hmx_worker_fn;

    int n_chunk_cnt = ceil_div(n, n_chunk_n_cols);
    for (size_t mr = 0; mr < m; mr += m_chunk_n_rows) {
      const size_t n_rows = smin(m - mr, m_chunk_n_rows);
      int64_t dma_issue_t0_us[n_chunk_cnt];
      for (int i = 0; i < n_chunk_cnt; ++i) {
        dma_issue_t0_us[i] = 0;
      }

      void *vtcm_qweight        = vtcm_weight;
      void *vtcm_weight_bufs[2] = { vtcm_scratch0, vtcm_scratch1 };
      void *vtcm_output_bufs[2] = { vtcm_output, vtcm_scratch2 };

      // prologue: A0
      const size_t n_cols_A0 = smin(n - 0 * n_chunk_n_cols, n_chunk_n_cols);
      {
        const size_t chunk_size_A0 = n_cols_A0 * k / QK_K * super_block_size;

        const uint8_t *qweight_chunk_A0 = permuted_weight;
        dma_issue_t0_us[0] = matmul_trace_now_us();
        dma_issue_load_from_ddr(&dma_desc, vtcm_qweight, qweight_chunk_A0, chunk_size_A0);
      }

      int64_t act_t0_us = matmul_trace_now_us();
      {
        const float *activation_chunk = activation + mr * k;
        transfer_activation_chunk_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);
      }
      int64_t act_t1_us = matmul_trace_now_us();
      record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_ACTIVATION_HVX_LOAD,
                          LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, -1, -1, n_rows, 0, k,
                          (int64_t) n_rows * k * (int64_t) sizeof(float), act_t0_us, act_t1_us);

      // prologue: B0, A1, C0, B1
      {
        // B0
        int64_t wait_t0_us = matmul_trace_now_us();
        dma_wait_for_idle();
        int64_t wait_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_DMA_INFLIGHT,
                            LLM_TRACE_UNIT_DMA, -1, m, k, n, mr, 0, -1, n_rows, n_cols_A0, k,
                            (int64_t) n_cols_A0 * k / QK_K * (int64_t) super_block_size, dma_issue_t0_us[0],
                            wait_t1_us);
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_DMA_WAIT,
                            LLM_TRACE_UNIT_DMA, -1, m, k, n, mr, 0, -1, n_rows, n_cols_A0, k,
                            (int64_t) n_cols_A0 * k / QK_K * (int64_t) super_block_size, wait_t0_us, wait_t1_us);
        int64_t deq_t0_us = matmul_trace_now_us();
        dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[0], NULL, n_cols_A0 * k, k, weight_type,
                                                          vtcm_qweight);
        int64_t deq_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_HVX_DEQUANT,
                            LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, 0, -1, n_rows, n_cols_A0, k,
                            (int64_t) n_cols_A0 * k / QK_K * (int64_t) super_block_size, deq_t0_us, deq_t1_us);

        // A1
        const size_t n_cols_A1 = smin(n - 1 * n_chunk_n_cols, n_chunk_n_cols);
        if (1 < n_chunk_cnt) {
          const size_t chunk_size_A1 = n_cols_A1 * k / QK_K * super_block_size;

          const uint8_t *qweight_chunk_A1 = permuted_weight + n_chunk_n_cols * k / QK_K * super_block_size;
          dma_issue_t0_us[1] = matmul_trace_now_us();
          dma_issue_load_from_ddr(&dma_desc, vtcm_qweight, qweight_chunk_A1, chunk_size_A1);
        }

        // C0
        {
          core_dot_fp16_task_state_t *s = &mm_task_state;

          s->c = (__fp16 *) vtcm_output_bufs[0];
          s->a = (__fp16 *) vtcm_activation;
          s->b = (__fp16 *) vtcm_weight_bufs[0];
          s->s = vtcm_scales;

          s->n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          s->n_col_tiles = ceil_div(n_cols_A0, HMX_FP16_TILE_N_COLS);
          s->n_dot_tiles = k / HMX_FP16_TILE_N_ROWS;
          s->profile     = profile;
          s->trace_id    = trace_id;
          s->mode_flags  = mode_flags;
          s->op_index    = op_index;
          s->m           = m;
          s->k           = k;
          s->n           = n;
          s->mr          = mr;
          s->nc          = 0;
          s->kk          = -1;
          s->chunk_m     = n_rows;
          s->chunk_n     = n_cols_A0;
          s->chunk_k     = k;
          s->bytes       = 2LL * n_rows * n_cols_A0 * k;

          worker_pool_synctoken_init(&s->sync_ctx, 1);
          worker_pool_submit(hmx_worker_pool_ctx, mm_task_job);
        }

        // B1
        if (1 < n_chunk_cnt) {
          wait_t0_us = matmul_trace_now_us();
          dma_wait_for_idle();
          wait_t1_us = matmul_trace_now_us();
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_DMA_INFLIGHT,
                              LLM_TRACE_UNIT_DMA, -1, m, k, n, mr, n_chunk_n_cols, -1, n_rows, n_cols_A1, k,
                              (int64_t) n_cols_A1 * k / QK_K * (int64_t) super_block_size, dma_issue_t0_us[1],
                              wait_t1_us);
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_DMA_WAIT,
                              LLM_TRACE_UNIT_DMA, -1, m, k, n, mr, n_chunk_n_cols, -1, n_rows, n_cols_A1, k,
                              (int64_t) n_cols_A1 * k / QK_K * (int64_t) super_block_size, wait_t0_us, wait_t1_us);
          deq_t0_us = matmul_trace_now_us();
          dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[1], NULL, n_cols_A1 * k, k, weight_type,
                                                            vtcm_qweight);
          deq_t1_us = matmul_trace_now_us();
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_HVX_DEQUANT,
                              LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, n_chunk_n_cols, -1, n_rows, n_cols_A1, k,
                              (int64_t) n_cols_A1 * k / QK_K * (int64_t) super_block_size, deq_t0_us, deq_t1_us);
        }
      }

      // main loop
      for (int i = 0; i < n_chunk_cnt; ++i) {
        const size_t nc    = i * n_chunk_n_cols;
        const size_t nc_p1 = nc + 1 * n_chunk_n_cols;
        const size_t nc_p2 = nc + 2 * n_chunk_n_cols;

        const size_t n_cols    = smin(n - nc, n_chunk_n_cols);
        const size_t n_cols_p1 = smin(n - nc_p1, n_chunk_n_cols);
        const size_t n_cols_p2 = smin(n - nc_p2, n_chunk_n_cols);

        // issue A_{i+2}
        if (i + 2 < n_chunk_cnt) {
          const size_t   chunk_size_p2    = n_cols_p2 * k / QK_K * super_block_size;
          const uint8_t *qweight_chunk_p2 = permuted_weight + nc_p2 * k / QK_K * super_block_size;
          dma_issue_t0_us[i + 2] = matmul_trace_now_us();
          dma_issue_load_from_ddr(&dma_desc, vtcm_qweight, qweight_chunk_p2, chunk_size_p2);
        }

        // wait for HMX (C_{i})
        worker_pool_synctoken_wait(&mm_task_state.sync_ctx);

        // result of B_{i+1} (input of C_{i+1}) should be ready now

        // issue C_{i+1}
        if (i + 1 < n_chunk_cnt) {
          core_dot_fp16_task_state_t *s = &mm_task_state;

          s->c = (__fp16 *) vtcm_output_bufs[(i + 1) % 2];
          s->a = (__fp16 *) vtcm_activation;
          s->b = (__fp16 *) vtcm_weight_bufs[(i + 1) % 2];
          s->s = vtcm_scales;

          s->n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          s->n_col_tiles = ceil_div(n_cols_p1, HMX_FP16_TILE_N_COLS);
          s->n_dot_tiles = k / HMX_FP16_TILE_N_ROWS;
          s->profile     = profile;
          s->trace_id    = trace_id;
          s->mode_flags  = mode_flags;
          s->op_index    = op_index;
          s->m           = m;
          s->k           = k;
          s->n           = n;
          s->mr          = mr;
          s->nc          = nc_p1;
          s->kk          = -1;
          s->chunk_m     = n_rows;
          s->chunk_n     = n_cols_p1;
          s->chunk_k     = k;
          s->bytes       = 2LL * n_rows * n_cols_p1 * k;

          worker_pool_synctoken_init(&s->sync_ctx, 1);
          worker_pool_submit(hmx_worker_pool_ctx, mm_task_job);
        }

        // compute D_{i}
        int64_t out_t0_us = matmul_trace_now_us();
        float *output_chunk = dst + (mr * n + nc);
        transfer_output_chunk_fp16_to_fp32(output_chunk, vtcm_output_bufs[i % 2], n_rows, n_cols, n);
        int64_t out_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_OUTPUT_STORE,
                            LLM_TRACE_UNIT_STORE, -1, m, k, n, mr, nc, -1, n_rows, n_cols, 0,
                            (int64_t) n_rows * n_cols * (int64_t) sizeof(float), out_t0_us, out_t1_us);

        // wait for DMA (A_{i+2}), compute B_{i+2}
        if (i + 2 < n_chunk_cnt) {
          int64_t wait_t0_us = matmul_trace_now_us();
          dma_wait_for_idle();
          int64_t wait_t1_us = matmul_trace_now_us();
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_DMA_INFLIGHT,
                              LLM_TRACE_UNIT_DMA, -1, m, k, n, mr, nc_p2, -1, n_rows, n_cols_p2, k,
                              (int64_t) n_cols_p2 * k / QK_K * (int64_t) super_block_size,
                              dma_issue_t0_us[i + 2], wait_t1_us);
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_DMA_WAIT,
                              LLM_TRACE_UNIT_DMA, -1, m, k, n, mr, nc_p2, -1, n_rows, n_cols_p2, k,
                              (int64_t) n_cols_p2 * k / QK_K * (int64_t) super_block_size, wait_t0_us, wait_t1_us);
          int64_t deq_t0_us = matmul_trace_now_us();
          dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[(i + 2) % 2], NULL, n_cols_p2 * k, k,
                                                            weight_type, vtcm_qweight);
          int64_t deq_t1_us = matmul_trace_now_us();
          record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_HVX_DEQUANT,
                              LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, nc_p2, -1, n_rows, n_cols_p2, k,
                              (int64_t) n_cols_p2 * k / QK_K * (int64_t) super_block_size, deq_t0_us, deq_t1_us);
        }
      }
    }
  }

  // FARF(ALWAYS, "%s: m = %d, k = %d, n = %d", __func__, m, k, n);
  // FARF(ALWAYS, "    activation load: %lld us", HAP_perf_qtimer_count_to_us(activation_load_time));
  // FARF(ALWAYS, "    weight     load: %lld us", HAP_perf_qtimer_count_to_us(weight_load_time));
  // FARF(ALWAYS, "    core     matmul: %lld us", HAP_perf_qtimer_count_to_us(hmx_core_time));
  // FARF(ALWAYS, "    output    store: %lld us", HAP_perf_qtimer_count_to_us(output_store_time));

  // size_t weight_size = k * n / QK_K * super_block_size;
  // float  bandwidth   = 1e-3 * weight_size / HAP_perf_qtimer_count_to_us(weight_load_time);
  // FARF(ALWAYS, "    weight load bandwidth: %.2f GB/s", bandwidth);

  return 0;
}

// C += AB
void core_mma_chunk_fp16(__fp16 *c, const __fp16 *a, const __fp16 *b, const __fp16 *col_scales, const __fp16 *eye_tile,
                         int n_row_tiles, int n_col_tiles, int n_dot_tiles, bool zero_init) {
  hmx_unit_acquire();

  asm volatile("mxclracc.hf");
  hmx_set_output_scales(col_scales);

  for (int i = 0; i < n_row_tiles; ++i) {
    for (int j = 0; j < n_col_tiles; ++j) {
      const __fp16 *row_tiles = a + i * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
      const __fp16 *col_tiles = b + j * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

      __fp16 *accum_tile = c + (i * n_col_tiles + j) * HMX_FP16_TILE_N_ELMS;
      if (!zero_init) {
        hmx_load_tiles_fp16(accum_tile, eye_tile, 1);
      }

      for (int k = 0; k < n_dot_tiles; k += 32) {
        int    offset  = k * HMX_FP16_TILE_N_ELMS;
        size_t n_tiles = smin(n_dot_tiles - k, 32);
        hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
      }

      hmx_consume_accumulator_fp16(accum_tile);
    }
  }

  hmx_unit_release();
}

// Only slightly faster than the common version (with L2 prefetch enabled) when doing VTCM to VTCM transfer
void transfer_activation_chunk_no_prefetch(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
                                           int k_block, int k_stride) {
  for (int r = 0; r < n_rows; r += 2) {
    int r0 = r / HMX_FP16_TILE_N_ROWS;  // tile row index
    int r1 = r % HMX_FP16_TILE_N_ROWS;  // intra-tile row idx

    const bool next_row_valid = (r + 1) < n_rows;

    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * k_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * k_stride);
    for (int c = 0; c < k_block; c += 32) {
      HVX_Vector v0 = *pv_in0++;
      HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

      HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

      // compute output position
      int c0       = c / HMX_FP16_TILE_N_COLS;  // tile column index
      int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2]     = v_out;
    }
  }
}

typedef struct {
  EXPAND_COMMON_TASK_STATE_MEMBERS
  __fp16      *dst;
  const float *src;
  int          k_block, k_stride;
} activation_transfer_task_state_t;

static void transfer_activation_chunk_worker_fn(void *data, int _worker_index) {
  (void) _worker_index;
  activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&st->task_id) - 1;
    if (task_id >= st->n_tasks) {
      break;
    }
    // one chunk: one row
    int    chunk_idx  = task_id * st->n_chunks_per_task;
    size_t chunk_size = smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

    __fp16      *dst = st->dst + chunk_idx * st->k_block;
    const float *src = st->src + chunk_idx * st->k_stride;
    transfer_activation_chunk_no_prefetch(dst, src, chunk_size, st->k_block, st->k_stride);
  }

  worker_pool_synctoken_jobdone(&st->sync_ctx);
}

void transfer_activation_chunk_multithread(__fp16 *dst, const float *src, int n_rows, int k_block, int k_stride) {
  int    n_workers         = num_hvx128_contexts;
  size_t n_tot_chunks      = n_rows;
  size_t n_chunks_per_task = 32;  // NOTE(hzx): must be multiple of 32 to ensure correct destination address

  activation_transfer_task_state_t state;
  INIT_COMMON_TASK_STATE_MEMBERS(state, n_tot_chunks, n_chunks_per_task);
  state.dst      = dst;
  state.src      = src;
  state.k_block  = k_block;
  state.k_stride = k_stride;

  worker_pool_job_t job;
  job.dptr = &state;
  job.fptr = &transfer_activation_chunk_worker_fn;

  worker_pool_synctoken_init(&state.sync_ctx, n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&state.sync_ctx);
}

#ifndef HTP_OS_M_BLOCK_SIZE
#define HTP_OS_M_BLOCK_SIZE 512
#endif

#ifndef HTP_OS_N_BLOCK_SIZE
#define HTP_OS_N_BLOCK_SIZE 512
#endif

#ifndef HTP_OS_K_BLOCK_SIZE
#define HTP_OS_K_BLOCK_SIZE 512
#endif

static void qk_os_init_eye_tile_fp16(__fp16 *eye_tile) {
  memset(eye_tile, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < HMX_FP16_TILE_N_ROWS; ++r) {
    for (int c = 0; c < HMX_FP16_TILE_N_COLS; ++c) {
      const int idx = (r & ~1) * HMX_FP16_TILE_N_COLS + c * 2 + (r & 1);
      eye_tile[idx] = (__fp16) (r == c ? 1.0f : 0.0f);
    }
  }
}

static void qk_os_init_eye_tile_fp16_layout(__fp16 *eye_tile, int layout) {
  memset(eye_tile, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < HMX_FP16_TILE_N_ROWS; ++r) {
    for (int c = 0; c < HMX_FP16_TILE_N_COLS; ++c) {
      int idx = 0;
      switch (layout) {
        case 1:
          idx = r * HMX_FP16_TILE_N_COLS + c;
          break;
        case 2:
          idx = c * HMX_FP16_TILE_N_ROWS + r;
          break;
        case 3:
          idx = (c & ~1) * HMX_FP16_TILE_N_ROWS + r * 2 + (c & 1);
          break;
        case 0:
        default:
          idx = (r & ~1) * HMX_FP16_TILE_N_COLS + c * 2 + (r & 1);
          break;
      }
      eye_tile[idx] = (__fp16) (r == c ? 1.0f : 0.0f);
    }
  }
}

static void qk_os_copy_fp16_weight_slice_from_full(__fp16 *restrict dst, const __fp16 *restrict full, int n_cols,
                                                   int full_k, int kk, int k_block) {
  const int full_dot_tiles  = full_k / HMX_FP16_TILE_N_COLS;
  const int slice_dot_tiles = k_block / HMX_FP16_TILE_N_COLS;
  const int n_col_tiles     = n_cols / HMX_FP16_TILE_N_COLS;
  const int k_tile_start    = kk / HMX_FP16_TILE_N_COLS;

  // The full dequantized qweight layout is [N tile][K tile][32x32 HMX tile].
  // This helper compacts a K slice into the same layout expected by core_mma,
  // removing qweight addressing/dequantization from the OS root-cause probe.
  for (int j = 0; j < n_col_tiles; ++j) {
    const __fp16 *src = full + (j * full_dot_tiles + k_tile_start) * HMX_FP16_TILE_N_ELMS;
    __fp16       *out = dst + j * slice_dot_tiles * HMX_FP16_TILE_N_ELMS;
    memcpy(out, src, (size_t) slice_dot_tiles * HMX_FP16_TILE_SIZE);
  }
}

static void qk_os_copy_contiguous_weight_k_block(uint8_t *restrict dst, const uint8_t *restrict src, int nc,
                                                 int n_cols, int k, int kk, int k_block,
                                                 size_t super_block_size) {
  const int n_k_super_per_col = k / QK_K;
  const int k_super_start     = kk / QK_K;
  const int k_super_count     = k_block / QK_K;
  const size_t n_bytes        = (size_t) n_cols * (size_t) k_super_count * super_block_size;
  const uint8_t *src_begin    = src + ((size_t) nc * n_k_super_per_col + k_super_start) * super_block_size;

  // This intentionally preserves the old OS bug for debug probes: it assumes
  // a K-slice over many output columns is contiguous in the repacked qweight
  // buffer. The fixed path below proves that assumption is false.
  memcpy(dst, src_begin, n_bytes);
}

static void qk_os_gather_weight_k_block(uint8_t *restrict dst, const uint8_t *restrict src, int nc, int n_cols,
                                        int k, int kk, int k_block, size_t super_block_size) {
  const int n_k_super_per_col = k / QK_K;
  const int k_super_start     = kk / QK_K;
  const int k_super_count     = k_block / QK_K;
  const size_t col_bytes      = (size_t) k_super_count * super_block_size;

  // Repacked qweight is column-major at the super-block level:
  //   [col0: ksuper0..ksuperN][col1: ksuper0..ksuperN]...
  // A true output-stationary K-block needs the same K-superblock range from
  // every output column, so it must gather strided column slices into a compact
  // temporary buffer before calling the existing HVX dequantizer.
  for (int c = 0; c < n_cols; ++c) {
    const uint8_t *src_col = src + ((size_t) (nc + c) * n_k_super_per_col + k_super_start) * super_block_size;
    memcpy(dst + (size_t) c * col_bytes, src_col, col_bytes);
  }
}

static void qk_os_copy_tiled_weight_k_block(uint8_t *restrict dst, const uint8_t *restrict src, int nc, int n_cols,
                                            int k, int kk, int k_block, size_t super_block_size) {
  const int n_k_super_per_tile = k / QK_K;
  const int k_super_start      = kk / QK_K;
  const int k_super_count      = k_block / QK_K;
  const int n_tile_start       = nc / HMX_FP16_TILE_N_COLS;
  const int n_tiles            = n_cols / HMX_FP16_TILE_N_COLS;
  const size_t tile_bytes      = (size_t) k_super_count * HMX_FP16_TILE_N_COLS * super_block_size;

  // The qweight buffer is not scalar-column-major. It is tiled for HMX as:
  //   [N tile of 32 columns][K superblock][32 columns within the tile].
  // A K-slice is therefore contiguous only inside each 32-column N tile. The
  // old OS path copied one large linear range across all N tiles, which pulled
  // unrelated K blocks from the first tile and skipped the same K range in
  // later tiles. Keep this helper local to the repaired OS path.
  for (int t = 0; t < n_tiles; ++t) {
    const size_t src_block = ((size_t) (n_tile_start + t) * n_k_super_per_tile + k_super_start) *
                             HMX_FP16_TILE_N_COLS;
    memcpy(dst + (size_t) t * tile_bytes, src + src_block * super_block_size, tile_bytes);
  }
}

static int qk_os_choose_block_size(int *m_block_out, int *n_block_out, size_t super_block_size) {
  const int k_block = HTP_OS_K_BLOCK_SIZE;
  size_t m_max = align_down(ACTIVATION_AREA_SIZE / ((size_t) k_block * sizeof(__fp16)), HMX_FP16_TILE_N_ROWS);
  size_t n_max = align_down(WEIGHT_AREA_SIZE / ((size_t) k_block * sizeof(__fp16)), HMX_FP16_TILE_N_COLS);
  m_max = smin(m_max, HTP_OS_M_BLOCK_SIZE);
  n_max = smin(n_max, HTP_OS_N_BLOCK_SIZE);

  size_t m_block = 0, n_block = 0;
  find_chunk_size(m_max, n_max, OUTPUT_AREA_SIZE / sizeof(__fp16), HMX_FP16_TILE_N_ROWS, HMX_FP16_TILE_N_COLS,
                  &m_block, &n_block);

  if (m_block == 0 || n_block == 0) {
    return -1;
  }

  const size_t scratch_need = n_block * (size_t) k_block / QK_K * super_block_size;
  if (scratch_need > SCRATCH_AREA_SIZE) {
    return -1;
  }

  *m_block_out = (int) m_block;
  *n_block_out = (int) n_block;
  return 0;
}

static int mat_mul_qk_0_d16a32_out_stationary_impl(float *restrict out, const float *restrict x,
                                                   const uint8_t *restrict w, int m, int k, int n,
                                                   enum ggml_type weight_type, int64_t trace_id, int mode_flags,
                                                   int op_index, struct LlmTraceProfileHeader *profile,
                                                   bool tile_k_slice) {
  assert(k < 16384);
  if (k % QK_K != 0 || n % HMX_FP16_TILE_N_COLS != 0) {
    return -1;
  }

  const size_t super_block_size = get_super_block_size(weight_type);
  if (super_block_size == 0) {
    return -1;
  }

  int m_block = 0, n_block = 0;
  if (qk_os_choose_block_size(&m_block, &n_block, super_block_size) != 0) {
    return -1;
  }

  const size_t required_vtcm_size =
    WEIGHT_AREA_SIZE + ACTIVATION_AREA_SIZE + OUTPUT_AREA_SIZE + SCRATCH_AREA_SIZE + HMX_FP16_TILE_SIZE + 256;
  if (matmul_check_vtcm_capacity(__func__, required_vtcm_size) != 0) {
    return -1;
  }

  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, WEIGHT_AREA_SIZE);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, ACTIVATION_AREA_SIZE);
  __fp16  *vtcm_output     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, OUTPUT_AREA_SIZE);
  uint8_t *vtcm_scratch0   = vtcm_seq_alloc(&vtcm_ptr, SCRATCH_AREA_SIZE);
  __fp16  *vtcm_eye_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  qk_os_init_eye_tile_fp16(vtcm_eye_tile);
  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0

  for (int mr = 0; mr < m; mr += m_block) {
    const int n_rows = smin(m - mr, m_block);

    for (int nc = 0; nc < n; nc += n_block) {
      const int n_cols = smin(n - nc, n_block);

      for (int kk = 0; kk < k; kk += HTP_OS_K_BLOCK_SIZE) {
        const int k_block = smin(k - kk, HTP_OS_K_BLOCK_SIZE);
        if (k_block % QK_K != 0) {
          return -1;
        }

        int64_t act_t0_us = matmul_trace_now_us();
        transfer_activation_chunk_fp32_to_fp16(vtcm_activation, x + (size_t) mr * k + kk, n_rows, k_block, k);
        int64_t act_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_ACTIVATION_HVX_LOAD,
                            LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, nc, kk, n_rows, n_cols, k_block,
                            (int64_t) n_rows * k_block * (int64_t) sizeof(float), act_t0_us, act_t1_us);

        const size_t weight_chunk_size = (size_t) n_cols * (size_t) k_block / QK_K * super_block_size;
        int64_t gather_t0_us = matmul_trace_now_us();
        if (tile_k_slice) {
          qk_os_copy_tiled_weight_k_block(vtcm_scratch0, w, nc, n_cols, k, kk, k_block, super_block_size);
        } else {
          qk_os_copy_contiguous_weight_k_block(vtcm_scratch0, w, nc, n_cols, k, kk, k_block, super_block_size);
        }
        int64_t gather_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_HVX_LOAD,
                            LLM_TRACE_UNIT_MEMORY, -1, m, k, n, mr, nc, kk, n_rows, n_cols, k_block,
                            weight_chunk_size, gather_t0_us, gather_t1_us);

        int64_t deq_t0_us = matmul_trace_now_us();
        dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight, NULL, n_cols * k_block, k_block, weight_type,
                                                          vtcm_scratch0);
        int64_t deq_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_WEIGHT_HVX_DEQUANT,
                            LLM_TRACE_UNIT_HVX, -1, m, k, n, mr, nc, kk, n_rows, n_cols, k_block,
                            weight_chunk_size, deq_t0_us, deq_t1_us);

        int64_t core_t0_us = matmul_trace_now_us();
        {
          const int n_row_tiles = ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
          const int n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
          core_mma_chunk_fp16(vtcm_output, vtcm_activation, vtcm_weight, vtcm_scales, vtcm_eye_tile, n_row_tiles,
                              n_col_tiles, k_block / HMX_FP16_TILE_N_COLS, kk == 0);
        }
        int64_t core_t1_us = matmul_trace_now_us();
        record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_HMX_MMA, LLM_TRACE_UNIT_HMX,
                            -1, m, k, n, mr, nc, kk, n_rows, n_cols, k_block,
                            2LL * n_rows * n_cols * k_block, core_t0_us, core_t1_us);
      }

      int64_t out_t0_us = matmul_trace_now_us();
      transfer_output_chunk_fp16_to_fp32(out + (size_t) mr * n + nc, vtcm_output, n_rows, n_cols, n);
      int64_t out_t1_us = matmul_trace_now_us();
      record_matmul_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_OUTPUT_STORE,
                          LLM_TRACE_UNIT_STORE, -1, m, k, n, mr, nc, -1, n_rows, n_cols, 0,
                          (int64_t) n_rows * n_cols * (int64_t) sizeof(float), out_t0_us, out_t1_us);
    }
  }

  return 0;
}

int mat_mul_qk_0_d16a32_out_stationary(float *restrict out, const float *restrict x, const uint8_t *restrict w, int m,
                                       int k, int n, enum ggml_type weight_type, int64_t trace_id, int mode_flags,
                                       int op_index, struct LlmTraceProfileHeader *profile) {
  // The repaired OS path keeps C[M,N] stationary across K-blocks, but fixes the
  // old qweight addressing bug by copying K-superblock slices per 32-column HMX
  // N tile before HVX dequantization.
  return mat_mul_qk_0_d16a32_out_stationary_impl(out, x, w, m, k, n, weight_type, trace_id, mode_flags, op_index,
                                                profile, true);
}

static float qk_os_probe_activation_value(int r, int c) {
  const int v = ((r * 17 + c * 13) % 97) - 48;
  return (float) v * 0.00390625f;
}

static void qk_os_probe_fill_iq4_weight(my_block_q4_0 *weight, int k, int n) {
  const int n_k_super_per_col = k / QK_K;
  for (int c = 0; c < n; ++c) {
    for (int ks = 0; ks < n_k_super_per_col; ++ks) {
      my_block_q4_0 *blk = weight + c * n_k_super_per_col + ks;
      for (int s = 0; s < 8; ++s) {
        const float scale = 0.00035f + 0.00001f * (float) ((c + ks + s) % 7);
        blk->scales[s] = (__fp16) scale;
      }
      for (int q = 0; q < (int) sizeof(blk->quants); ++q) {
        const uint8_t lo = (uint8_t) ((c * 3 + ks * 5 + q * 7) & 15);
        const uint8_t hi = (uint8_t) ((c * 11 + ks * 13 + q * 3 + 1) & 15);
        blk->quants[q] = (uint8_t) ((hi << 4) | lo);
      }
    }
  }
}

static void qk_os_probe_record(struct HmxInt8GateResult *results, int *idx, int max_results, int selector, int variant,
                               int first_bad, int count, float first_got, float first_ref, float max_abs,
                               float rmse) {
  if (*idx >= max_results) {
    return;
  }
  struct HmxInt8GateResult *r = results + *idx;
  memset(r, 0, sizeof(*r));
  r->selector    = selector;
  r->variant     = variant;
  r->output_kind = first_bad < 0 ? 0 : 1;
  r->tile_bytes  = first_bad;
  r->reserved    = 21;
  r->expected    = (float) count;
  r->first8[0]   = first_got;
  r->first8[1]   = first_ref;
  r->first8[2]   = max_abs;
  r->first8[3]   = rmse;
  r->min_value   = first_got;
  r->max_value   = max_abs;
  r->mean_value  = first_ref;
  r->rmse        = rmse;
  ++(*idx);
}

static void qk_os_probe_compare_fp16_slice(struct HmxInt8GateResult *results, int *idx, int max_results, int selector,
                                           const __fp16 *got, const __fp16 *full, int k, int n, int kk,
                                           int k_block) {
  const int full_dot_tiles  = k / HMX_FP16_TILE_N_COLS;
  const int slice_dot_tiles = k_block / HMX_FP16_TILE_N_COLS;
  const int n_col_tiles     = n / HMX_FP16_TILE_N_COLS;
  const int k_tile_start    = kk / HMX_FP16_TILE_N_COLS;

  int first_bad = -1;
  int count     = 0;
  float first_got = 0.0f, first_ref = 0.0f, max_abs = 0.0f;
  double sq_err = 0.0;

  for (int j = 0; j < n_col_tiles; ++j) {
    for (int d = 0; d < slice_dot_tiles; ++d) {
      const __fp16 *got_tile  = got + (j * slice_dot_tiles + d) * HMX_FP16_TILE_N_ELMS;
      const __fp16 *full_tile = full + (j * full_dot_tiles + k_tile_start + d) * HMX_FP16_TILE_N_ELMS;
      for (int e = 0; e < HMX_FP16_TILE_N_ELMS; ++e) {
        const float g   = (float) got_tile[e];
        const float ref = (float) full_tile[e];
        const float err = fabsf(g - ref);
        if (count == 0) {
          first_got = g;
          first_ref = ref;
        }
        if (err > max_abs) {
          max_abs = err;
          first_bad = count;
        }
        sq_err += (double) err * (double) err;
        ++count;
      }
    }
  }

  qk_os_probe_record(results, idx, max_results, selector, kk, first_bad, count, first_got, first_ref, max_abs,
                     count > 0 ? (float) sqrt(sq_err / (double) count) : 0.0f);
}

static void qk_os_probe_compare_f32(struct HmxInt8GateResult *results, int *idx, int max_results, int selector,
                                    const float *got, const float *ref, int count) {
  int first_bad = -1;
  float first_got = 0.0f, first_ref = 0.0f, max_abs = 0.0f;
  double sq_err = 0.0;
  for (int i = 0; i < count; ++i) {
    const float err = fabsf(got[i] - ref[i]);
    if (i == 0) {
      first_got = got[i];
      first_ref = ref[i];
    }
    if (err > max_abs) {
      max_abs = err;
      first_bad = i;
    }
    sq_err += (double) err * (double) err;
  }
  qk_os_probe_record(results, idx, max_results, selector, 0, first_bad, count, first_got, first_ref, max_abs,
                     count > 0 ? (float) sqrt(sq_err / (double) count) : 0.0f);
}

int qk_os_debug_probe_run(struct HmxInt8GateResult *results, int max_results, int mode) {
  (void) mode;
  if (!results || max_results < 8) {
    return -1;
  }

  const int m = 64;
  const int k = 1024;
  const int n = 64;
  const enum ggml_type weight_type = GGML_TYPE_IQ4_NL;
  const size_t super_block_size = get_super_block_size(weight_type);
  const size_t activation_size  = (size_t) m * k * sizeof(float);
  const size_t output_size      = (size_t) m * n * sizeof(float);
  const size_t weight_size      = (size_t) n * (k / QK_K) * super_block_size;

  float *activation = NULL, *ref = NULL, *old_os = NULL, *fixed_os = NULL;
  float *full_mma = NULL, *split_full = NULL;
  uint8_t *weight = NULL;
  int result_idx = 0;

  if (posix_memalign((void **) &activation, VLEN, activation_size) ||
      posix_memalign((void **) &weight, VLEN, weight_size) ||
      posix_memalign((void **) &ref, VLEN, output_size) ||
      posix_memalign((void **) &old_os, VLEN, output_size) ||
      posix_memalign((void **) &fixed_os, VLEN, output_size) ||
      posix_memalign((void **) &full_mma, VLEN, output_size) ||
      posix_memalign((void **) &split_full, VLEN, output_size)) {
    return -1;
  }

  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < k; ++c) {
      activation[(size_t) r * k + c] = qk_os_probe_activation_value(r, c);
    }
  }
  qk_os_probe_fill_iq4_weight((my_block_q4_0 *) weight, k, n);
  memset(ref, 0, output_size);
  memset(old_os, 0, output_size);
  memset(fixed_os, 0, output_size);
  memset(full_mma, 0, output_size);
  memset(split_full, 0, output_size);

  const size_t probe_weight_fp16_size = align_up((size_t) n * k * sizeof(__fp16), VLEN);
  const size_t probe_scratch_size     = align_up(weight_size, VLEN);
  uint8_t *vtcm_ptr          = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight_full  = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, probe_weight_fp16_size);
  __fp16  *vtcm_weight_old   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, probe_weight_fp16_size);
  __fp16  *vtcm_weight_fixed = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, probe_weight_fp16_size);
  uint8_t *vtcm_scratch_full = vtcm_seq_alloc(&vtcm_ptr, probe_scratch_size);
  uint8_t *vtcm_scratch_old  = vtcm_seq_alloc(&vtcm_ptr, probe_scratch_size);
  uint8_t *vtcm_scratch_fix  = vtcm_seq_alloc(&vtcm_ptr, probe_scratch_size);
  __fp16  *vtcm_activation_probe = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_size / sizeof(float) * sizeof(__fp16));
  __fp16  *vtcm_weight_slice     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, probe_weight_fp16_size);
  __fp16  *vtcm_output_probe     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_size / sizeof(float) * sizeof(__fp16));
  __fp16  *vtcm_restore_ref      = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  __fp16  *vtcm_restore_got      = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  __fp16  *vtcm_eye_probe        = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, HMX_FP16_TILE_SIZE);
  __fp16  *vtcm_scales_probe     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  memcpy(vtcm_scratch_full, weight, weight_size);
  dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_full, NULL, n * k, k, weight_type,
                                                    vtcm_scratch_full);

  for (int kk = 0; kk < k; kk += HTP_OS_K_BLOCK_SIZE) {
    const int k_block = smin(k - kk, HTP_OS_K_BLOCK_SIZE);
    const size_t slice_size = (size_t) n * k_block / QK_K * super_block_size;

    qk_os_copy_contiguous_weight_k_block(vtcm_scratch_old, weight, 0, n, k, kk, k_block, super_block_size);
    dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_old, NULL, n * k_block, k_block, weight_type,
                                                      vtcm_scratch_old);
    qk_os_probe_compare_fp16_slice(results, &result_idx, max_results, kk == 0 ? 1 : 3, vtcm_weight_old,
                                   vtcm_weight_full, k, n, kk, k_block);

    memset(vtcm_scratch_fix, 0, slice_size);
    qk_os_gather_weight_k_block(vtcm_scratch_fix, weight, 0, n, k, kk, k_block, super_block_size);
    dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_fixed, NULL, n * k_block, k_block, weight_type,
                                                      vtcm_scratch_fix);
    qk_os_probe_compare_fp16_slice(results, &result_idx, max_results, kk == 0 ? 2 : 4, vtcm_weight_fixed,
                                   vtcm_weight_full, k, n, kk, k_block);

    memset(vtcm_scratch_fix, 0, slice_size);
    qk_os_copy_tiled_weight_k_block(vtcm_scratch_fix, weight, 0, n, k, kk, k_block, super_block_size);
    dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_fixed, NULL, n * k_block, k_block, weight_type,
                                                      vtcm_scratch_fix);
    qk_os_probe_compare_fp16_slice(results, &result_idx, max_results, kk == 0 ? 5 : 6, vtcm_weight_fixed,
                                   vtcm_weight_full, k, n, kk, k_block);
  }

  hmx_init_column_scales(vtcm_scales_probe, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0
  transfer_activation_chunk_fp32_to_fp16(vtcm_activation_probe, activation, m, k, k);
  qk_os_init_eye_tile_fp16_layout(vtcm_eye_probe, 0);
  memset(vtcm_output_probe, 0, output_size / sizeof(float) * sizeof(__fp16));
  core_mma_chunk_fp16(vtcm_output_probe, vtcm_activation_probe, vtcm_weight_full, vtcm_scales_probe,
                      vtcm_eye_probe, ceil_div(m, HMX_FP16_TILE_N_ROWS), ceil_div(n, HMX_FP16_TILE_N_COLS),
                      k / HMX_FP16_TILE_N_COLS, true);
  transfer_output_chunk_fp16_to_fp32(full_mma, vtcm_output_probe, m, n, n);
  // Use direct full-dequant + full-K HMX as the standalone probe reference.
  // The generic safe function uses DMA from its weight pointer; that is valid
  // for server FastRPC shared buffers, but not for this synthetic DSP heap
  // allocation without an explicit cache clean before DMA.
  memcpy(ref, full_mma, output_size);
  qk_os_probe_compare_f32(results, &result_idx, max_results, 7, full_mma, ref, m * n);

  for (int i = 0; i < HMX_FP16_TILE_N_ELMS; ++i) {
    const int r = i / HMX_FP16_TILE_N_COLS;
    const int c = i % HMX_FP16_TILE_N_COLS;
    vtcm_restore_ref[i] = (__fp16) (((r * 19 + c * 7) % 41) * 0.03125f - 0.625f);
  }
  for (int layout = 0; layout < 4; ++layout) {
    memcpy(vtcm_restore_got, vtcm_restore_ref, HMX_FP16_TILE_SIZE);
    qk_os_init_eye_tile_fp16_layout(vtcm_eye_probe, layout);
    core_mma_chunk_fp16(vtcm_restore_got, vtcm_restore_got, vtcm_eye_probe, vtcm_scales_probe, vtcm_eye_probe, 1, 1,
                        0, false);
    qk_os_probe_compare_fp16_slice(results, &result_idx, max_results, 8 + layout, vtcm_restore_got, vtcm_restore_ref,
                                   HMX_FP16_TILE_N_COLS, HMX_FP16_TILE_N_COLS, 0, HMX_FP16_TILE_N_COLS);
  }

  for (int layout = 0; layout < 4; ++layout) {
    qk_os_init_eye_tile_fp16_layout(vtcm_eye_probe, layout);
    memset(vtcm_output_probe, 0, output_size / sizeof(float) * sizeof(__fp16));
    memset(split_full, 0, output_size);
    for (int kk = 0; kk < k; kk += HTP_OS_K_BLOCK_SIZE) {
      const int k_block = smin(k - kk, HTP_OS_K_BLOCK_SIZE);
      transfer_activation_chunk_fp32_to_fp16(vtcm_activation_probe, activation + kk, m, k_block, k);
      qk_os_copy_fp16_weight_slice_from_full(vtcm_weight_slice, vtcm_weight_full, n, k, kk, k_block);
      core_mma_chunk_fp16(vtcm_output_probe, vtcm_activation_probe, vtcm_weight_slice, vtcm_scales_probe,
                          vtcm_eye_probe, ceil_div(m, HMX_FP16_TILE_N_ROWS), ceil_div(n, HMX_FP16_TILE_N_COLS),
                          k_block / HMX_FP16_TILE_N_COLS, kk == 0);
    }
    transfer_output_chunk_fp16_to_fp32(split_full, vtcm_output_probe, m, n, n);
    qk_os_probe_compare_f32(results, &result_idx, max_results, 12 + layout, split_full, ref, m * n);
  }

  qk_os_init_eye_tile_fp16_layout(vtcm_eye_probe, 0);
  memset(vtcm_output_probe, 0, output_size / sizeof(float) * sizeof(__fp16));
  memset(split_full, 0, output_size);
  for (int kk = 0; kk < k; kk += HTP_OS_K_BLOCK_SIZE) {
    const int k_block = smin(k - kk, HTP_OS_K_BLOCK_SIZE);
    const size_t slice_size = (size_t) n * k_block / QK_K * super_block_size;
    transfer_activation_chunk_fp32_to_fp16(vtcm_activation_probe, activation + kk, m, k_block, k);
    qk_os_copy_tiled_weight_k_block(vtcm_scratch_fix, weight, 0, n, k, kk, k_block, super_block_size);
    dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_slice, NULL, n * k_block, k_block, weight_type,
                                                      vtcm_scratch_fix);
    core_mma_chunk_fp16(vtcm_output_probe, vtcm_activation_probe, vtcm_weight_slice, vtcm_scales_probe,
                        vtcm_eye_probe, ceil_div(m, HMX_FP16_TILE_N_ROWS), ceil_div(n, HMX_FP16_TILE_N_COLS),
                        k_block / HMX_FP16_TILE_N_COLS, kk == 0);
    (void) slice_size;
  }
  transfer_output_chunk_fp16_to_fp32(split_full, vtcm_output_probe, m, n, n);
  qk_os_probe_compare_f32(results, &result_idx, max_results, 18, split_full, ref, m * n);

  qk_os_init_eye_tile_fp16_layout(vtcm_eye_probe, 0);
  memset(vtcm_output_probe, 0, output_size / sizeof(float) * sizeof(__fp16));
  memset(split_full, 0, output_size);
  for (int kk = 0; kk < k; kk += HTP_OS_K_BLOCK_SIZE) {
    const int k_block = smin(k - kk, HTP_OS_K_BLOCK_SIZE);
    const size_t fp16_slice_bytes = (size_t) n * k_block * sizeof(__fp16);
    transfer_activation_chunk_fp32_to_fp16(vtcm_activation_probe, activation + kk, m, k_block, k);
    qk_os_copy_tiled_weight_k_block(vtcm_scratch_fix, weight, 0, n, k, kk, k_block, super_block_size);
    dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_slice, NULL, n * k_block, k_block, weight_type,
                                                      vtcm_scratch_fix);
    qk_os_probe_compare_fp16_slice(results, &result_idx, max_results, kk == 0 ? 19 : 20, vtcm_weight_slice,
                                   vtcm_weight_full, k, n, kk, k_block);
    memcpy(vtcm_weight_fixed, vtcm_weight_slice, fp16_slice_bytes);
    core_mma_chunk_fp16(vtcm_output_probe, vtcm_activation_probe, vtcm_weight_fixed, vtcm_scales_probe,
                        vtcm_eye_probe, ceil_div(m, HMX_FP16_TILE_N_ROWS), ceil_div(n, HMX_FP16_TILE_N_COLS),
                        k_block / HMX_FP16_TILE_N_COLS, kk == 0);
  }
  transfer_output_chunk_fp16_to_fp32(split_full, vtcm_output_probe, m, n, n);
  qk_os_probe_compare_f32(results, &result_idx, max_results, 21, split_full, ref, m * n);

  mat_mul_qk_0_d16a32_out_stationary_impl(old_os, activation, weight, m, k, n, weight_type, 0, 0, 0, NULL, false);
  mat_mul_qk_0_d16a32_out_stationary_impl(fixed_os, activation, weight, m, k, n, weight_type, 0, 0, 0, NULL, true);

  qk_os_probe_compare_f32(results, &result_idx, max_results, 16, old_os, ref, m * n);
  qk_os_probe_compare_f32(results, &result_idx, max_results, 17, fixed_os, ref, m * n);

  FARF(ALWAYS,
       "QK_OS_DEBUG_PROBE selector meanings: 1 old linear weight kk0, 2 scalar-gather weight kk0, 3 old linear "
       "weight kk512, 4 scalar-gather weight kk512, 5 tiled weight kk0, 6 tiled weight kk512, 7 full-K core_mma, "
       "8..11 identity restore layouts, 12..15 split-K from full dequant, 16 old OS output, 17 tiled OS output, "
       "18 manual tiled-qweight split-K, 19..20 manual tiled weights, 21 manual tiled with post-dequant memcpy");

  free(activation);
  free(weight);
  free(ref);
  free(old_os);
  free(fixed_os);
  free(full_mma);
  free(split_full);
  return 0;
}
