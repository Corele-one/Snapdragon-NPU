#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"
#include "dsp/scna_exp2.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "op_reg.h"

// for debug
#include <HAP_farf.h>
#include <HAP_perf.h>

typedef struct {
  worker_synctoken_t sync_ctx;
  unsigned int       task_id;
  int                n_tasks;
  // int                n_tot_chunks;
  // int                n_chunks_per_task;
  uint8_t           *vtcm_base;
  size_t             vtcm_size_per_thread;
  // params
  __fp16            *O;
  const __fp16      *Q, *K, *V, *mask;
  int                qo_len, kv_len, n_heads, n_kv_heads, head_dim;
  int                mode_flags;
  struct Figure8ProfileHeader *profile;
  struct LlmTraceProfileHeader *llm_profile;
  int64_t            trace_id;
  int                op_index;
} simple_fa_task_state_t;

static inline void swap_ptr(__fp16 **p0, __fp16 **p1) {
  __fp16 *t = *p0;
  *p0       = *p1;
  *p1       = t;
}

static inline void hvx_fill_uh(void *p, uint16_t v, size_t size) {
  assert(size % VLEN == 0);
  assert(is_aligned(p, VLEN));
  HVX_Vector  v_v    = Q6_Vh_vsplat_R(v);
  HVX_Vector *pv_out = (HVX_Vector *) p;
  for (int i = 0; i < size / VLEN; ++i) {
    *pv_out++ = v_v;
  }
}

static inline void hvx_fill_uw(void *p, uint32_t v, size_t size) {
  assert(size % VLEN == 0);
  assert(is_aligned(p, VLEN));
  HVX_Vector  v_v    = Q6_V_vsplat_R(v);
  HVX_Vector *pv_out = (HVX_Vector *) p;
  for (int i = 0; i < size / VLEN; ++i) {
    *pv_out++ = v_v;
  }
}

size_t fa_f16_compute_vtcm_usage(int gqa_factor, int head_dim, int n_rows, int n_cols) {
  const size_t g_br = align_up(gqa_factor * n_rows, HMX_FP16_TILE_N_ROWS);

  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(__fp16), 4096);    // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(n_cols * head_dim * sizeof(__fp16), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * n_cols * sizeof(__fp16), 4096);      // S, P: [Br', Bc]
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);        // D: [Br', Br']
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);                // m, l, rowmax, rowsum: [Br']
  const size_t row_vec_size   = align_up(n_cols * sizeof(__fp16), 256);

  size_t total = qo_tile_size * 3 /* Q, O0, O1 */ + kv_tile_size * 2 /* K, V */ + core_tile_size * 2 /* S, P */ +
                 d_tile_size /* D */ + col_vec_size * 4 + row_vec_size * 2 /* 2x row buffer */ +
                 512 /* HMX column scales */;
  return total;
}

#define MAX_G_BR        256
#define __vec_aligned__ __attribute__((aligned(VLEN)))

void find_chunk_size_common(size_t *blk_r, size_t *blk_c, int gqa_factor, int head_dim, int qo_len, int kv_len,
                            size_t limit, int nr_unit, int nc_unit, size_t (*compute_vtcm_usage)(int, int, int, int)) {
  size_t nr = nr_unit, nc = nc_unit;
  size_t nr_ok = nr, nc_ok = nc;
  assert(compute_vtcm_usage(gqa_factor, head_dim, nr, nc) <= limit);

  const size_t max_g_nr = MAX_G_BR;
  const size_t max_nr   = align_up(qo_len, nr_unit);
  const size_t max_nc   = align_up(kv_len, nc_unit);

  // increase Br first
  for (; nr <= max_nr && gqa_factor * nr <= max_g_nr; nr += nr_unit) {
    if (compute_vtcm_usage(gqa_factor, head_dim, nr, nc) > limit) {
      break;
    }

    nr_ok = nr;
  }

  // then increase Bc
  for (; nc <= max_nc; nc += nc_unit) {
    if (compute_vtcm_usage(gqa_factor, head_dim, nr_ok, nc) > limit) {
      break;
    }

    nc_ok = nc;
  }

  *blk_r = nr_ok, *blk_c = nc_ok;
}

void fa_f16_find_chunk_size(size_t *blk_r, size_t *blk_c, int gqa_factor, int head_dim, int qo_len, int kv_len,
                            size_t limit) {
  const int nr_unit = ceil_div(HMX_FP16_TILE_N_ROWS, gqa_factor);
  const int nc_unit = 64;

  find_chunk_size_common(blk_r, blk_c, gqa_factor, head_dim, qo_len, kv_len, limit, nr_unit, nc_unit,
                         fa_f16_compute_vtcm_usage);
}

#ifndef FIGURE8_ENABLE_PROFILE_TIMERS
#  define FIGURE8_ENABLE_PROFILE_TIMERS 1
#endif

#ifndef FIGURE8_ENABLE_LUT_EXP
#  define FIGURE8_ENABLE_LUT_EXP 0
#endif

#if FIGURE8_ENABLE_PROFILE_TIMERS
#  define ENABLE_PROFILE_TIMERS
#endif

#if defined(ENABLE_PROFILE_TIMERS)

#  define TIMER_DEFINE(name) int64_t name##_ticks = 0
#  define TIMER_START(name)  int64_t name##_t0 = HAP_perf_get_qtimer_count()
#  define TIMER_STOP(name)   name##_ticks += HAP_perf_get_qtimer_count() - name##_t0
#  define TIMER_US(name)     HAP_perf_qtimer_count_to_us(name##_ticks)
#  define TIMER_STOP_EVENT(name, component, block_r, block_c)                                                        \
    do {                                                                                                             \
      const int64_t name##_t1 = HAP_perf_get_qtimer_count();                                                         \
      name##_ticks += name##_t1 - name##_t0;                                                                         \
      figure8_profile_record_event(profile, component, enable_vgather_exp ? 1 : 0, qo_len, kv_len, n_heads,           \
                                   n_kv_heads, head_dim, kv_head_idx, worker_index, block_r, block_c, name##_t0,      \
                                   name##_t1, enable_scna_exp ? scna_hvx_params.layout : SCNA_LAYOUT_SERIAL,          \
                                   enable_scna_exp ? scna_hvx_params.width : 0);                                       \
    } while (0)

#else

#  define TIMER_DEFINE(name)
#  define TIMER_START(name)
#  define TIMER_STOP(name)
#  define TIMER_US(name)
#  define TIMER_STOP_EVENT(name, component, block_r, block_c)

#endif

#if defined(ENABLE_PROFILE_TIMERS)
static inline void figure8_profile_record_event(struct Figure8ProfileHeader *profile, int component, int lut_exp,
                                                int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
                                                int kv_head_idx, int worker_index, int block_r, int block_c,
                                                int64_t t0_tick, int64_t t1_tick, int scna_layout, int scna_width) {
  if (profile == NULL || profile->max_events <= 0) {
    return;
  }

  const int idx = __sync_fetch_and_add(&(profile->event_count), 1);
  if (idx < 0 || idx >= profile->max_events) {
    __sync_fetch_and_add(&(profile->event_overflow), 1);
    return;
  }

  struct Figure8ProfileEvent *events = figure8_profile_events(profile);
  const int64_t               dur_us = HAP_perf_qtimer_count_to_us(t1_tick - t0_tick);
  events[idx]                        = (struct Figure8ProfileEvent) {
                           .component  = component,
                           .lut_exp    = lut_exp,
                           .qo_len     = qo_len,
                           .kv_len     = kv_len,
                           .n_heads    = n_heads,
                           .n_kv_heads = n_kv_heads,
                           .head_dim   = head_dim,
                           .kv_head    = kv_head_idx,
                           .worker     = worker_index,
                           .block_r    = block_r,
                           .block_c    = block_c,
                           .scna_layout = scna_layout,
                           .scna_width = scna_width,
                           .reserved   = 0,
                           .t0_us      = HAP_perf_qtimer_count_to_us(t0_tick),
                           .t1_us      = HAP_perf_qtimer_count_to_us(t1_tick),
                           .dur_us     = dur_us,
  };
}
#endif

static inline void llm_trace_profile_record_flash_event(struct LlmTraceProfileHeader *profile, int64_t trace_id,
                                                        int mode_flags, int op_index, int stage, int unit, int worker,
                                                        int qo_len, int kv_len, int n_heads, int n_kv_heads,
                                                        int head_dim, int block_r, int block_c, int chunk_r,
                                                        int chunk_c, int64_t bytes, int64_t t0_tick,
                                                        int64_t t1_tick) {
  if (profile == NULL || (mode_flags & LLM_NPU_MODE_DETAILED_TRACE) == 0 ||
      profile->magic != LLM_TRACE_PROFILE_MAGIC || profile->max_events <= 0) {
    return;
  }
  if (t1_tick < t0_tick) {
    t1_tick = t0_tick;
  }

  const int idx = __sync_fetch_and_add(&(profile->event_count), 1);
  if (idx < 0 || idx >= profile->max_events) {
    __sync_fetch_and_add(&(profile->event_overflow), 1);
    return;
  }

  const int64_t t0_us = HAP_perf_qtimer_count_to_us(t0_tick);
  const int64_t t1_us = HAP_perf_qtimer_count_to_us(t1_tick);
  struct LlmTraceProfileEvent *events = llm_trace_profile_events(profile);
  events[idx]                        = (struct LlmTraceProfileEvent) {
                           .trace_id   = trace_id,
                           .op_index   = op_index,
                           .stage      = stage,
                           .unit       = unit,
                           .worker     = worker,
                           .m          = qo_len,
                           .k          = head_dim,
                           .n          = n_heads * head_dim,
                           .qo_len     = qo_len,
                           .kv_len     = kv_len,
                           .n_heads    = n_heads,
                           .n_kv_heads = n_kv_heads,
                           .head_dim   = head_dim,
                           .mr         = block_r,
                           .nc         = block_c,
                           .kk         = -1,
                           .chunk_m    = chunk_r,
                           .chunk_n    = chunk_c,
                           .chunk_k    = head_dim,
                           .flags      = mode_flags,
                           .bytes      = bytes,
                           .t0_us      = t0_us,
                           .t1_us      = t1_us,
                           .dur_us     = t1_us - t0_us,
  };
}

// pre-assert: D is multiple of 64
void simple_flash_attn_f16_core(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O,
                                const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                                const __fp16 *restrict qk_mask, int qo_len, int kv_len, int n_heads, int n_kv_heads,
                                int head_dim, int worker_index, int mode_flags, struct Figure8ProfileHeader *profile,
                                struct LlmTraceProfileHeader *llm_profile, int64_t trace_id, int op_index) {
  // "compile-time" configs
  // TODO: make them real compile-time constants (constexpr or template parameters)
  const int G = n_heads / n_kv_heads;  // GQA factor
  const int D = head_dim;

  const bool   qo_fp32_element = true;  // Q/O storage; SCNA evaluator itself is FP16
  const size_t qo_element_size = qo_fp32_element ? sizeof(float) : sizeof(__fp16);

  // NOTE(hzx): confirmed that non-constant `has_qk_mask` affects softmax performance, disable it for now
  assert(qk_mask != NULL);
  // const bool   has_qk_mask = (qk_mask != NULL);
  const bool   has_qk_mask = true;
  const size_t kv_pad_len  = align_up(kv_len, 64);

  const bool enable_vgather_exp = (mode_flags & LLM_NPU_MODE_LUT_EXP) != 0;  // use table lookup (vgather) to compute exp
  const bool enable_scna_exp    = scna_exp2_enabled(mode_flags);
  const bool use_fp32_exp       = false;  // compute FP32 exp
  scna_exp2_hvx_params_t scna_hvx_params;
  scna_exp2_prepare_hvx_params(&scna_hvx_params, mode_flags);
  assert(!enable_scna_exp || scna_hvx_params.width == 8);

  // determine block sizes
  size_t blk_sz_r, blk_sz_c;  // Br, Bc
  fa_f16_find_chunk_size(&blk_sz_r, &blk_sz_c, G, head_dim, qo_len, kv_len, vtcm_limit - vtcm);
  assert(blk_sz_c % 64 == 0);

  const size_t g_br = align_up(G * blk_sz_r, HMX_FP16_TILE_N_ROWS);  // Br'

  // FARF(ALWAYS, "%s: Br=%d Bc=%d Br'=%d", __func__, blk_sz_r, blk_sz_c, g_br);

  const size_t n_tiles_per_blk_r = g_br / HMX_FP16_TILE_N_ROWS;
  const size_t n_tiles_per_blk_c = blk_sz_c / HMX_FP16_TILE_N_COLS;

  // compute tile/vector sizes
  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(__fp16), 4096);      // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(blk_sz_c * head_dim * sizeof(__fp16), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * blk_sz_c * sizeof(__fp16), 4096);      // S, P: [Br', Bc]
  const size_t d_tile_size    = align_up(g_br * g_br * sizeof(__fp16), 4096);          // D: [Br', Br']
  const size_t col_vec_size   = align_up(g_br * sizeof(__fp16), 256);                  // m, l, rowmax, rowsum: [Br']
  const size_t row_vec_size   = align_up(blk_sz_c * sizeof(__fp16), 256);

  const size_t kv_ld_blk_sz   = head_dim;               // no * element_size
  const size_t kv_ld_stride   = n_kv_heads * head_dim;  // no * element_size
  const size_t qo_ldst_blk_sz = G * head_dim;           // no * element_size
  const size_t qo_ldst_stride = n_heads * head_dim;     // no * element_size

  // begin VTCM allocation
  uint8_t *vtcm_cur = vtcm;
  __fp16  *q_tile   = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile0  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  __fp16  *o_tile1  = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);

  __fp16 *k_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  __fp16 *v_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);

  __fp16 *s_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);
  __fp16 *p_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);

  __fp16 *d_tile = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, d_tile_size);

  HVX_Vector *mvec_m        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_l        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_s_rowmax = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);

  HVX_Vector *row_buffer0 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);
  HVX_Vector *row_buffer1 = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_size);

  uint8_t *hmx_output_scales_id = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);
  uint8_t *hmx_output_scales_qk = (uint8_t *) vtcm_seq_alloc(&vtcm_cur, 256);

  // end VTCM allocation
  assert(vtcm_cur <= vtcm_limit);

  float  qk_scale    = 1.0f / sqrtf(head_dim) * 1.44269504f;  // log2(e) = 1.44269504
  __fp16 qk_scale_hf = (__fp16) qk_scale;                     // NOTE: this conversion can be very slow

  // NOTE: there are 32 effective elements in scales, use 4 bytes splat (not Q6_Vh_vsplat_R)
  hmx_init_column_scales(hmx_output_scales_id, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0
  hmx_init_column_scales(hmx_output_scales_qk, Q6_V_vsplat_R(fp16_to_bits(&qk_scale_hf)));

  // prepare constants
  static int32_t transpose_vscatter_indices_base[32] __vec_aligned__;
  for (int i = 0; i < 32; ++i) {
    transpose_vscatter_indices_base[i] = i * 128;  // range [0, 4096), two HMX tiles
  }

  static int16_t d_tile_vscatter_offsets[64] __vec_aligned__;
  for (int i = 0; i < 16; ++i) {
    // offsets within the first tile
    d_tile_vscatter_offsets[i * 2 + 0] = i * 136;
    d_tile_vscatter_offsets[i * 2 + 1] = i * 136 + 6;
  }

  // alternative computation method
  const int    sub_table_idx  = worker_index % 4;
  const size_t sub_table_size = 65536;

  uint8_t *vtcm_exp2_table_base = vtcm_manager_query_area("safe_softmax::exp2_hf_qf16");
  uint8_t *vtcm_exp2_table      = vtcm_exp2_table_base + sub_table_idx * sub_table_size;
  if (!enable_vgather_exp) {
    (void) vtcm_exp2_table;
  }

  // profile timers
  TIMER_DEFINE(q_load);
  TIMER_DEFINE(k_load);
  TIMER_DEFINE(v_load);
  TIMER_DEFINE(qk_dot);
  TIMER_DEFINE(safe_sm);   // safe softmax
  TIMER_DEFINE(core_acc);  // core accumulation
  TIMER_DEFINE(o_scale);
  TIMER_DEFINE(o_store);
  TIMER_DEFINE(scna_exp);

  /////////////// CORE LOGIC BEGIN

  for (int ir = 0; ir < qo_len; ir += blk_sz_r) {
    const size_t n_rows        = smin(qo_len - ir, blk_sz_r);
    const size_t n_rows_g      = n_rows * G;
    const size_t n_row_tiles   = ceil_div(n_rows_g, HMX_FP16_TILE_N_ROWS);
    const size_t n_row_vec_cnt = ceil_div(n_rows_g, 64);

    // load [n_rows*G, D] tile of Q into VTCM
    TIMER_START(q_load);
    int64_t q_load_stage_t0 = HAP_perf_get_qtimer_count();
    {
      // load block size: G*D elements
      const size_t q_ld_blk_sz_bytes = qo_ldst_blk_sz * qo_element_size;
      const size_t q_ld_stride_bytes = qo_ldst_stride * qo_element_size;  // a.k.a. hidden_size

      const uint8_t *q_ld_base = ((uint8_t *) Q) + ir * q_ld_stride_bytes + kv_head_idx * q_ld_blk_sz_bytes;

      // FIXME(hzx): This L2 fetch may not be very useful
      // NOTE(hzx): what about prefetching in reverse order?
      l2fetch(q_ld_base, q_ld_stride_bytes, q_ld_blk_sz_bytes, n_rows, 1);

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        // input positions
        int q_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int q_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        const HVX_Vector *pv_in0 =
          (const HVX_Vector *) (q_ld_base + q_idx0 * q_ld_stride_bytes + h_idx0 * head_dim * qo_element_size);
        const HVX_Vector *pv_in1 =
          (const HVX_Vector *) (q_ld_base + q_idx1 * q_ld_stride_bytes + h_idx1 * head_dim * qo_element_size);

        // output positions
        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        __fp16 *out_base = q_tile + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector v0 = *pv_in0++;
            const HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

            const HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

            HVX_Vector *out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]     = v_out;
          }
        } else {
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            // locate target dual-tile
            __fp16     *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector *pv_out0       = ((HVX_Vector *) out_dual_tile) + r1 / 2;
            HVX_Vector *pv_out1       = pv_out0 + 16;  // 16 * 128B = 2048B (1 tile)

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
          }
        }
        // clang-format on
      }
    }
    int64_t q_load_stage_t1 = HAP_perf_get_qtimer_count();
    llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_Q_LOAD,
                                         LLM_TRACE_UNIT_MEMORY, worker_index, qo_len, kv_len, n_heads, n_kv_heads,
                                         head_dim, ir, -1, n_rows_g, 0,
                                         (int64_t) n_rows_g * head_dim * (int64_t) qo_element_size, q_load_stage_t0,
                                         q_load_stage_t1);
    TIMER_STOP_EVENT(q_load, FIGURE8_COMP_Q_LOAD, ir, -1);

    hvx_fill_uh(mvec_m, 0xfbff, col_vec_size);  // init to -65504 (-inf)
    hvx_fill_uh(mvec_l, 0, col_vec_size);       // init: 0

    __fp16 *o_tile_prev = o_tile0;
    __fp16 *o_tile_curr = o_tile1;

    hvx_fill_uh(o_tile_prev, 0, qo_tile_size);
    hvx_fill_uh(d_tile, 0, d_tile_size);

    // inner loop over kv_len
    for (int jc = 0; jc < kv_len; jc += blk_sz_c) {
      const size_t n_cols      = smin(kv_len - jc, blk_sz_c);
      const size_t n_col_tiles = ceil_div(n_cols, HMX_FP16_TILE_N_COLS);

      // load [Bc, D] tile of K^T into VTCM
      // TODO(hzx): use DMA? if DMA used, we should read from VTCM
      TIMER_START(k_load);
      int64_t k_load_stage_t0 = HAP_perf_get_qtimer_count();
      {
        const __fp16 *k_ld_base = K + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        // FIXME: Is this necessary?
        l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 1);

        const HVX_Vector v_step         = Q6_V_vsplat_R(4);
        const HVX_Vector v_offsets_base = vmem(transpose_vscatter_indices_base);

        // continuous fetch loop: [Bc/32, 32, D]
        for (int r0 = 0; r0 < n_col_tiles; ++r0) {
          __fp16 *out_base = k_tile + r0 * HMX_FP16_TILE_N_COLS * head_dim;  // transposed [D, 32] column chunk

          HVX_Vector v_offsets = v_offsets_base;                             // reset to base offsets

          for (int r1 = 0; r1 < HMX_FP16_TILE_N_COLS; ++r1) {
            int r = r0 * HMX_FP16_TILE_N_COLS + r1;
            if (r >= n_cols) {
              break;
            }

            const HVX_Vector *pv_in = (const HVX_Vector *) (k_ld_base + r * kv_ld_stride);

            // clang-format off
            #pragma unroll
            for (int d = 0; d < D / 64; ++d) {
              __fp16 *out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
              Q6_vscatter_RMVwV((size_t) out_dual_tile, HMX_FP16_TILE_SIZE * 2 - 1, v_offsets, *pv_in++);
            }
            // clang-format on

            v_offsets = Q6_Vw_vadd_VwVw(v_offsets, v_step);
          }
        }
      }
      int64_t k_load_stage_t1 = HAP_perf_get_qtimer_count();
      llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_K_LOAD,
                                           LLM_TRACE_UNIT_MEMORY, worker_index, qo_len, kv_len, n_heads, n_kv_heads,
                                           head_dim, ir, jc, n_rows_g, n_cols,
                                           (int64_t) n_cols * head_dim * (int64_t) sizeof(__fp16), k_load_stage_t0,
                                           k_load_stage_t1);
      TIMER_STOP_EVENT(k_load, FIGURE8_COMP_K_LOAD, ir, jc);

      // issue L2 prefetch of V tile
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 0);
      }

      // compute dot product of tiles: dot(Q[Br', D], K[Bc, D]) ==> [Br', Bc]
      TIMER_START(qk_dot);
      int64_t qk_dot_stage_t0 = HAP_perf_get_qtimer_count();
      {
        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_qk);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < n_col_tiles; ++c) {
              const __fp16 *row_tiles = q_tile + r * HMX_FP16_TILE_N_ROWS * head_dim;
              const __fp16 *col_tiles = k_tile + c * HMX_FP16_TILE_N_COLS * head_dim;

              // NOTE: we use `n_tiles_per_blk_c` instead of `n_col_tiles` here
              __fp16 *out_tile = s_tile + (r * n_tiles_per_blk_c + c) * HMX_FP16_TILE_N_ELMS;
              hmx_dot_fp16(out_tile, row_tiles, col_tiles, head_dim / 32);
            }
          }
        }
        hmx_unit_release();
      }
      int64_t qk_dot_stage_t1 = HAP_perf_get_qtimer_count();
      llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_QK_DOT,
                                           LLM_TRACE_UNIT_HMX, worker_index, qo_len, kv_len, n_heads, n_kv_heads,
                                           head_dim, ir, jc, n_rows_g, n_cols,
                                           2LL * n_rows_g * n_cols * head_dim, qk_dot_stage_t0, qk_dot_stage_t1);
      TIMER_STOP_EVENT(qk_dot, FIGURE8_COMP_QK_DOT, ir, jc);

      // core softmax computation
      TIMER_START(safe_sm);
      int64_t safe_sm_stage_t0 = HAP_perf_get_qtimer_count();
      {
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);  // fp16: -65504

        // read from S tile, process 2 rows at a time, generate P tile
        for (int r_vec_idx = 0; r_vec_idx < n_row_vec_cnt; ++r_vec_idx) {
          // vector registers, empty when initialized, fill in 2 rows at a time
          HVX_Vector v_s_rowmax_local = v_neg_inf;
          HVX_Vector v_p_rowsum_local = Q6_V_vzero();

          for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            int r = r_vec_idx * 64 + r_vec_off;
            if (r >= align_up(n_rows_g, 2)) {
              break;
            }

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;

            // NOTE: make sure this match with S tile generation logic
            __fp16 *s_ld_base = s_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;
            __fp16 *p_st_base = p_tile + r0 * HMX_FP16_TILE_N_ROWS * blk_sz_c;

            // decode 2 rows into row buffers
            HVX_Vector *pv_row_buf0 = row_buffer0;
            HVX_Vector *pv_row_buf1 = row_buffer1;
            for (int c = 0; c < n_cols; c += 64) {
              const __fp16     *in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              const HVX_Vector *pv_s_in0     = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
              const HVX_Vector *pv_s_in1     = pv_s_in0 + 16;  // 16 * 128B = 2048B (1 tile)

              HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
              *pv_row_buf0++               = Q6_V_lo_W(vp_s_dual_row);
              *pv_row_buf1++               = Q6_V_hi_W(vp_s_dual_row);
            }

            // apply mask & compute rowmax(S)
            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            // reduction phase 1: inter-vector
            for (int c = 0; c < n_cols; c += 64) {
              int q_idx0 = ir + (r + 0) / G;
              int q_idx1 = ir + (r + 1) / G;
              int k_idx  = jc + c;

              HVX_VectorPred q_mask_keep0, q_mask_keep1;
              if (has_qk_mask) {
                HVX_Vector v_mask0 = vmemu(qk_mask + q_idx0 * kv_pad_len + k_idx);
                HVX_Vector v_mask1 = vmemu(qk_mask + q_idx1 * kv_pad_len + k_idx);

                const HVX_Vector v_fp16_mask_threshold = Q6_Vh_vsplat_R(0xcc00);  // fp16: -16.0

                q_mask_keep0 = Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_fp16_mask_threshold);
                q_mask_keep1 = Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_fp16_mask_threshold);
              } else {
                const size_t ne = smin(n_cols - c, 64);

                q_mask_keep0 = q_mask_keep1 = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
              }

              HVX_Vector v_s_row0 = Q6_V_vmux_QVV(q_mask_keep0, row_buffer0[c / 64], v_neg_inf);
              HVX_Vector v_s_row1 = Q6_V_vmux_QVV(q_mask_keep1, row_buffer1[c / 64], v_neg_inf);

              row_buffer0[c / 64] = v_s_row0;
              row_buffer1[c / 64] = v_s_row1;

              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_s_row0);
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_s_row1);
            }

            // clang-format off
            // reduction phase 2: intra-vector
            #pragma unroll
            for (int s = 64; s >= 2; s >>= 1) {
              v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, Q6_V_vlalign_VVR(v_s_rowmax0, v_neg_inf, s));
              v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, Q6_V_vlalign_VVR(v_s_rowmax1, v_neg_inf, s));
            }
            // clang-format on
            // now, v_s_rowmax0[63] = rowmax(S)_0, v_s_rowmax1[63] = rowmax(S)_1

            // shift rowmax(S_i^j) into v_s_rowmax_local
            HVX_Vector v_s_rowmax_pack2 =
              Q6_V_hi_W(Q6_W_vshuff_VVR(v_s_rowmax1, v_s_rowmax0, -2));    // highest 4 bytes are valid
            HVX_Vector v_s_rowmax_pack2_rot =
              Q6_V_vror_VR(v_s_rowmax_pack2, VLEN - 2 * sizeof(__fp16));   // lowest 4 bytes valid
            HVX_Vector v_s_rowmax_local_rot =
              Q6_V_vror_VR(v_s_rowmax_local, r_vec_off * sizeof(__fp16));  // highest r*2 bytes valid
            const HVX_VectorPred q_two_rows = Q6_Q_vsetq_R(2 * sizeof(__fp16));
            v_s_rowmax_local_rot = Q6_V_vmux_QVV(q_two_rows, v_s_rowmax_pack2_rot, v_s_rowmax_local_rot);
            v_s_rowmax_local = Q6_V_vror_VR(v_s_rowmax_local_rot,
                                             (VLEN - r_vec_off * sizeof(__fp16)) % VLEN);

            // compute m_i^j = max(m_i^{j-1}, rowmax(S_i^j))
            HVX_Vector v_m_cur = Q6_Vhf_vmax_VhfVhf(mvec_m[r_vec_idx], v_s_rowmax_local);

            // broadcast new m_0^j and m_1^j to whole vectors using LUT
            HVX_Vector v_m_lut  = Q6_V_vror_VR(v_m_cur, r_vec_off * sizeof(__fp16));  // lowest 4 bytes are valid
            HVX_Vector v_dup_m0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 0));
            HVX_Vector v_dup_m1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(Q6_V_vzero(), v_m_lut, 2));

            // compute rows of P_i^j = exp(S_i^j - m_i^j)
            // write permuted rows of P tile into VTCM
            // compute rowsum(P)
            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;  // qfloat
            HVX_Vector       v_p_rowsum1 = v_zero;  // qfloat

            if (enable_vgather_exp) {
              for (int c = 0; c < n_cols; c += 64) {
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(row_buffer0[c / 64], v_dup_m0);
                HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(row_buffer1[c / 64], v_dup_m1);

                HVX_Vector v_s_minus_m0_hf = Q6_Vhf_equals_Vqf16(v_s_minus_m0);
                HVX_Vector v_s_minus_m1_hf = Q6_Vhf_equals_Vqf16(v_s_minus_m1);
                HVX_Vector v_gather_input0 = Q6_Vh_vasl_VhR(v_s_minus_m0_hf, 1);
                HVX_Vector v_gather_input1 = Q6_Vh_vasl_VhR(v_s_minus_m1_hf, 1);

                Q6_vgather_ARMVh(&row_buffer0[c / 64], (size_t) vtcm_exp2_table, 65535, v_gather_input0);
                Q6_vgather_ARMVh(&row_buffer1[c / 64], (size_t) vtcm_exp2_table, 65535, v_gather_input1);
              }

              const int sync_idx = (n_cols - 64) / 64;
              volatile HVX_Vector *sync0 = (volatile HVX_Vector *) &row_buffer0[sync_idx];
              volatile HVX_Vector *sync1 = (volatile HVX_Vector *) &row_buffer1[sync_idx];
              HVX_Vector           sync_v0 = *sync0;
              HVX_Vector           sync_v1 = *sync1;
              (void) sync_v0;
              (void) sync_v1;
            }

            const int64_t scna_score_t0 = enable_scna_exp ? HAP_perf_get_qtimer_count() : 0;
            for (int c = 0; c < n_cols; c += 64) {
              HVX_Vector v_p_row0_hf, v_p_row1_hf;

              if (enable_scna_exp) {
                HVX_Vector v_s_minus_m0 = Q6_Vhf_equals_Vqf16(
                  Q6_Vqf16_vsub_VhfVhf(row_buffer0[c / 64], v_dup_m0));
                HVX_Vector v_s_minus_m1 = Q6_Vhf_equals_Vqf16(
                  Q6_Vqf16_vsub_VhfVhf(row_buffer1[c / 64], v_dup_m1));
                const HVX_VectorPred q_valid0 = Q6_Q_vcmp_gt_VhfVhf(row_buffer0[c / 64], v_neg_inf);
                const HVX_VectorPred q_valid1 = Q6_Q_vcmp_gt_VhfVhf(row_buffer1[c / 64], v_neg_inf);
                const HVX_Vector v_zero_scna = Q6_V_vzero();
                __fp16 scna_min_hf = (__fp16) SCNA_MIN_INPUT;
                const HVX_Vector v_scna_min = Q6_Vh_vsplat_R(fp16_to_bits(&scna_min_hf));
                v_s_minus_m0 = Q6_V_vmux_QVV(q_valid0, v_s_minus_m0, v_scna_min);
                v_s_minus_m1 = Q6_V_vmux_QVV(q_valid1, v_s_minus_m1, v_scna_min);
                HVX_Vector v_scna_row0, v_scna_row1;
                hvx_scna_exp2_pair_vhf(v_s_minus_m0, v_s_minus_m1, &scna_hvx_params,
                                        &v_scna_row0, &v_scna_row1);
                v_p_row0_hf = Q6_V_vmux_QVV(q_valid0, v_scna_row0, v_zero_scna);
                v_p_row1_hf = Q6_V_vmux_QVV(q_valid1, v_scna_row1, v_zero_scna);
              } else if (enable_vgather_exp) {
                v_p_row0_hf = row_buffer0[c / 64];
                v_p_row1_hf = row_buffer1[c / 64];
              } else {
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(row_buffer0[c / 64], v_dup_m0);  // qf16
                HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(row_buffer1[c / 64], v_dup_m1);  // qf16

                if (use_fp32_exp) {
                  HVX_VectorPair vp_s_minus_m0_sf = hvx_my_vqf16_to_wsf(v_s_minus_m0);
                  HVX_VectorPair vp_s_minus_m1_sf = hvx_my_vqf16_to_wsf(v_s_minus_m1);

                  HVX_Vector v_p_row00_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m0_sf));
                  HVX_Vector v_p_row01_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m0_sf));
                  HVX_Vector v_p_row10_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp_s_minus_m1_sf));
                  HVX_Vector v_p_row11_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp_s_minus_m1_sf));

                  v_p_row0_hf = hvx_my_wsf_to_vhf(v_p_row01_sf, v_p_row00_sf);
                  v_p_row1_hf = hvx_my_wsf_to_vhf(v_p_row11_sf, v_p_row10_sf);
                } else {
                  v_p_row0_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m0);
                  v_p_row1_hf = hvx_my_exp2_vhf_vqf16(v_s_minus_m1);
                }
              }

              // compute P tile output positions
              __fp16     *out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
              HVX_Vector *pv_p_out0     = ((HVX_Vector *) out_dual_tile) + r1 / 2;
              HVX_Vector *pv_p_out1     = pv_p_out0 + 16;  // 16 * 128B = 2048B (1 tile)

              // write to P tile
              HVX_VectorPair vp_p_dual_row = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
              *pv_p_out0                   = Q6_V_lo_W(vp_p_dual_row);
              *pv_p_out1                   = Q6_V_hi_W(vp_p_dual_row);

              // rowsum(P) phase 1 reduction
              // v_p_rowsum0 = Q6_Vqf16_vadd_Vqf16Vhf(v_p_rowsum0, v_p_row0_hf);
              // v_p_rowsum1 = Q6_Vqf16_vadd_Vqf16Vhf(v_p_rowsum1, v_p_row1_hf);

              // reduce sum using qf32 precision
              HVX_VectorPair vp_p_row0 = hvx_my_vhf_to_wqf32(v_p_row0_hf);
              HVX_VectorPair vp_p_row1 = hvx_my_vhf_to_wqf32(v_p_row1_hf);

              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row0), Q6_V_hi_W(vp_p_row0)));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(
                v_p_rowsum1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(vp_p_row1), Q6_V_hi_W(vp_p_row1)));
            }
            if (enable_scna_exp) {
              const int64_t scna_score_t1 = HAP_perf_get_qtimer_count();
              scna_exp_ticks += scna_score_t1 - scna_score_t0;
              figure8_profile_record_event(profile, FIGURE8_COMP_SCNA_EXP, 0, qo_len, kv_len, n_heads,
                                           n_kv_heads, head_dim, kv_head_idx, worker_index, ir + r, jc,
                                           scna_score_t0, scna_score_t1, scna_hvx_params.layout,
                                           scna_hvx_params.width);
            }

            // clang-format off
            // rowsum(P) phase 2 reduction
            // #pragma unroll
            // for (int s = 64; s >= 2; s >>= 1) {
            //   v_p_rowsum0 = Q6_Vqf16_vadd_Vqf16Vqf16(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
            //   v_p_rowsum1 = Q6_Vqf16_vadd_Vqf16Vqf16(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
            // }
            // clang-format on
            // now, v_p_rowsum0[63] = rowsum(P)_0, v_p_rowsum1[63] = rowsum(P)_1

#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_V_vlalign_VVR(v_p_rowsum0, v_zero, s));
              v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_V_vlalign_VVR(v_p_rowsum1, v_zero, s));
            }
            HVX_Vector v_p_rowsum_pack2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v_p_rowsum1, v_p_rowsum0));

            // shift rowsum(P) into v_p_rowsum_local
            // HVX_Vector v_p_rowsum_pack2     = Q6_V_hi_W(Q6_W_vshuff_VVR(v_p_rowsum1, v_p_rowsum0, -2));
            HVX_Vector v_p_rowsum_pack2_rot = Q6_V_vror_VR(v_p_rowsum_pack2, VLEN - 2 * sizeof(__fp16));
            HVX_Vector v_p_rowsum_local_rot = Q6_V_vror_VR(v_p_rowsum_local, r_vec_off * sizeof(__fp16));
            v_p_rowsum_local_rot = Q6_V_vmux_QVV(q_two_rows, v_p_rowsum_pack2_rot, v_p_rowsum_local_rot);
            v_p_rowsum_local = Q6_V_vror_VR(v_p_rowsum_local_rot,
                                             (VLEN - r_vec_off * sizeof(__fp16)) % VLEN);
          }

          // write local vector registers back to VTCM
          mvec_s_rowmax[r_vec_idx] = v_s_rowmax_local;
          mvec_p_rowsum[r_vec_idx] = v_p_rowsum_local;
        }
      }
      int64_t safe_sm_stage_t1 = HAP_perf_get_qtimer_count();
      llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_SAFE_SM,
                                           LLM_TRACE_UNIT_HVX, worker_index, qo_len, kv_len, n_heads, n_kv_heads,
                                           head_dim, ir, jc, n_rows_g, n_cols,
                                           (int64_t) n_rows_g * n_cols * (int64_t) sizeof(__fp16), safe_sm_stage_t0,
                                           safe_sm_stage_t1);
      TIMER_STOP_EVENT(safe_sm, FIGURE8_COMP_SAFE_SM, ir, jc);

      // load [Bc, D] tile of V into VTCM
      TIMER_START(v_load);
      int64_t v_load_stage_t0 = HAP_perf_get_qtimer_count();
      {
        // NOTE: at tile granularity, tile V's layout is column-major rather than row-major
        // because V tile is an RHS of matmul and HMX's dot RHS operands are column-major tiles
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        for (int r = 0; r < n_cols; r += 2) {
          const bool next_row_valid = (r + 1) < n_cols;

          const HVX_Vector *pv_in0 = (const HVX_Vector *) (v_ld_base + (r + 0) * kv_ld_stride);
          const HVX_Vector *pv_in1 = (const HVX_Vector *) (v_ld_base + (r + 1) * kv_ld_stride);

          // clang-format off
          #pragma unroll
          for (int c = 0; c < D; c += 64) {
            const HVX_Vector     v0 = *pv_in0++;
            const HVX_Vector     v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
            const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;
            int c0 = c / HMX_FP16_TILE_N_COLS;

            // transposed tile index: (c0, r0) => c0 * Bc/32 + r0
            int     tile_idx0  = (c0 + 0) * n_tiles_per_blk_c + r0;
            int     tile_idx1  = (c0 + 1) * n_tiles_per_blk_c + r0;
            __fp16 *tile_base0 = v_tile + tile_idx0 * HMX_FP16_TILE_N_ELMS;
            __fp16 *tile_base1 = v_tile + tile_idx1 * HMX_FP16_TILE_N_ELMS;

            HVX_Vector *pv_out0 = ((HVX_Vector *) tile_base0) + r1 / 2;
            HVX_Vector *pv_out1 = ((HVX_Vector *) tile_base1) + r1 / 2;
            *pv_out0            = Q6_V_lo_W(vp);
            *pv_out1            = Q6_V_hi_W(vp);
          }
          // clang-format on
        }
      }
      int64_t v_load_stage_t1 = HAP_perf_get_qtimer_count();
      llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_V_LOAD,
                                           LLM_TRACE_UNIT_MEMORY, worker_index, qo_len, kv_len, n_heads, n_kv_heads,
                                           head_dim, ir, jc, n_rows_g, n_cols,
                                           (int64_t) n_cols * head_dim * (int64_t) sizeof(__fp16), v_load_stage_t0,
                                           v_load_stage_t1);
      TIMER_STOP_EVENT(v_load, FIGURE8_COMP_V_LOAD, ir, jc);

      // issue L2 prefetch of the next K tile
      {
        int jc_next = jc + blk_sz_c;
        if (jc_next < kv_len) {
          const size_t n_cols_next = smin(kv_len - jc_next, blk_sz_c);

          const __fp16 *k_ld_base = K + jc_next * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
          l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols_next, 0);
        }
      }

      // NOTE: after the use of rowmax(S), store exp(m_i^{j-1} - m_i^j) in the very same VTCM buffer
      HVX_Vector *mvec_exp_m_diff = mvec_s_rowmax;

      TIMER_START(core_acc);
      int64_t core_acc_hvx_stage_t0 = HAP_perf_get_qtimer_count();
      // update rowmax vector m_i and vector l_i
      {
        for (int i = 0; i < n_row_vec_cnt; ++i) {  // i => r_vec_idx?
          HVX_Vector v_m_prev = mvec_m[i];
          HVX_Vector v_m_curr = Q6_Vhf_vmax_VhfVhf(v_m_prev, mvec_s_rowmax[i]);
          HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);  // qf16

          HVX_Vector v_exp_m_diff_hf;
          if (enable_scna_exp) {
            const int64_t scna_online_t0 = HAP_perf_get_qtimer_count();
            HVX_Vector v_m_diff_hf = Q6_Vhf_equals_Vqf16(v_m_diff);
            __fp16 scna_min_hf = (__fp16) SCNA_MIN_INPUT;
            const HVX_Vector v_scna_min = Q6_Vh_vsplat_R(fp16_to_bits(&scna_min_hf));
            const HVX_Vector v_scna_neg_inf = Q6_Vh_vsplat_R(0xfc00);
            const HVX_VectorPred q_finite = Q6_Q_vcmp_gt_VhfVhf(v_m_diff_hf, v_scna_neg_inf);
            v_m_diff_hf = Q6_V_vmux_QVV(q_finite, v_m_diff_hf, v_scna_min);
            v_exp_m_diff_hf = hvx_scna_exp2_vhf(v_m_diff_hf, &scna_hvx_params);
            const int64_t scna_online_t1 = HAP_perf_get_qtimer_count();
            scna_exp_ticks += scna_online_t1 - scna_online_t0;
            figure8_profile_record_event(profile, FIGURE8_COMP_SCNA_EXP, 0, qo_len, kv_len, n_heads,
                                         n_kv_heads, head_dim, kv_head_idx, worker_index, ir + i * 64, jc,
                                         scna_online_t0, scna_online_t1, scna_hvx_params.layout,
                                         scna_hvx_params.width);
          } else {
            v_exp_m_diff_hf = hvx_my_exp2_vhf_vqf16(v_m_diff);    // fp16
          }

          // l_i^j = exp(m_i^{j-1} - m_i^j) * l_i^{j-1} + rowsum(P_i^j)
          HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(mvec_l[i], v_exp_m_diff_hf);  // qf16
          v_l_curr            = Q6_Vqf16_vadd_Vqf16Vhf(v_l_curr, mvec_p_rowsum[i]);

          mvec_m[i] = v_m_curr;
          mvec_l[i] = v_l_curr;

          mvec_exp_m_diff[i] = v_exp_m_diff_hf;  // fp16
        }
      }
      int64_t core_acc_hvx_stage_t1 = HAP_perf_get_qtimer_count();
      llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index,
                                           LLM_TRACE_STAGE_FLASH_CORE_ACC, LLM_TRACE_UNIT_HVX, worker_index, qo_len,
                                           kv_len, n_heads, n_kv_heads, head_dim, ir, jc, n_rows_g, n_cols,
                                           (int64_t) n_rows_g * (int64_t) sizeof(__fp16), core_acc_hvx_stage_t0,
                                           core_acc_hvx_stage_t1);

      // compute O_i^j = diag(exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + P_i^j V_j
      int64_t core_acc_hmx_stage_t0 = HAP_perf_get_qtimer_count();
      {
        // generate D tile = diag(exp(m_i^{j-1} - m_i^j))
        const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
        const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        for (int i = 0; i < n_row_tiles; ++i) {
          const HVX_Vector v_content = Q6_V_vror_VR(mvec_exp_m_diff[i / 2], (i % 2) * 64);

          __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
          Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        hmx_unit_acquire();
        {
          hmx_set_output_scales(hmx_output_scales_id);
          for (int r = 0; r < n_row_tiles; ++r) {
            for (int c = 0; c < head_dim / 32; ++c) {
              __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;  // D: [Br', Br']
              __fp16 *o_tile_in =
                o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;  // O: [Br', D] --T-> [D, Br']
              hmx_load_tiles_fp16(d_tile_in, o_tile_in, n_row_tiles);

              __fp16 *p_tile_in = p_tile + (r * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;  // P: [Br', Bc]
              __fp16 *v_tile_in = v_tile + (c * n_tiles_per_blk_c) * HMX_FP16_TILE_N_ELMS;  // V: [Bc, D] --T-> [D, Bc]
              // NOTE: `n_col_tiles` may exceed 32, we need to explicitly split and accumulate
              for (int k = 0; k < n_col_tiles; k += 32) {
                int    offset  = k * HMX_FP16_TILE_N_ELMS;
                size_t n_tiles = smin(n_col_tiles - k, 32);
                hmx_load_tiles_fp16(p_tile_in + offset, v_tile_in + offset, n_tiles);
              }

              // NOTE: O's layout is also column-major as O is always on the RHS
              __fp16 *o_tile_out = o_tile_curr + (c * n_tiles_per_blk_r + r) * HMX_FP16_TILE_N_ELMS;
              hmx_consume_accumulator_fp16(o_tile_out);
            }
          }
        }
        hmx_unit_release();

        swap_ptr(&o_tile_curr, &o_tile_prev);
      }
      int64_t core_acc_hmx_stage_t1 = HAP_perf_get_qtimer_count();
      llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index,
                                           LLM_TRACE_STAGE_FLASH_CORE_ACC, LLM_TRACE_UNIT_HMX, worker_index, qo_len,
                                           kv_len, n_heads, n_kv_heads, head_dim, ir, jc, n_rows_g, n_cols,
                                           2LL * n_rows_g * n_cols * head_dim, core_acc_hmx_stage_t0,
                                           core_acc_hmx_stage_t1);
      TIMER_STOP_EVENT(core_acc, FIGURE8_COMP_CORE_ACC, ir, jc);
    }

    // generate final output: scale O_i = diag(l_i^{-1}) O_i
    TIMER_START(o_scale);
    int64_t o_scale_stage_t0 = HAP_perf_get_qtimer_count();
    {
      const HVX_Vector     v_offsets       = vmem(d_tile_vscatter_offsets);
      const HVX_VectorPred q_32_elems_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

      HVX_Vector v_content;
      for (int i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
          v_content = hvx_my_inv_vhf(Q6_Vhf_equals_Vqf16(mvec_l[i / 2]));
        } else {
          v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 *out_base = d_tile + i * (n_tiles_per_blk_r + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_elems_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
      }

      hmx_unit_acquire();
      {
        hmx_set_output_scales(hmx_output_scales_id);
        for (int r = 0; r < n_row_tiles; ++r) {
          for (int c = 0; c < head_dim / 32; ++c) {
            __fp16 *d_tile_in = d_tile + (r * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *o_tile_in = o_tile_prev + (c * n_tiles_per_blk_r) * HMX_FP16_TILE_N_ELMS;

            // NOTE: to simplify final output procedure, we turn final O into row-major layout
            __fp16 *o_tile_out = o_tile_curr + (r * head_dim / 32 + c) * HMX_FP16_TILE_N_ELMS;

            hmx_dot_fp16(o_tile_out, d_tile_in, o_tile_in, n_row_tiles);
          }
        }
      }
      hmx_unit_release();
    }
    int64_t o_scale_stage_t1 = HAP_perf_get_qtimer_count();
    llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_O_SCALE,
                                         LLM_TRACE_UNIT_HMX, worker_index, qo_len, kv_len, n_heads, n_kv_heads,
                                         head_dim, ir, -1, n_rows_g, 0, 2LL * n_rows_g * head_dim, o_scale_stage_t0,
                                         o_scale_stage_t1);
    TIMER_STOP_EVENT(o_scale, FIGURE8_COMP_O_SCALE, ir, -1);

    // store [n_rows*G, D] tile of O back to memory
    TIMER_START(o_store);
    int64_t o_store_stage_t0 = HAP_perf_get_qtimer_count();
    {
      const size_t o_st_blk_sz_bytes = qo_ldst_blk_sz * qo_element_size;
      const size_t o_st_stride_bytes = qo_ldst_stride * qo_element_size;

      uint8_t *o_st_base = ((uint8_t *) O) + ir * o_st_stride_bytes + kv_head_idx * o_st_blk_sz_bytes;

      for (int r = 0; r < n_rows_g; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        int o_idx0 = (r + 0) / G;
        int h_idx0 = (r + 0) % G;
        int o_idx1 = (r + 1) / G;
        int h_idx1 = (r + 1) % G;

        HVX_Vector *pv_out0 =
          (HVX_Vector *) (o_st_base + o_idx0 * o_st_stride_bytes + h_idx0 * head_dim * qo_element_size);
        HVX_Vector *pv_out1 =
          (HVX_Vector *) (o_st_base + o_idx1 * o_st_stride_bytes + h_idx1 * head_dim * qo_element_size);

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 *in_base = o_tile_curr + r0 * HMX_FP16_TILE_N_ROWS * head_dim;  // [32, D] row chunk

        // clang-format off
        if (qo_fp32_element) {
          #pragma unroll
          for (int d = 0; d < D / 32; ++d) {
            const HVX_Vector *in_tile = (const HVX_Vector *) (in_base + d * HMX_FP16_TILE_N_ELMS);
        
            const HVX_VectorPair vp = hvx_my_vhf_to_wsf(in_tile[r1 / 2]);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        } else {
          #pragma unroll
          for (int d = 0; d < D / 64; ++d) {
            const __fp16     *in_dual_tile = in_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector *pv_in0       = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
            const HVX_Vector *pv_in1       = pv_in0 + 16;

            const HVX_VectorPair vp = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);

            *pv_out0++ = Q6_V_lo_W(vp);
            if (next_row_valid) {
              *pv_out1++ = Q6_V_hi_W(vp);
            }
          }
        }
        // clang-format on
      }
    }
    int64_t o_store_stage_t1 = HAP_perf_get_qtimer_count();
    llm_trace_profile_record_flash_event(llm_profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_O_STORE,
                                         LLM_TRACE_UNIT_STORE, worker_index, qo_len, kv_len, n_heads, n_kv_heads,
                                         head_dim, ir, -1, n_rows_g, 0,
                                         (int64_t) n_rows_g * head_dim * (int64_t) qo_element_size, o_store_stage_t0,
                                         o_store_stage_t1);
    TIMER_STOP_EVENT(o_store, FIGURE8_COMP_O_STORE, ir, -1);
  }

#if defined(ENABLE_PROFILE_TIMERS)
  {
    const int64_t q_load_us   = TIMER_US(q_load);
    const int64_t k_load_us   = TIMER_US(k_load);
    const int64_t v_load_us   = TIMER_US(v_load);
    const int64_t qk_dot_us   = TIMER_US(qk_dot);
    const int64_t safe_sm_us  = TIMER_US(safe_sm);
    const int64_t core_acc_us = TIMER_US(core_acc);
    const int64_t o_scale_us  = TIMER_US(o_scale);
    const int64_t o_store_us  = TIMER_US(o_store);
    const int64_t scna_exp_us = TIMER_US(scna_exp);
    const int64_t profiled_total_us =
      q_load_us + k_load_us + v_load_us + qk_dot_us + safe_sm_us + core_acc_us + o_scale_us + o_store_us;

    FARF(ALWAYS,
         "FIG8_ATTENTION_TIMERS lut_exp=%d scna_mode=%d scna_layout=%d scna_width=%d qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d kv_head=%d "
         "worker=%d profiled_total=%lld q_load=%lld k_load=%lld v_load=%lld qk_dot=%lld safe_sm=%lld core_acc=%lld "
         "o_scale=%lld o_store=%lld scna_exp=%lld",
         enable_vgather_exp ? 1 : 0, enable_scna_exp ? 1 : 0,
         enable_scna_exp ? scna_hvx_params.layout : SCNA_LAYOUT_SERIAL,
         enable_scna_exp ? scna_hvx_params.width : 0,
         qo_len, kv_len, n_heads, n_kv_heads, head_dim, kv_head_idx, worker_index,
         profiled_total_us, q_load_us, k_load_us, v_load_us, qk_dot_us, safe_sm_us, core_acc_us, o_scale_us,
         o_store_us, scna_exp_us);

    if (profile != NULL) {
      int idx = __sync_fetch_and_add(&(profile->record_count), 1);
      if (idx >= 0 && idx < profile->max_records) {
        struct Figure8ProfileRecord *record = (struct Figure8ProfileRecord *) (profile + 1);
        record[idx]                         = (struct Figure8ProfileRecord) {
          .lut_exp        = enable_vgather_exp ? 1 : 0,
          .qo_len         = qo_len,
          .kv_len         = kv_len,
          .n_heads        = n_heads,
          .n_kv_heads     = n_kv_heads,
          .head_dim       = head_dim,
          .kv_head        = kv_head_idx,
          .worker         = worker_index,
          .profiled_total = profiled_total_us,
          .q_load         = q_load_us,
          .k_load         = k_load_us,
          .v_load         = v_load_us,
          .qk_dot         = qk_dot_us,
          .safe_sm        = safe_sm_us,
          .core_acc       = core_acc_us,
          .o_scale        = o_scale_us,
          .o_store        = o_store_us,
          .scna_exp       = scna_exp_us,
          .scna_layout    = enable_scna_exp ? scna_hvx_params.layout : SCNA_LAYOUT_SERIAL,
          .scna_width     = enable_scna_exp ? scna_hvx_params.width : 0,
        };
      }
    }
  }
#endif
}

size_t fa_f32_compute_vtcm_usage(int group_size, int head_dim, int n_rows, int n_cols) {
  const size_t g_br = align_up(group_size * n_rows, 32);

  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(float), 4096);    // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(n_cols * head_dim * sizeof(float), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * n_cols * sizeof(float), 4096);      // S, P: [Br', Bc]
  const size_t col_vec_size   = align_up(g_br * sizeof(float), 256);                // m_prev, m_curr, l, rowsum: [Br']

  size_t total =
    qo_tile_size * 2 /* Q, O */ + kv_tile_size * 2 /* K, V */ + core_tile_size * 1 /* S/P */ + col_vec_size * 4;
  return total;
}

void fa_f32_find_chunk_size(size_t *blk_r, size_t *blk_c, int group_size, int head_dim, int qo_len, int kv_len,
                            size_t limit) {
  const int nr_unit = 32 / group_size;
  const int nc_unit = 32;

  find_chunk_size_common(blk_r, blk_c, group_size, head_dim, qo_len, kv_len, limit, nr_unit, nc_unit,
                         fa_f32_compute_vtcm_usage);
}

void simple_flash_attn_f32_core(int kv_head_idx, uint8_t *vtcm, uint8_t *vtcm_limit, __fp16 *restrict O,
                                const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                                const __fp16 *restrict qk_mask, int qo_len, int kv_len, int n_heads, int n_kv_heads,
                                int head_dim) {
  const int G = n_heads / n_kv_heads;  // group size
  const int D = head_dim;

  const bool   has_qk_mask = (qk_mask != NULL);
  const size_t kv_pad_len  = align_up(kv_len, 64);

  const float qk_scale = 1.0f / sqrtf(head_dim) * 1.44269504f;

  const size_t qo_ldst_stride = n_heads * head_dim;
  const size_t qo_ldst_blk_sz = G * head_dim;
  const size_t kv_ld_stride   = n_kv_heads * head_dim;
  const size_t kv_ld_blk_sz   = head_dim;

  size_t blk_sz_r, blk_sz_c;  // Br, Bc
  fa_f32_find_chunk_size(&blk_sz_r, &blk_sz_c, G, head_dim, qo_len, kv_len, vtcm_limit - vtcm);

  const size_t g_br = align_up(G * blk_sz_r, 32);  // Br'
  FARF(ALWAYS, "%s: Br=%d Bc=%d Br'=%d", __func__, blk_sz_r, blk_sz_c, g_br);

  const size_t qo_tile_size   = align_up(g_br * head_dim * sizeof(float), 4096);      // Q, O: [Br', D]
  const size_t kv_tile_size   = align_up(blk_sz_c * head_dim * sizeof(float), 4096);  // K, V: [Bc, D]
  const size_t core_tile_size = align_up(g_br * blk_sz_c * sizeof(float), 4096);      // S, P: [Br', Bc]
  const size_t col_vec_size   = align_up(g_br * sizeof(float), 256);  // m_prev, m_curr, l, rowsum: [Br']

  uint8_t *vtcm_cur = vtcm;

  float *q_tile = (float *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  float *o_tile = (float *) vtcm_seq_alloc(&vtcm_cur, qo_tile_size);
  float *k_tile = (float *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  float *v_tile = (float *) vtcm_seq_alloc(&vtcm_cur, kv_tile_size);
  float *s_tile = (float *) vtcm_seq_alloc(&vtcm_cur, core_tile_size);

  HVX_Vector *mvec_m_prev   = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_m_curr   = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_p_rowsum = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);
  HVX_Vector *mvec_l        = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_size);

  assert(vtcm_cur <= vtcm_limit);

  for (int ir = 0; ir < qo_len; ir += blk_sz_r) {
    const size_t n_rows   = smin(qo_len - ir, blk_sz_r);
    const size_t n_rows_g = n_rows * G;

    // load [n_rows*G, D] tile of Q into VTCM
    {
      const float *q_ld_base = ((const float *) Q) + ir * qo_ldst_stride + kv_head_idx * qo_ldst_blk_sz;

      // l2fetch(q_ld_base, qo_ldst_stride * sizeof(float), qo_ldst_blk_sz * sizeof(float), n_rows, 1);

      // HVX_Vector *pv_out = (HVX_Vector *) q_tile;
      // for (int r = 0; r < n_rows; ++r) {
      //   const HVX_Vector *pv_in = (const HVX_Vector *) (q_ld_base + r * qo_ldst_stride);
      //   for (int gd = 0; gd < G * D; gd += 32) {
      //     *pv_out++ = *pv_in++;
      //   }
      // }

      _Alignas(64) dma_desc_2d_t desc = { 0 };

      desc.next       = 0;  // no next descriptor
      desc.length     = 0;  // not used
      desc.type       = DMA_DESC_TYPE_2D;
      desc.dst_bypass = 0;
      desc.src_bypass = 1;                        // bypass L2 cache
      desc.ordered    = 1;                        // ordered, doesn't matter
      desc.dstate     = DMA_DESC_DSTATE_PENDING;  // to be processed

      desc.src        = (uint32_t) q_ld_base;
      desc.dst        = (uint32_t) q_tile;
      desc.roi_width  = qo_ldst_blk_sz * sizeof(float);
      desc.roi_height = n_rows;
      desc.src_stride = qo_ldst_stride * sizeof(float);
      desc.dst_stride = qo_ldst_blk_sz * sizeof(float);

      desc.src_width_offset = 0;
      desc.dst_width_offset = 0;

      dma_wait_for_idle();
      dma_submit_one((dma_desc_1d_t *) &desc);
      dma_wait_for_idle();
    }

    hvx_fill_uw(mvec_m_prev, 0xff7fffff, col_vec_size);
    hvx_fill_uw(mvec_l, 0, col_vec_size);

    hvx_fill_uw(o_tile, 0, qo_tile_size);

    for (int jc = 0; jc < kv_len; jc += blk_sz_c) {
      const size_t n_cols = smin(kv_len - jc, blk_sz_c);

      // load and convert [Bc, D] tile of K into VTCM
      {
        const __fp16 *k_ld_base = K + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 1);

        /*
        HVX_Vector *pv_out = (HVX_Vector *) k_tile;
        for (int c = 0; c < n_cols; ++c) {
          const HVX_Vector *pv_in = (const HVX_Vector *) (k_ld_base + c * kv_ld_stride);

#pragma unroll
          for (int d = 0; d < D; d += 64) {
            HVX_VectorPair vp = hvx_my_vhf_to_wsf(*pv_in++);
            vp                = Q6_W_vshuff_VVR(Q6_V_hi_W(vp), Q6_V_lo_W(vp), -4);
            *pv_out++         = Q6_V_lo_W(vp);
            *pv_out++         = Q6_V_hi_W(vp);
          }
        }
        */

        // NOTE: This scalar version is slow but more accurate
        for (int c = 0; c < n_cols; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            k_tile[c * head_dim + d] = (float) k_ld_base[c * kv_ld_stride + d];
          }
        }
      }

      // issue L2 prefetch of V tile
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
        l2fetch(v_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols, 0);
      }

      // compute QK^T
      {
        const HVX_Vector v_zero          = Q6_V_vzero();
        const HVX_Vector v_qk_scale_sf   = Q6_V_vsplat_R(*(uint32_t *) (&qk_scale));  // fp32_to_bits
        const HVX_Vector v_qk_scale_qf32 = Q6_Vqf32_vadd_VsfVsf(v_qk_scale_sf, v_zero);

        for (int r = 0; r < n_rows_g; ++r) {
          for (int c0 = 0; c0 < n_cols; c0 += 32) {
            _Alignas(VLEN) float qk_dot_local[32], tmp[32];

            for (int c1 = 0; c1 < 32; ++c1) {
              int c = c0 + c1;
              if (c >= n_cols) {
                break;
              }

              const HVX_Vector *pv_row = (const HVX_Vector *) (q_tile + r * head_dim);
              const HVX_Vector *pv_col = (const HVX_Vector *) (k_tile + c * head_dim);

              HVX_Vector v_sum = v_zero;
              for (int d = 0; d < D; d += 32) {
                HVX_Vector v_prod;
                v_prod = Q6_Vqf32_vmpy_VsfVsf(*pv_row++, *pv_col++);
                v_sum  = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, v_prod);
              }
#pragma unroll
              for (int s = 64; s >= 4; s >>= 1) {
                v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_V_vlalign_VVR(v_sum, v_zero, s));
              }
              vmem(tmp)        = v_sum;
              qk_dot_local[c1] = tmp[31];
            }

            HVX_Vector v_scaled_qk           = Q6_Vqf32_vmpy_Vqf32Vqf32(vmem(qk_dot_local), v_qk_scale_qf32);
            vmem(&s_tile[r * blk_sz_c + c0]) = Q6_Vsf_equals_Vqf32(v_scaled_qk);
          }
        }

        // for (int r = 0; r < n_rows_g; ++r) {
        //   for (int c = 0; c < n_cols; ++c) {
        //     float s = 0.0f;
        //     for (int d = 0; d < head_dim; ++d) {
        //       s += q_tile[r * head_dim + d] * k_tile[c * head_dim + d];
        //     }
        //     s_tile[r * blk_sz_c + c] = s * qk_scale;
        //   }
        // }
      }

      // core softmax computation
      {
        const HVX_Vector v_neg_inf = Q6_V_vsplat_R(0xff7fffff);  // 1 11111110 11...1

        for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
          _Alignas(VLEN) float m_prev_local[32], m_curr_local[32], p_rowsum_local[32], tmp[32];

          vmem(m_prev_local) = mvec_m_prev[r0 / 32];

          for (int r1 = 0; r1 < 32; ++r1) {
            int r = r0 + r1;
            if (r >= n_rows_g) {
              break;
            }

            HVX_Vector *const pv_s_row = (HVX_Vector *) (s_tile + r * blk_sz_c);

            // apply mask & compute rowmax(S)
            HVX_Vector v_s_rowmax = v_neg_inf;
            for (int c = 0; c < n_cols; c += 32) {
              int q_idx = ir + r / G;
              int k_idx = jc + c;

              // TODO(hzx): handle leftover mask when qk_mask == null
              HVX_Vector     v_mask_hf  = has_qk_mask ? vmemu(qk_mask + q_idx * kv_pad_len + k_idx) : Q6_V_vzero();
              HVX_VectorPair vp_mask_sf = hvx_my_vhf_to_wsf(v_mask_hf);
              HVX_Vector     v_mask_sf  = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(vp_mask_sf), Q6_V_lo_W(vp_mask_sf), -4));

              const HVX_Vector v_fp32_mask_threshold = Q6_V_vsplat_R(0xc3000000);  // fp32: -128.0
              HVX_VectorPred   q_mask_out            = Q6_Q_vcmp_gt_VsfVsf(v_fp32_mask_threshold, v_mask_sf);

              HVX_Vector v_s_row = Q6_V_vmux_QVV(q_mask_out, v_neg_inf, pv_s_row[c / 32]);
              pv_s_row[c / 32]   = v_s_row;

              v_s_rowmax = Q6_Vsf_vmax_VsfVsf(v_s_rowmax, v_s_row);
            }
#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_s_rowmax = Q6_Vsf_vmax_VsfVsf(v_s_rowmax, Q6_V_vlalign_VVR(v_s_rowmax, v_neg_inf, s));
            }
            vmem(tmp)      = v_s_rowmax;
            float s_rowmax = tmp[31];

            float m_cur      = s_rowmax > m_prev_local[r1] ? s_rowmax : m_prev_local[r1];  // fmaxf
            m_curr_local[r1] = m_cur;

            HVX_Vector v_dup_m = Q6_V_vsplat_R(*(uint32_t *) (&m_cur));  // fp32_to_bits

            // compute rows of P = exp(S - m)
            HVX_Vector *const pv_p_row   = pv_s_row;  // inplace replacement of S tile
            const HVX_Vector  v_zero     = Q6_V_vzero();
            HVX_Vector        v_p_rowsum = v_zero;    // qf32
            for (int c = 0; c < n_cols; c += 32) {
              HVX_Vector v_s_minus_m = Q6_Vqf32_vsub_VsfVsf(pv_s_row[c / 32], v_dup_m);
              HVX_Vector v_p_row_sf  = hvx_my_exp2_vsf_vqf32(v_s_minus_m);

              pv_p_row[c / 32] = v_p_row_sf;

              v_p_rowsum = Q6_Vqf32_vadd_Vqf32Vsf(v_p_rowsum, v_p_row_sf);
            }
#pragma unroll
            for (int s = 64; s >= 4; s >>= 1) {
              v_p_rowsum = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum, Q6_V_vlalign_VVR(v_p_rowsum, v_zero, s));
            }
            vmem(tmp)      = v_p_rowsum;
            float p_rowsum = tmp[31];

            p_rowsum_local[r1] = p_rowsum;
          }

          mvec_m_curr[r0 / 32]   = vmem(m_curr_local);
          mvec_p_rowsum[r0 / 32] = vmem(p_rowsum_local);
        }
      }

      // load and convert [Bc, D] tile of V into VTCM
      {
        const __fp16 *v_ld_base = V + jc * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;

        /*
        HVX_Vector *pv_out = (HVX_Vector *) v_tile;
        for (int c = 0; c < n_cols; ++c) {
          const HVX_Vector *pv_in = (const HVX_Vector *) (v_ld_base + c * kv_ld_stride);

#pragma unroll
          for (int d = 0; d < D; d += 64) {
            HVX_VectorPair vp = hvx_my_vhf_to_wsf(*pv_in++);
            vp                = Q6_W_vshuff_VVR(Q6_V_hi_W(vp), Q6_V_lo_W(vp), -4);
            *pv_out++         = Q6_V_lo_W(vp);
            *pv_out++         = Q6_V_hi_W(vp);
          }
        }
        */

        // NOTE: This scalar version is slow but more accurate
        for (int c = 0; c < n_cols; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            v_tile[c * head_dim + d] = (float) v_ld_base[c * kv_ld_stride + d];
          }
        }
      }

      // issue L2 prefetch of the next K tile
      {
        int jc_next = jc + blk_sz_c;
        if (jc_next < kv_len) {
          const size_t n_cols_next = smin(kv_len - jc_next, blk_sz_c);

          const __fp16 *k_ld_base = K + jc_next * kv_ld_stride + kv_head_idx * kv_ld_blk_sz;
          l2fetch(k_ld_base, kv_ld_stride * sizeof(__fp16), kv_ld_blk_sz * sizeof(__fp16), n_cols_next, 0);
        }
      }

      // core accumulation (update)
      {
        HVX_Vector *mvec_exp_m_diff = mvec_m_curr;  // inplace replacement, qf32
        // update vector m_i, l_i
        for (int r = 0; r < n_rows_g; r += 32) {
          int i = r / 32;

          HVX_Vector v_m_prev = mvec_m_prev[i];
          HVX_Vector v_m_curr = mvec_m_curr[i];
          HVX_Vector v_m_diff = Q6_Vqf32_vsub_VsfVsf(v_m_prev, v_m_curr);

          HVX_Vector v_exp_m_diff = hvx_my_exp2_vqf32(v_m_diff);

          // l_i^j = exp(m_i^{j-1} - m_i^j) * l_i^{j-1} + rowsum(P_i^j)
          HVX_Vector v_l_curr = Q6_Vqf32_vmpy_Vqf32Vqf32(mvec_l[i], v_exp_m_diff);
          v_l_curr            = Q6_Vqf32_vadd_Vqf32Vqf32(v_l_curr, mvec_p_rowsum[i]);

          mvec_m_prev[i] = v_m_curr;
          mvec_l[i]      = v_l_curr;

          mvec_exp_m_diff[i] = v_exp_m_diff;  // qf32
        }

        // compute O_i^j = diag(exp(m_i^{j-1} - m_i^j)) O_i^{j-1} + P_i^j V_j

        /*
        // scalar impl ref: assume o_tile fp32
        _Alignas(VLEN) float o_row_scale_local[32];
        for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
          vmem(o_row_scale_local) = Q6_Vsf_equals_Vqf32(mvec_exp_m_diff[r0 / 32]);

          for (int r1 = 0; r1 < 32; ++r1) {
            int r = r0 + r1;
            if (r >= n_rows_g) {
              break;
            }

            float *p_row = s_tile + r * blk_sz_c;
            float *o_row = o_tile + r * head_dim;

            for (int d = 0; d < head_dim; ++d) {
              o_row[d] *= o_row_scale_local[r1];

              float sum = 0.0f;
              for (int c = 0; c < n_cols; ++c) {
                sum += p_row[c] * v_tile[c * head_dim + d];
              }
              o_row[d] += sum;
            }
          }
        }
        */

        _Alignas(VLEN) int32_t o_row_scale_qf32_local[32], p_row_local[32];
        for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
          vmem(o_row_scale_qf32_local) = mvec_exp_m_diff[r0 / 32];

          for (int r1 = 0; r1 < 32; ++r1) {
            int r = r0 + r1;
            if (r >= n_rows_g) {
              break;
            }

            int32_t          o_row_scale_qf32   = o_row_scale_qf32_local[r1];
            const HVX_Vector v_o_row_scale_qf32 = Q6_V_vsplat_R(o_row_scale_qf32);

            const HVX_Vector *const pv_p_row = (const HVX_Vector *) (s_tile + r * blk_sz_c);
            HVX_Vector *const       pv_o_row = (HVX_Vector *) (o_tile + r * head_dim);

            for (int d = 0; d < head_dim; d += 32) {
              HVX_Vector v_sum = Q6_V_vzero();

              // reduction axis: column (along `kv_len` axis)
              for (int c0 = 0; c0 < n_cols; c0 += 32) {
                // NOTE(hzx): consider alternative solution
                // sol 1# outer loop: VTCM -> vec reg -> L2; inner loop: L2 -> L1 -> scalar reg -> vec reg
                // sol 2# outer loop: VTCM -> vec reg; inner loop: vec reg ---> vec reg
                // NOTE: broadcasting a 4B value to the whole vector register may have long critical path
                //     example instruction sequence: vror + 2x vlut16 + vmux
                vmem(p_row_local) = pv_p_row[c0 / 32];

                for (int c1 = 0; c1 < 32; ++c1) {
                  int c = c0 + c1;
                  if (c >= n_cols) {
                    break;
                  }

                  HVX_Vector v_p_elem = Q6_V_vsplat_R(p_row_local[c1]);
                  HVX_Vector v_v_vec  = vmem(&v_tile[c * head_dim + d]);

                  v_sum = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_Vqf32_vmpy_VsfVsf(v_p_elem, v_v_vec));
                }
              }

              // now `v_sum` contains the dot product of P[1, Bc] & V[Bc, 32]
              HVX_Vector v_scaled_o = Q6_Vqf32_vmpy_Vqf32Vqf32(v_o_row_scale_qf32, pv_o_row[d / 32]);
              pv_o_row[d / 32]      = Q6_Vqf32_vadd_Vqf32Vqf32(v_scaled_o, v_sum);
            }
          }
        }
      }
    }

    // scale O tile: O_i = diag(l_i^{-1}) O_i
    {
      _Alignas(VLEN) int32_t inv_l_local[32];
      for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
        HVX_Vector v_inv_l_qf32 = hvx_my_inv_vqf32_vsf(Q6_Vsf_equals_Vqf32(mvec_l[r0 / 32]));
        vmem(inv_l_local)       = v_inv_l_qf32;

        for (int r1 = 0; r1 < 32; ++r1) {
          int r = r0 + r1;
          if (r >= n_rows_g) {
            break;
          }

          const HVX_Vector v_o_row_scale_qf32 = Q6_V_vsplat_R(inv_l_local[r1]);

          HVX_Vector *pv_o_row = (HVX_Vector *) (o_tile + r * head_dim);
          for (int d = 0; d < head_dim; d += 32) {
            HVX_Vector v_scaled_o = Q6_Vqf32_vmpy_Vqf32Vqf32(v_o_row_scale_qf32, pv_o_row[d / 32]);
            pv_o_row[d / 32]      = Q6_Vsf_equals_Vqf32(v_scaled_o);
          }
        }
      }

      /*
      // scalar impl ref: assume o_tile fp32
      _Alignas(VLEN) float inv_l_local[32];
      for (int r0 = 0; r0 < n_rows_g; r0 += 32) {
        // vmem(inv_l_local) = Q6_Vsf_equals_Vqf32(hvx_my_inv_vqf32_vsf(Q6_Vsf_equals_Vqf32(mvec_l[r0 / 32])));
        vmem(inv_l_local) = Q6_Vsf_equals_Vqf32(mvec_l[r0 / 32]);

        for (int r1 = 0; r1 < 32; ++r1) {
          int r = r0 + r1;
          if (r >= n_rows_g) {
            break;
          }

          inv_l_local[r1] = 1.0f / inv_l_local[r1];

          float *o_row = o_tile + r * head_dim;
          for (int d = 0; d < head_dim; ++d) {
            o_row[d] *= inv_l_local[r1];
          }
        }
      }
      */
    }

    // store [n_rows*G, D] tile of O back to DDR memory
    {
      float *o_st_base = ((float *) O) + ir * qo_ldst_stride + kv_head_idx * qo_ldst_blk_sz;

      const HVX_Vector *pv_in = (HVX_Vector *) o_tile;
      for (int r = 0; r < n_rows; ++r) {
        HVX_Vector *pv_out = (HVX_Vector *) (o_st_base + r * qo_ldst_stride);
        for (int gd = 0; gd < G * D; gd += 32) {
          *pv_out++ = *pv_in++;
        }
      }

      // TODO(hzx): investigate why DMA is not working here
      /*
      _Alignas(64) dma_desc_2d_t desc = { 0 };

      desc.next       = 0;
      desc.length     = 0;
      desc.type       = DMA_DESC_TYPE_2D;
      desc.dst_bypass = 1;
      desc.src_bypass = 0;
      desc.order      = 1;
      desc.dstate     = DMA_DESC_DSTATE_PENDING;

      desc.src        = (uint32_t) o_tile;
      desc.dst        = (uint32_t) o_st_base;
      desc.roi_width  = qo_ldst_blk_sz * sizeof(float);
      desc.roi_height = n_rows;
      desc.src_stride = qo_ldst_blk_sz * sizeof(float);
      desc.dst_stride = qo_ldst_stride * sizeof(float);

      desc.src_width_offset = 0;
      desc.dst_width_offset = 0;

      dma_wait_for_idle();
      dma_submit_one((dma_desc_1d_t *) &desc);
      dma_wait_for_idle();
      */
    }
  }
}

void simple_flash_attn_worker(void *data, int worker_index) {
  simple_fa_task_state_t *s = (simple_fa_task_state_t *) data;

  uint8_t *vtcm       = s->vtcm_base + worker_index * s->vtcm_size_per_thread;
  uint8_t *vtcm_limit = vtcm + s->vtcm_size_per_thread;

  hmx_manager_enable_execution();

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(s->task_id)) - 1;
    if (task_id >= s->n_tasks) {
      break;
    }

    int kv_head_idx = task_id;
    simple_flash_attn_f16_core(kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->mask, s->qo_len, s->kv_len,
                               s->n_heads, s->n_kv_heads, s->head_dim, worker_index, s->mode_flags, s->profile,
                               s->llm_profile, s->trace_id, s->op_index);
  }

  hmx_manager_disable_execution();

  worker_pool_synctoken_jobdone(&(s->sync_ctx));
}

void simple_flash_attn_f32_worker(void *data, int worker_index) {
  simple_fa_task_state_t *s = (simple_fa_task_state_t *) data;

  uint8_t *vtcm       = s->vtcm_base + worker_index * s->vtcm_size_per_thread;
  uint8_t *vtcm_limit = vtcm + s->vtcm_size_per_thread;

  while (1) {
    unsigned int task_id = worker_pool_atomic_inc_return(&(s->task_id)) - 1;
    if (task_id >= s->n_tasks) {
      break;
    }

    int kv_head_idx = task_id;
    simple_flash_attn_f32_core(kv_head_idx, vtcm, vtcm_limit, s->O, s->Q, s->K, s->V, s->mask, s->qo_len, s->kv_len,
                               s->n_heads, s->n_kv_heads, s->head_dim);
  }

  worker_pool_synctoken_jobdone(&(s->sync_ctx));
}

int simple_flash_attn_sp_hdim(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                              const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len, int kv_len,
                              int n_heads, int n_kv_heads, int head_dim);

/**
 * Simple llama.cpp-style FlashAttention implementation
 *
 * batch_size dimension is omitted
 * 
 * Q: [qo_len, n_heads, head_dim], K/V: [kv_len, n_kv_heads, head_dim]
 * mask: [qo_len*, kv_len] broadcast to each head (first dimension maybe larger than qo_len)
 */
static int simple_flash_attn_impl(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                                  const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len, int kv_len,
                                  int n_heads, int n_kv_heads, int head_dim, int mode_flags,
                                  struct Figure8ProfileHeader *profile, struct LlmTraceProfileHeader *llm_profile,
                                  int64_t trace_id, int op_index) {
  if (head_dim % 64 != 0) {
    return simple_flash_attn_sp_hdim(O, Q, K, V, mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim);
  }
  if (n_heads % n_kv_heads != 0) {
    FARF(ALWAYS, "FA not supported: head_dim=%d n_heads=%d n_kv_heads=%d", head_dim, n_heads, n_kv_heads);
    return -1;
  }

  const int    n_workers            = num_hvx128_contexts;
  const size_t vtcm_size_per_thread = 1024 * 1024;
  assert(n_workers * vtcm_size_per_thread <= 6 * 1024 * 1024);  // don't use too much VTCM

  simple_fa_task_state_t state;
  state.O          = O;
  state.Q          = Q;
  state.K          = K;
  state.V          = V;
  state.mask       = mask;
  state.qo_len     = qo_len;
  state.kv_len     = kv_len;
  state.n_heads    = n_heads;
  state.n_kv_heads = n_kv_heads;
  state.head_dim   = head_dim;
  state.mode_flags = mode_flags;
  state.profile    = profile;
  state.llm_profile = llm_profile;
  state.trace_id   = trace_id;
  state.op_index   = op_index;

  // TODO(hzx): parallelize along query_len x n_kv_heads dimension
  // size_t n_tot_chunks      = qo_len * n_kv_heads;
  // size_t n_chunks_per_task = ceil_div(n_tot_chunks, n_workers);

  state.task_id              = 0;
  state.n_tasks              = n_kv_heads;
  state.vtcm_base            = (uint8_t *) vtcm_manager_get_vtcm_base();
  state.vtcm_size_per_thread = vtcm_size_per_thread;

  worker_pool_job_t job;
  job.fptr = simple_flash_attn_worker;
  job.dptr = &state;

  int64_t t0 = HAP_perf_get_time_us();

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);  // use default worker pool
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  int64_t elapsed_us = HAP_perf_get_time_us() - t0;
  FARF(ALWAYS,
       "FIG8_ATTENTION_KERNEL lut_exp=%d total_kernel=%lld qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d",
       (mode_flags & LLM_NPU_MODE_LUT_EXP) ? 1 : 0, elapsed_us, qo_len, kv_len, n_heads, n_kv_heads, head_dim);

  return 0;
}

int simple_flash_attn(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                      const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
  const int mode_flags = FIGURE8_ENABLE_LUT_EXP ? LLM_NPU_MODE_LUT_EXP : 0;
  return simple_flash_attn_impl(O, Q, K, V, mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, mode_flags, NULL,
                                NULL, 0, 0);
}

int simple_flash_attn_with_flags(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                                 const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len, int kv_len,
                                 int n_heads, int n_kv_heads, int head_dim, int mode_flags) {
  return simple_flash_attn_impl(O, Q, K, V, mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, mode_flags, NULL,
                                NULL, 0, 0);
}

int simple_flash_attn_qo_f32_kv_f16_with_flags(float *restrict O, const float *restrict Q, const __fp16 *restrict K,
                                               const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len,
                                               int kv_len, int n_heads, int n_kv_heads, int head_dim, int mode_flags) {
  (void) mode_flags;

  if (head_dim % 32 != 0 || n_heads % n_kv_heads != 0) {
    FARF(ALWAYS, "F32 FA not supported: head_dim=%d n_heads=%d n_kv_heads=%d", head_dim, n_heads, n_kv_heads);
    return -1;
  }

  const int    n_workers            = num_hvx128_contexts;
  const size_t vtcm_size_per_thread = 1024 * 1024;
  assert(n_workers * vtcm_size_per_thread <= 6 * 1024 * 1024);

  simple_fa_task_state_t state;
  state.O          = (__fp16 *) O;
  state.Q          = (const __fp16 *) Q;
  state.K          = K;
  state.V          = V;
  state.mask       = mask;
  state.qo_len     = qo_len;
  state.kv_len     = kv_len;
  state.n_heads    = n_heads;
  state.n_kv_heads = n_kv_heads;
  state.head_dim   = head_dim;
  state.mode_flags = mode_flags;
  state.profile    = NULL;
  state.llm_profile = NULL;
  state.trace_id   = 0;
  state.op_index   = 0;

  state.task_id              = 0;
  state.n_tasks              = n_kv_heads;
  state.vtcm_base            = (uint8_t *) vtcm_manager_get_vtcm_base();
  state.vtcm_size_per_thread = vtcm_size_per_thread;

  worker_pool_job_t job;
  job.fptr = simple_flash_attn_f32_worker;
  job.dptr = &state;

  int64_t t0 = HAP_perf_get_time_us();

  worker_pool_synctoken_init(&(state.sync_ctx), n_workers);
  for (int i = 0; i < n_workers; ++i) {
    worker_pool_submit(NULL, job);
  }
  worker_pool_synctoken_wait(&(state.sync_ctx));

  int64_t elapsed_us = HAP_perf_get_time_us() - t0;
  FARF(ALWAYS,
       "F32_ATTENTION_KERNEL total_kernel=%lld qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d",
       elapsed_us, qo_len, kv_len, n_heads, n_kv_heads, head_dim);

  return 0;
}

int simple_flash_attn_profiled(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                               const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len, int kv_len,
                               int n_heads, int n_kv_heads, int head_dim, int mode_flags,
                               struct Figure8ProfileHeader *profile) {
  return simple_flash_attn_impl(O, Q, K, V, mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, mode_flags, profile,
                                NULL, 0, 0);
}

int simple_flash_attn_llm_profiled(__fp16 *restrict O, const __fp16 *restrict Q, const __fp16 *restrict K,
                                   const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len, int kv_len,
                                   int n_heads, int n_kv_heads, int head_dim, int64_t trace_id, int mode_flags,
                                   int op_index, struct LlmTraceProfileHeader *profile) {
  return simple_flash_attn_impl(O, Q, K, V, mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, mode_flags, NULL,
                                profile, trace_id, op_index);
}

static inline int64_t llm_trace_now_us(void) {
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
}

static inline bool llm_detailed_trace_enabled(int mode_flags, struct LlmTraceProfileHeader *profile) {
  return (mode_flags & LLM_NPU_MODE_DETAILED_TRACE) != 0 && profile != NULL &&
         profile->magic == LLM_TRACE_PROFILE_MAGIC && profile->max_events > 0;
}

static void record_flash_stage(struct LlmTraceProfileHeader *profile, int64_t trace_id, int mode_flags, int op_index,
                               int stage, int unit, int worker, int qo_len, int kv_len, int n_heads, int n_kv_heads,
                               int head_dim, int block_r, int block_c, int chunk_r, int chunk_c, int64_t bytes,
                               int64_t t0_us, int64_t t1_us) {
  if (!llm_detailed_trace_enabled(mode_flags, profile)) {
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
                           .m          = qo_len,
                           .k          = head_dim,
                           .n          = n_heads * head_dim,
                           .qo_len     = qo_len,
                           .kv_len     = kv_len,
                           .n_heads    = n_heads,
                           .n_kv_heads = n_kv_heads,
                           .head_dim   = head_dim,
                           .mr         = block_r,
                           .nc         = block_c,
                           .kk         = -1,
                           .chunk_m    = chunk_r,
                           .chunk_n    = chunk_c,
                           .chunk_k    = head_dim,
                           .flags      = mode_flags,
                           .bytes      = bytes,
                           .t0_us      = t0_us,
                           .t1_us      = t1_us,
                           .dur_us     = t1_us - t0_us,
  };
}

#define Br 32
#define Bc 256
#define D  128

int naive_flash_attn_profiled(float *restrict O, const float *restrict Q, const __fp16 *restrict K,
                              const __fp16 *restrict V, const __fp16 *restrict mask, int qo_len, int kv_len,
                              int n_heads, int n_kv_heads, int head_dim, int64_t trace_id, int mode_flags,
                              int op_index, struct LlmTraceProfileHeader *profile) {
  if (n_heads % n_kv_heads != 0) {
    return -1;
  }
  if (head_dim > D) {
    return -1;
  }

  const int gqa_factor = n_heads / n_kv_heads;

  const size_t kv_pad_len = align_up(kv_len, 64);
  const size_t qo_stride  = n_heads * head_dim;
  const size_t kv_stride  = n_kv_heads * head_dim;
  const float  qk_scale   = 1.0f / sqrtf(head_dim) * 1.44269504f;

  for (int h = 0; h < n_heads; ++h) {
    const int h_kv = h / gqa_factor;

    for (int i = 0; i < qo_len; i += Br) {
      const int q_start = i;
      const int q_end   = (i + Br) < qo_len ? (i + Br) : qo_len;
      const int br      = q_end - q_start;
      float    *O_dst   = O + q_start * qo_stride + h * head_dim;

      static float Qi[Br][D];
      static float Oi[Br][D];
      static float mi[Br];
      static float li[Br];

      int64_t t0_us = llm_trace_now_us();
      const float *Q_src = Q + q_start * qo_stride + h * head_dim;
      for (int r = 0; r < br; ++r) {
        for (int d = 0; d < head_dim; ++d) {
          Qi[r][d] = Q_src[r * qo_stride + d];
        }
      }
      int64_t t1_us = llm_trace_now_us();
      record_flash_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_Q_LOAD, LLM_TRACE_UNIT_MEMORY,
                         -1, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q_start, -1, br, 0,
                         (int64_t) br * head_dim * (int64_t) sizeof(float), t0_us, t1_us);

      for (int r = 0; r < br; ++r) {
        mi[r] = -INFINITY;
        li[r] = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          Oi[r][d] = 0.0f;
        }
      }

      for (int j = 0; j < kv_len; j += Bc) {
        const int k_start = j;
        const int k_end   = (j + Bc) < kv_len ? (j + Bc) : kv_len;
        const int bc      = k_end - k_start;

        static float Kj[Bc][D];
        const __fp16 *K_src = K + k_start * kv_stride + h_kv * head_dim;
        t0_us = llm_trace_now_us();
        for (int c = 0; c < bc; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            Kj[c][d] = (float) K_src[c * kv_stride + d];
          }
        }
        t1_us = llm_trace_now_us();
        record_flash_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_K_LOAD, LLM_TRACE_UNIT_MEMORY,
                           -1, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q_start, k_start, br, bc,
                           (int64_t) bc * head_dim * (int64_t) sizeof(__fp16), t0_us, t1_us);

        static float Sij[Br][Bc];
        t0_us = llm_trace_now_us();
        for (int r = 0; r < br; ++r) {
          for (int c = 0; c < bc; ++c) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              sum += Qi[r][d] * Kj[c][d];
            }
            Sij[r][c] = sum * qk_scale;
          }
        }
        t1_us = llm_trace_now_us();
        record_flash_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_QK_DOT, LLM_TRACE_UNIT_SCALAR,
                           -1, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q_start, k_start, br, bc,
                           2LL * br * bc * head_dim, t0_us, t1_us);

        if (mask != NULL) {
          t0_us = llm_trace_now_us();
          for (int r = 0; r < br; ++r) {
            for (int c = 0; c < bc; ++c) {
              const int mask_idx = (q_start + r) * kv_pad_len + (k_start + c);
              Sij[r][c] += (float) mask[mask_idx];
            }
          }
          t1_us = llm_trace_now_us();
          record_flash_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_SAFE_SM,
                             LLM_TRACE_UNIT_SCALAR, -1, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q_start, k_start,
                             br, bc, (int64_t) br * bc * (int64_t) sizeof(__fp16), t0_us, t1_us);
        }

        static float Vj[Bc][D];
        const __fp16 *V_src = V + k_start * kv_stride + h_kv * head_dim;
        t0_us = llm_trace_now_us();
        for (int c = 0; c < bc; ++c) {
          for (int d = 0; d < head_dim; ++d) {
            Vj[c][d] = (float) V_src[c * kv_stride + d];
          }
        }
        t1_us = llm_trace_now_us();
        record_flash_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_V_LOAD, LLM_TRACE_UNIT_MEMORY,
                           -1, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q_start, k_start, br, bc,
                           (int64_t) bc * head_dim * (int64_t) sizeof(__fp16), t0_us, t1_us);

        t0_us = llm_trace_now_us();
        for (int r = 0; r < br; ++r) {
          float m_curr = -INFINITY;
          for (int c = 0; c < bc; ++c) {
            if (Sij[r][c] > m_curr) {
              m_curr = Sij[r][c];
            }
          }

          const float m_new = fmaxf(mi[r], m_curr);
          float exp_sum = 0.0f;
          float exp_values[Bc];
          for (int c = 0; c < bc; ++c) {
            exp_values[c] = exp2f(Sij[r][c] - m_new);
            exp_sum += exp_values[c];
          }

          const float f = exp2f(mi[r] - m_new);
          const float l_new = li[r] * f + exp_sum;

          for (int d = 0; d < head_dim; ++d) {
            Oi[r][d] *= f;
            float sum = 0.0f;
            for (int c = 0; c < bc; ++c) {
              sum += exp_values[c] * Vj[c][d];
            }
            Oi[r][d] += sum;
          }

          mi[r] = m_new;
          li[r] = l_new;
        }
        t1_us = llm_trace_now_us();
        record_flash_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_CORE_ACC,
                           LLM_TRACE_UNIT_SCALAR, -1, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q_start, k_start,
                           br, bc, 2LL * br * bc * head_dim, t0_us, t1_us);
      }

      t0_us = llm_trace_now_us();
      for (int r = 0; r < br; ++r) {
        const float scale = 1.0f / li[r];
        for (int d = 0; d < head_dim; ++d) {
          O_dst[r * qo_stride + d] = Oi[r][d] * scale;
        }
      }
      t1_us = llm_trace_now_us();
      record_flash_stage(profile, trace_id, mode_flags, op_index, LLM_TRACE_STAGE_FLASH_O_STORE,
                         LLM_TRACE_UNIT_STORE, -1, qo_len, kv_len, n_heads, n_kv_heads, head_dim, q_start, -1, br, 0,
                         (int64_t) br * head_dim * (int64_t) sizeof(float), t0_us, t1_us);
    }
  }
  return 0;
}

int naive_flash_attn(float *restrict O, const float *restrict Q, const __fp16 *restrict K, const __fp16 *restrict V,
                     const __fp16 *restrict mask, int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim) {
  return naive_flash_attn_profiled(O, Q, K, V, mask, qo_len, kv_len, n_heads, n_kv_heads, head_dim, 0, 0,
                                   HTP_OPS_FLASH_ATTN_QO_F32_KV_F16, NULL);
}

#undef Br
#undef Bc
#undef D
