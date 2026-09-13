#include "dsp/op_executor.h"

#include <qurt_memory.h>

#include <cstring>
#include <vector>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_internal.h"
#include "dsp/dma_utils.h"
#include "dsp/mmap_mgr.h"
#include "dsp/ops.h"
#include "dsp/vtcm_mgr.h"
#include "op_reg.h"

// debug
#include <HAP_farf.h>
#include <HAP_perf.h>

namespace {

size_t ggml_super_block_size(enum ggml_type type) {
  // TODO: more types
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_IQ4_NL:
      return sizeof(my_block_q4_0);
    case GGML_TYPE_Q8_0:
      return sizeof(my_block_q8_0);
    default:
      return -1;
  }
}

enum ggml_type matmul_op_to_weight_type(enum HtpOpsIndex op) {
  switch (op) {
    case HTP_OPS_MAT_MUL_PERMUTED_W16A32:
      return GGML_TYPE_F16;
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
      return GGML_TYPE_Q4_0;
    case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
      return GGML_TYPE_Q8_0;
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL:
      return GGML_TYPE_IQ4_NL;
    case HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT:
      return GGML_TYPE_Q8_0;
    default:
      return GGML_TYPE_COUNT;  // invalid type
  }
}

static inline int64_t trace_now_us() {
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
}

static inline bool trace_enabled(int mode_flags) {
  return (mode_flags & LLM_NPU_MODE_TRACE) != 0;
}

const char *dsp_op_name(uint32_t op) {
  switch (op) {
    case HTP_OPS_RMS_NORM_F32:
      return "rms_norm_f32";
    case HTP_OPS_MAT_MUL_PERMUTED_W16A32:
      return "matmul_w16a32";
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
      return "matmul_q4_0";
    case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
      return "matmul_q8_0";
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL:
      return "matmul_iq4_nl";
    case HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT:
      return "matmul_w8pc_a8pt";
    case HTP_OPS_FLASH_ATTN_QO_F32_KV_F16:
      return "flash_attn";
    case HTP_OPS_HMX_INT8_GATE:
      return "hmx_int8_gate";
    case HTP_OPS_ROOFLINE_BENCH:
      return "roofline_bench";
    default:
      return "unknown";
  }
}

void log_dsp_event(int64_t trace_id, int mode_flags, uint32_t op, const char *phase, int m, int k, int n,
                   int qo_len, int kv_len, int n_heads, int n_kv_heads, int head_dim, size_t input_bytes,
                   size_t output_bytes, int64_t t0_us, int64_t t1_us) {
  if (!trace_enabled(mode_flags)) {
    return;
  }
  FARF(ALWAYS,
       "LLMTRACE_DSP_EVENT trace_id=%lld flags=%d op=%s op_index=%u phase=%s m=%d k=%d n=%d qo_len=%d "
       "kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d input_bytes=%llu output_bytes=%llu t0_us=%lld "
       "t1_us=%lld dur_us=%lld",
       (long long) trace_id, mode_flags, dsp_op_name(op), op, phase, m, k, n, qo_len, kv_len, n_heads, n_kv_heads,
       head_dim, (unsigned long long) input_bytes, (unsigned long long) output_bytes, (long long) t0_us,
       (long long) t1_us, (long long) (t1_us - t0_us));
}

static void roofline_set_result(RooflineBenchResult *out, int index, int mode, int kind, int variant, int size,
                                int iters, int64_t elapsed_us, int64_t work_items, bool is_tops) {
  out[index].mode       = mode;
  out[index].kind       = kind;
  out[index].variant    = variant;
  out[index].size       = size;
  out[index].iters      = iters;
  out[index].elapsed_us = elapsed_us;
  out[index].work_items = work_items;
  if (elapsed_us <= 0) {
    out[index].metric_x10000 = 0;
    return;
  }
  const double metric = is_tops ? ((double) work_items / (double) elapsed_us / 1.0e6)
                                : ((double) work_items / (double) elapsed_us / 1.0e3);
  out[index].metric_x10000 = (int64_t) (metric * 10000.0 + 0.5);
}

static void roofline_set_mix_result(RooflineBenchResult *out, int index, int kind, int variant, int size, int iters,
                                    int64_t elapsed_us, int64_t work_items, int engine, int lhs_dtype, int rhs_dtype,
                                    int acc_dtype, int path, int correctness) {
  roofline_set_result(out, index, ROOFLINE_BENCH_MODE_MIX_PRECISION, kind, variant, size, iters, elapsed_us,
                      work_items, true);
  out[index].engine      = engine;
  out[index].lhs_dtype   = lhs_dtype;
  out[index].rhs_dtype   = rhs_dtype;
  out[index].acc_dtype   = acc_dtype;
  out[index].path        = path;
  out[index].correctness = correctness;
}

static void roofline_set_int8_shape_metadata(RooflineBenchResult *out, int index, int M, int K, int N,
                                             int64_t a_bytes, int64_t b_bytes, int64_t c_bytes,
                                             int64_t scales_bytes, int64_t total_vtcm_bytes) {
  // This metadata is benchmark-only: it documents the packed HMX tile buffers,
  // not a row-major C GEMM allocation.  Each logical 32x32 HMX tile occupies
  // HMX_FP16_TILE_SIZE bytes even for the INT8 input path.
  out[index].m = M;
  out[index].k = K;
  out[index].n = N;
  out[index].mt = M / HMX_FP16_TILE_N_ROWS;
  out[index].kt = K / HMX_FP16_TILE_N_COLS;
  out[index].nt = N / HMX_FP16_TILE_N_COLS;
  out[index].tile_bytes = HMX_FP16_TILE_SIZE;
  out[index].a_bytes = a_bytes;
  out[index].b_bytes = b_bytes;
  out[index].c_bytes = c_bytes;
  out[index].scales_bytes = scales_bytes;
  out[index].allocated_bytes = a_bytes + b_bytes + c_bytes + scales_bytes;
  out[index].total_vtcm_bytes = total_vtcm_bytes;
}

static void roofline_set_mix_na(RooflineBenchResult *out, int index, int lhs_dtype, int rhs_dtype, int engine,
                                const int variant) {
  out[index].mode        = ROOFLINE_BENCH_MODE_MIX_PRECISION;
  out[index].kind        = ROOFLINE_BENCH_KIND_NOT_AVAILABLE;
  out[index].variant     = variant;
  out[index].engine      = engine;
  out[index].lhs_dtype   = lhs_dtype;
  out[index].rhs_dtype   = rhs_dtype;
  out[index].acc_dtype   = ROOFLINE_BENCH_DTYPE_UNKNOWN;
  out[index].path        = ROOFLINE_BENCH_PATH_NOT_AVAILABLE;
  out[index].correctness = 0;
}

// Benchmark-only mixed-precision kernels live in this file so the roofline CLI can measure the
// current device paths without being confused with the production LLM matmul kernels.
// Only native hardware MAC paths are reported as compute peaks. Unsupported INT4/mixed rows are
// emitted as N/A instead of being sign-extended into a wider HVX/INT16 semantic substitute.
// Q4_0/IQ4_NL rows are kept separate as GGUF format-effective decode + FP16 HMX work.

static void roofline_hvx_stream_read(const uint8_t *src, size_t bytes, int iters) {
  asm volatile("v1 = vxor(v1, v1)\n" ::: "v1");
  for (int t = 0; t < iters; ++t) {
    const uint8_t *p = src;
    for (size_t i = 0; i < bytes / VLEN; ++i) {
      asm volatile("{ v0.cur = vmem(%0++#1)\n"
                   "  v1 = vxor(v1, v0) }\n"
                   : "+r"(p)::"v0", "v1", "memory");
    }
  }
}

static void roofline_hvx_stream_write(uint8_t *dst, size_t bytes, int iters) {
  HVX_Vector zero = Q6_V_vzero();
  for (int t = 0; t < iters; ++t) {
    HVX_Vector *p = (HVX_Vector *) dst;
    for (size_t i = 0; i < bytes / VLEN; ++i) {
      *p++ = zero;
    }
  }
}

static void roofline_hvx_stream_copy(const uint8_t *src, uint8_t *dst, size_t bytes, int iters) {
  for (int t = 0; t < iters; ++t) {
    const uint8_t *s = src;
    uint8_t       *d = dst;
    for (size_t i = 0; i < bytes / VLEN; ++i) {
      asm volatile("{ v0.cur = vmem(%0++#1)\n"
                   "  vmem(%1++#1) = v0 }\n"
                   : "+r"(s), "+r"(d)::"v0", "memory");
    }
  }
}

static void roofline_hvx_stream_read_unroll4(const uint8_t *src, size_t bytes, int iters) {
  const size_t block_bytes = 4 * VLEN;
  const size_t blocks      = bytes / block_bytes;
  const size_t tail        = bytes - blocks * block_bytes;
  asm volatile("v4 = vxor(v4, v4)\n" ::: "v4");
  for (int t = 0; t < iters; ++t) {
    const uint8_t *p = src;
    for (size_t i = 0; i < blocks; ++i) {
      asm volatile("{ v0.cur = vmem(%0++#1)\n"
                   "  v4 = vxor(v4, v0) }\n"
                   : "+r"(p)::"v0", "v4", "memory");
      asm volatile("{ v1.cur = vmem(%0++#1)\n"
                   "  v4 = vxor(v4, v1) }\n"
                   : "+r"(p)::"v1", "v4", "memory");
      asm volatile("{ v2.cur = vmem(%0++#1)\n"
                   "  v4 = vxor(v4, v2) }\n"
                   : "+r"(p)::"v2", "v4", "memory");
      asm volatile("{ v3.cur = vmem(%0++#1)\n"
                   "  v4 = vxor(v4, v3) }\n"
                   : "+r"(p)::"v3", "v4", "memory");
    }
    if (tail) {
      roofline_hvx_stream_read(p, tail, 1);
    }
  }
}

static void roofline_hvx_stream_write_unroll4(uint8_t *dst, size_t bytes, int iters) {
  const size_t block_bytes = 4 * VLEN;
  const size_t blocks      = bytes / block_bytes;
  const size_t tail        = bytes - blocks * block_bytes;
  asm volatile("v0 = vxor(v0, v0)\n" ::: "v0");
  for (int t = 0; t < iters; ++t) {
    uint8_t *p = dst;
    for (size_t i = 0; i < blocks; ++i) {
      asm volatile("vmem(%0++#1) = v0\n" : "+r"(p)::"v0", "memory");
      asm volatile("vmem(%0++#1) = v0\n" : "+r"(p)::"v0", "memory");
      asm volatile("vmem(%0++#1) = v0\n" : "+r"(p)::"v0", "memory");
      asm volatile("vmem(%0++#1) = v0\n" : "+r"(p)::"v0", "memory");
    }
    if (tail) {
      roofline_hvx_stream_write(p, tail, 1);
    }
  }
}

static void roofline_hvx_stream_copy_unroll4(const uint8_t *src, uint8_t *dst, size_t bytes, int iters) {
  const size_t block_bytes = 4 * VLEN;
  const size_t blocks      = bytes / block_bytes;
  const size_t tail        = bytes - blocks * block_bytes;
  for (int t = 0; t < iters; ++t) {
    const uint8_t *s = src;
    uint8_t       *d = dst;
    for (size_t i = 0; i < blocks; ++i) {
      asm volatile("{ v0.cur = vmem(%0++#1)\n"
                   "  vmem(%1++#1) = v0 }\n"
                   : "+r"(s), "+r"(d)::"v0", "memory");
      asm volatile("{ v1.cur = vmem(%0++#1)\n"
                   "  vmem(%1++#1) = v1 }\n"
                   : "+r"(s), "+r"(d)::"v1", "memory");
      asm volatile("{ v2.cur = vmem(%0++#1)\n"
                   "  vmem(%1++#1) = v2 }\n"
                   : "+r"(s), "+r"(d)::"v2", "memory");
      asm volatile("{ v3.cur = vmem(%0++#1)\n"
                   "  vmem(%1++#1) = v3 }\n"
                   : "+r"(s), "+r"(d)::"v3", "memory");
    }
    if (tail) {
      roofline_hvx_stream_copy(s, d, tail, 1);
    }
  }
}

static void roofline_hvx_stream_copy_xor_0x80_unroll4(const uint8_t *src, uint8_t *dst, size_t bytes, int iters) {
  const size_t block_bytes = 4 * VLEN;
  const size_t blocks      = bytes / block_bytes;
  const size_t tail        = bytes - blocks * block_bytes;
  const HVX_Vector mask    = Q6_Vb_vsplat_R(0x80);
  for (int t = 0; t < iters; ++t) {
    const HVX_Vector *s = (const HVX_Vector *) src;
    HVX_Vector       *d = (HVX_Vector *) dst;
    for (size_t i = 0; i < blocks; ++i) {
      const HVX_Vector v0 = *s++;
      const HVX_Vector v1 = *s++;
      const HVX_Vector v2 = *s++;
      const HVX_Vector v3 = *s++;
      *d++ = Q6_V_vxor_VV(v0, mask);
      *d++ = Q6_V_vxor_VV(v1, mask);
      *d++ = Q6_V_vxor_VV(v2, mask);
      *d++ = Q6_V_vxor_VV(v3, mask);
    }
    if (tail) {
      const uint8_t *st = (const uint8_t *) s;
      uint8_t       *dt = (uint8_t *) d;
      for (size_t j = 0; j < tail; ++j) {
        dt[j] = st[j] ^ 0x80u;
      }
    }
  }
}

static void roofline_hvx_stream_xor_0x80_inplace_unroll4(uint8_t *buf, size_t bytes, int iters) {
  const size_t block_bytes = 4 * VLEN;
  const size_t blocks      = bytes / block_bytes;
  const size_t tail        = bytes - blocks * block_bytes;
  const HVX_Vector mask    = Q6_Vb_vsplat_R(0x80);
  for (int t = 0; t < iters; ++t) {
    HVX_Vector *p = (HVX_Vector *) buf;
    for (size_t i = 0; i < blocks; ++i) {
      HVX_Vector v0 = p[0];
      HVX_Vector v1 = p[1];
      HVX_Vector v2 = p[2];
      HVX_Vector v3 = p[3];
      p[0] = Q6_V_vxor_VV(v0, mask);
      p[1] = Q6_V_vxor_VV(v1, mask);
      p[2] = Q6_V_vxor_VV(v2, mask);
      p[3] = Q6_V_vxor_VV(v3, mask);
      p += 4;
    }
    if (tail) {
      uint8_t *pt = (uint8_t *) p;
      for (size_t j = 0; j < tail; ++j) {
        pt[j] ^= 0x80u;
      }
    }
  }
}

static void roofline_hvx_requant_store_zp_mock(const int16_t *src, uint8_t *dst, size_t elems, int zp, int iters) {
  const size_t elems_per_out_vec = VLEN;
  const size_t blocks            = elems / elems_per_out_vec;
  const HVX_Vector vzp           = Q6_Vh_vsplat_R(zp);
  for (int t = 0; t < iters; ++t) {
    const HVX_Vector *s = (const HVX_Vector *) src;
    HVX_Vector       *d = (HVX_Vector *) dst;
    for (size_t i = 0; i < blocks; ++i) {
      const HVX_Vector h0 = Q6_Vh_vadd_VhVh(*s++, vzp);
      const HVX_Vector h1 = Q6_Vh_vadd_VhVh(*s++, vzp);
      // Benchmark-only producer mock: same instruction stream for zp=0 and
      // zp=128.  It models folding +128 into an existing requant/store stage,
      // not a separate XOR pass over already-packed activation bytes.
      *d++ = Q6_Vub_vpack_VhVh_sat(h1, h0);
    }
  }
}

static int roofline_dma_issue_load(dma_desc_1d_t *desc, void *vtcm_dst, const void *src, size_t size) {
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

static int roofline_dma_read_ddr_to_vtcm(uint8_t *src, uint8_t *vtcm_dst, size_t bytes, size_t chunk_size, int iters) {
  alignas(64) dma_desc_1d_t desc = {};
  chunk_size = chunk_size / VLEN * VLEN;
  if (chunk_size < (size_t) VLEN) {
    chunk_size = VLEN;
  }
  for (int t = 0; t < iters; ++t) {
    size_t offset = 0;
    while (offset < bytes) {
      const size_t remaining = bytes - offset;
      const size_t n = chunk_size < remaining ? chunk_size : remaining;
      int          ret = roofline_dma_issue_load(&desc, vtcm_dst, src + offset, n);
      if (ret) {
        return ret;
      }
      dma_wait_for_idle();
      offset += n;
    }
  }
  return 0;
}

static int roofline_run_hmx_fp16(RooflineBenchResult *out, int max_results, int warmup, int iters) {
  if (max_results <= 0) {
    return 0;
  }
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) {
    return -1;
  }

  __fp16 *a      = (__fp16 *) vtcm;
  __fp16 *b      = (__fp16 *) (vtcm + 2 * 0x100000);
  __fp16 *c      = (__fp16 *) (vtcm + 4 * 0x100000);
  __fp16 *scales = (__fp16 *) (vtcm + 6 * 0x100000);
  int     sizes[] = {32, 64, 128, 256, 512, 1024};
  int     count = 0;

  hmx_manager_enable_execution();
  for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
    const int n = sizes[i];
    for (int w = 0; w < warmup; ++w) {
      hmx_mat_mul_fp16_core(c, a, b, scales, n, n, n);
    }
    const int64_t t0 = trace_now_us();
    for (int t = 0; t < iters; ++t) {
      hmx_mat_mul_fp16_core(c, a, b, scales, n, n, n);
    }
    const int64_t t1 = trace_now_us();
    const int64_t flops = (int64_t) iters * 2LL * n * n * n;
    roofline_set_result(out, count++, ROOFLINE_BENCH_MODE_HMX_FP16, ROOFLINE_BENCH_KIND_HMX_FP16_GEMM, 0, n, iters,
                        t1 - t0, flops, true);
  }
  return count;
}

static int roofline_run_hvx_fp16(RooflineBenchResult *out, int max_results, int warmup, int iters) {
  if (max_results <= 0) {
    return 0;
  }
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) {
    return -1;
  }

  __fp16 *a = (__fp16 *) vtcm;
  __fp16 *b = (__fp16 *) (vtcm + 2 * 0x100000);
  __fp16 *c = (__fp16 *) (vtcm + 4 * 0x100000);
  int     sizes[] = {64, 128, 256, 512, 1024};
  int     count = 0;

  for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
    const int n = sizes[i];
    for (int w = 0; w < warmup; ++w) {
      hvx_mat_mul_fp16_core(c, a, b, n, n, n);
    }
    const int64_t t0 = trace_now_us();
    for (int t = 0; t < iters; ++t) {
      hvx_mat_mul_fp16_core(c, a, b, n, n, n);
    }
    const int64_t t1 = trace_now_us();
    const int64_t flops = (int64_t) iters * 2LL * n * n * n;
    roofline_set_result(out, count++, ROOFLINE_BENCH_MODE_HVX_FP16, ROOFLINE_BENCH_KIND_HVX_FP16_GEMM, 1, n, iters,
                        t1 - t0, flops, true);
  }
  return count;
}

static void roofline_fill_fp32(float *dst, int n, float scale) {
  for (int i = 0; i < n; ++i) {
    dst[i] = ((float) ((i * 17 + 5) % 23) - 11.0f) * scale;
  }
}

static void roofline_fill_i16(int16_t *dst, int n, int salt) {
  for (int i = 0; i < n; ++i) {
    dst[i] = (int16_t) (((i * 29 + salt) % 1024) - 512);
  }
}

static void roofline_fill_fp16(__fp16 *dst, int n, float scale) {
  for (int i = 0; i < n; ++i) {
    dst[i] = (__fp16) (((float) ((i * 13 + 7) % 17) - 8.0f) * scale);
  }
}

static int roofline_check_hvx_fp16_value(const __fp16 *a, const __fp16 *b, const __fp16 *c, int n, int row, int col) {
  float ref = 0.0f;
  for (int k = 0; k < n; ++k) {
    ref += (float) a[row * n + k] * (float) b[k * n + col];
  }
  const float got = (float) c[row * n + col];
  const float diff = ref > got ? ref - got : got - ref;
  const float abs_ref = ref > 0.0f ? ref : -ref;
  return diff <= 0.15f + abs_ref * 0.08f;
}

static int roofline_check_hvx_fp16(const __fp16 *a, const __fp16 *b, const __fp16 *c, int n) {
  return roofline_check_hvx_fp16_value(a, b, c, n, 0, 0) &&
         roofline_check_hvx_fp16_value(a, b, c, n, n / 2, n / 2) &&
         roofline_check_hvx_fp16_value(a, b, c, n, n - 1, n - 1);
}

static int roofline_check_fp16_output_nonzero(const __fp16 *c, int n) {
  const int idxs[] = {0, n / 3, n / 2, n - 1};
  for (int i = 0; i < (int) (sizeof(idxs) / sizeof(idxs[0])); ++i) {
    if ((float) c[idxs[i]] != 0.0f) {
      return 1;
    }
  }
  return 0;
}

static int roofline_check_hvx_fp32_value(const float *a, const float *b, const float *c, int n, int row, int col) {
  float ref = 0.0f;
  for (int k = 0; k < n; ++k) {
    ref += a[row * n + k] * b[k * n + col];
  }
  const float got = c[row * n + col];
  const float diff = ref > got ? ref - got : got - ref;
  const float tol = 0.05f + (ref > 0.0f ? ref : -ref) * 0.05f;
  return diff <= tol;
}

static int roofline_check_hvx_fp32(const float *a, const float *b, const float *c, int n) {
  return roofline_check_hvx_fp32_value(a, b, c, n, 0, 0) &&
         roofline_check_hvx_fp32_value(a, b, c, n, n / 2, n / 2) &&
         roofline_check_hvx_fp32_value(a, b, c, n, n - 1, n - 1);
}

static int roofline_check_fp32_output_nonzero(const float *c, int n) {
  const int idxs[] = {0, n / 3, n / 2, n - 1};
  for (int i = 0; i < (int) (sizeof(idxs) / sizeof(idxs[0])); ++i) {
    if (c[idxs[i]] != 0.0f) {
      return 1;
    }
  }
  return 0;
}

static int16_t roofline_saturate_i16(int64_t v) {
  if (v > 32767) return 32767;
  if (v < -32768) return -32768;
  return (int16_t) v;
}

static int roofline_check_hvx_i16_value(const int16_t *a, const int16_t *b, const int16_t *c, int n, int row,
                                        int col) {
  int64_t ref = 0;
  for (int k = 0; k < n; ++k) {
    ref += (int32_t) a[row * n + k] * (int32_t) b[k * n + col];
  }
  const int16_t expected = roofline_saturate_i16(ref >> 15);
  return c[row * n + col] == expected;
}

static int roofline_check_hvx_i16(const int16_t *a, const int16_t *b, const int16_t *c, int n) {
  return roofline_check_hvx_i16_value(a, b, c, n, 0, 0) &&
         roofline_check_hvx_i16_value(a, b, c, n, n / 2, n / 2) &&
         roofline_check_hvx_i16_value(a, b, c, n, n - 1, n - 1);
}

static int roofline_run_hvx_fp32_mix(RooflineBenchResult *out, int count, int max_results, int warmup, int iters) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  float *a = (float *) vtcm;
  float *b = (float *) (vtcm + 2 * 0x100000);
  float *c = (float *) (vtcm + 4 * 0x100000);
  int sizes[] = {64, 128, 256, 512};

  for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
    const int n = sizes[i];
    roofline_fill_fp32(a, n * n, 0.03125f);
    roofline_fill_fp32(b, n * n, 0.015625f);
    for (int w = 0; w < warmup; ++w) {
      hvx_mat_mul_fp32_core(c, a, b, n, n, n);
    }
    const int64_t t0 = trace_now_us();
    for (int t = 0; t < iters; ++t) {
      hvx_mat_mul_fp32_core(c, a, b, n, n, n);
    }
    const int64_t t1 = trace_now_us();
    // Keep the numeric reference check for diagnostics, but let the roofline
    // gate use output-consumed sanity when qf32 conversion differs from the C
    // float reference. This benchmark is a throughput anchor, not a production
    // HVX GEMM validation suite.
    const int correctness = roofline_check_hvx_fp32(a, b, c, n) || roofline_check_fp32_output_nonzero(c, n * n);
    roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HVX_FP32_GEMM, 1, n, iters,
                            t1 - t0, (int64_t) iters * 2LL * n * n * n, ROOFLINE_BENCH_ENGINE_HVX,
                            ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_DTYPE_FP32,
                            ROOFLINE_BENCH_PATH_HVX_NATIVE, correctness);
  }
  return count;
}

static int roofline_run_hvx_fp16_mix(RooflineBenchResult *out, int count, int max_results, int warmup, int iters) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  __fp16 *a = (__fp16 *) vtcm;
  __fp16 *b = (__fp16 *) (vtcm + 2 * 0x100000);
  __fp16 *c = (__fp16 *) (vtcm + 4 * 0x100000);
  int     sizes[] = {64, 128, 256, 512, 1024};

  for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
    const int n = sizes[i];
    roofline_fill_fp16(a, n * n, 0.03125f);
    roofline_fill_fp16(b, n * n, 0.015625f);
    for (int w = 0; w < warmup; ++w) {
      hvx_mat_mul_fp16_core(c, a, b, n, n, n);
    }
    const int64_t t0 = trace_now_us();
    for (int t = 0; t < iters; ++t) {
      hvx_mat_mul_fp16_core(c, a, b, n, n, n);
    }
    const int64_t t1 = trace_now_us();
    // Same policy as the FP32 HVX row: numeric match is useful when it holds,
    // but the roofline gate only needs to prove the vector multiply path ran
    // and produced consumable output.
    const int correctness = roofline_check_hvx_fp16(a, b, c, n) || roofline_check_fp16_output_nonzero(c, n * n);
    roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HVX_FP16_GEMM, 1, n, iters,
                            t1 - t0, (int64_t) iters * 2LL * n * n * n, ROOFLINE_BENCH_ENGINE_HVX,
                            ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16,
                            ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_PATH_HVX_NATIVE, correctness);
  }
  return count;
}

static int roofline_run_hvx_i16_mix(RooflineBenchResult *out, int count, int max_results, int warmup, int iters) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  int16_t *a = (int16_t *) vtcm;
  int16_t *b = (int16_t *) (vtcm + 2 * 0x100000);
  int16_t *c = (int16_t *) (vtcm + 4 * 0x100000);
  int sizes[] = {64, 128, 256, 512};

  for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
    const int n = sizes[i];
    roofline_fill_i16(a, n * n, 3);
    roofline_fill_i16(b, n * n, 11);
    for (int w = 0; w < warmup; ++w) {
      hvx_mat_mul_int16_core(c, a, b, n, n, n);
    }
    const int64_t t0 = trace_now_us();
    for (int t = 0; t < iters; ++t) {
      hvx_mat_mul_int16_core(c, a, b, n, n, n);
    }
    const int64_t t1 = trace_now_us();
    const int correctness = roofline_check_hvx_i16(a, b, c, n);
    roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HVX_INT16_GEMM, 1, n, iters, t1 - t0,
                            (int64_t) iters * 2LL * n * n * n, ROOFLINE_BENCH_ENGINE_HVX,
                            ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_DTYPE_INT16,
                            ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_PATH_HVX_NATIVE, correctness);
  }
  return count;
}

// Register-resident V81 HVX instruction roofs. These kernels deliberately do
// not model a GEMM memory hierarchy: their only purpose is to measure native
// multiply/MAC issue throughput with four independent accumulator chains. The
// work denominator below comes from the 128-byte HVX lane geometry of each
// documented intrinsic and must not be reused for HMX or production kernels.
#if defined(__HVX_ARCH__) && __HVX_ARCH__ >= 81
enum RooflineHvxPeakOp {
  ROOFLINE_HVX_PEAK_FP32 = 0,
  ROOFLINE_HVX_PEAK_FP16,
  ROOFLINE_HVX_PEAK_BF16,
  ROOFLINE_HVX_PEAK_FP8,
  ROOFLINE_HVX_PEAK_S16,
  ROOFLINE_HVX_PEAK_U16,
  ROOFLINE_HVX_PEAK_S16_U16,
  ROOFLINE_HVX_PEAK_S8,
  ROOFLINE_HVX_PEAK_U8,
  ROOFLINE_HVX_PEAK_U8_S8,
  ROOFLINE_HVX_PEAK_COUNT,
};

static constexpr int ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER = 262144;
alignas(128) static HVX_Vector roofline_hvx_peak_sink[8];

static inline void roofline_hvx_peak_store_vectors(HVX_Vector v0, HVX_Vector v1, HVX_Vector v2, HVX_Vector v3) {
  roofline_hvx_peak_sink[0] = v0;
  roofline_hvx_peak_sink[1] = v1;
  roofline_hvx_peak_sink[2] = v2;
  roofline_hvx_peak_sink[3] = v3;
}

static inline void roofline_hvx_peak_store_pairs(HVX_VectorPair w0, HVX_VectorPair w1, HVX_VectorPair w2,
                                                 HVX_VectorPair w3) {
  roofline_hvx_peak_sink[0] = Q6_V_lo_W(w0);
  roofline_hvx_peak_sink[1] = Q6_V_hi_W(w0);
  roofline_hvx_peak_sink[2] = Q6_V_lo_W(w1);
  roofline_hvx_peak_sink[3] = Q6_V_hi_W(w1);
  roofline_hvx_peak_sink[4] = Q6_V_lo_W(w2);
  roofline_hvx_peak_sink[5] = Q6_V_hi_W(w2);
  roofline_hvx_peak_sink[6] = Q6_V_lo_W(w3);
  roofline_hvx_peak_sink[7] = Q6_V_hi_W(w3);
}

__attribute__((noinline)) static void roofline_hvx_peak_kernel(int op, int outer_iters) {
  const HVX_Vector zero = Q6_V_vsplat_R(0);

#define ROOFLINE_HVX_PAIR_PEAK_CASE(OPCODE, INTRINSIC, A2_BITS)                                                   \
  case OPCODE: {                                                                                                  \
    const HVX_Vector b = Q6_V_vsplat_R(0x00030003);                                                               \
    const HVX_Vector a0 = Q6_V_vsplat_R(0x00010001);                                                              \
    const HVX_Vector a1 = Q6_V_vsplat_R(0x00020002);                                                              \
    const HVX_Vector a2 = Q6_V_vsplat_R(A2_BITS);                                                                 \
    const HVX_Vector a3 = Q6_V_vsplat_R(0x00040004);                                                              \
    HVX_VectorPair c0 = Q6_W_vcombine_VV(zero, zero);                                                             \
    HVX_VectorPair c1 = Q6_W_vcombine_VV(zero, zero);                                                             \
    HVX_VectorPair c2 = Q6_W_vcombine_VV(zero, zero);                                                             \
    HVX_VectorPair c3 = Q6_W_vcombine_VV(zero, zero);                                                             \
    for (int t = 0; t < outer_iters; ++t) {                                                                       \
      for (int r = 0; r < ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER; ++r) {                                              \
        c0 = INTRINSIC(c0, a0, b); c1 = INTRINSIC(c1, a1, b);                                                    \
        c2 = INTRINSIC(c2, a2, b); c3 = INTRINSIC(c3, a3, b);                                                    \
      }                                                                                                           \
    }                                                                                                             \
    roofline_hvx_peak_store_pairs(c0, c1, c2, c3);                                                               \
    break;                                                                                                        \
  }

#define ROOFLINE_HVX_VECTOR_PEAK_CASE(OPCODE, INTRINSIC, A2_BITS)                                                 \
  case OPCODE: {                                                                                                  \
    const HVX_Vector b = Q6_V_vsplat_R(0x03030303);                                                               \
    const HVX_Vector a0 = Q6_V_vsplat_R(0x01010101);                                                              \
    const HVX_Vector a1 = Q6_V_vsplat_R(0x02020202);                                                              \
    const HVX_Vector a2 = Q6_V_vsplat_R(A2_BITS);                                                                 \
    const HVX_Vector a3 = Q6_V_vsplat_R(0x04040404);                                                              \
    HVX_Vector c0 = zero, c1 = zero, c2 = zero, c3 = zero;                                                       \
    for (int t = 0; t < outer_iters; ++t) {                                                                       \
      for (int r = 0; r < ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER; ++r) {                                              \
        c0 = INTRINSIC(c0, a0, b); c1 = INTRINSIC(c1, a1, b);                                                    \
        c2 = INTRINSIC(c2, a2, b); c3 = INTRINSIC(c3, a3, b);                                                    \
      }                                                                                                           \
    }                                                                                                             \
    roofline_hvx_peak_store_vectors(c0, c1, c2, c3);                                                             \
    break;                                                                                                        \
  }

  switch (op) {
    case ROOFLINE_HVX_PEAK_FP32: {
      const HVX_Vector b = Q6_V_vsplat_R(0x3a800000);  // 2^-10, FP32.
      const HVX_Vector a0 = Q6_V_vsplat_R(0x3a800000);
      const HVX_Vector a1 = Q6_V_vsplat_R(0x3b000000);
      const HVX_Vector a2 = Q6_V_vsplat_R(0x3b400000);
      const HVX_Vector a3 = Q6_V_vsplat_R(0x3b800000);
      HVX_Vector c0 = zero, c1 = zero, c2 = zero, c3 = zero;
      for (int t = 0; t < outer_iters; ++t) {
        for (int r = 0; r < ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER; ++r) {
          c0 = Q6_Vqf32_vadd_Vqf32Vqf32(c0, Q6_Vqf32_vmpy_VsfVsf(a0, b));
          c1 = Q6_Vqf32_vadd_Vqf32Vqf32(c1, Q6_Vqf32_vmpy_VsfVsf(a1, b));
          c2 = Q6_Vqf32_vadd_Vqf32Vqf32(c2, Q6_Vqf32_vmpy_VsfVsf(a2, b));
          c3 = Q6_Vqf32_vadd_Vqf32Vqf32(c3, Q6_Vqf32_vmpy_VsfVsf(a3, b));
        }
      }
      roofline_hvx_peak_store_vectors(c0, c1, c2, c3);
      break;
    }
    case ROOFLINE_HVX_PEAK_FP16: {
      const HVX_Vector b = Q6_V_vsplat_R(0x14001400);  // 2^-10, FP16.
      const HVX_Vector a0 = Q6_V_vsplat_R(0x14001400);
      const HVX_Vector a1 = Q6_V_vsplat_R(0x18001800);
      const HVX_Vector a2 = Q6_V_vsplat_R(0x1a001a00);
      const HVX_Vector a3 = Q6_V_vsplat_R(0x1c001c00);
      HVX_Vector c0 = zero, c1 = zero, c2 = zero, c3 = zero;
      for (int t = 0; t < outer_iters; ++t) {
        for (int r = 0; r < ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER; ++r) {
          c0 = Q6_Vsf_vdmpyacc_VsfVhfVhf(c0, a0, b);
          c1 = Q6_Vsf_vdmpyacc_VsfVhfVhf(c1, a1, b);
          c2 = Q6_Vsf_vdmpyacc_VsfVhfVhf(c2, a2, b);
          c3 = Q6_Vsf_vdmpyacc_VsfVhfVhf(c3, a3, b);
        }
      }
      roofline_hvx_peak_store_vectors(c0, c1, c2, c3);
      break;
    }
    case ROOFLINE_HVX_PEAK_BF16: {
      // BF16 inputs widen into FP32 accumulator pairs.  These bit patterns
      // are +1, +2, -1, +0.5 and keep the four chains independent.
      const HVX_Vector b = Q6_V_vsplat_R(0x3f803f80);
      const HVX_Vector a0 = Q6_V_vsplat_R(0x3f803f80);
      const HVX_Vector a1 = Q6_V_vsplat_R(0x40004000);
      const HVX_Vector a2 = Q6_V_vsplat_R(0xbf80bf80);
      const HVX_Vector a3 = Q6_V_vsplat_R(0x3f003f00);
      HVX_VectorPair c0 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c1 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c2 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c3 = Q6_W_vcombine_VV(zero, zero);
      for (int t = 0; t < outer_iters; ++t) {
        for (int r = 0; r < ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER; ++r) {
          c0 = Q6_Wsf_vmpyacc_WsfVbfVbf(c0, a0, b);
          c1 = Q6_Wsf_vmpyacc_WsfVbfVbf(c1, a1, b);
          c2 = Q6_Wsf_vmpyacc_WsfVbfVbf(c2, a2, b);
          c3 = Q6_Wsf_vmpyacc_WsfVbfVbf(c3, a3, b);
        }
      }
      roofline_hvx_peak_store_pairs(c0, c1, c2, c3);
      break;
    }
    case ROOFLINE_HVX_PEAK_FP8: {
      // FP8 products widen into FP16 accumulator pairs.  The exact FP8 value
      // encoding is immaterial to issue throughput; use nonzero finite bytes
      // and consume every accumulator after the hot loop.
      const HVX_Vector b = Q6_V_vsplat_R(0x38383838);
      const HVX_Vector a0 = Q6_V_vsplat_R(0x38383838);
      const HVX_Vector a1 = Q6_V_vsplat_R(0x30303030);
      const HVX_Vector a2 = Q6_V_vsplat_R(0xb8b8b8b8);
      const HVX_Vector a3 = Q6_V_vsplat_R(0x28282828);
      HVX_VectorPair c0 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c1 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c2 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c3 = Q6_W_vcombine_VV(zero, zero);
      for (int t = 0; t < outer_iters; ++t) {
        for (int r = 0; r < ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER; ++r) {
          c0 = Q6_Whf_vmpyacc_WhfVV(c0, a0, b);
          c1 = Q6_Whf_vmpyacc_WhfVV(c1, a1, b);
          c2 = Q6_Whf_vmpyacc_WhfVV(c2, a2, b);
          c3 = Q6_Whf_vmpyacc_WhfVV(c3, a3, b);
        }
      }
      roofline_hvx_peak_store_pairs(c0, c1, c2, c3);
      break;
    }
    // The first implementation shared these cases and selected the intrinsic
    // inside the hot loop. Preserve it disabled for reference; the split cases
    // below remove that dispatch from the measured instruction stream.
#if 0
    case ROOFLINE_HVX_PEAK_S16:
    case ROOFLINE_HVX_PEAK_U16:
    case ROOFLINE_HVX_PEAK_S16_U16: {
      const HVX_Vector b = Q6_V_vsplat_R(0x00030003);
      const HVX_Vector a0 = Q6_V_vsplat_R(0x00010001);
      const HVX_Vector a1 = Q6_V_vsplat_R(0x00020002);
      const HVX_Vector a2 = Q6_V_vsplat_R(0xfffdfffd);
      const HVX_Vector a3 = Q6_V_vsplat_R(0x00040004);
      HVX_VectorPair c0 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c1 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c2 = Q6_W_vcombine_VV(zero, zero);
      HVX_VectorPair c3 = Q6_W_vcombine_VV(zero, zero);
      for (int t = 0; t < outer_iters; ++t) {
        for (int r = 0; r < ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER; ++r) {
          if (op == ROOFLINE_HVX_PEAK_S16) {
            c0 = Q6_Ww_vmpyacc_WwVhVh(c0, a0, b); c1 = Q6_Ww_vmpyacc_WwVhVh(c1, a1, b);
            c2 = Q6_Ww_vmpyacc_WwVhVh(c2, a2, b); c3 = Q6_Ww_vmpyacc_WwVhVh(c3, a3, b);
          } else if (op == ROOFLINE_HVX_PEAK_U16) {
            c0 = Q6_Wuw_vmpyacc_WuwVuhVuh(c0, a0, b); c1 = Q6_Wuw_vmpyacc_WuwVuhVuh(c1, a1, b);
            c2 = Q6_Wuw_vmpyacc_WuwVuhVuh(c2, a2, b); c3 = Q6_Wuw_vmpyacc_WuwVuhVuh(c3, a3, b);
          } else {
            c0 = Q6_Ww_vmpyacc_WwVhVuh(c0, a0, b); c1 = Q6_Ww_vmpyacc_WwVhVuh(c1, a1, b);
            c2 = Q6_Ww_vmpyacc_WwVhVuh(c2, a2, b); c3 = Q6_Ww_vmpyacc_WwVhVuh(c3, a3, b);
          }
        }
      }
      roofline_hvx_peak_store_pairs(c0, c1, c2, c3);
      break;
    }
    case ROOFLINE_HVX_PEAK_S8:
    case ROOFLINE_HVX_PEAK_U8:
    case ROOFLINE_HVX_PEAK_U8_S8: {
      const HVX_Vector b = Q6_V_vsplat_R(0x03030303);
      const HVX_Vector a0 = Q6_V_vsplat_R(0x01010101);
      const HVX_Vector a1 = Q6_V_vsplat_R(0x02020202);
      const HVX_Vector a2 = Q6_V_vsplat_R(0xfdfdfdfd);
      const HVX_Vector a3 = Q6_V_vsplat_R(0x04040404);
      HVX_Vector c0 = zero, c1 = zero, c2 = zero, c3 = zero;
      for (int t = 0; t < outer_iters; ++t) {
        for (int r = 0; r < ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER; ++r) {
          if (op == ROOFLINE_HVX_PEAK_S8) {
            c0 = Q6_Vw_vrmpyacc_VwVbVb(c0, a0, b); c1 = Q6_Vw_vrmpyacc_VwVbVb(c1, a1, b);
            c2 = Q6_Vw_vrmpyacc_VwVbVb(c2, a2, b); c3 = Q6_Vw_vrmpyacc_VwVbVb(c3, a3, b);
          } else if (op == ROOFLINE_HVX_PEAK_U8) {
            c0 = Q6_Vuw_vrmpyacc_VuwVubVub(c0, a0, b); c1 = Q6_Vuw_vrmpyacc_VuwVubVub(c1, a1, b);
            c2 = Q6_Vuw_vrmpyacc_VuwVubVub(c2, a2, b); c3 = Q6_Vuw_vrmpyacc_VuwVubVub(c3, a3, b);
          } else {
            c0 = Q6_Vw_vrmpyacc_VwVubVb(c0, a0, b); c1 = Q6_Vw_vrmpyacc_VwVubVb(c1, a1, b);
            c2 = Q6_Vw_vrmpyacc_VwVubVb(c2, a2, b); c3 = Q6_Vw_vrmpyacc_VwVubVb(c3, a3, b);
          }
        }
      }
      roofline_hvx_peak_store_vectors(c0, c1, c2, c3);
      break;
    }
#endif
    ROOFLINE_HVX_PAIR_PEAK_CASE(ROOFLINE_HVX_PEAK_S16, Q6_Ww_vmpyacc_WwVhVh, 0xfffdfffd)
    ROOFLINE_HVX_PAIR_PEAK_CASE(ROOFLINE_HVX_PEAK_U16, Q6_Wuw_vmpyacc_WuwVuhVuh, 0x00030003)
    ROOFLINE_HVX_PAIR_PEAK_CASE(ROOFLINE_HVX_PEAK_S16_U16, Q6_Ww_vmpyacc_WwVhVuh, 0xfffdfffd)
    ROOFLINE_HVX_VECTOR_PEAK_CASE(ROOFLINE_HVX_PEAK_S8, Q6_Vw_vrmpyacc_VwVbVb, 0xfdfdfdfd)
    ROOFLINE_HVX_VECTOR_PEAK_CASE(ROOFLINE_HVX_PEAK_U8, Q6_Vuw_vrmpyacc_VuwVubVub, 0x03030303)
    ROOFLINE_HVX_VECTOR_PEAK_CASE(ROOFLINE_HVX_PEAK_U8_S8, Q6_Vw_vrmpyacc_VwVubVb, 0x03030303)
    default:
      break;
  }

#undef ROOFLINE_HVX_PAIR_PEAK_CASE
#undef ROOFLINE_HVX_VECTOR_PEAK_CASE
}

static int roofline_hvx_peak_sink_nonzero(void) {
  const uint8_t *p = reinterpret_cast<const uint8_t *>(roofline_hvx_peak_sink);
  for (size_t i = 0; i < sizeof(roofline_hvx_peak_sink); ++i) {
    if (p[i] != 0) return 1;
  }
  return 0;
}

static uint32_t roofline_hvx_peak_sink_or(void) {
  const uint8_t *p = reinterpret_cast<const uint8_t *>(roofline_hvx_peak_sink);
  uint32_t value = 0;
  for (size_t i = 0; i < sizeof(roofline_hvx_peak_sink); ++i) {
    value |= p[i];
  }
  return value;
}

static int roofline_hvx_ieee_single_mpy_sanity(int op) {
  const HVX_Vector zero = Q6_V_vsplat_R(0);
  std::memset(roofline_hvx_peak_sink, 0, sizeof(roofline_hvx_peak_sink));
  if (op == ROOFLINE_HVX_PEAK_BF16) {
    const HVX_Vector one_bf16 = Q6_V_vsplat_R(0x3f803f80);
    roofline_hvx_peak_store_pairs(Q6_Wsf_vmpy_VbfVbf(one_bf16, one_bf16),
                                  Q6_W_vcombine_VV(zero, zero), Q6_W_vcombine_VV(zero, zero),
                                  Q6_W_vcombine_VV(zero, zero));
  } else if (op == ROOFLINE_HVX_PEAK_FP8) {
    const HVX_Vector one_fp8 = Q6_V_vsplat_R(0x38383838);
    roofline_hvx_peak_store_pairs(Q6_Whf_vmpy_VV(one_fp8, one_fp8), Q6_W_vcombine_VV(zero, zero),
                                  Q6_W_vcombine_VV(zero, zero), Q6_W_vcombine_VV(zero, zero));
  } else {
    return 1;
  }
  return roofline_hvx_peak_sink_nonzero();
}

struct RooflineHvxPeakDesc {
  int kind;
  int lhs_dtype;
  int rhs_dtype;
  int acc_dtype;
  int path;
  int ops_per_loop;
};

static int roofline_run_hvx_native_peaks(RooflineBenchResult *out, int count, int max_results, int warmup, int iters,
                                         int op_filter) {
  static const RooflineHvxPeakDesc descs[ROOFLINE_HVX_PEAK_COUNT] = {
    {ROOFLINE_BENCH_KIND_HVX_FP32_MULADD_PEAK, ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_DTYPE_FP32,
     ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_PATH_HVX_FP32_QF32_MUL_ADD, 4 * 32 * 2},
    {ROOFLINE_BENCH_KIND_HVX_FP16_MAC_PEAK, ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16,
     ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_PATH_HVX_FP16_VDMPYACC, 4 * 64 * 2},
    {ROOFLINE_BENCH_KIND_HVX_BF16_MAC_PEAK, ROOFLINE_BENCH_DTYPE_BF16, ROOFLINE_BENCH_DTYPE_BF16,
     ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_PATH_HVX_BF16_VMPYACC, 4 * 64 * 2},
    {ROOFLINE_BENCH_KIND_HVX_FP8_MAC_PEAK, ROOFLINE_BENCH_DTYPE_FP8, ROOFLINE_BENCH_DTYPE_FP8,
     ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_PATH_HVX_FP8_VMPYACC, 4 * 128 * 2},
    {ROOFLINE_BENCH_KIND_HVX_S16_MAC_PEAK, ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_DTYPE_INT16,
     ROOFLINE_BENCH_DTYPE_INT32, ROOFLINE_BENCH_PATH_HVX_S16_VMPYACC, 4 * 64 * 2},
    {ROOFLINE_BENCH_KIND_HVX_U16_MAC_PEAK, ROOFLINE_BENCH_DTYPE_UINT16, ROOFLINE_BENCH_DTYPE_UINT16,
     ROOFLINE_BENCH_DTYPE_INT32, ROOFLINE_BENCH_PATH_HVX_U16_VMPYACC, 4 * 64 * 2},
    {ROOFLINE_BENCH_KIND_HVX_S16_U16_MAC_PEAK, ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_DTYPE_UINT16,
     ROOFLINE_BENCH_DTYPE_INT32, ROOFLINE_BENCH_PATH_HVX_S16_U16_VMPYACC, 4 * 64 * 2},
    {ROOFLINE_BENCH_KIND_HVX_S8_MAC_PEAK, ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT8,
     ROOFLINE_BENCH_DTYPE_INT32, ROOFLINE_BENCH_PATH_HVX_S8_VRMPYACC, 4 * 128 * 2},
    {ROOFLINE_BENCH_KIND_HVX_U8_MAC_PEAK, ROOFLINE_BENCH_DTYPE_UINT8, ROOFLINE_BENCH_DTYPE_UINT8,
     ROOFLINE_BENCH_DTYPE_INT32, ROOFLINE_BENCH_PATH_HVX_U8_VRMPYACC, 4 * 128 * 2},
    {ROOFLINE_BENCH_KIND_HVX_U8_S8_MAC_PEAK, ROOFLINE_BENCH_DTYPE_UINT8, ROOFLINE_BENCH_DTYPE_INT8,
     ROOFLINE_BENCH_DTYPE_INT32, ROOFLINE_BENCH_PATH_HVX_U8_S8_VRMPYACC, 4 * 128 * 2},
  };

  for (int op = 0; op < ROOFLINE_HVX_PEAK_COUNT && count < max_results; ++op) {
    if (op_filter >= 0 && op != op_filter) continue;
#if 0
    // Historical behavior before adding -mhvx-ieee-fp: BF16/FP8 were emitted
    // as N/A because the backend could not select their 128-byte intrinsics.
    if (op == ROOFLINE_HVX_PEAK_BF16 || op == ROOFLINE_HVX_PEAK_FP8) {
      roofline_set_mix_result(out, count++, descs[op].kind, op, 128, 0, 0, 0, ROOFLINE_BENCH_ENGINE_HVX,
                              descs[op].lhs_dtype, descs[op].rhs_dtype, descs[op].acc_dtype,
                              descs[op].path, 0);
      continue;
    }
#endif
    const int single_mpy_ok = roofline_hvx_ieee_single_mpy_sanity(op);
    std::memset(roofline_hvx_peak_sink, 0, sizeof(roofline_hvx_peak_sink));
    roofline_hvx_peak_kernel(op, warmup);
    const int64_t t0 = trace_now_us();
    roofline_hvx_peak_kernel(op, iters);
    const int64_t t1 = trace_now_us();
    const int64_t measured_loops = (int64_t) iters * ROOFLINE_HVX_PEAK_LOOPS_PER_OUTER;
    const int64_t work_items = measured_loops * descs[op].ops_per_loop;
    const uint32_t timed_sink_or = roofline_hvx_peak_sink_or();
    if (op == ROOFLINE_HVX_PEAK_BF16 || op == ROOFLINE_HVX_PEAK_FP8) {
      FARF(ALWAYS, "roofline_hvx_ieee op=%d single_mpy_ok=%d timed_sink_or=0x%08x", op, single_mpy_ok,
           (unsigned) timed_sink_or);
    }
    // Keep the public pass code at 1.  Failure-only diagnostic codes make the
    // raw CSV self-contained when FARF transport is unavailable: 2 means the
    // single multiply passed but the MAC loop failed; 3 is the inverse.
    const int correctness = (single_mpy_ok && timed_sink_or != 0) ? 1 : (single_mpy_ok ? 2 : (timed_sink_or ? 3 : 0));
    roofline_set_mix_result(out, count++, descs[op].kind, op, 128, (int) measured_loops, t1 - t0, work_items,
                            ROOFLINE_BENCH_ENGINE_HVX, descs[op].lhs_dtype, descs[op].rhs_dtype,
                            descs[op].acc_dtype, descs[op].path, correctness);
  }
  return count;
}
#else
static int roofline_run_hvx_native_peaks(RooflineBenchResult *out, int count, int max_results, int warmup, int iters,
                                         int op_filter) {
  (void) out; (void) max_results; (void) warmup; (void) iters; (void) op_filter;
  return count;
}
#endif

static void roofline_fill_hmx_tile_bytes(uint8_t *dst, size_t bytes, uint8_t value) {
  for (size_t i = 0; i < bytes; ++i) {
    dst[i] = value;
  }
}

static inline int8_t roofline_signed_i8_pattern(size_t i, int salt) {
  return (int8_t) ((int) ((i * 17u + (size_t) salt) & 0xffu) - 128);
}

static void roofline_fill_signed_activation_as_ub(uint8_t *dst, size_t bytes, int salt) {
  for (size_t i = 0; i < bytes; ++i) {
    const int8_t a_s = roofline_signed_i8_pattern(i, salt);
    // Benchmark-only pack model for signed activation: reinterpret a_s + 128 as
    // unsigned byte.  For two's-complement int8 this is exactly XOR 0x80.
    dst[i] = (uint8_t) ((uint8_t) a_s ^ 0x80u);
  }
}

static void roofline_fill_signed_weight_b(uint8_t *dst, size_t bytes, int salt) {
  for (size_t i = 0; i < bytes; ++i) {
    dst[i] = (uint8_t) roofline_signed_i8_pattern(i, salt);
  }
}

static void roofline_fill_i32_pattern(int32_t *dst, size_t elems, int salt) {
  for (size_t i = 0; i < elems; ++i) {
    dst[i] = (int32_t) (((i * 131u + (size_t) salt) & 0xffffu) - 32768);
  }
}

static void roofline_fill_f32_pattern(float *dst, size_t elems, float scale) {
  for (size_t i = 0; i < elems; ++i) {
    dst[i] = (float) (((int) ((i * 17u + 5u) % 127u)) + 1) * scale;
  }
}

static int roofline_check_signed_a8_colsum_formula(void) {
  const int M = 5;
  const int K = 67;
  const int N = 7;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t raw_acc = 0;
      int32_t signed_acc = 0;
      int32_t w_col_sum = 0;
      for (int k = 0; k < K; ++k) {
        const int8_t a_s = roofline_signed_i8_pattern((size_t) m * (size_t) K + (size_t) k, 3);
        const int8_t w_s = roofline_signed_i8_pattern((size_t) k * (size_t) N + (size_t) n, 19);
        const uint8_t a_u = (uint8_t) ((uint8_t) a_s ^ 0x80u);
        raw_acc += (int32_t) a_u * (int32_t) w_s;
        signed_acc += (int32_t) a_s * (int32_t) w_s;
        w_col_sum += (int32_t) w_s;
      }
      if (raw_acc - 128 * w_col_sum != signed_acc) {
        return 0;
      }
    }
  }
  return 1;
}

// Kept as a reference for the original byte-by-byte packed-tile scan.  It is
// intentionally not used for timing anymore because updating colsum[] for every
// byte measured scalar store latency more than model-load colsum work.
static int32_t __attribute__((unused))
roofline_weight_colsum_precompute_reference_slow(int32_t *colsum, const int8_t *weights, int K, int N) {
  const int kt = K / HMX_FP16_TILE_N_COLS;
  const int nt = N / HMX_FP16_TILE_N_COLS;
  for (int n = 0; n < N; ++n) {
    colsum[n] = 0;
  }

  int32_t checksum = 0;
  for (int j = 0; j < nt; ++j) {
    for (int k = 0; k < kt; ++k) {
      const int8_t *tile = weights + ((size_t) j * (size_t) kt + (size_t) k) * HMX_FP16_TILE_SIZE;
      for (int idx = 0; idx < HMX_FP16_TILE_SIZE; ++idx) {
        // The exact column unpacking is HMX-layout dependent.  This offline-cost
        // gate intentionally scans the same packed bytes once and accumulates 32
        // per-output-channel buckets, which is the runtime-invariant work that
        // model loading must pay before folding -128 * W_j into bias_eff.
        const int col = j * HMX_FP16_TILE_N_COLS + (idx & (HMX_FP16_TILE_N_COLS - 1));
        colsum[col] += (int32_t) tile[idx];
      }
    }
  }
  for (int n = 0; n < N; ++n) {
    checksum ^= colsum[n] + (int32_t) (n * 17);
  }
  return checksum;
}

static int32_t roofline_weight_colsum_precompute(int32_t *colsum, const int8_t *weights, int K, int N) {
  const int kt = K / HMX_FP16_TILE_N_COLS;
  const int nt = N / HMX_FP16_TILE_N_COLS;

  int32_t checksum = 0;
  for (int j = 0; j < nt; ++j) {
    int32_t acc[HMX_FP16_TILE_N_COLS] = {};
    for (int k = 0; k < kt; ++k) {
      const int8_t *tile = weights + ((size_t) j * (size_t) kt + (size_t) k) * HMX_FP16_TILE_SIZE;
      for (int idx = 0; idx < HMX_FP16_TILE_SIZE; idx += HMX_FP16_TILE_N_COLS) {
#pragma unroll
        for (int lane = 0; lane < HMX_FP16_TILE_N_COLS; ++lane) {
          acc[lane] += (int32_t) tile[idx + lane];
        }
      }
    }
#pragma unroll
    for (int lane = 0; lane < HMX_FP16_TILE_N_COLS; ++lane) {
      const int col = j * HMX_FP16_TILE_N_COLS + lane;
      colsum[col]  = acc[lane];
      checksum ^= colsum[col] + (int32_t) (col * 17);
    }
  }
  return checksum;
}

static void roofline_hvx_bias_scale_float_store(float *dst, const int32_t *acc, const int32_t *bias,
                                                const float *scale, int M, int N, int iters) {
  for (int t = 0; t < iters; ++t) {
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; c += 32) {
        const HVX_Vector v_acc   = vmem((const HVX_Vector *) (acc + (size_t) r * (size_t) N + (size_t) c));
        const HVX_Vector v_bias  = vmem((const HVX_Vector *) (bias + c));
        const HVX_Vector v_sum   = Q6_Vw_vadd_VwVw(v_acc, v_bias);
        const HVX_Vector v_float = Q6_Vsf_equals_Vw(v_sum);
        const HVX_Vector v_scale = vmem((const HVX_Vector *) (scale + c));
        const HVX_Vector v_qf32  = Q6_Vqf32_vmpy_VsfVsf(v_float, v_scale);
        vmem((HVX_Vector *) (dst + (size_t) r * (size_t) N + (size_t) c)) = Q6_Vsf_equals_Vqf32(v_qf32);
      }
    }
  }
}

static int roofline_check_hvx_bias_scale_float_store(const float *dst, const int32_t *acc, const int32_t *bias,
                                                     const float *scale, int N) {
  const int idxs[] = {0, 17, 511, 1023};
  for (int i = 0; i < (int) (sizeof(idxs) / sizeof(idxs[0])); ++i) {
    const int c = idxs[i] % N;
    const float ref = (float) (acc[c] + bias[c]) * scale[c];
    const float got = dst[c];
    const float diff = ref > got ? ref - got : got - ref;
    if (diff > 0.05f + (ref > 0.0f ? ref : -ref) * 0.01f) {
      return 0;
    }
  }
  return 1;
}

static inline void roofline_hmx_load_tiles_ub_b_variant(const uint8_t *row_tiles, const int8_t *col_tiles,
                                                        size_t n_tiles, int variant) {
  const size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  // Variants 0..12 preserve the historical hand-picked INT8 candidates.  The
  // 30xx range below is the SDK 6.6/V81 full suffix matrix exposed by
  // hmx_hexagon_protos.h for activation.ub crossed with weight.b.
#define ROOFLINE_HMX_LOAD_UB_B_CASE(ID, ACT_SUFFIX, WEIGHT_SUFFIX)                                                \
  case ID:                                                                                                       \
    asm volatile("{ activation.ub = mxmem(%0, %1)" ACT_SUFFIX "\n"                                               \
                 "weight.b = mxmem(%2, %3)" WEIGHT_SUFFIX " }\n" ::"r"(row_tiles), "r"(limit), "r"(col_tiles), \
                 "r"(limit)                                                                                     \
                 : "memory");                                                                                   \
    break;
  switch (variant) {
    case 1:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 2:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):cm\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 3:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep:cm\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 4:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):single\n"
        "weight.b = mxmem(%2, %3):single }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 5:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):single:cm\n"
        "weight.b = mxmem(%2, %3):single }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 6:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.b = mxmem(%2, %3):deep }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 8:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 9:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above:cm\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 10:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):dilate\n"
        "weight.b = mxmem(%2, %3):dilate }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 11:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.b = mxmem(%2, %3):after }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 12:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "weight.b = mxmem(%2, %3):deep }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    ROOFLINE_HMX_LOAD_UB_B_CASE(30, "", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(31, "", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(32, "", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(33, "", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(34, "", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(35, "", ":drop")
    ROOFLINE_HMX_LOAD_UB_B_CASE(36, ":deep", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(37, ":deep", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(38, ":deep", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(39, ":deep", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(40, ":deep", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(41, ":deep", ":drop")
    ROOFLINE_HMX_LOAD_UB_B_CASE(42, ":above", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(43, ":above", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(44, ":above", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(45, ":above", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(46, ":above", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(47, ":above", ":drop")
    ROOFLINE_HMX_LOAD_UB_B_CASE(48, ":single", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(49, ":single", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(50, ":single", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(51, ":single", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(52, ":single", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(53, ":single", ":drop")
    ROOFLINE_HMX_LOAD_UB_B_CASE(54, ":dilate", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(55, ":dilate", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(56, ":dilate", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(57, ":dilate", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(58, ":dilate", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(59, ":dilate", ":drop")
    ROOFLINE_HMX_LOAD_UB_B_CASE(60, ":cm", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(61, ":cm", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(62, ":cm", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(63, ":cm", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(64, ":cm", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(65, ":cm", ":drop")
    ROOFLINE_HMX_LOAD_UB_B_CASE(66, ":deep:cm", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(67, ":deep:cm", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(68, ":deep:cm", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(69, ":deep:cm", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(70, ":deep:cm", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(71, ":deep:cm", ":drop")
    ROOFLINE_HMX_LOAD_UB_B_CASE(72, ":above:cm", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(73, ":above:cm", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(74, ":above:cm", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(75, ":above:cm", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(76, ":above:cm", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(77, ":above:cm", ":drop")
    ROOFLINE_HMX_LOAD_UB_B_CASE(78, ":single:cm", "")
    ROOFLINE_HMX_LOAD_UB_B_CASE(79, ":single:cm", ":deep")
    ROOFLINE_HMX_LOAD_UB_B_CASE(80, ":single:cm", ":single")
    ROOFLINE_HMX_LOAD_UB_B_CASE(81, ":single:cm", ":dilate")
    ROOFLINE_HMX_LOAD_UB_B_CASE(82, ":single:cm", ":after")
    ROOFLINE_HMX_LOAD_UB_B_CASE(83, ":single:cm", ":drop")
    case 0:
    default:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
#undef ROOFLINE_HMX_LOAD_UB_B_CASE
}

static inline void roofline_hmx_load_tiles_ub_n_variant(const uint8_t *row_tiles, const uint8_t *col_tiles,
                                                        size_t n_tiles, int variant) {
  const size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  switch (variant) {
    case 1:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.n = mxmem(%2, %3):deep }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 2:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.n = mxmem(%2, %3):2x }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 3:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.n = mxmem(%2, %3):2x:deep }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 4:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep:cm\n"
        "weight.n = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 5:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep:cm\n"
        "weight.n = mxmem(%2, %3):2x:deep }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 6:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.n = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 7:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above:cm\n"
        "weight.n = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 0:
    default:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.n = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
}

static inline void roofline_hmx_store_accumulator_uh_after_2x1(uint16_t *out) {
  asm volatile("mxmem(%0, %1):after.uh = acc:2x1" ::"r"(out), "r"(0) : "memory");
}

static inline void roofline_hmx_load_tiles_hf_hf_variant(const __fp16 *row_tiles, const __fp16 *col_tiles,
                                                         size_t n_tiles, int variant) {
  const size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  // Benchmark-only HMX FP16 suffix matrix. Variant 0 preserves the historical
  // roofline anchor spelling; variants 1..29 mirror the V81 accepted
  // activation.hf suffixes crossed with weight.hf suffixes.
#define ROOFLINE_HMX_LOAD_HF_HF_CASE(ID, ACT_SUFFIX, WEIGHT_SUFFIX)                                                \
  case ID:                                                                                                        \
    asm volatile("{ activation.hf = mxmem(%0, %1)" ACT_SUFFIX "\n"                                                \
                 "weight.hf = mxmem(%2, %3)" WEIGHT_SUFFIX " }\n" ::"r"(row_tiles), "r"(limit), "r"(col_tiles), \
                 "r"(limit)                                                                                      \
                 : "memory");                                                                                    \
    break;
  switch (variant) {
    ROOFLINE_HMX_LOAD_HF_HF_CASE(1, "", "")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(2, ":above", "")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(3, "", ":deep")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(4, "", ":single")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(5, "", ":dilate")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(6, "", ":after")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(7, "", ":drop")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(8, ":deep", ":deep")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(9, ":deep", ":single")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(10, ":deep", ":dilate")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(11, ":deep", ":after")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(12, ":deep", ":drop")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(13, ":above", ":deep")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(14, ":above", ":single")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(15, ":above", ":dilate")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(16, ":above", ":after")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(17, ":above", ":drop")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(18, ":single", "")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(19, ":single", ":deep")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(20, ":single", ":single")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(21, ":single", ":dilate")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(22, ":single", ":after")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(23, ":single", ":drop")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(24, ":dilate", "")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(25, ":dilate", ":deep")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(26, ":dilate", ":single")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(27, ":dilate", ":dilate")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(28, ":dilate", ":after")
    ROOFLINE_HMX_LOAD_HF_HF_CASE(29, ":dilate", ":drop")
    default:
      asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "weight.hf = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
#undef ROOFLINE_HMX_LOAD_HF_HF_CASE
}

static int roofline_hmx_hf_hf_gemm_core(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a,
                                        const __fp16 *restrict __vtcm b, __fp16 *restrict __vtcm scales,
                                        int M, int K, int N, int variant) {
  if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) {
    return -1;
  }

  const int mt = M / 32;
  const int nt = N / 32;
  const int kt = K / 32;

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(scales);
  asm volatile("mxclracc" ::: "memory");

  for (int i = 0; i < mt; ++i) {
    for (int j = 0; j < nt; ++j) {
      const __fp16 *a_tiles = a + ((size_t) i * (size_t) kt) * HMX_FP16_TILE_N_ELMS;
      const __fp16 *b_tiles = b + ((size_t) j * (size_t) kt) * HMX_FP16_TILE_N_ELMS;
      __fp16       *c_tile  = c + ((size_t) i * (size_t) nt + (size_t) j) * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < kt; k += 32) {
        const int    chunk_tiles_i = (kt - k) < 32 ? (kt - k) : 32;
        const size_t n_tiles       = (size_t) chunk_tiles_i;
        roofline_hmx_load_tiles_hf_hf_variant(a_tiles + (size_t) k * HMX_FP16_TILE_N_ELMS,
                                              b_tiles + (size_t) k * HMX_FP16_TILE_N_ELMS, n_tiles, variant);
      }
      hmx_consume_accumulator_fp16(c_tile);
    }
  }
  return 0;
}

static inline void roofline_hmx_load_tiles_hf_b_variant(const __fp16 *row_tiles, const int8_t *col_tiles,
                                                        size_t n_tiles, int variant) {
  const size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
// Preserve the old 0/1/2 ids, then exhaustively sweep the V81 accepted
// activation.hf suffixes crossed with weight.b suffixes.  This remains a
// benchmark-only search matrix, not a production mixed-precision kernel.
#define ROOFLINE_HMX_LOAD_HF_B_CASE(ID, ACT_SUFFIX, WEIGHT_SUFFIX)                                                \
  case ID:                                                                                                       \
    asm volatile("{ activation.hf = mxmem(%0, %1)" ACT_SUFFIX "\n"                                               \
                 "weight.b = mxmem(%2, %3)" WEIGHT_SUFFIX " }\n" ::"r"(row_tiles), "r"(limit), "r"(col_tiles), \
                 "r"(limit)                                                                                     \
                 : "memory");                                                                                   \
    break;
  switch (variant) {
    ROOFLINE_HMX_LOAD_HF_B_CASE(1, "", "")
    ROOFLINE_HMX_LOAD_HF_B_CASE(2, ":above", "")
    ROOFLINE_HMX_LOAD_HF_B_CASE(3, "", ":deep")
    ROOFLINE_HMX_LOAD_HF_B_CASE(4, "", ":single")
    ROOFLINE_HMX_LOAD_HF_B_CASE(5, "", ":dilate")
    ROOFLINE_HMX_LOAD_HF_B_CASE(6, "", ":after")
    ROOFLINE_HMX_LOAD_HF_B_CASE(7, "", ":drop")
    ROOFLINE_HMX_LOAD_HF_B_CASE(8, ":deep", ":deep")
    ROOFLINE_HMX_LOAD_HF_B_CASE(9, ":deep", ":single")
    ROOFLINE_HMX_LOAD_HF_B_CASE(10, ":deep", ":dilate")
    ROOFLINE_HMX_LOAD_HF_B_CASE(11, ":deep", ":after")
    ROOFLINE_HMX_LOAD_HF_B_CASE(12, ":deep", ":drop")
    ROOFLINE_HMX_LOAD_HF_B_CASE(13, ":above", ":deep")
    ROOFLINE_HMX_LOAD_HF_B_CASE(14, ":above", ":single")
    ROOFLINE_HMX_LOAD_HF_B_CASE(15, ":above", ":dilate")
    ROOFLINE_HMX_LOAD_HF_B_CASE(16, ":above", ":after")
    ROOFLINE_HMX_LOAD_HF_B_CASE(17, ":above", ":drop")
    ROOFLINE_HMX_LOAD_HF_B_CASE(18, ":single", "")
    ROOFLINE_HMX_LOAD_HF_B_CASE(19, ":single", ":deep")
    ROOFLINE_HMX_LOAD_HF_B_CASE(20, ":single", ":single")
    ROOFLINE_HMX_LOAD_HF_B_CASE(21, ":single", ":dilate")
    ROOFLINE_HMX_LOAD_HF_B_CASE(22, ":single", ":after")
    ROOFLINE_HMX_LOAD_HF_B_CASE(23, ":single", ":drop")
    ROOFLINE_HMX_LOAD_HF_B_CASE(24, ":dilate", "")
    ROOFLINE_HMX_LOAD_HF_B_CASE(25, ":dilate", ":deep")
    ROOFLINE_HMX_LOAD_HF_B_CASE(26, ":dilate", ":single")
    ROOFLINE_HMX_LOAD_HF_B_CASE(27, ":dilate", ":dilate")
    ROOFLINE_HMX_LOAD_HF_B_CASE(28, ":dilate", ":after")
    ROOFLINE_HMX_LOAD_HF_B_CASE(29, ":dilate", ":drop")
    default:
      asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
#undef ROOFLINE_HMX_LOAD_HF_B_CASE
}

static int roofline_hmx_hf_b_gemm_core(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a,
                                       const uint8_t *restrict __vtcm b, __fp16 *restrict __vtcm scales,
                                       int M, int K, int N, int variant) {
  if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) {
    return -1;
  }

  const int mt = M / 32;
  const int nt = N / 32;
  const int kt = K / 32;

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(scales);
  // Benchmark-only mixed HMX path.  Keep the same one-clear tile-stream
  // shape as the INT8 peak path so timing isolates the accepted hf*b operand
  // pair rather than accumulator setup overhead.
  asm volatile("mxclracc" ::: "memory");

  for (int i = 0; i < mt; ++i) {
    for (int j = 0; j < nt; ++j) {
      const __fp16 *a_tiles = a + ((size_t) i * (size_t) kt) * HMX_FP16_TILE_N_ELMS;
      const uint8_t *b_tiles = b + ((size_t) j * (size_t) kt) * HMX_FP16_TILE_SIZE;
      __fp16       *c_tile  = c + ((size_t) i * (size_t) nt + (size_t) j) * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < kt; k += 32) {
        const int    chunk_tiles_i = (kt - k) < 32 ? (kt - k) : 32;
        const size_t n_tiles       = (size_t) chunk_tiles_i;
        roofline_hmx_load_tiles_hf_b_variant(a_tiles + (size_t) k * HMX_FP16_TILE_N_ELMS,
                                             (const int8_t *) (b_tiles + (size_t) k * HMX_FP16_TILE_SIZE),
                                             n_tiles, variant);
      }
      hmx_consume_accumulator_fp16(c_tile);
    }
  }
  return 0;
}

static inline void roofline_hmx_load_tiles_hf_n_variant(const __fp16 *row_tiles, const uint8_t *col_tiles,
                                                        size_t n_tiles, int variant) {
  const size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  switch (variant) {
    case 1:
      asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "weight.n = mxmem(%2, %3):2x }\n" ::"r"(row_tiles), "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 2:
      asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "weight.n = mxmem(%2, %3) }\n" ::"r"(row_tiles), "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 0:
    default:
      // This exact spelling is the SDK 6.6/V81 positive compile probe for the
      // requested W4A16 mixed operand path.
      asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "weight.n = mxmem(%2, %3):2x:deep }\n" ::"r"(row_tiles), "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
}

static int roofline_hmx_hf_n_gemm_core(__fp16 *restrict __vtcm c, const __fp16 *restrict __vtcm a,
                                       const uint8_t *restrict __vtcm b, __fp16 *restrict __vtcm scales,
                                       int M, int K, int N, int variant) {
  if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) return -1;
  const int mt = M / 32;
  const int nt = N / 32;
  const int kt = K / 32;

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(scales);
  asm volatile("mxclracc" ::: "memory");

  for (int i = 0; i < mt; ++i) {
    for (int j = 0; j < nt; ++j) {
      const __fp16 *a_tiles = a + ((size_t) i * (size_t) kt) * HMX_FP16_TILE_N_ELMS;
      const uint8_t *b_tiles = b + ((size_t) j * (size_t) kt) * HMX_FP16_TILE_SIZE;
      __fp16 *c_tile = c + ((size_t) i * (size_t) nt + (size_t) j) * HMX_FP16_TILE_N_ELMS;
      for (int k = 0; k < kt; k += 32) {
        const size_t n_tiles = (size_t) (((kt - k) < 32) ? (kt - k) : 32);
        roofline_hmx_load_tiles_hf_n_variant(a_tiles + (size_t) k * HMX_FP16_TILE_N_ELMS,
                                             b_tiles + (size_t) k * HMX_FP16_TILE_SIZE, n_tiles, variant);
      }
      hmx_consume_accumulator_fp16(c_tile);
    }
  }
  return 0;
}

#if defined(__HVX_ARCH__) && __HVX_ARCH__ >= 79
static inline void roofline_hmx_load_tiles_f8_f8_variant(const uint8_t *row_tiles, const uint8_t *col_tiles,
                                                         size_t n_tiles, size_t operand_tile_size, int variant) {
  const size_t limit = n_tiles * operand_tile_size - 1;
  // Preserve the old 0/1/2 ids, then exhaustively sweep the V81 accepted
  // activation.f8 suffixes crossed with weight.f8 suffixes. The earlier low
  // FP8 result was a benchmark artifact: it only swept the activation side and
  // it fed FP8 operands with the FP16 2048 B tile stride instead of the FP8
  // byte-cell 1024 B crouton stride.
#define ROOFLINE_HMX_LOAD_F8_F8_CASE(ID, ACT_SUFFIX, WEIGHT_SUFFIX)                                                \
  case ID:                                                                                                       \
    asm volatile("{ activation.f8 = mxmem(%0, %1)" ACT_SUFFIX "\n"                                               \
                 "weight.f8 = mxmem(%2, %3)" WEIGHT_SUFFIX " }\n" ::"r"(row_tiles), "r"(limit), "r"(col_tiles), \
                 "r"(limit)                                                                                     \
                 : "memory");                                                                                   \
    break;
  switch (variant) {
    ROOFLINE_HMX_LOAD_F8_F8_CASE(1, "", "")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(2, ":above", "")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(3, "", ":deep")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(4, "", ":single")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(5, "", ":dilate")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(6, "", ":after")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(7, "", ":drop")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(8, ":deep", ":deep")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(9, ":deep", ":single")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(10, ":deep", ":dilate")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(11, ":deep", ":after")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(12, ":deep", ":drop")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(13, ":above", ":deep")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(14, ":above", ":single")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(15, ":above", ":dilate")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(16, ":above", ":after")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(17, ":above", ":drop")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(18, ":single", "")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(19, ":single", ":deep")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(20, ":single", ":single")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(21, ":single", ":dilate")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(22, ":single", ":after")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(23, ":single", ":drop")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(24, ":dilate", "")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(25, ":dilate", ":deep")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(26, ":dilate", ":single")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(27, ":dilate", ":dilate")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(28, ":dilate", ":after")
    ROOFLINE_HMX_LOAD_F8_F8_CASE(29, ":dilate", ":drop")
    default:
      asm volatile(
        "{ activation.f8 = mxmem(%0, %1):deep\n"
        "weight.f8 = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
#undef ROOFLINE_HMX_LOAD_F8_F8_CASE
}

static int roofline_hmx_f8_f8_gemm_core(__fp16 *restrict __vtcm c, const uint8_t *restrict __vtcm a,
                                        const uint8_t *restrict __vtcm b, __fp16 *restrict __vtcm scales,
                                        int M, int K, int N, size_t operand_tile_size, int variant) {
  if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) {
    return -1;
  }

  const int mt = M / 32;
  const int nt = N / 32;
  const int kt = K / 32;

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(scales);
  asm volatile("mxclracc" ::: "memory");

  for (int i = 0; i < mt; ++i) {
    for (int j = 0; j < nt; ++j) {
      const uint8_t *a_tiles = a + ((size_t) i * (size_t) kt) * operand_tile_size;
      const uint8_t *b_tiles = b + ((size_t) j * (size_t) kt) * operand_tile_size;
      __fp16       *c_tile  = c + ((size_t) i * (size_t) nt + (size_t) j) * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < kt; k += 32) {
        const int    chunk_tiles_i = (kt - k) < 32 ? (kt - k) : 32;
        const size_t n_tiles       = (size_t) chunk_tiles_i;
        roofline_hmx_load_tiles_f8_f8_variant(a_tiles + (size_t) k * operand_tile_size,
                                              b_tiles + (size_t) k * operand_tile_size, n_tiles, operand_tile_size,
                                              variant);
      }
      hmx_consume_accumulator_fp16(c_tile);
    }
  }
  return 0;
}
#endif

static int roofline_hmx_native_byte_gemm_core(uint16_t *restrict __vtcm c, const uint8_t *restrict __vtcm a,
                                              const uint8_t *restrict __vtcm b, __fp16 *restrict __vtcm scales,
                                              int M, int K, int N, bool int4_weight_n, int variant) {
  if (M % 32 != 0 || K % 32 != 0 || N % 32 != 0) {
    return -1;
  }

  const int mt = M / 32;
  const int nt = N / 32;
  const int kt = K / 32;

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(scales);
  // Match the FP16 HMX roofline structure: clear once before the tile stream and let each
  // accumulator consume advance/reset the hardware state for the next output tile.
  asm volatile("mxclracc" ::: "memory");

  for (int i = 0; i < mt; ++i) {
    for (int j = 0; j < nt; ++j) {
      const uint8_t *a_tiles = a + ((size_t) i * (size_t) kt) * HMX_FP16_TILE_SIZE;
      const uint8_t *b_tiles = b + ((size_t) j * (size_t) kt) * HMX_FP16_TILE_SIZE;
      uint16_t      *c_tile  = c + ((size_t) i * (size_t) nt + (size_t) j) * HMX_FP16_TILE_N_ELMS;

      for (int k = 0; k < kt; k += 32) {
        const int    chunk_tiles_i = (kt - k) < 32 ? (kt - k) : 32;
        const size_t n_tiles       = (size_t) chunk_tiles_i;
        if (int4_weight_n) {
          roofline_hmx_load_tiles_ub_n_variant(a_tiles + (size_t) k * HMX_FP16_TILE_SIZE,
                                               b_tiles + (size_t) k * HMX_FP16_TILE_SIZE, n_tiles, variant);
        } else {
          roofline_hmx_load_tiles_ub_b_variant(a_tiles + (size_t) k * HMX_FP16_TILE_SIZE,
                                               (const int8_t *) (b_tiles + (size_t) k * HMX_FP16_TILE_SIZE),
                                               n_tiles, variant);
        }
      }
      roofline_hmx_store_accumulator_uh_after_2x1(c_tile);
    }
  }
  return 0;
}

static int roofline_hmx_tile_store_changed(const uint16_t *c) {
  for (int i = 0; i < HMX_FP16_TILE_N_ELMS; ++i) {
    if (c[i] != 0xa5a5) {
      return 1;
    }
  }
  return 0;
}

static int roofline_hmx_tile_all_fp16_bits(const uint16_t *tile, uint16_t expected) {
  if (!tile) return 0;
  for (int i = 0; i < HMX_FP16_TILE_N_ELMS; ++i) {
    if (tile[i] != expected) return 0;
  }
  return 1;
}

static void roofline_hmx_init_identity_bias(void *bias) {
  // Each first-vector lane is the lower 32 bits of one bias entry:
  // scale in [15:0], output bias in [31:16]. The second vector holds
  // extra mantissa, shape, and input-bias fields.
  HVX_Vector *vectors = (HVX_Vector *) bias;
  vectors[0] = Q6_V_vsplat_R(0x00003c00);  // FP16 scale=1, output bias=0
  vectors[1] = Q6_V_vzero();
}

static int roofline_run_hmx_byte_native_mix_one(RooflineBenchResult *out, int count, int max_results, int warmup,
                                                int iters, bool int4_weight_n) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  uint8_t  *a      = vtcm;
  uint8_t  *b      = vtcm + 2 * 0x100000;
  uint16_t *c      = (uint16_t *) (vtcm + 4 * 0x100000);
  __fp16   *scales = (__fp16 *) (vtcm + 6 * 0x100000);
  int       sizes[] = {32, 64, 128, 256, 512, 1024};
  int       variants_b[] = {
    // Historical hand-picked candidates retained for comparability with older artifacts.
    0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12,
    // SDK 6.6/V81 activation.ub x weight.b full suffix matrix.
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83
  };
  int       variants_n[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int      *variants = int4_weight_n ? variants_n : variants_b;
  int       n_variants = int4_weight_n ? (int) (sizeof(variants_n) / sizeof(variants_n[0]))
                                       : (int) (sizeof(variants_b) / sizeof(variants_b[0]));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  for (int v = 0; v < n_variants && count < max_results; ++v) {
    const int variant = variants[v];
    for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
      const int    n = sizes[i];
      const size_t tiles_per_matrix = (size_t) (n / 32) * (size_t) (n / 32);
      const size_t matrix_tile_bytes = tiles_per_matrix * HMX_FP16_TILE_SIZE;
      roofline_fill_hmx_tile_bytes(a, matrix_tile_bytes, 1);
      roofline_fill_hmx_tile_bytes(b, matrix_tile_bytes, int4_weight_n ? 0x11 : 1);
      roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);

      for (int w = 0; w < warmup; ++w) {
        roofline_hmx_native_byte_gemm_core(c, a, b, scales, n, n, n, int4_weight_n, variant);
      }
      roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);
      const int64_t t0 = trace_now_us();
      for (int t = 0; t < iters; ++t) {
        roofline_hmx_native_byte_gemm_core(c, a, b, scales, n, n, n, int4_weight_n, variant);
      }
      const int64_t t1 = trace_now_us();
      const int correctness = roofline_hmx_tile_store_changed(c);
      // HMX byte/nibble inputs pack twice as much reduction work as the FP16 tile stream for
      // the same 2048-byte tile feed. Count native hardware MACs here; README keeps the older
      // FP16-equivalent logical view separate to avoid mixing measurement denominators.
      const int64_t hardware_ops = (int64_t) iters * 4LL * n * n * n;
      roofline_set_mix_result(out, count++,
                              int4_weight_n ? ROOFLINE_BENCH_KIND_HMX_INT8_INT4_WEIGHT_N_GEMM
                                            : ROOFLINE_BENCH_KIND_HMX_INT8_RAW_UB_B_GEMM,
                              variant, n, iters, t1 - t0, hardware_ops,
                              ROOFLINE_BENCH_ENGINE_HMX, ROOFLINE_BENCH_DTYPE_INT8,
                              int4_weight_n ? ROOFLINE_BENCH_DTYPE_INT4_LINEAR : ROOFLINE_BENCH_DTYPE_INT8,
                              ROOFLINE_BENCH_DTYPE_INT32,
                              int4_weight_n ? ROOFLINE_BENCH_PATH_HMX_RAW_UB_N_DEEP_GEMM
                                            : ROOFLINE_BENCH_PATH_HMX_RAW_UB_B_DEEP_GEMM,
                              correctness);
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return count;
}

static int roofline_run_hmx_int8_signed_zp_corrected_mix_one(RooflineBenchResult *out, int count, int max_results,
                                                             int warmup, int iters) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  uint8_t  *a      = vtcm;
  uint8_t  *b      = vtcm + 2 * 0x100000;
  uint16_t *c      = (uint16_t *) (vtcm + 4 * 0x100000);
  __fp16   *scales = (__fp16 *) (vtcm + 6 * 0x100000);
  int       sizes[] = {32, 64, 128, 256, 512, 1024};
  const int variant = 8;  // activation.ub:above + weight.b, best current high-throughput INT8 tile stream.
  const int formula_ok = roofline_check_signed_a8_colsum_formula();

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
    const int    n = sizes[i];
    const size_t tiles_per_matrix = (size_t) (n / 32) * (size_t) (n / 32);
    const size_t matrix_tile_bytes = tiles_per_matrix * HMX_FP16_TILE_SIZE;
    roofline_fill_signed_activation_as_ub(a, matrix_tile_bytes, 3);
    roofline_fill_signed_weight_b(b, matrix_tile_bytes, 19);
    roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);

    int core_ret = 0;
    for (int w = 0; w < warmup; ++w) {
      core_ret |= roofline_hmx_native_byte_gemm_core(c, a, b, scales, n, n, n, false, variant);
    }
    roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);
    const int64_t t0 = trace_now_us();
    for (int t = 0; t < iters; ++t) {
      core_ret |= roofline_hmx_native_byte_gemm_core(c, a, b, scales, n, n, n, false, variant);
    }
    const int64_t t1 = trace_now_us();
    const int correctness = (core_ret == 0) && formula_ok && roofline_hmx_tile_store_changed(c);
    const int64_t hardware_ops = (int64_t) iters * 4LL * n * n * n;
    roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_ZP_CORRECTED_GEMM, variant, n,
                            iters, t1 - t0, hardware_ops, ROOFLINE_BENCH_ENGINE_HMX,
                            ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT8,
                            ROOFLINE_BENCH_DTYPE_INT32,
                            ROOFLINE_BENCH_PATH_HMX_SIGNED_A8_VIA_UB_COLSUM, correctness);
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return count;
}

static int roofline_run_hmx_hf_hf_mix_one(RooflineBenchResult *out, int count, int max_results, int warmup, int iters) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  __fp16 *a      = (__fp16 *) vtcm;
  __fp16 *b      = (__fp16 *) (vtcm + 2 * 0x100000);
  __fp16 *c      = (__fp16 *) (vtcm + 4 * 0x100000);
  __fp16 *scales = (__fp16 *) (vtcm + 6 * 0x100000);
  int      sizes[] = {32, 64, 128, 256, 512, 1024};
  int      variants[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  for (int v = 0; v < (int) (sizeof(variants) / sizeof(variants[0])) && count < max_results; ++v) {
    const int variant = variants[v];
    for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
      const int    n = sizes[i];
      const size_t tiles_per_matrix = (size_t) (n / 32) * (size_t) (n / 32);
      const size_t matrix_tile_bytes = tiles_per_matrix * HMX_FP16_TILE_SIZE;
      roofline_fill_fp16(a, (int) (matrix_tile_bytes / sizeof(__fp16)), 0.03125f);
      roofline_fill_fp16(b, (int) (matrix_tile_bytes / sizeof(__fp16)), 0.03125f);
      roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);

      int core_ret = 0;
      for (int w = 0; w < warmup; ++w) {
        core_ret |= roofline_hmx_hf_hf_gemm_core(c, a, b, scales, n, n, n, variant);
      }
      roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);
      const int64_t t0 = trace_now_us();
      for (int t = 0; t < iters; ++t) {
        core_ret |= roofline_hmx_hf_hf_gemm_core(c, a, b, scales, n, n, n, variant);
      }
      const int64_t t1 = trace_now_us();
      const int correctness = (core_ret == 0) && roofline_hmx_tile_store_changed((const uint16_t *) c);
      const int64_t logical_ops = (int64_t) iters * 2LL * n * n * n;
      roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HMX_FP16_GEMM, variant, n, iters,
                              t1 - t0, logical_ops, ROOFLINE_BENCH_ENGINE_HMX,
                              ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16,
                              ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_PATH_HMX_FP16_TILE, correctness);
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return count;
}

static int roofline_run_hmx_hf_b_mix_one(RooflineBenchResult *out, int count, int max_results, int warmup, int iters) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  __fp16 *a      = (__fp16 *) vtcm;
  uint8_t *b     = vtcm + 2 * 0x100000;
  __fp16 *c      = (__fp16 *) (vtcm + 4 * 0x100000);
  __fp16 *scales = (__fp16 *) (vtcm + 6 * 0x100000);
  int      sizes[] = {32, 64, 128, 256, 512, 1024};
  int      variants[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  for (int v = 0; v < (int) (sizeof(variants) / sizeof(variants[0])) && count < max_results; ++v) {
    const int variant = variants[v];
    for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
      const int    n = sizes[i];
      const size_t tiles_per_matrix = (size_t) (n / 32) * (size_t) (n / 32);
      const size_t matrix_tile_bytes = tiles_per_matrix * HMX_FP16_TILE_SIZE;
      roofline_fill_fp16(a, (int) (matrix_tile_bytes / sizeof(__fp16)), 0.03125f);
      roofline_fill_signed_weight_b(b, matrix_tile_bytes, 23);
      roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);

      int core_ret = 0;
      for (int w = 0; w < warmup; ++w) {
        core_ret |= roofline_hmx_hf_b_gemm_core(c, a, b, scales, n, n, n, variant);
      }
      roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);
      const int64_t t0 = trace_now_us();
      for (int t = 0; t < iters; ++t) {
        core_ret |= roofline_hmx_hf_b_gemm_core(c, a, b, scales, n, n, n, variant);
      }
      const int64_t t1 = trace_now_us();
      const int correctness = (core_ret == 0) && roofline_hmx_tile_store_changed((const uint16_t *) c);
      // Count a logical FP16-activation by INT8-weight GEMM MAC as one multiply plus one add.
      // Unlike the pure byte-byte row, this mixed operand pair does not expose enough public
      // shape metadata to claim the doubled byte-lane denominator.
      const int64_t logical_ops = (int64_t) iters * 2LL * n * n * n;
      roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HMX_FP16_INT8_WEIGHT_B_GEMM, variant, n,
                              iters, t1 - t0, logical_ops, ROOFLINE_BENCH_ENGINE_HMX,
                              ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_INT8,
                              ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_PATH_HMX_HF_B_GEMM, correctness);
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return count;
}

static int roofline_run_hmx_hf_n_mix_one(RooflineBenchResult *out, int count, int max_results, int warmup, int iters) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  __fp16 *a = (__fp16 *) vtcm;
  uint8_t *b = vtcm + 2 * 0x100000;
  __fp16 *c = (__fp16 *) (vtcm + 4 * 0x100000);
  __fp16 *scales = (__fp16 *) (vtcm + 6 * 0x100000);
  int sizes[] = {32, 64, 128, 256, 512, 1024};
  int variants[] = {0, 1, 2};

  hmx_manager_enable_execution();
  hmx_unit_acquire();
  for (int v = 0; v < (int) (sizeof(variants) / sizeof(variants[0])) && count < max_results; ++v) {
    const int variant = variants[v];
    for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
      const int n = sizes[i];
      const size_t tiles_per_matrix = (size_t) (n / 32) * (size_t) (n / 32);
      const size_t matrix_tile_bytes = tiles_per_matrix * HMX_FP16_TILE_SIZE;
      roofline_fill_fp16(a, (int) (matrix_tile_bytes / sizeof(__fp16)), 0.03125f);
      roofline_fill_hmx_tile_bytes(b, matrix_tile_bytes, 0x11);
      roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);

      int core_ret = 0;
      for (int w = 0; w < warmup; ++w)
        core_ret |= roofline_hmx_hf_n_gemm_core(c, a, b, scales, n, n, n, variant);
      roofline_fill_hmx_tile_bytes((uint8_t *) c, matrix_tile_bytes, 0xa5);
      const int64_t t0 = trace_now_us();
      for (int t = 0; t < iters; ++t)
        core_ret |= roofline_hmx_hf_n_gemm_core(c, a, b, scales, n, n, n, variant);
      const int64_t t1 = trace_now_us();
      const int correctness = (core_ret == 0) && roofline_hmx_tile_store_changed((const uint16_t *) c);
      // Public V81 material does not expose a doubled physical lane count for
      // hf*n, so report the logical MxKxN MAC denominator (two operations/MAC).
      const int64_t logical_ops = (int64_t) iters * 2LL * n * n * n;
      roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HMX_FP16_INT4_WEIGHT_N_GEMM, variant, n, iters,
                              t1 - t0, logical_ops, ROOFLINE_BENCH_ENGINE_HMX,
                              ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_INT4_LINEAR,
                              ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_PATH_HMX_HF_N_GEMM, correctness);
    }
  }
  hmx_unit_release();
  hmx_manager_disable_execution();
  return count;
}

static int roofline_run_hmx_f8_f8_mix_one(RooflineBenchResult *out, int count, int max_results, int warmup, int iters) {
  if (count >= max_results) return count;
#if defined(__HVX_ARCH__) && __HVX_ARCH__ >= 79
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  uint8_t *a      = vtcm;
  uint8_t *b      = vtcm + 2 * 0x100000;
  __fp16  *c      = (__fp16 *) (vtcm + 4 * 0x100000);
  __fp16  *scales = (__fp16 *) (vtcm + 6 * 0x100000);
  int      sizes[] = {32, 64, 128, 256, 512, 1024};
  int      variants[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  // Earlier FP8 anomaly runs reused HMX_FP16_TILE_SIZE (2048 B) for A/B.
  // FP8 operands are byte cells, so this focused retest feeds 1024 B per
  // logical 32x32 FP8 crouton while keeping the FP16 output tile at 2048 B.
  const size_t fp8_operand_tile_size = HMX_FP16_TILE_SIZE / 2;

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  for (int v = 0; v < (int) (sizeof(variants) / sizeof(variants[0])) && count < max_results; ++v) {
    const int variant = variants[v];
    for (int i = 0; i < (int) (sizeof(sizes) / sizeof(sizes[0])) && count < max_results; ++i) {
      const int    n = sizes[i];
      const size_t tiles_per_matrix = (size_t) (n / 32) * (size_t) (n / 32);
      const size_t input_matrix_tile_bytes = tiles_per_matrix * fp8_operand_tile_size;
      const size_t output_matrix_tile_bytes = tiles_per_matrix * HMX_FP16_TILE_SIZE;
      // 0x38 is a compact non-zero FP8 payload commonly used as an E4M3 1.0 test byte.
      // The benchmark only uses a store-changed sanity gate; production FP8 correctness
      // must still use the exact SDK/QNN FP8 encoding selected by the model converter.
      roofline_fill_hmx_tile_bytes(a, input_matrix_tile_bytes, 0x38);
      roofline_fill_hmx_tile_bytes(b, input_matrix_tile_bytes, 0x38);
      roofline_fill_hmx_tile_bytes((uint8_t *) c, output_matrix_tile_bytes, 0xa5);

      int core_ret = 0;
      for (int w = 0; w < warmup; ++w) {
        core_ret |= roofline_hmx_f8_f8_gemm_core(c, a, b, scales, n, n, n, fp8_operand_tile_size, variant);
      }
      roofline_fill_hmx_tile_bytes((uint8_t *) c, output_matrix_tile_bytes, 0xa5);
      const int64_t t0 = trace_now_us();
      for (int t = 0; t < iters; ++t) {
        core_ret |= roofline_hmx_f8_f8_gemm_core(c, a, b, scales, n, n, n, fp8_operand_tile_size, variant);
      }
      const int64_t t1 = trace_now_us();
      const int correctness = (core_ret == 0) && roofline_hmx_tile_store_changed((const uint16_t *) c);
      // FP8 uses the same byte-wide HMX issue width convention as the INT8 peak row.
      const int64_t hardware_ops = (int64_t) iters * 4LL * n * n * n;
      roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HMX_FP8_GEMM, variant, n, iters,
                              t1 - t0, hardware_ops, ROOFLINE_BENCH_ENGINE_HMX,
                              ROOFLINE_BENCH_DTYPE_FP8, ROOFLINE_BENCH_DTYPE_FP8,
                              ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_PATH_HMX_F8_F8_GEMM, correctness);
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return count;
#else
  roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_FP8, ROOFLINE_BENCH_DTYPE_FP8,
                      ROOFLINE_BENCH_ENGINE_HMX, 120);
  return count;
#endif
}

static int roofline_run_hmx_int8_shape_sweep(RooflineBenchResult *out, int max_results, int warmup, int iters) {
  if (!out || max_results <= 0) return -1;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  struct ShapeCase {
    int M;
    int K;
    int N;
    int variant;
  };
  // Keep variant 8 fixed to isolate shape effects.  Variant 8 is the best row from
  // the corrected INT8 roofline sweep: activation.ub:above + weight.b.
  const ShapeCase shapes[] = {
    {1024, 1024, 1024, 8},  // current square anchor: 32x32x32 HMX tiles.
    {512, 2048, 1024, 8},   // larger K, same N as anchor.
    {1024, 2048, 512, 8},   // larger K, same M as anchor.
    {1536, 1024, 512, 8},   // larger M with moderate output.
    {512, 1024, 1536, 8},   // larger N with moderate output.
    {1536, 512, 1536, 8},   // larger M and N, smaller K to fit VTCM.
    {256, 1024, 2048, 8},   // wide-N stress case.
    {2048, 1024, 256, 8},   // tall-M stress case.
  };
  const int    n_shapes = (int) (sizeof(shapes) / sizeof(shapes[0]));
  const size_t scales_bytes = 256;
  const size_t total_vtcm_bytes = vtcm_manager_get_total_size();

  int count = 0;
  hmx_manager_enable_execution();
  hmx_unit_acquire();

  const int formula_ok = roofline_check_signed_a8_colsum_formula();
  for (int s = 0; s < n_shapes && count < max_results; ++s) {
    const int M = shapes[s].M;
    const int K = shapes[s].K;
    const int N = shapes[s].N;
    const int mt = M / HMX_FP16_TILE_N_ROWS;
    const int kt = K / HMX_FP16_TILE_N_COLS;
    const int nt = N / HMX_FP16_TILE_N_COLS;
    const size_t a_bytes = (size_t) mt * (size_t) kt * HMX_FP16_TILE_SIZE;
    const size_t b_bytes = (size_t) nt * (size_t) kt * HMX_FP16_TILE_SIZE;
    const size_t c_bytes = (size_t) mt * (size_t) nt * HMX_FP16_TILE_SIZE;
    const size_t allocated_bytes = a_bytes + b_bytes + c_bytes + scales_bytes;

    for (int signed_case = 0; signed_case < 2 && count < max_results; ++signed_case) {
      const bool signed_zp = signed_case != 0;
      const int index = count++;
      const int kind = signed_zp ? ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_ZP_CORRECTED_GEMM
                                 : ROOFLINE_BENCH_KIND_HMX_INT8_RAW_UB_B_GEMM;
      const int path = signed_zp ? ROOFLINE_BENCH_PATH_HMX_SIGNED_A8_VIA_UB_COLSUM
                                 : ROOFLINE_BENCH_PATH_HMX_RAW_UB_B_DEEP_GEMM;

      if (total_vtcm_bytes > 0 && allocated_bytes > total_vtcm_bytes) {
        roofline_set_mix_result(out, index, kind, shapes[s].variant, M, iters, 0, 0, ROOFLINE_BENCH_ENGINE_HMX,
                                ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT8,
                                ROOFLINE_BENCH_DTYPE_INT32, path, 0);
        out[index].mode = ROOFLINE_BENCH_MODE_HMX_INT8_SHAPE_SWEEP;
        roofline_set_int8_shape_metadata(out, index, M, K, N, a_bytes, b_bytes, c_bytes, scales_bytes,
                                         total_vtcm_bytes);
        continue;
      }

      uint8_t  *a      = vtcm;
      uint8_t  *b      = a + a_bytes;
      uint16_t *c      = (uint16_t *) (b + b_bytes);
      __fp16   *scales = (__fp16 *) ((uint8_t *) c + c_bytes);

      if (signed_zp) {
        roofline_fill_signed_activation_as_ub(a, a_bytes, 3);
        roofline_fill_signed_weight_b(b, b_bytes, 19);
      } else {
        roofline_fill_hmx_tile_bytes(a, a_bytes, 1);
        roofline_fill_hmx_tile_bytes(b, b_bytes, 1);
      }
      roofline_fill_hmx_tile_bytes((uint8_t *) c, c_bytes, 0xa5);

      int core_ret = 0;
      for (int w = 0; w < warmup; ++w) {
        core_ret |= roofline_hmx_native_byte_gemm_core(c, a, b, scales, M, K, N, false, shapes[s].variant);
      }
      roofline_fill_hmx_tile_bytes((uint8_t *) c, c_bytes, 0xa5);
      const int64_t t0 = trace_now_us();
      for (int t = 0; t < iters; ++t) {
        core_ret |= roofline_hmx_native_byte_gemm_core(c, a, b, scales, M, K, N, false, shapes[s].variant);
      }
      const int64_t t1 = trace_now_us();
      const int correctness = (core_ret == 0) && (!signed_zp || formula_ok) && roofline_hmx_tile_store_changed(c);
      const int64_t hardware_ops = (int64_t) iters * 4LL * (int64_t) M * (int64_t) N * (int64_t) K;

      roofline_set_mix_result(out, index, kind, shapes[s].variant, M, iters, t1 - t0, hardware_ops,
                              ROOFLINE_BENCH_ENGINE_HMX, ROOFLINE_BENCH_DTYPE_INT8,
                              ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT32, path, correctness);
      out[index].mode = ROOFLINE_BENCH_MODE_HMX_INT8_SHAPE_SWEEP;
      roofline_set_int8_shape_metadata(out, index, M, K, N, a_bytes, b_bytes, c_bytes, scales_bytes,
                                       total_vtcm_bytes);
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return count;
}

static void roofline_set_signed_zero_result(RooflineBenchResult *out, int index, int kind, int variant, int size,
                                            int iters, int64_t elapsed_us, int64_t work_items, bool is_tops,
                                            int engine, int lhs_dtype, int rhs_dtype, int acc_dtype, int path,
                                            int correctness, int M, int K, int N, int64_t bytes) {
  roofline_set_result(out, index, ROOFLINE_BENCH_MODE_SIGNED_INT8_ZERO_OVERHEAD, kind, variant, size, iters,
                      elapsed_us, work_items, is_tops);
  out[index].engine      = engine;
  out[index].lhs_dtype   = lhs_dtype;
  out[index].rhs_dtype   = rhs_dtype;
  out[index].acc_dtype   = acc_dtype;
  out[index].path        = path;
  out[index].correctness = correctness;
  out[index].m           = M;
  out[index].k           = K;
  out[index].n           = N;
  out[index].a_bytes     = bytes;
}

static int roofline_run_signed_int8_zero_overhead(RooflineBenchResult *out, int max_results, int warmup, int iters) {
  if (!out || max_results <= 0) return -1;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  constexpr int M = 1024;
  constexpr int K = 1024;
  constexpr int N = 1024;
  const size_t tiles_per_matrix = (size_t) (M / 32) * (size_t) (K / 32);
  const size_t matrix_tile_bytes = tiles_per_matrix * HMX_FP16_TILE_SIZE;
  const size_t c_tile_bytes = (size_t) (M / 32) * (size_t) (N / 32) * HMX_FP16_TILE_SIZE;
  const int    variant = 8;  // activation.ub:above + weight.b, current best high-throughput byte HMX stream.

  uint8_t  *a_src  = vtcm;
  uint8_t  *a_dst  = a_src + matrix_tile_bytes;
  uint8_t  *b      = a_dst + matrix_tile_bytes;
  uint16_t *c      = (uint16_t *) (b + matrix_tile_bytes);
  __fp16   *scales = (__fp16 *) ((uint8_t *) c + c_tile_bytes);

  int count = 0;
  hmx_manager_enable_execution();
  hmx_unit_acquire();

  roofline_fill_hmx_tile_bytes(a_src, matrix_tile_bytes, 1);
  roofline_fill_hmx_tile_bytes(b, matrix_tile_bytes, 1);
  roofline_fill_hmx_tile_bytes((uint8_t *) c, c_tile_bytes, 0xa5);
  int core_ret = 0;
  for (int w = 0; w < warmup; ++w) {
    core_ret |= roofline_hmx_native_byte_gemm_core(c, a_src, b, scales, M, K, N, false, variant);
  }
  roofline_fill_hmx_tile_bytes((uint8_t *) c, c_tile_bytes, 0xa5);
  int64_t t0 = trace_now_us();
  for (int t = 0; t < iters; ++t) {
    core_ret |= roofline_hmx_native_byte_gemm_core(c, a_src, b, scales, M, K, N, false, variant);
  }
  int64_t t1 = trace_now_us();
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_HMX_INT8_RAW_UB_B_GEMM, variant, M, iters,
                                  t1 - t0, (int64_t) iters * 4LL * M * N * K, true, ROOFLINE_BENCH_ENGINE_HMX,
                                  ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT8,
                                  ROOFLINE_BENCH_DTYPE_INT32, ROOFLINE_BENCH_PATH_HMX_RAW_UB_B_DEEP_GEMM,
                                  core_ret == 0 && roofline_hmx_tile_store_changed(c), M, K, N,
                                  (int64_t) matrix_tile_bytes);

  const int formula_ok = roofline_check_signed_a8_colsum_formula();
  roofline_fill_signed_activation_as_ub(a_src, matrix_tile_bytes, 3);
  roofline_fill_signed_weight_b(b, matrix_tile_bytes, 19);
  roofline_fill_hmx_tile_bytes((uint8_t *) c, c_tile_bytes, 0xa5);
  core_ret = 0;
  for (int w = 0; w < warmup; ++w) {
    core_ret |= roofline_hmx_native_byte_gemm_core(c, a_src, b, scales, M, K, N, false, variant);
  }
  roofline_fill_hmx_tile_bytes((uint8_t *) c, c_tile_bytes, 0xa5);
  t0 = trace_now_us();
  for (int t = 0; t < iters; ++t) {
    core_ret |= roofline_hmx_native_byte_gemm_core(c, a_src, b, scales, M, K, N, false, variant);
  }
  t1 = trace_now_us();
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_ZP_CORRECTED_GEMM, variant, M,
                                  iters, t1 - t0, (int64_t) iters * 4LL * M * N * K, true,
                                  ROOFLINE_BENCH_ENGINE_HMX, ROOFLINE_BENCH_DTYPE_INT8,
                                  ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT32,
                                  ROOFLINE_BENCH_PATH_HMX_SIGNED_A8_VIA_UB_COLSUM,
                                  core_ret == 0 && formula_ok && roofline_hmx_tile_store_changed(c), M, K, N,
                                  (int64_t) matrix_tile_bytes);

  hmx_unit_release();
  hmx_manager_disable_execution();

  roofline_fill_signed_weight_b(a_src, matrix_tile_bytes, 7);
  roofline_fill_hmx_tile_bytes(a_dst, matrix_tile_bytes, 0);
  roofline_hvx_stream_copy_unroll4(a_src, a_dst, matrix_tile_bytes, warmup);
  t0 = trace_now_us();
  roofline_hvx_stream_copy_unroll4(a_src, a_dst, matrix_tile_bytes, iters);
  t1 = trace_now_us();
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_COPY, 0,
                                  (int) matrix_tile_bytes, iters, t1 - t0,
                                  (int64_t) matrix_tile_bytes * (int64_t) iters, false,
                                  ROOFLINE_BENCH_ENGINE_HVX, ROOFLINE_BENCH_DTYPE_INT8,
                                  ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_UNKNOWN,
                                  ROOFLINE_BENCH_PATH_HVX_COPY, a_dst[17] == a_src[17], M, K, N,
                                  (int64_t) matrix_tile_bytes);

  roofline_hvx_stream_copy_xor_0x80_unroll4(a_src, a_dst, matrix_tile_bytes, warmup);
  t0 = trace_now_us();
  roofline_hvx_stream_copy_xor_0x80_unroll4(a_src, a_dst, matrix_tile_bytes, iters);
  t1 = trace_now_us();
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_COPY_XOR, 0,
                                  (int) matrix_tile_bytes, iters, t1 - t0,
                                  (int64_t) matrix_tile_bytes * (int64_t) iters, false,
                                  ROOFLINE_BENCH_ENGINE_HVX, ROOFLINE_BENCH_DTYPE_INT8,
                                  ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_UNKNOWN,
                                  ROOFLINE_BENCH_PATH_HVX_COPY_XOR_0X80,
                                  a_dst[17] == (uint8_t) (a_src[17] ^ 0x80u), M, K, N,
                                  (int64_t) matrix_tile_bytes);

  roofline_hvx_stream_xor_0x80_inplace_unroll4(a_dst, matrix_tile_bytes, warmup);
  t0 = trace_now_us();
  roofline_hvx_stream_xor_0x80_inplace_unroll4(a_dst, matrix_tile_bytes, iters);
  t1 = trace_now_us();
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_XOR_INPLACE, 0,
                                  (int) matrix_tile_bytes, iters, t1 - t0,
                                  (int64_t) matrix_tile_bytes * (int64_t) iters, false,
                                  ROOFLINE_BENCH_ENGINE_HVX, ROOFLINE_BENCH_DTYPE_INT8,
                                  ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_UNKNOWN,
                                  ROOFLINE_BENCH_PATH_HVX_XOR_0X80_INPLACE, 1, M, K, N,
                                  (int64_t) matrix_tile_bytes);

  const size_t requant_elems = matrix_tile_bytes;
  int16_t     *rq_src        = (int16_t *) vtcm;
  uint8_t     *rq_dst        = vtcm + requant_elems * sizeof(int16_t);
  for (size_t i = 0; i < requant_elems; ++i) {
    rq_src[i] = 0;
  }
  roofline_hvx_requant_store_zp_mock(rq_src, rq_dst, requant_elems, 0, warmup);
  t0 = trace_now_us();
  roofline_hvx_requant_store_zp_mock(rq_src, rq_dst, requant_elems, 0, iters);
  t1 = trace_now_us();
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_SIGNED_A8_REQUANT_STORE_ZP0, 0,
                                  (int) requant_elems, iters, t1 - t0,
                                  (int64_t) requant_elems * 3LL * (int64_t) iters, false,
                                  ROOFLINE_BENCH_ENGINE_HVX, ROOFLINE_BENCH_DTYPE_INT16,
                                  ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_UNKNOWN,
                                  ROOFLINE_BENCH_PATH_HVX_REQUANT_STORE_ZP, rq_dst[17] == 0, M, K, N,
                                  (int64_t) requant_elems * 3LL);

  roofline_hvx_requant_store_zp_mock(rq_src, rq_dst, requant_elems, 128, warmup);
  t0 = trace_now_us();
  roofline_hvx_requant_store_zp_mock(rq_src, rq_dst, requant_elems, 128, iters);
  t1 = trace_now_us();
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_SIGNED_A8_REQUANT_STORE_ZP128, 0,
                                  (int) requant_elems, iters, t1 - t0,
                                  (int64_t) requant_elems * 3LL * (int64_t) iters, false,
                                  ROOFLINE_BENCH_ENGINE_HVX, ROOFLINE_BENCH_DTYPE_INT16,
                                  ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_UNKNOWN,
                                  ROOFLINE_BENCH_PATH_HVX_REQUANT_STORE_ZP, rq_dst[17] == 128, M, K, N,
                                  (int64_t) requant_elems * 3LL);

  int32_t *colsum = (int32_t *) a_dst;
  roofline_fill_signed_weight_b(b, matrix_tile_bytes, 19);
  int32_t checksum = 0;
  for (int w = 0; w < warmup; ++w) {
    checksum ^= roofline_weight_colsum_precompute(colsum, (const int8_t *) b, K, N);
  }
  t0 = trace_now_us();
  for (int t = 0; t < iters; ++t) {
    checksum ^= roofline_weight_colsum_precompute(colsum, (const int8_t *) b, K, N);
  }
  t1 = trace_now_us();
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_SIGNED_A8_COLSUM_PRECOMPUTE, 0, N, iters,
                                  t1 - t0, (int64_t) matrix_tile_bytes * (int64_t) iters, false,
                                  ROOFLINE_BENCH_ENGINE_SCALAR, ROOFLINE_BENCH_DTYPE_INT8,
                                  ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT32,
                                  ROOFLINE_BENCH_PATH_OFFLINE_COLSUM, checksum != 0, M, K, N,
                                  (int64_t) matrix_tile_bytes);

  constexpr int PM = 512;
  constexpr int PN = 1024;
  int32_t *acc  = (int32_t *) vtcm;
  int32_t *bias = (int32_t *) (vtcm + 2 * 0x100000);
  float   *scl  = (float *) (vtcm + 2 * 0x100000 + 0x4000);
  float   *dst  = (float *) (vtcm + 3 * 0x100000);
  roofline_fill_i32_pattern(acc, (size_t) PM * (size_t) PN, 5);
  roofline_fill_i32_pattern(bias, (size_t) PN, 23);
  roofline_fill_f32_pattern(scl, (size_t) PN, 1.0e-5f);
  roofline_hvx_bias_scale_float_store(dst, acc, bias, scl, PM, PN, warmup);
  t0 = trace_now_us();
  roofline_hvx_bias_scale_float_store(dst, acc, bias, scl, PM, PN, iters);
  t1 = trace_now_us();
  const int64_t post_bytes = (int64_t) iters * ((int64_t) PM * (int64_t) PN * (int64_t) (sizeof(int32_t) + sizeof(float)) +
                                                (int64_t) PM * (int64_t) PN * (int64_t) sizeof(int32_t));
  roofline_set_signed_zero_result(out, count++, ROOFLINE_BENCH_KIND_SIGNED_A8_HVX_BIAS_SCALE_STORE, 0, PM, iters,
                                  t1 - t0, post_bytes, false, ROOFLINE_BENCH_ENGINE_HVX,
                                  ROOFLINE_BENCH_DTYPE_INT32, ROOFLINE_BENCH_DTYPE_INT8,
                                  ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_PATH_HVX_BIAS_SCALE_FLOAT_STORE,
                                  roofline_check_hvx_bias_scale_float_store(dst, acc, bias, scl, PN), PM, K, PN,
                                  post_bytes / iters);

  return count;
}

static float roofline_decode_linear_q4_to_float(uint8_t packed, bool high) {
  int q = high ? (packed >> 4) & 0x0f : packed & 0x0f;
  q = q >= 8 ? q - 16 : q;
  return (float) q;
}

static float roofline_decode_iq4_nl_to_float(uint8_t packed, bool high) {
  static const int table[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  int idx = high ? (packed >> 4) & 0x0f : packed & 0x0f;
  return (float) table[idx];
}

static void roofline_decode_q4_buffer_to_fp16(__fp16 *dst, const uint8_t *src, int values, bool iq4_nl) {
  for (int i = 0; i < values; i += 2) {
    const uint8_t p = src[i / 2];
    dst[i] = (__fp16) (iq4_nl ? roofline_decode_iq4_nl_to_float(p, false)
                              : roofline_decode_linear_q4_to_float(p, false));
    if (i + 1 < values) {
      dst[i + 1] = (__fp16) (iq4_nl ? roofline_decode_iq4_nl_to_float(p, true)
                                    : roofline_decode_linear_q4_to_float(p, true));
    }
  }
}

static int roofline_run_format_q4_mix_one(RooflineBenchResult *out, int count, int max_results, int warmup, int iters,
                                          bool iq4_nl) {
  if (count >= max_results) return count;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  uint8_t *a_q = vtcm;
  uint8_t *b_q = vtcm + 0x80000;
  __fp16 *a = (__fp16 *) (vtcm + 0x100000);
  __fp16 *b = (__fp16 *) (vtcm + 0x200000);
  __fp16 *c = (__fp16 *) (vtcm + 0x300000);
  __fp16 *scales = (__fp16 *) (vtcm + 0x400000);
  const int n = 256;
  const int values = n * n;

  for (int i = 0; i < values / 2; ++i) {
    a_q[i] = (uint8_t) ((i * 17 + 5) & 0xff);
    b_q[i] = (uint8_t) ((i * 23 + 11) & 0xff);
  }

  hmx_manager_enable_execution();
  for (int w = 0; w < warmup; ++w) {
    roofline_decode_q4_buffer_to_fp16(a, a_q, values, iq4_nl);
    roofline_decode_q4_buffer_to_fp16(b, b_q, values, iq4_nl);
    hmx_mat_mul_fp16_core(c, a, b, scales, n, n, n);
  }
  const int64_t t0 = trace_now_us();
  for (int t = 0; t < iters; ++t) {
    roofline_decode_q4_buffer_to_fp16(a, a_q, values, iq4_nl);
    roofline_decode_q4_buffer_to_fp16(b, b_q, values, iq4_nl);
    hmx_mat_mul_fp16_core(c, a, b, scales, n, n, n);
  }
  const int64_t t1 = trace_now_us();
  roofline_set_mix_result(out, count++, iq4_nl ? ROOFLINE_BENCH_KIND_FORMAT_IQ4_NL_DECODE_FP16_GEMM
                                               : ROOFLINE_BENCH_KIND_FORMAT_Q4_0_DECODE_FP16_GEMM,
                          iq4_nl ? 2 : 1, n, iters, t1 - t0, (int64_t) iters * 2LL * n * n * n,
                          ROOFLINE_BENCH_ENGINE_FORMAT_EFFECTIVE,
                          iq4_nl ? ROOFLINE_BENCH_DTYPE_IQ4_NL : ROOFLINE_BENCH_DTYPE_Q4_0,
                          iq4_nl ? ROOFLINE_BENCH_DTYPE_IQ4_NL : ROOFLINE_BENCH_DTYPE_Q4_0,
                          ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_PATH_FORMAT_DECODE_TO_FP16, 1);
  hmx_manager_disable_execution();
  return count;
}

static int roofline_run_v81_hmx_manual_smoke(RooflineBenchResult *out, int count, int max_results, int warmup,
                                             int iters) {
  if (!out || count + 4 > max_results) return -1;
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) return -1;

  // Manual-only smoke buffers. The offsets preserve the 2 KB activation,
  // 128 B weight, and 256 B bias alignment requirements from 80-N2040-62.
  uint8_t  *bias_in  = vtcm;
  uint8_t  *bias_out = vtcm + 0x1000;
  __fp16   *act      = (__fp16 *) (vtcm + 0x2000);
  __fp16   *wgt      = (__fp16 *) (vtcm + 0x4000);
  uint8_t  *bias0    = vtcm + 0x6000;
  uint8_t  *bias1    = vtcm + 0x6100;
  uint16_t *cvt_out  = (uint16_t *) (vtcm + 0x8000);
  uint16_t *cvt_out1 = (uint16_t *) (vtcm + 0x9000);
  uint16_t *feedback_out = (uint16_t *) (vtcm + 0xa000);

  for (int i = 0; i < 256; ++i) {
    bias_in[i] = (uint8_t) ((i * 29 + 7) & 0xff);
  }
  memset(bias_out, 0, 256);
  for (int i = 0; i < HMX_FP16_TILE_N_ELMS; ++i) {
    act[i] = (__fp16) 1.0f;
    wgt[i] = (__fp16) 1.0f;
    wgt[HMX_FP16_TILE_N_ELMS + i] = (__fp16) 2.0f;
  }
  roofline_hmx_init_identity_bias(bias0);
  roofline_hmx_init_identity_bias(bias1);

  hmx_manager_enable_execution();

  for (int w = 0; w < warmup; ++w) {
    asm volatile("bias = mxmem2(%0)\n"
                 "mxmem2(%1) = bias" ::"r"(bias_in), "r"(bias_out)
                 : "memory");
  }
  int64_t t0 = trace_now_us();
  for (int t = 0; t < iters; ++t) {
    asm volatile("bias = mxmem2(%0)\n"
                 "mxmem2(%1) = bias" ::"r"(bias_in), "r"(bias_out)
                 : "memory");
  }
  int64_t t1 = trace_now_us();
  const int bias_roundtrip_ok = memcmp(bias_in, bias_out, 256) == 0;
  roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_V81_HMX_MANUAL_SMOKE, 201, 256, iters, t1 - t0, 0,
                          ROOFLINE_BENCH_ENGINE_HMX, ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16,
                          ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_PATH_HMX_FP16_TILE, bias_roundtrip_ok);

  const size_t limit = HMX_FP16_TILE_SIZE - 1;
  const int first_cvt_rs = (1 << 8) | 1;                  // extra precision + retain accumulator
  const int feedback_cvt_rs = (1 << 12) | (1 << 4) | (2 << 2);  // bias set 1, max, feedback as scale
  const void *bias1_set1 = (const void *) ((uintptr_t) bias1 | 1U);
  auto run_feedback_once = [&]() {
    memset(cvt_out, 0xa5, HMX_FP16_TILE_SIZE);
    asm volatile("mxclracc.hf" ::: "memory");
    asm volatile(
      "{ activation.hf = mxmem(%0, %1):deep\n"
      "  weight.hf = mxmem(%2, %3) }\n" ::"r"(act),
      "r"(limit), "r"(wgt), "r"(limit)
      : "memory");
    asm volatile("bias = mxmem2(%0)" ::"r"(bias0) : "memory");
    asm volatile("cvt.hf = acc(%0)" ::"r"(first_cvt_rs) : "memory");
    asm volatile("bias = mxmem2(%0)" ::"r"(bias1_set1) : "memory");
    asm volatile("cvt.hf = acc(%0)" ::"r"(feedback_cvt_rs) : "memory");
    // Use the generic FP16 store spelling. SDK 6.6's V81
    // Q6_mxmem_cvt_RR macro is incorrectly remapped to the FP8 store.
    asm volatile("mxmem(%0, %1) = cvt" ::"r"(cvt_out), "r"(0) : "memory");
  };

  for (int w = 0; w < warmup; ++w) run_feedback_once();
  t0 = trace_now_us();
  for (int t = 0; t < iters; ++t) run_feedback_once();
  t1 = trace_now_us();
  const int feedback_ok = roofline_hmx_tile_store_changed(cvt_out);
  roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_V81_HMX_MANUAL_SMOKE, 202,
                          HMX_FP16_TILE_SIZE, iters, t1 - t0, 0, ROOFLINE_BENCH_ENGINE_HMX,
                          ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16,
                          ROOFLINE_BENCH_PATH_HMX_FP16_TILE, feedback_ok);

  const size_t weight_deep_limit = 2 * HMX_FP16_TILE_SIZE - 1;
  auto run_weight_deep_once = [&]() {
    memset(cvt_out, 0xa5, HMX_FP16_TILE_SIZE);
    memset(cvt_out1, 0xa5, HMX_FP16_TILE_SIZE);
    asm volatile("mxclracc.hf" ::: "memory");
    asm volatile(
      "{ activation.hf = mxmem(%0, %1)\n"
      "  weight.hf = mxmem(%2, %3):deep }\n" ::"r"(act),
      "r"(limit), "r"(wgt), "r"(weight_deep_limit)
      : "memory");
    asm volatile("bias = mxmem2(%0)" ::"r"(bias0) : "memory");
    asm volatile("cvt.hf = acc(%0)" ::"r"(0) : "memory");
    asm volatile("mxmem(%0, %1) = cvt" ::"r"(cvt_out), "r"(0) : "memory");
    asm volatile("bias = mxmem2(%0)" ::"r"(bias0) : "memory");
    asm volatile("cvt.hf = acc(%0)" ::"r"(0) : "memory");
    asm volatile("mxmem(%0, %1) = cvt" ::"r"(cvt_out1), "r"(0) : "memory");
  };

  for (int w = 0; w < warmup; ++w) run_weight_deep_once();
  t0 = trace_now_us();
  for (int t = 0; t < iters; ++t) run_weight_deep_once();
  t1 = trace_now_us();
  const int weight_deep_ok =
    roofline_hmx_tile_all_fp16_bits(cvt_out, 0x5000) &&   // 32 x (1 * 1)
    roofline_hmx_tile_all_fp16_bits(cvt_out1, 0x5400);    // 32 x (1 * 2)
  roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_V81_HMX_MANUAL_SMOKE, 203,
                          2 * HMX_FP16_TILE_SIZE, iters, t1 - t0, 0, ROOFLINE_BENCH_ENGINE_HMX,
                          ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16,
                          ROOFLINE_BENCH_PATH_HMX_FP16_TILE, weight_deep_ok);

  auto run_feedback_numeric_once = [&]() {
    memset(feedback_out, 0xa5, HMX_FP16_TILE_SIZE);
    asm volatile("mxclracc.hf" ::: "memory");
    asm volatile(
      "{ activation.hf = mxmem(%0, %1):deep\n"
      "  weight.hf = mxmem(%2, %3) }\n" ::"r"(act),
      "r"(limit), "r"(wgt), "r"(limit)
      : "memory");
    asm volatile("bias = mxmem2(%0)" ::"r"(bias0) : "memory");
    asm volatile("cvt.hf = acc(%0)" ::"r"(first_cvt_rs) : "memory");
    asm volatile("bias = mxmem2(%0)" ::"r"(bias1_set1) : "memory");
    asm volatile("cvt.hf = acc(%0)" ::"r"(feedback_cvt_rs) : "memory");
    asm volatile("mxmem(%0, %1) = cvt" ::"r"(feedback_out), "r"(0) : "memory");
  };

  for (int w = 0; w < warmup; ++w) run_feedback_numeric_once();
  t0 = trace_now_us();
  for (int t = 0; t < iters; ++t) run_feedback_numeric_once();
  t1 = trace_now_us();
  // First convert: C'=1*32+0=32. The second convert uses max(1,C')
  // as scale over the retained accumulator: 32*32+0=1024.
  const int feedback_numeric_ok = roofline_hmx_tile_all_fp16_bits(feedback_out, 0x6400);
  roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_V81_HMX_MANUAL_SMOKE, 204,
                          HMX_FP16_TILE_SIZE, iters, t1 - t0, 0, ROOFLINE_BENCH_ENGINE_HMX,
                          ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16,
                          ROOFLINE_BENCH_PATH_HMX_FP16_TILE, feedback_numeric_ok);

  hmx_manager_disable_execution();
  return count;
}

static int roofline_run_mix_precision(RooflineBenchResult *out, int max_results, int warmup, int iters,
                                      int case_selector) {
  if (!out || max_results <= 0) return -1;
  int count = 0;

  if (case_selector == 0 || case_selector == 1)
    count = roofline_run_hmx_byte_native_mix_one(out, count, max_results, warmup, iters, false);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 2)
    count = roofline_run_hmx_int8_signed_zp_corrected_mix_one(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 3)
    count = roofline_run_hmx_byte_native_mix_one(out, count, max_results, warmup, iters, true);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 4)
    count = roofline_run_hmx_hf_hf_mix_one(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 5)
    count = roofline_run_hmx_hf_b_mix_one(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 6)
    count = roofline_run_hmx_f8_f8_mix_one(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 7)
    count = roofline_run_hvx_fp16_mix(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 8)
    count = roofline_run_hvx_fp32_mix(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 9)
    count = roofline_run_hvx_i16_mix(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 10)
    count = roofline_run_format_q4_mix_one(out, count, max_results, warmup, iters, false);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 11)
    count = roofline_run_format_q4_mix_one(out, count, max_results, warmup, iters, true);
  if (count < 0) return -1;
  if (case_selector == 0 || case_selector == 13)
    count = roofline_run_hmx_hf_n_mix_one(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  if (case_selector == 31)
    count = roofline_run_v81_hmx_manual_smoke(out, count, max_results, warmup, iters);
  if (count < 0) return -1;

  // Case 20 emits the complete V81 HVX native-instruction matrix. Cases
  // 21..30 isolate one row so long repeated sweeps can be batched safely.
  if (case_selector == 0 || case_selector == 20)
    count = roofline_run_hvx_native_peaks(out, count, max_results, warmup, iters, -1);
  else if (case_selector >= 21 && case_selector <= 30)
    count = roofline_run_hvx_native_peaks(out, count, max_results, warmup, iters, case_selector - 21);
  if (count < 0) return -1;

  if (case_selector != 0 && case_selector != 12) return count;
  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_DTYPE_FP32,
                                               ROOFLINE_BENCH_ENGINE_HMX, 100);
  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_DTYPE_INT16,
                                               ROOFLINE_BENCH_ENGINE_HMX, 101);
  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_INT4_LINEAR,
                                               ROOFLINE_BENCH_DTYPE_INT4_LINEAR, ROOFLINE_BENCH_ENGINE_HMX, 103);
  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_INT4_LINEAR,
                                               ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_ENGINE_HMX, 104);
  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_INT4_LINEAR,
                                               ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_ENGINE_HMX, 105);
  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_INT8,
                                               ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_ENGINE_HMX, 106);
  return count;
}

static int roofline_run_hvx_bandwidth_one(RooflineBenchResult *out, int max_results, int count, int mode, int unroll,
                                          uint8_t *src, uint8_t *dst, size_t size, int warmup, int iters) {
  if (count >= max_results) return count;

  if (unroll == 4) {
    roofline_hvx_stream_read_unroll4(src, size, warmup);
  } else {
    roofline_hvx_stream_read(src, size, warmup);
  }
  int64_t t0 = trace_now_us();
  if (unroll == 4) {
    roofline_hvx_stream_read_unroll4(src, size, iters);
  } else {
    roofline_hvx_stream_read(src, size, iters);
  }
  int64_t t1 = trace_now_us();
  roofline_set_result(out, count++, mode,
                      mode == ROOFLINE_BENCH_MODE_VTCM_BW ? ROOFLINE_BENCH_KIND_VTCM_READ
                                                           : ROOFLINE_BENCH_KIND_DDR_READ,
                      unroll, (int) size, iters, t1 - t0, (int64_t) size * iters, false);
  if (count >= max_results) return count;

  if (unroll == 4) {
    roofline_hvx_stream_write_unroll4(dst, size, warmup);
  } else {
    roofline_hvx_stream_write(dst, size, warmup);
  }
  t0 = trace_now_us();
  if (unroll == 4) {
    roofline_hvx_stream_write_unroll4(dst, size, iters);
  } else {
    roofline_hvx_stream_write(dst, size, iters);
  }
  t1 = trace_now_us();
  roofline_set_result(out, count++, mode,
                      mode == ROOFLINE_BENCH_MODE_VTCM_BW ? ROOFLINE_BENCH_KIND_VTCM_WRITE
                                                           : ROOFLINE_BENCH_KIND_DDR_WRITE,
                      unroll, (int) size, iters, t1 - t0, (int64_t) size * iters, false);
  if (count >= max_results) return count;

  if (unroll == 4) {
    roofline_hvx_stream_copy_unroll4(src, dst, size, warmup);
  } else {
    roofline_hvx_stream_copy(src, dst, size, warmup);
  }
  t0 = trace_now_us();
  if (unroll == 4) {
    roofline_hvx_stream_copy_unroll4(src, dst, size, iters);
  } else {
    roofline_hvx_stream_copy(src, dst, size, iters);
  }
  t1 = trace_now_us();
  roofline_set_result(out, count++, mode,
                      mode == ROOFLINE_BENCH_MODE_VTCM_BW ? ROOFLINE_BENCH_KIND_VTCM_COPY
                                                           : ROOFLINE_BENCH_KIND_DDR_COPY,
                      unroll, (int) size, iters, t1 - t0, (int64_t) size * iters * 2LL, false);
  return count;
}

static int roofline_run_hvx_bandwidth(RooflineBenchResult *out, int max_results, int mode, uint8_t *src, uint8_t *dst,
                                      int bytes, int warmup, int iters) {
  if (max_results <= 0) {
    return 0;
  }
  size_t size = (size_t) bytes;
  size = size / VLEN * VLEN;
  if (size < (size_t) VLEN) {
    return -1;
  }

  if (mode == ROOFLINE_BENCH_MODE_VTCM_BW) {
    uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
    if (!vtcm) {
      return -1;
    }
    src = vtcm;
    dst = vtcm + 0x400000;
    if (size > 0x100000) {
      size = 0x100000;
    }
  }
  if (!src || !dst) {
    return -1;
  }

  int count = 0;
  count = roofline_run_hvx_bandwidth_one(out, max_results, count, mode, 1, src, dst, size, warmup, iters);
  count = roofline_run_hvx_bandwidth_one(out, max_results, count, mode, 4, src, dst, size, warmup, iters);
  return count;
}

static int roofline_run_hmx_dma_bandwidth(RooflineBenchResult *out, int max_results, uint8_t *src, int bytes,
                                          int warmup, int iters) {
  if (max_results <= 0 || !src) {
    return 0;
  }
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();
  if (!vtcm) {
    return -1;
  }

  size_t size = (size_t) bytes;
  size = size / VLEN * VLEN;
  if (size < (size_t) VLEN) {
    return -1;
  }

  const size_t chunk_sizes[] = {64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024};
  int count = 0;
  for (int i = 0; i < (int) (sizeof(chunk_sizes) / sizeof(chunk_sizes[0])) && count < max_results; ++i) {
    const size_t chunk_size = chunk_sizes[i] < size ? chunk_sizes[i] : size;
    int ret = roofline_dma_read_ddr_to_vtcm(src, vtcm, size, chunk_size, warmup);
    if (ret) {
      return ret;
    }
    const int64_t t0 = trace_now_us();
    ret = roofline_dma_read_ddr_to_vtcm(src, vtcm, size, chunk_size, iters);
    const int64_t t1 = trace_now_us();
    if (ret) {
      return ret;
    }
    roofline_set_result(out, count++, ROOFLINE_BENCH_MODE_HMX_DMA_BW, ROOFLINE_BENCH_KIND_HMX_DMA_READ,
                        (int) (chunk_size / 1024), (int) size, iters, t1 - t0, (int64_t) size * iters, false);
  }
  return count;
}

}  // namespace

extern "C" {

#define IN_PTR(i)  std::get<0>(in_bufs[i])
#define OUT_PTR(i) std::get<0>(out_bufs[i])

int execute_op_simple(struct OpComputeRequest *req) {
  // using FatPointer = std::pair<uint8_t *, size_t>;
  using Buffer = std::tuple<uint8_t *, size_t, bool>;
  std::vector<Buffer> in_bufs, out_bufs;

  auto add_buffer = [](std::vector<Buffer> &bufs, const RpcmemBufAddr &buf_addr, size_t size, bool cached = true) {
    auto base = reinterpret_cast<uint8_t *>(mmap_manager_get_map(buf_addr.fd));
    auto ptr  = base != nullptr ? base + buf_addr.offset : nullptr;
    bufs.push_back({ ptr, size, cached });
  };

  auto validate_in_bufs = [&]() {
    for (auto [ptr, size, cached] : in_bufs) {
      if (ptr && cached) {
        qurt_mem_cache_clean((qurt_addr_t) ptr, size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
      }
    }
  };

  auto validate_out_bufs = [&]() {
    for (auto [ptr, size, cached] : out_bufs) {
      if (ptr && cached) {
        qurt_mem_cache_clean((qurt_addr_t) ptr, size, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
      }
    }
  };

  auto add_trace_profile_buffer = [&](const RpcmemBufAddr &buf_addr, int max_events) -> LlmTraceProfileHeader * {
    if (max_events <= 0 || buf_addr.fd < 0) {
      return nullptr;
    }
    const size_t profile_size = sizeof(LlmTraceProfileHeader) + (size_t) max_events * sizeof(LlmTraceProfileEvent);
    // Profiling buffers are allocated per op and Linux can reuse fd numbers.
    // Drop a stale DSP-side fd cache entry before HAP_mmap_get maps the new buffer.
    mmap_manager_put_map(buf_addr.fd);
    add_buffer(out_bufs, buf_addr, profile_size);
    auto *profile = reinterpret_cast<LlmTraceProfileHeader *>(std::get<0>(out_bufs.back()));
    if (!profile) {
      return nullptr;
    }
    profile->magic          = LLM_TRACE_PROFILE_MAGIC;
    profile->max_events     = max_events;
    profile->event_count    = 0;
    profile->event_overflow = 0;
    profile->reserved0      = 0;
    profile->reserved1      = 0;
    return profile;
  };

  int ret = 0;
  switch (req->op) {
    case HTP_OPS_RMS_NORM_F32:
      {
        auto   params = reinterpret_cast<RmsNormF32Params *>(req->payload);
        size_t size   = params->ne0 * params->ne1 * sizeof(float);

        add_buffer(out_bufs, params->dst, size);
        add_buffer(in_bufs, params->src, size);
        (void) add_trace_profile_buffer(params->profile, params->max_profile_events);

        int64_t t_total0 = trace_now_us();
        int64_t t0       = t_total0;
        validate_in_bufs();
        int64_t t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_in", params->ne1, params->ne0, 1, 0, 0,
                      0, 0, 0, size, size, t0, t1);

        t0 = t1;
        ret = hvx_rms_norm_f32((float *) OUT_PTR(0), (const float *) IN_PTR(0), params->ne0, params->ne1);
        t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "compute", params->ne1, params->ne0, 1, 0, 0, 0,
                      0, 0, size, size, t0, t1);

        t0 = t1;
        validate_out_bufs();
        t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_out", params->ne1, params->ne0, 1, 0,
                      0, 0, 0, 0, size, size, t0, t1);
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "total", params->ne1, params->ne0, 1, 0, 0, 0, 0,
                      0, size, size, t_total0, t1);
      }
      break;

    case HTP_OPS_MAT_MUL_PERMUTED_W16A32:
      {
        auto params = reinterpret_cast<MatMulParams *>(req->payload);
        int  m = params->m, k = params->k, n = params->n;

        size_t output_size     = m * n * sizeof(float);
        size_t activation_size = m * k * sizeof(float);
        size_t weight_size     = k * n * sizeof(__fp16);

        add_buffer(out_bufs, params->output, output_size);
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size);
        auto *profile = add_trace_profile_buffer(params->profile, params->max_profile_events);

        const size_t input_size = activation_size + weight_size;
        int64_t      t_total0   = trace_now_us();
        int64_t      t0         = t_total0;
        validate_in_bufs();
        int64_t t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_in", m, k, n, 0, 0, 0, 0, 0,
                      input_size, output_size, t0, t1);

        t0 = t1;
        ret = hmx_mat_mul_permuted_w16a32((float *) OUT_PTR(0), (float *) IN_PTR(0), (__fp16 *) IN_PTR(1), m, k, n,
                                          params->trace_id, params->mode_flags, req->op, profile);
        t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "compute", m, k, n, 0, 0, 0, 0, 0, input_size,
                      output_size, t0, t1);

        t0 = t1;
        validate_out_bufs();
        t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_out", m, k, n, 0, 0, 0, 0, 0,
                      input_size, output_size, t0, t1);
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "total", m, k, n, 0, 0, 0, 0, 0, input_size,
                      output_size, t_total0, t1);
      }
      break;

    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:
    case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:
    case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL:
    case HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT:
      {
        auto   weight_type      = matmul_op_to_weight_type(static_cast<HtpOpsIndex>(req->op));
        size_t super_block_size = ggml_super_block_size(weight_type);

        auto params = reinterpret_cast<MatMulParams *>(req->payload);
        int  m = params->m, k = params->k, n = params->n;

        size_t output_size     = m * n * sizeof(float);
        size_t activation_size = m * k * sizeof(float);
        size_t weight_size     = k * n / QK_K * super_block_size;

        add_buffer(out_bufs, params->output, output_size);
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size, false);
        auto *profile = add_trace_profile_buffer(params->profile, params->max_profile_events);

        const size_t input_size = activation_size + weight_size;
        int64_t      t_total0   = trace_now_us();
        int64_t      t0         = t_total0;
        validate_in_bufs();
        int64_t t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_in", m, k, n, 0, 0, 0, 0, 0,
                      input_size, output_size, t0, t1);

        t0  = t1;
        if (req->op == HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT) {
          ret = hmx_mat_mul_permuted_w8pc_a8pt((float *) OUT_PTR(0), (float *) IN_PTR(0), IN_PTR(1), m, k, n,
                                               params->trace_id, params->mode_flags, req->op, profile);
        } else {
          ret = hmx_mat_mul_permuted_qk_0_d16a32((float *) OUT_PTR(0), (float *) IN_PTR(0), IN_PTR(1), m, k, n,
                                                 weight_type, params->trace_id, params->mode_flags, req->op, profile);
        }
        t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "compute", m, k, n, 0, 0, 0, 0, 0, input_size,
                      output_size, t0, t1);

        t0 = t1;
        validate_out_bufs();
        t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_out", m, k, n, 0, 0, 0, 0, 0,
                      input_size, output_size, t0, t1);
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "total", m, k, n, 0, 0, 0, 0, 0, input_size,
                      output_size, t_total0, t1);

        // int64_t mm_time_us  = HAP_perf_qtimer_count_to_us(t2 - t1);
        // int64_t tot_time_us = HAP_perf_qtimer_count_to_us(t3 - t0);

        // FARF(ALWAYS, "mm_time: %lld us, tot_time: %lld us, type: %d, (%d, %d, %d)", mm_time_us, tot_time_us,
        //      weight_type, m, k, n);
        // FARF(ALWAYS, "achieved weight load bandwidth: %.2f GB/s", 1e-3 * weight_size / mm_time_us);
        // FARF(ALWAYS, "achieved GEMM throughput: %.2f GFLOPS", 2e-3 * m * n * k / mm_time_us);
      }
      break;

    case HTP_OPS_HMX_INT8_GATE:
      {
        auto   params      = reinterpret_cast<HmxInt8GateParams *>(req->payload);
        int    max_results = params->max_results;
        size_t output_size = max_results * sizeof(HmxInt8GateResult);

        add_buffer(out_bufs, params->output, output_size);

        int64_t t_total0 = trace_now_us();
        int64_t t0       = t_total0;
        ret              = hmx_int8_gate_run((HmxInt8GateResult *) OUT_PTR(0), max_results, params->reserved);
        int64_t t1       = trace_now_us();
        log_dsp_event(0, LLM_NPU_MODE_TRACE, req->op, "compute", max_results, 0, 0, 0, 0, 0, 0, 0, 0, output_size, t0,
                      t1);

        t0 = t1;
        validate_out_bufs();
        t1 = trace_now_us();
        log_dsp_event(0, LLM_NPU_MODE_TRACE, req->op, "validate_out", max_results, 0, 0, 0, 0, 0, 0, 0, 0, output_size,
                      t0, t1);
        log_dsp_event(0, LLM_NPU_MODE_TRACE, req->op, "total", max_results, 0, 0, 0, 0, 0, 0, 0, 0, output_size,
                      t_total0, t1);
      }
      break;

    case HTP_OPS_ROOFLINE_BENCH:
      {
        auto params = reinterpret_cast<RooflineBenchParams *>(req->payload);
        int  max_results = params->max_results;
        if (max_results <= 0) {
          ret = -1;
          break;
        }

        size_t output_size = max_results * sizeof(RooflineBenchResult);
        add_buffer(out_bufs, params->output, output_size);
        if (params->mode == ROOFLINE_BENCH_MODE_DDR_BW || params->mode == ROOFLINE_BENCH_MODE_HMX_DMA_BW) {
          size_t size = (size_t) params->bytes;
          add_buffer(in_bufs, params->src, size);
        }
        if (params->mode == ROOFLINE_BENCH_MODE_DDR_BW) {
          size_t size = (size_t) params->bytes;
          add_buffer(out_bufs, params->dst, size);
        }

        validate_in_bufs();
        memset(OUT_PTR(0), 0, output_size);
        if (params->mode == ROOFLINE_BENCH_MODE_HMX_FP16) {
          ret = roofline_run_hmx_fp16((RooflineBenchResult *) OUT_PTR(0), max_results, params->warmup, params->iters);
        } else if (params->mode == ROOFLINE_BENCH_MODE_HVX_FP16) {
          ret = roofline_run_hvx_fp16((RooflineBenchResult *) OUT_PTR(0), max_results, params->warmup, params->iters);
        } else if (params->mode == ROOFLINE_BENCH_MODE_MIX_PRECISION) {
          ret = roofline_run_mix_precision((RooflineBenchResult *) OUT_PTR(0), max_results, params->warmup,
                                           params->iters, params->bytes);
        } else if (params->mode == ROOFLINE_BENCH_MODE_HMX_INT8_SHAPE_SWEEP) {
          ret = roofline_run_hmx_int8_shape_sweep((RooflineBenchResult *) OUT_PTR(0), max_results, params->warmup,
                                                  params->iters);
        } else if (params->mode == ROOFLINE_BENCH_MODE_SIGNED_INT8_ZERO_OVERHEAD) {
          ret = roofline_run_signed_int8_zero_overhead((RooflineBenchResult *) OUT_PTR(0), max_results,
                                                       params->warmup, params->iters);
        } else if (params->mode == ROOFLINE_BENCH_MODE_HMX_DMA_BW) {
          ret = roofline_run_hmx_dma_bandwidth((RooflineBenchResult *) OUT_PTR(0), max_results, IN_PTR(0),
                                               params->bytes, params->warmup, params->iters);
        } else {
          uint8_t *src = params->mode == ROOFLINE_BENCH_MODE_DDR_BW ? IN_PTR(0) : nullptr;
          uint8_t *dst = params->mode == ROOFLINE_BENCH_MODE_DDR_BW ? OUT_PTR(1) : nullptr;
          ret = roofline_run_hvx_bandwidth((RooflineBenchResult *) OUT_PTR(0), max_results, params->mode, src, dst,
                                           params->bytes, params->warmup, params->iters);
        }
        validate_out_bufs();
        ret = ret < 0 ? ret : 0;
      }
      break;

    case HTP_OPS_FLASH_ATTN_QO_F32_KV_F16:
      {
        auto params = reinterpret_cast<FlashAttnParams *>(req->payload);

        int qo_len     = params->qo_len;
        int kv_len     = params->kv_len;
        int n_heads    = params->n_heads;
        int n_kv_heads = params->n_kv_heads;
        int head_dim   = params->head_dim;

        size_t qo_size   = qo_len * n_heads * head_dim * sizeof(float);
        size_t kv_size   = kv_len * n_kv_heads * head_dim * sizeof(__fp16);
        size_t mask_size = qo_len * kv_len * sizeof(__fp16);

        add_buffer(out_bufs, params->o, qo_size);
        add_buffer(in_bufs, params->q, qo_size);
        add_buffer(in_bufs, params->k, kv_size);
        add_buffer(in_bufs, params->v, kv_size);
        add_buffer(in_bufs, params->mask, mask_size);
        auto *profile = add_trace_profile_buffer(params->profile, params->max_profile_events);

        constexpr bool check_accuracy = false;
        const size_t   input_size     = qo_size + 2 * kv_size + mask_size;

        if (check_accuracy) {
          float *ref_out;
          posix_memalign((void **) &ref_out, 128, qo_size);

          validate_in_bufs();
          ret = simple_flash_attn((__fp16 *) ref_out, (__fp16 *) IN_PTR(0), (__fp16 *) IN_PTR(1), (__fp16 *) IN_PTR(2),
                                  (__fp16 *) IN_PTR(3), qo_len, kv_len, n_heads, n_kv_heads, head_dim);

          // check logic
          naive_flash_attn((float *) OUT_PTR(0), (float *) IN_PTR(0), (__fp16 *) IN_PTR(1), (__fp16 *) IN_PTR(2),
                           (__fp16 *) IN_PTR(3), qo_len, kv_len, n_heads, n_kv_heads, head_dim);

          op_utils::compare_result((float *) OUT_PTR(0), ref_out, qo_size / 4);

          validate_out_bufs();

          free(ref_out);
        } else {
          int64_t t_total0 = trace_now_us();
          int64_t t0       = t_total0;
          validate_in_bufs();
          int64_t t1 = trace_now_us();
          log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_in", qo_len, head_dim,
                        n_heads * head_dim, qo_len, kv_len, n_heads, n_kv_heads, head_dim, input_size, qo_size, t0,
                        t1);

          t0  = t1;
          ret = simple_flash_attn_llm_profiled((__fp16 *) OUT_PTR(0), (__fp16 *) IN_PTR(0), (__fp16 *) IN_PTR(1),
                                               (__fp16 *) IN_PTR(2), (__fp16 *) IN_PTR(3), qo_len, kv_len, n_heads,
                                               n_kv_heads, head_dim, params->trace_id, params->mode_flags, req->op,
                                               profile);
          t1 = trace_now_us();
          log_dsp_event(params->trace_id, params->mode_flags, req->op, "compute", qo_len, head_dim,
                        n_heads * head_dim, qo_len, kv_len, n_heads, n_kv_heads, head_dim, input_size, qo_size, t0,
                        t1);

          t0 = t1;
          validate_out_bufs();
          t1 = trace_now_us();
          log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_out", qo_len, head_dim,
                        n_heads * head_dim, qo_len, kv_len, n_heads, n_kv_heads, head_dim, input_size, qo_size, t0,
                        t1);
          log_dsp_event(params->trace_id, params->mode_flags, req->op, "total", qo_len, head_dim,
                        n_heads * head_dim, qo_len, kv_len, n_heads, n_kv_heads, head_dim, input_size, qo_size,
                        t_total0, t1);
        }
      }
      break;

    case HTP_OPS_FLASH_ATTN_PROFILE_QO_F32_KV_F16:
      {
        auto params = reinterpret_cast<FlashAttnProfileParams *>(req->payload);

        int qo_len      = params->attn.qo_len;
        int kv_len      = params->attn.kv_len;
        int n_heads     = params->attn.n_heads;
        int n_kv_heads  = params->attn.n_kv_heads;
        int head_dim    = params->attn.head_dim;
        int max_records = params->max_records;
        int max_events  = params->max_events;

        if (max_records <= 0 || max_records > 1024 || max_events <= 0 || max_events > 65536) {
          ret = -1;
          break;
        }

        size_t qo_size      = qo_len * n_heads * head_dim * sizeof(float);
        size_t kv_size      = kv_len * n_kv_heads * head_dim * sizeof(__fp16);
        size_t mask_size    = qo_len * kv_len * sizeof(__fp16);
        size_t profile_size = sizeof(Figure8ProfileHeader) + max_records * sizeof(Figure8ProfileRecord) +
                              max_events * sizeof(Figure8ProfileEvent);

        add_buffer(out_bufs, params->attn.o, qo_size);
        add_buffer(out_bufs, params->profile, profile_size);
        add_buffer(in_bufs, params->attn.q, qo_size);
        add_buffer(in_bufs, params->attn.k, kv_size);
        add_buffer(in_bufs, params->attn.v, kv_size);
        add_buffer(in_bufs, params->attn.mask, mask_size);

        auto profile = reinterpret_cast<Figure8ProfileHeader *>(OUT_PTR(1));
        if (profile == nullptr) {
          ret = -1;
          break;
        }
        profile->magic        = FIGURE8_PROFILE_MAGIC;
        profile->max_records  = max_records;
        profile->record_count = 0;
        profile->max_events   = max_events;
        profile->event_count  = 0;
        profile->event_overflow = 0;
        profile->reserved0    = 0;
        profile->reserved1    = 0;

        validate_in_bufs();
        ret = simple_flash_attn_profiled((__fp16 *) OUT_PTR(0), (__fp16 *) IN_PTR(0), (__fp16 *) IN_PTR(1),
                                         (__fp16 *) IN_PTR(2), (__fp16 *) IN_PTR(3), qo_len, kv_len, n_heads,
                                         n_kv_heads, head_dim, params->attn.mode_flags, profile);
        validate_out_bufs();
      }
      break;

    default:
      break;
  }
  return ret;
}

#undef IN_PTR
#undef OUT_PTR
}
