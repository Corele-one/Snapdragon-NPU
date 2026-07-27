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

extern "C" void hmx_mat_mul_lpbq_a8w8_get_last_hmx_path_diag(int64_t out_diag[8]);

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
    case HTP_OPS_MAT_MUL_LPBQ_A8W8:
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
    case HTP_OPS_MAT_MUL_LPBQ_A8W8:
      return "matmul_lpbq_a8w8";
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

const char *dsp_op_trace_name(uint32_t op, int mode_flags) {
  if (op == HTP_OPS_MAT_MUL_PERMUTED_W16A32 && (mode_flags & LLM_NPU_MODE_PURE_FP16)) {
    return "matmul_f16";
  }
  return dsp_op_name(op);
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
       (long long) trace_id, mode_flags, dsp_op_trace_name(op, mode_flags), op, phase, m, k, n, qo_len, kv_len, n_heads, n_kv_heads,
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
      // Benchmark-only producer mock: zp=0 and zp=128 use the same instruction
      // stream, modeling +128 folded into an existing requant/store stage.
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
    const int correctness = roofline_check_hvx_fp32(a, b, c, n);
    roofline_set_mix_result(out, count++, ROOFLINE_BENCH_KIND_HVX_FP32_GEMM, 1, n, iters,
                            t1 - t0, (int64_t) iters * 2LL * n * n * n, ROOFLINE_BENCH_ENGINE_HVX,
                            ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_DTYPE_FP32,
                            ROOFLINE_BENCH_PATH_HVX_NATIVE, correctness);
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
    // Signed A8 benchmark model: HMX still consumes activation.ub, so the
    // producer supplies the zero-point-128 view a_u = a_s + 128 = a_s ^ 0x80.
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

// Kept as a reference for the original byte-by-byte packed-tile scan. It is
// intentionally not used for timing because model-load/offline colsum should
// not pay scalar DSP store latency on the token hot path.
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
    case 0:
    default:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
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
  int       variants_b[] = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12};
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
                              ROOFLINE_BENCH_DTYPE_INT16,
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
  const int variant = 8;  // activation.ub:above + weight.b, current best high-throughput INT8 tile stream.
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
    const int index = count++;

    if (total_vtcm_bytes > 0 && allocated_bytes > total_vtcm_bytes) {
      roofline_set_mix_result(out, index, ROOFLINE_BENCH_KIND_HMX_INT8_RAW_UB_B_GEMM, shapes[s].variant, M,
                              iters, 0, 0, ROOFLINE_BENCH_ENGINE_HMX, ROOFLINE_BENCH_DTYPE_INT8,
                              ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT16,
                              ROOFLINE_BENCH_PATH_HMX_RAW_UB_B_DEEP_GEMM, 0);
      out[index].mode = ROOFLINE_BENCH_MODE_HMX_INT8_SHAPE_SWEEP;
      roofline_set_int8_shape_metadata(out, index, M, K, N, a_bytes, b_bytes, c_bytes, scales_bytes,
                                       total_vtcm_bytes);
      continue;
    }

    uint8_t  *a      = vtcm;
    uint8_t  *b      = a + a_bytes;
    uint16_t *c      = (uint16_t *) (b + b_bytes);
    __fp16   *scales = (__fp16 *) ((uint8_t *) c + c_bytes);

    roofline_fill_hmx_tile_bytes(a, a_bytes, 1);
    roofline_fill_hmx_tile_bytes(b, b_bytes, 1);
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
    const int correctness = (core_ret == 0) && roofline_hmx_tile_store_changed(c);
    const int64_t hardware_ops = (int64_t) iters * 4LL * (int64_t) M * (int64_t) N * (int64_t) K;

    roofline_set_mix_result(out, index, ROOFLINE_BENCH_KIND_HMX_INT8_RAW_UB_B_GEMM, shapes[s].variant, M,
                            iters, t1 - t0, hardware_ops, ROOFLINE_BENCH_ENGINE_HMX,
                            ROOFLINE_BENCH_DTYPE_INT8, ROOFLINE_BENCH_DTYPE_INT8,
                            ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_PATH_HMX_RAW_UB_B_DEEP_GEMM,
                            correctness);
    out[index].mode = ROOFLINE_BENCH_MODE_HMX_INT8_SHAPE_SWEEP;
    roofline_set_int8_shape_metadata(out, index, M, K, N, a_bytes, b_bytes, c_bytes, scales_bytes,
                                     total_vtcm_bytes);
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

  // NOTE(hzx): the original signed roofline probe used 1024^3:
  //   constexpr int M = 1024;
  //   constexpr int K = 1024;
  //   constexpr int N = 1024;
  // On the V73 isolated LPBQ deployment, mode 31 reports only about 7.75 MiB
  // usable VTCM after reservations.  The 1024^3 layout places the HMX scales
  // buffer just past the usable VTCM end and causes a DSP reply timeout before
  // the signed A8W8 rows can be reported.  Keep K at 1024 so the HMX byte
  // stream still issues a full 32-Ktile chunk; reduce M/N to keep the standalone
  // probe inside VTCM while preserving a large HMX workload.
  constexpr int M = 960;
  constexpr int K = 1024;
  constexpr int N = 960;
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

static int roofline_run_mix_precision(RooflineBenchResult *out, int max_results, int warmup, int iters) {
  if (!out || max_results <= 0) return -1;
  int count = 0;

  count = roofline_run_hmx_byte_native_mix_one(out, count, max_results, warmup, iters, false);
  if (count < 0) return -1;
  count = roofline_run_hmx_byte_native_mix_one(out, count, max_results, warmup, iters, true);
  if (count < 0) return -1;
  count = roofline_run_hmx_int8_signed_zp_corrected_mix_one(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  count = roofline_run_hvx_fp32_mix(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  count = roofline_run_hvx_i16_mix(out, count, max_results, warmup, iters);
  if (count < 0) return -1;
  count = roofline_run_format_q4_mix_one(out, count, max_results, warmup, iters, false);
  if (count < 0) return -1;
  count = roofline_run_format_q4_mix_one(out, count, max_results, warmup, iters, true);
  if (count < 0) return -1;

  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_FP32, ROOFLINE_BENCH_DTYPE_FP32,
                                               ROOFLINE_BENCH_ENGINE_HMX, 100);
  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_INT16, ROOFLINE_BENCH_DTYPE_INT16,
                                               ROOFLINE_BENCH_ENGINE_HMX, 101);
  if (count < max_results) roofline_set_mix_na(out, count++, ROOFLINE_BENCH_DTYPE_FP16, ROOFLINE_BENCH_DTYPE_FP16,
                                               ROOFLINE_BENCH_ENGINE_HVX, 102);
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
  if (req->op != HTP_OPS_MAT_MUL_PERMUTED_W16A32) {
    // The pure-FP16 activation cache is only valid across immediately adjacent
    // W16A32 matmuls. Any other DSP op may reuse the same VTCM slots or imply
    // that host-side tensor data has advanced, so drop the cache conservatively.
    mat_mul_fp16_activation_cache_invalidate();
  }
  if (req->op != HTP_OPS_MAT_MUL_LPBQ_A8W8) {
    // LPBQ activation HMX cache follows the same VTCM lifetime rule as the
    // FP16 activation cache: only adjacent LPBQ matmuls may reuse it. Flash
    // attention, RMSNorm, FP16, or legacy quantized ops can overwrite VTCM.
    hmx_mat_mul_lpbq_a8w8_activation_cache_invalidate();
  }
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
        // Pure-FP16 weights are immutable model tensors. The hot W16A32 kernel
        // reads them through DMA/HVX just like quantized weights, so invalidating
        // the full FP16 weight tensor before every layer only adds short-prefill
        // latency without improving correctness. Keep activation cached because
        // the host writes it between graph nodes; keep weight uncached here to
        // match the established Q4/Q8 path.
        add_buffer(in_bufs, params->weight, weight_size, false);
        auto *profile = add_trace_profile_buffer(params->profile, params->max_profile_events);

        // pure_fp16 keeps the production HMX W16A32 kernel: FP32 activation,
        // HMX-layout FP16 weight, HMX MMA, FP32 output. Quantized W4/W8 paths
        // remain in the following switch cases and are not used for this op.
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

    case HTP_OPS_MAT_MUL_LPBQ_A8W8:
      {
        auto params = reinterpret_cast<LpbqA8W8MatMulParams *>(req->payload);
        int  m = params->m, k = params->k, n = params->n;

        size_t output_size     = m * n * sizeof(float);
        size_t activation_size = m * k * sizeof(float);
        const bool q8_0_weight_container = (params->mode_flags & LLM_NPU_MODE_LPBQ_INT8) != 0;
        // LPBQ deploy-v1 stores the offline-expanded int8 payload inside a
        // GGUF Q8_0 container.  Keep the declared RPC span aligned with that
        // container instead of the raw k*n byte payload; older code used the
        // smaller raw size and made trace/input accounting misleading.
        size_t weight_size     = q8_0_weight_container ?
                                   (size_t) n * (size_t) k / QK_0 * sizeof(block_q8_0) :
                                   (size_t) k * (size_t) n * sizeof(int8_t);
        size_t scale2_size     = (size_t) n * sizeof(float);
        size_t bias_size       = (size_t) n * sizeof(float);
        const bool has_folded_dequant = params->out_scale.fd >= 0 && params->bias_eff.fd >= 0;
        size_t out_scale_size  = has_folded_dequant ? (size_t) n * sizeof(float) : 0u;
        size_t bias_eff_size   = has_folded_dequant ? (size_t) n * sizeof(float) : 0u;
        const bool has_online_sum_w = params->sum_w.fd >= 0;
        const bool has_packed_weight =
          params->packed_weight.fd >= 0 && (has_online_sum_w || has_folded_dequant);
        const bool packed_weight_v6_full =
          has_packed_weight && ((params->mode_flags & LLM_NPU_MODE_LPBQ_PACKED_V6_FULL) != 0);
        size_t packed_weight_size = has_packed_weight ?
                                      (packed_weight_v6_full ? 2u : 1u) *
                                        (size_t) k * (size_t) n * sizeof(int8_t) :
                                      0u;
        size_t sum_w_size         = has_online_sum_w ? (size_t) n * sizeof(int32_t) : 0u;
        const bool has_k32_safe   = has_packed_weight && params->k32_safe.fd >= 0;
        size_t k32_safe_size      = has_k32_safe ? (size_t) (k / 32) * (size_t) (n / 32) * sizeof(uint8_t) : 0u;
        const bool has_k64_safe   = has_packed_weight && params->k64_safe.fd >= 0 && (k % 64) == 0;
        size_t k64_safe_size      = has_k64_safe ? (size_t) (k / 64) * (size_t) (n / 32) * sizeof(uint8_t) : 0u;
        const bool r4_folded_input_scale =
          (params->mode_flags & LLM_NPU_MODE_LPBQ_R4_FOLDED_INPUT_SCALE) != 0;
        const size_t r4_n_blocks =
          (r4_folded_input_scale && params->r4_block > 0 && (k % params->r4_block) == 0) ?
            (size_t) (k / params->r4_block) : 1u;
        size_t r4_size         = (params->r4.fd >= 0 && params->r4_block > 0) ?
                                   r4_n_blocks * (size_t) params->r4_block *
                                     (size_t) params->r4_block * sizeof(float) : 0u;
        const bool has_r4_hmx_dense_fp16 =
          params->r4_hmx_dense_fp16.fd >= 0 &&
          ((params->mode_flags & LLM_NPU_MODE_LPBQ_R4_HMX_DENSE_FP16_SIDECAR) != 0) &&
          params->r4_block == 128 && (k % params->r4_block) == 0;
        const size_t r4_hmx_dense_fp16_size =
          has_r4_hmx_dense_fp16 ?
            (size_t) (k / params->r4_block) * 16u * (size_t) HMX_FP16_TILE_SIZE : 0u;
        size_t input_scale_size = (params->input_scale.fd >= 0) ? (size_t) k * sizeof(float) : 0u;

        add_buffer(out_bufs, params->output, output_size);
        add_buffer(in_bufs, params->activation, activation_size);
        add_buffer(in_bufs, params->weight, weight_size, false);
        const int packed_weight_index = has_packed_weight ? (int) in_bufs.size() : -1;
        if (packed_weight_index >= 0) {
          // LPBQ deploy-v1: packed weights are immutable sidecars.  Avoid a
          // per-op cache invalidate over hundreds of MB; the buffer is mapped
          // once by FastRPC and read-only on DSP.
          add_buffer(in_bufs, params->packed_weight, packed_weight_size, false);
        }
        const bool immutable_lpbq_sidecar_cache = HTP_LPBQ_IMMUTABLE_SIDECAR_CACHE != 0;
        const bool lpbq_sidecar_cached = !immutable_lpbq_sidecar_cache;

        const int sum_w_index = has_online_sum_w ? (int) in_bufs.size() : -1;
        if (sum_w_index >= 0) {
          // LPBQ A/B: sum_w is a read-only sidecar when it is still sent online.
          // Keep the original cache invalidate path available through
          // HTP_LPBQ_IMMUTABLE_SIDECAR_CACHE=0 for quick correctness bisection.
          add_buffer(in_bufs, params->sum_w, sum_w_size, lpbq_sidecar_cached);
        }
        const int k32_safe_index = has_k32_safe ? (int) in_bufs.size() : -1;
        if (k32_safe_index >= 0) {
          // Host precomputes this immutable byte table while loading the
          // packed sidecar; DSP only reads one flag per (Ktile,Ntile).
          add_buffer(in_bufs, params->k32_safe, k32_safe_size, false);
        }
        const int k64_safe_index = has_k64_safe ? (int) in_bufs.size() : -1;
        if (k64_safe_index >= 0) {
          // Same lifetime model as k32_safe, but each flag covers two adjacent
          // K32 groups so the exact K64 candidate avoids online packed-weight scans.
          add_buffer(in_bufs, params->k64_safe, k64_safe_size, false);
        }
        const int scale2_index = (int) in_bufs.size();
        // LPBQ offline/QAT constants are loaded once and never modified by the
        // host during inference.  The old per-op invalidate path is preserved by
        // the compile-time guard above, but the A/B path avoids repeatedly
        // paying cache-maintenance cost for sidecar constants.
        add_buffer(in_bufs, params->scale2, scale2_size, lpbq_sidecar_cached);
        const int bias_index = (params->bias.fd >= 0) ? (int) in_bufs.size() : -1;
        if (bias_index >= 0) {
          add_buffer(in_bufs, params->bias, bias_size, lpbq_sidecar_cached);
        }
        const int r4_index = (r4_size > 0) ? (int) in_bufs.size() : -1;
        if (r4_index >= 0) {
          add_buffer(in_bufs, params->r4, r4_size, lpbq_sidecar_cached);
        }
        const int r4_hmx_dense_fp16_index = has_r4_hmx_dense_fp16 ? (int) in_bufs.size() : -1;
        if (r4_hmx_dense_fp16_index >= 0) {
          // No-quality/performance-first Stage-A sidecar.  The buffer is an
          // immutable HMX FP16 tile stream; keep it out of the old dense-R4 fd
          // so fallback semantics stay explicit.
          add_buffer(in_bufs, params->r4_hmx_dense_fp16, r4_hmx_dense_fp16_size, false);
        }
        const int input_scale_index = (input_scale_size > 0) ? (int) in_bufs.size() : -1;
        if (input_scale_index >= 0) {
          add_buffer(in_bufs, params->input_scale, input_scale_size, lpbq_sidecar_cached);
        }
        const int out_scale_index = has_folded_dequant ? (int) in_bufs.size() : -1;
        if (out_scale_index >= 0) {
          add_buffer(in_bufs, params->out_scale, out_scale_size, lpbq_sidecar_cached);
        }
        const int bias_eff_index = has_folded_dequant ? (int) in_bufs.size() : -1;
        if (bias_eff_index >= 0) {
          add_buffer(in_bufs, params->bias_eff, bias_eff_size, lpbq_sidecar_cached);
        }
        auto *profile = add_trace_profile_buffer(params->profile, params->max_profile_events);

        const size_t input_size = activation_size + weight_size + packed_weight_size + sum_w_size + k32_safe_size +
                                  k64_safe_size +
                                  scale2_size +
                                  (bias_index >= 0 ? bias_size : 0) +
                                  (r4_index >= 0 ? r4_size : 0) +
                                  (r4_hmx_dense_fp16_index >= 0 ? r4_hmx_dense_fp16_size : 0) +
                                  (input_scale_index >= 0 ? input_scale_size : 0) +
                                  (out_scale_index >= 0 ? out_scale_size : 0) +
                                  (bias_eff_index >= 0 ? bias_eff_size : 0);
        int64_t      t_total0   = trace_now_us();
        int64_t      t0         = t_total0;
        validate_in_bufs();
        int64_t t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_in", m, k, n, 0, 0, 0, 0, 0,
                      input_size, output_size, t0, t1);

        t0 = t1;
        if (trace_enabled(params->mode_flags)) {
          FARF(ALWAYS,
               "LPBQ_DSP_ENTER trace_id=%lld op=%s flags=%d m=%d k=%d n=%d r4_block=%d packed=%d k32=%d k64=%d "
               "folded=%d act_scale=%f input_bytes=%lld output_bytes=%lld",
               (long long) params->trace_id, dsp_op_trace_name(req->op, params->mode_flags), params->mode_flags,
               m, k, n, params->r4_block, packed_weight_index >= 0 ? 1 : 0, k32_safe_index >= 0 ? 1 : 0,
               k64_safe_index >= 0 ? 1 : 0, has_folded_dequant ? 1 : 0, params->act_scale,
               (long long) input_size, (long long) output_size);
        }
        ret = hmx_mat_mul_lpbq_a8w8((float *) OUT_PTR(0), (float *) IN_PTR(0), (int8_t *) IN_PTR(1),
                                     packed_weight_index >= 0 ? (int8_t *) IN_PTR(packed_weight_index) : nullptr,
                                     sum_w_index >= 0 ? (int32_t *) IN_PTR(sum_w_index) : nullptr,
                                     k32_safe_index >= 0 ? (uint8_t *) IN_PTR(k32_safe_index) : nullptr,
                                     k64_safe_index >= 0 ? (uint8_t *) IN_PTR(k64_safe_index) : nullptr,
                                     (float *) IN_PTR(scale2_index),
                                     bias_index >= 0 ? (float *) IN_PTR(bias_index) : nullptr,
                                     r4_index >= 0 ? (float *) IN_PTR(r4_index) : nullptr,
                                     r4_hmx_dense_fp16_index >= 0 ?
                                       (__fp16 *) IN_PTR(r4_hmx_dense_fp16_index) : nullptr,
                                     input_scale_index >= 0 ? (float *) IN_PTR(input_scale_index) : nullptr,
                                     out_scale_index >= 0 ? (float *) IN_PTR(out_scale_index) : nullptr,
                                     bias_eff_index >= 0 ? (float *) IN_PTR(bias_eff_index) : nullptr,
                                     params->r4_block, params->act_scale, m, k, n, params->trace_id,
                                     params->mode_flags, req->op, profile);
        t1 = trace_now_us();
        if (trace_enabled(params->mode_flags)) {
          int64_t lpbq_path_diag[8] = {};
          hmx_mat_mul_lpbq_a8w8_get_last_hmx_path_diag(lpbq_path_diag);
          FARF(ALWAYS,
               "LPBQ_DSP_EXIT trace_id=%lld op=%s flags=%d m=%d k=%d n=%d ret=%d compute_us=%lld",
               (long long) params->trace_id, dsp_op_trace_name(req->op, params->mode_flags), params->mode_flags,
               m, k, n, ret, (long long) (t1 - t0));
          FARF(ALWAYS,
               "LPBQ_DSP_PATH_DIAG trace_id=%lld m=%d k=%d n=%d k64_total=%lld k64_safe=%lld "
               "k64_unsafe=%lld fallback_k32=%lld fallback_k16=%lld est_drains=%lld est_k4_issues=%lld flags=%lld "
               "diag0=%lld diag1=%lld diag2=%lld diag3=%lld diag4=%lld diag5=%lld diag6=%lld diag7=%lld",
               (long long) params->trace_id, m, k, n,
               (long long) lpbq_path_diag[0], (long long) lpbq_path_diag[1],
               (long long) lpbq_path_diag[2], (long long) lpbq_path_diag[3],
               (long long) lpbq_path_diag[4], (long long) lpbq_path_diag[5],
               (long long) lpbq_path_diag[6], (long long) lpbq_path_diag[7],
               (long long) lpbq_path_diag[0], (long long) lpbq_path_diag[1],
               (long long) lpbq_path_diag[2], (long long) lpbq_path_diag[3],
               (long long) lpbq_path_diag[4], (long long) lpbq_path_diag[5],
               (long long) lpbq_path_diag[6], (long long) lpbq_path_diag[7]);
        }
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "compute", m, k, n, 0, 0, 0, 0, 0,
                      input_size, output_size, t0, t1);

        t0 = t1;
        validate_out_bufs();
        t1 = trace_now_us();
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "validate_out", m, k, n, 0, 0, 0, 0, 0,
                      input_size, output_size, t0, t1);
        log_dsp_event(params->trace_id, params->mode_flags, req->op, "total", m, k, n, 0, 0, 0, 0, 0,
                      input_size, output_size, t_total0, t1);
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
                                           params->iters);
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
