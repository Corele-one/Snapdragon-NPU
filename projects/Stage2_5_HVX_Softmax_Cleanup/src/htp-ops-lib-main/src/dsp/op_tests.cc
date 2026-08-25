#include <HAP_farf.h>
#include <HAP_perf.h>

// std headers
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_math.h"
#include "dsp/ops.h"
#include "dsp/vtcm_mgr.h"

namespace op_utils {

float compute_rmse(const float *x, const float *y, int n) {
  float squared_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    float err = x[i] - y[i];
    squared_error += err * err;
  }
  float rmse = sqrtf(squared_error / n);
  return rmse;
}

int compare_result(const float *x, const float *y, int n_elems) {
  static int counter = 0;

  int layer = counter++ % 28;
  FARF(ALWAYS, "layer %d attention compare:", layer);

  // hard-coded constants
  constexpr int D = 128;
  constexpr int H = 12;  // 12 query heads
  for (int h = 0; h < n_elems / D; ++h) {
    int q    = h / H;
    int head = h % H;

    float rmse = compute_rmse(&x[h * D], &y[h * D], D);
    FARF(ALWAYS, "query %d head %d RMSE: %g", q, head, rmse);
  }
  return 0;
}

}  // namespace op_utils

namespace internal {

namespace {

constexpr int HMX_INT8_TILE_N_ELMS = 32 * 32;
constexpr int HMX_INT8_TILE_SIZE   = HMX_INT8_TILE_N_ELMS * sizeof(uint8_t);

static inline void hmx_load_tiles_ub_b_variant(const uint8_t *row_tiles, const int8_t *col_tiles, size_t limit,
                                               int variant) {
  switch (variant) {
    case 0:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):deep\n"
        "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
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
    default:
      if (variant == 8) {
        asm volatile(
          "{ activation.ub = mxmem(%0, %1):above\n"
          "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
          "r"(limit), "r"(col_tiles), "r"(limit)
          : "memory");
        break;
      }
      if (variant == 9) {
        asm volatile(
          "{ activation.ub = mxmem(%0, %1):above:cm\n"
          "weight.b = mxmem(%2, %3) }\n" ::"r"(row_tiles),
          "r"(limit), "r"(col_tiles), "r"(limit)
          : "memory");
        break;
      }
      if (variant == 10) {
        asm volatile(
          "{ activation.ub = mxmem(%0, %1):dilate\n"
          "weight.b = mxmem(%2, %3):dilate }\n" ::"r"(row_tiles),
          "r"(limit), "r"(col_tiles), "r"(limit)
          : "memory");
        break;
      }
      if (variant == 11) {
        asm volatile(
          "{ activation.ub = mxmem(%0, %1):deep\n"
          "weight.b = mxmem(%2, %3):after }\n" ::"r"(row_tiles),
          "r"(limit), "r"(col_tiles), "r"(limit)
          : "memory");
        break;
      }
      asm volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "weight.b = mxmem(%2, %3):deep }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
}

static inline void hmx_consume_accumulator_hf_selector(__fp16 *out, int selector) {
  asm volatile(
    "cvt.hf = acc(%0)\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(selector),
    "r"(out), "r"(0)
    : "memory");
}

static inline void hmx_consume_accumulator_ub_selector(uint8_t *out, int selector) {
  asm volatile(
    "cvt.ub = acc(%0)\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(selector),
    "r"(out), "r"(0)
    : "memory");
}

static inline void hmx_consume_accumulator_ub_sc0_selector(uint8_t *out, int selector) {
  asm volatile(
    "cvt.ub = acc(%0):sc0\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(selector),
    "r"(out), "r"(0)
    : "memory");
}

static inline void hmx_consume_accumulator_ub_sc1_selector(uint8_t *out, int selector) {
  asm volatile(
    "cvt.ub = acc(%0):sc1\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(selector),
    "r"(out), "r"(0)
    : "memory");
}

static inline void hmx_consume_accumulator_uh_2x1_selector(uint16_t *out, int selector) {
  asm volatile(
    "cvt.uh = acc(%0):2x1\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(selector),
    "r"(out), "r"(0)
    : "memory");
}

static inline void hmx_consume_accumulator_uh_2x2_selector(uint16_t *out, int selector) {
  asm volatile(
    "cvt.uh = acc(%0):2x2\n"
    "mxmem(%1, %2):2x2 = cvt\n" ::"r"(selector),
    "r"(out), "r"(0)
    : "memory");
}

static inline void hmx_store_accumulator_direct(uint8_t *out, int output_kind) {
  switch (output_kind) {
    case 9:
      asm volatile("mxmem(%0, %1):before.hf = acc" ::"r"(out), "r"(0) : "memory");
      break;
    case 10:
      asm volatile("mxmem(%0, %1):after.hf = acc" ::"r"(out), "r"(0) : "memory");
      break;
    case 11:
      asm volatile("mxmem(%0, %1):before.ub = acc" ::"r"(out), "r"(0) : "memory");
      break;
    case 12:
      asm volatile("mxmem(%0, %1):after.ub = acc" ::"r"(out), "r"(0) : "memory");
      break;
    case 13:
      asm volatile("mxmem(%0, %1):before:sat.ub = acc" ::"r"(out), "r"(0) : "memory");
      break;
    case 14:
      asm volatile("mxmem(%0, %1):after:sat.ub = acc" ::"r"(out), "r"(0) : "memory");
      break;
    case 15:
      asm volatile("mxmem(%0, %1):before.uh = acc:2x1" ::"r"(out), "r"(0) : "memory");
      break;
    case 16:
      asm volatile("mxmem(%0, %1):after.uh = acc:2x1" ::"r"(out), "r"(0) : "memory");
      break;
    case 17:
      asm volatile("mxmem(%0, %1):before.uh = acc:2x2" ::"r"(out), "r"(0) : "memory");
      break;
    case 18:
      asm volatile("mxmem(%0, %1):after.uh = acc:2x2" ::"r"(out), "r"(0) : "memory");
      break;
  }
}

static inline float hmx_gate_read_output_sample(const void *out, int output_kind, int idx) {
  if (output_kind == 0 || output_kind == 9 || output_kind == 10) {
    return (float) ((const __fp16 *) out)[idx];
  }
  if (output_kind == 7 || output_kind == 8 || output_kind >= 15) {
    return (float) ((const uint16_t *) out)[idx];
  }
  return (float) ((const uint8_t *) out)[idx];
}

static inline uint16_t hmx_gate_fp16_bits(float value) {
  __fp16   h = (__fp16) value;
  uint16_t bits;
  memcpy(&bits, &h, sizeof(bits));
  return bits;
}

static inline void hmx_gate_init_scales(void *scales, int scale_mode) {
  memset(scales, 0, 256);

  if (scale_mode == 0) {
    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));  // fp16: 1.0
    return;
  }

  float scale = 1.0f;
  float bias  = 0.0f;
  if (scale_mode == 2) {
    bias = -64.0f;
  } else if (scale_mode == 3) {
    scale = 0.5f;
  } else if (scale_mode == 4) {
    scale = 0.01f;
    bias  = -204.8f;
  }

  const uint16_t scale_bits = hmx_gate_fp16_bits(scale);
  const uint16_t bias_bits  = hmx_gate_fp16_bits(bias);
  uint32_t      *words      = (uint32_t *) scales;

  for (int i = 0; i < 32; ++i) {
    words[i]      = ((uint32_t) bias_bits << 16) | scale_bits;
    words[32 + i] = 0;
  }
}

static inline float hmx_gate_fill_inputs(uint8_t *a, int8_t *b, int input_case) {
  uint8_t a_value = 1;
  int8_t  b_value = 1;
  float   expected = 32.0f;

  switch (input_case) {
    case 1:
      b_value  = 2;
      expected = 64.0f;
      break;
    case 2:
      a_value  = 2;
      b_value  = 3;
      expected = 192.0f;
      break;
    case 3:
      a_value  = 255;
      expected = 8160.0f;
      break;
    case 4:
      b_value  = -1;
      expected = -32.0f;
      break;
    case 5:
      a_value  = 2;
      b_value  = -3;
      expected = -192.0f;
      break;
    case 6:
      for (int i = 0; i < HMX_FP16_TILE_SIZE; ++i) {
        a[i] = 1;
        b[i] = (i & 1) ? -1 : 1;
      }
      return 0.0f;
    case 7:
      a_value  = 127;
      b_value  = 127;
      expected = 516128.0f;
      break;
    case 8:
      a_value  = 64;
      b_value  = 127;
      expected = 260096.0f;
      break;
    case 9:
      a_value  = 126;   // signed activation -2 encoded as uint8 offset by 128
      b_value  = 5;
      expected = 20160.0f;
      break;
  }

  memset(a, a_value, HMX_FP16_TILE_SIZE);
  memset(b, b_value, HMX_FP16_TILE_SIZE);
  return expected;
}

static inline void hmx_int8_gate_run_one(struct HmxInt8GateResult *res, __fp16 *out, const __fp16 *scales,
                                         const uint8_t *a, const int8_t *b, int input_case, int variant,
                                         int tile_bytes, int output_kind, int selector, float expected) {
  memset(res, 0, sizeof(*res));
  res->selector    = selector;
  res->variant     = variant;
  res->output_kind = output_kind;
  res->tile_bytes  = tile_bytes;
  res->reserved    = input_case;
  res->expected    = expected;

  memset(out, 0, HMX_FP16_TILE_SIZE);
  asm volatile("mxclracc" ::: "memory");
  hmx_set_output_scales(scales);
  hmx_load_tiles_ub_b_variant(a, b, tile_bytes - 1, variant);
  if (output_kind <= 3) {
    hmx_consume_accumulator_hf_selector(out, selector);
  } else if (output_kind == 4) {
    hmx_consume_accumulator_ub_selector((uint8_t *) out, selector);
  } else if (output_kind == 5) {
    hmx_consume_accumulator_ub_sc0_selector((uint8_t *) out, selector);
  } else if (output_kind == 6) {
    hmx_consume_accumulator_ub_sc1_selector((uint8_t *) out, selector);
  } else if (output_kind == 7) {
    hmx_consume_accumulator_uh_2x1_selector((uint16_t *) out, selector);
  } else if (output_kind == 8) {
    hmx_consume_accumulator_uh_2x2_selector((uint16_t *) out, selector);
  } else {
    hmx_store_accumulator_direct((uint8_t *) out, output_kind);
  }

  float min_v = 1e30f;
  float max_v = -1e30f;
  float sum_v = 0.0f;
  float rmse  = 0.0f;
  int   n_nan = 0;
  for (int i = 0; i < 8; ++i) {
    res->first8[i] = hmx_gate_read_output_sample(out, output_kind, i);
  }
  for (int i = 0; i < HMX_FP16_TILE_N_ELMS; ++i) {
    float v = hmx_gate_read_output_sample(out, output_kind, i);
    if (!isfinite(v)) {
      ++n_nan;
      continue;
    }
    min_v = fminf(min_v, v);
    max_v = fmaxf(max_v, v);
    sum_v += v;
    float e = v - expected;
    rmse += e * e;
  }
  rmse = sqrtf(rmse / HMX_FP16_TILE_N_ELMS);
  res->nan_count  = n_nan;
  res->min_value  = min_v;
  res->max_value  = max_v;
  res->mean_value = sum_v / HMX_FP16_TILE_N_ELMS;
  res->rmse       = rmse;
}

}  // namespace

static int hmx_int8_gate_search_run(struct HmxInt8GateResult *results, int max_results) {
  const int input_cases[]  = {0, 4, 7, 9};
  const int variants[]     = {8, 9};
  const int output_kinds[] = {0, 7, 16};
  constexpr int n_selectors = 64;
  constexpr int n_scales    = 3;

  const int n_results = (int) (sizeof(input_cases) / sizeof(input_cases[0])) *
                        (int) (sizeof(variants) / sizeof(variants[0])) *
                        ((2 * n_selectors) + 1) * n_scales;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t *a      = vtcm;
  int8_t  *b      = (int8_t *) (vtcm + 0x1000);
  __fp16  *out    = (__fp16 *) (vtcm + 0x2000);
  __fp16  *scales = (__fp16 *) (vtcm + 0x3000);

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  const int scale_modes[] = {0, 2, 4};
  for (int scale_mode : scale_modes) {
    hmx_gate_init_scales(scales, scale_mode);
    for (int input_case : input_cases) {
      float expected = hmx_gate_fill_inputs(a, b, input_case);
      for (int variant : variants) {
        for (int output_kind : output_kinds) {
          const bool selector_output = output_kind == 0 || output_kind == 7 || output_kind == 8;
          const int  selector_count  = selector_output ? n_selectors : 1;
          for (int selector = 0; selector < selector_count; ++selector) {
            HmxInt8GateResult *res = &results[result_idx++];
            hmx_int8_gate_run_one(res, out, scales, a, b, input_case, variant, HMX_FP16_TILE_SIZE, output_kind,
                                  selector, expected);
            res->tile_bytes = scale_mode;
          }
        }
      }
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static inline int hmx_gate_decode_uh_signed(uint16_t raw) {
  return raw < 32768 ? (int) raw : (int) raw - 65536;
}

static inline void hmx_load_tiles_ub_bit_weight_variant(const uint8_t *row_tiles, const uint8_t *col_tiles,
                                                        size_t limit, int variant) {
  switch (variant) {
    case 100:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.ubit = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 101:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.sbit = mxmem(%2, %3) }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 102:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.ubit = mxmem(%2, %3):after }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 103:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.sbit = mxmem(%2, %3):after }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 104:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.ubit = mxmem(%2, %3):deep }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 105:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.sbit = mxmem(%2, %3):deep }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 106:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.ubit = mxmem(%2, %3):single }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
    case 107:
      asm volatile(
        "{ activation.ub = mxmem(%0, %1):above\n"
        "weight.sbit = mxmem(%2, %3):single }\n" ::"r"(row_tiles),
        "r"(limit), "r"(col_tiles), "r"(limit)
        : "memory");
      break;
  }
}

static int hmx_int8_bitop_raw_dot_first(uint8_t *a, uint8_t *b, uint16_t *out, const __fp16 *scales, int variant) {
  memset(out, 0, HMX_FP16_TILE_SIZE);

  asm volatile("mxclracc" ::: "memory");
  hmx_set_output_scales(scales);
  hmx_load_tiles_ub_bit_weight_variant(a, b, HMX_FP16_TILE_SIZE - 1, variant);
  hmx_store_accumulator_direct((uint8_t *) out, 16);
  return hmx_gate_decode_uh_signed(out[0]);
}

static int hmx_int8_bitop_gate_run(struct HmxInt8GateResult *results, int max_results) {
  const int variants[] = {100, 101, 102, 103, 104, 105, 106, 107};
  const uint8_t patterns[] = {0x00, 0x01, 0x03, 0x55, 0x80, 0xaa, 0xff};
  const int n_results = (int) (sizeof(variants) / sizeof(variants[0])) *
                        (int) (sizeof(patterns) / sizeof(patterns[0]));
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a      = vtcm;
  uint8_t  *b      = vtcm + 0x1000;
  uint16_t *out    = (uint16_t *) (vtcm + 0x2000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x3000);
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  for (int variant : variants) {
    for (uint8_t pattern : patterns) {
      memset(a, 1, HMX_FP16_TILE_SIZE);
      memset(b, pattern, HMX_FP16_TILE_SIZE);
      int got = hmx_int8_bitop_raw_dot_first(a, b, out, scales, variant);

      HmxInt8GateResult *res = &results[result_idx++];
      memset(res, 0, sizeof(*res));
      res->selector    = 0;
      res->variant     = variant;
      res->output_kind = 16;
      res->tile_bytes  = pattern;
      res->reserved    = pattern;
      res->expected    = 0.0f;
      res->first8[0]   = (float) got;
      for (int i = 0; i < 7; ++i) {
        res->first8[i + 1] = (float) hmx_gate_decode_uh_signed(out[i + 1]);
      }
      res->min_value  = (float) got;
      res->max_value  = (float) got;
      res->mean_value = (float) got;
      res->rmse       = fabsf((float) got);
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static inline int hmx_tile_idx_32(int row, int col) {
  return (row & ~1) * 32 + col * 2 + (row & 1);
}

static void hmx_int8_bitop_raw_dot_tile(const uint8_t *a, const uint8_t *b, uint16_t *out, const __fp16 *scales);
static void hmx_int8_raw_dot_tile_b(const uint8_t *a, const int8_t *b, uint16_t *out, const __fp16 *scales);

static void __attribute__((unused)) hmx_int8_pack_activation_mag(uint8_t *dst, const int8_t *src, bool negative) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      const int8_t v = src[r * 32 + k];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag = v < 0 ? -v : v;
      const int base = 2 * hmx_tile_idx_32(r, k);
      if (base + 1 < HMX_FP16_TILE_SIZE) {
        dst[base + 1] = (uint8_t) (mag << 1);
      }
    }
  }
}

static void __attribute__((unused)) hmx_int8_pack_activation_bit(uint8_t *dst, const int8_t *src, int bit,
                                                                 bool negative) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      const int8_t v = src[r * 32 + k];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag = v < 0 ? -v : v;
      if ((mag & (1 << bit)) == 0) {
        continue;
      }
      const int base = 2 * hmx_tile_idx_32(r, k);
      if (base + 1 < HMX_FP16_TILE_SIZE) {
        dst[base + 1] = 2;
      }
    }
  }
}

static void hmx_int8_pack_activation_raw_lane(uint8_t *dst, int r, int k, int lane, uint8_t value) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  const int base = 2 * hmx_tile_idx_32(r, k);
  if (base + lane >= 0 && base + lane < HMX_FP16_TILE_SIZE) {
    dst[base + lane] = value;
  }
}

static void __attribute__((unused)) hmx_int8_pack_weight_bit(uint8_t *dst, const int8_t *src, int bit, bool negative) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      const int8_t v = src[k * 32 + c];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag = v < 0 ? -v : v;
      if ((mag & (1 << bit)) == 0) {
        continue;
      }
      const int base = 2 * hmx_tile_idx_32(k, c);
      for (int lane = 0; lane < 4; ++lane) {
        if (base + lane < HMX_FP16_TILE_SIZE) {
          dst[base + lane] = 1;
        }
        if (base + 128 + lane < HMX_FP16_TILE_SIZE) {
          dst[base + 128 + lane] = 1;
        }
      }
    }
  }
}

static void __attribute__((unused)) hmx_int8_pack_weight_b_bit(int8_t *dst, const int8_t *src, int bit, bool negative) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      const int8_t v = src[k * 32 + c];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag = v < 0 ? -v : v;
      if ((mag & (1 << bit)) == 0) {
        continue;
      }
      const int base = 128 * (k / 2) + 4 * c + 2 * (k & 1);
      if (base >= 0 && base + 1 < HMX_FP16_TILE_SIZE) {
        dst[base + 0] = 1;
        dst[base + 1] = 1;
      }
    }
  }
}

static void hmx_int8_pack_weight_b_raw_pair(int8_t *dst, int k, int c, int8_t value) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  const int base = 128 * (k / 2) + 4 * c + 2 * (k & 1);
  if (base >= 0 && base + 1 < HMX_FP16_TILE_SIZE) {
    dst[base + 0] = value;
    dst[base + 1] = value;
  }
}

static void hmx_int8_pack_activation_mag_k4(uint8_t *dst, const int8_t *src, int k_base, bool negative) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < 32; ++r) {
    for (int kk = 0; kk < 4; ++kk) {
      const int k = k_base + kk;
      const int8_t v = src[r * 32 + k];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag  = v < 0 ? -v : v;
      const int base = 128 * (r / 2) + 4 * kk + 2 * (r & 1) + 1;
      if (base >= 0 && base < HMX_FP16_TILE_SIZE) {
        dst[base] = (uint8_t) (mag << 1);
      }
    }
  }
}

static void hmx_int8_pack_activation_mag_k2(uint8_t *dst, const int8_t *src, int k_base, bool negative) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < 32; ++r) {
    for (int kk = 0; kk < 2; ++kk) {
      const int k = k_base + kk;
      const int8_t v = src[r * 32 + k];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag  = v < 0 ? -v : v;
      const int base = 128 * (r / 2) + 4 * kk + 2 * (r & 1) + 1;
      if (base >= 0 && base < HMX_FP16_TILE_SIZE) {
        dst[base] = (uint8_t) (mag << 1);
      }
    }
  }
}

static void hmx_int8_pack_weight_b_bit_k4(int8_t *dst, const int8_t *src, int k_base, int bit, bool negative) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int kk = 0; kk < 4; ++kk) {
    const int k = k_base + kk;
    for (int c = 0; c < 32; ++c) {
      const int8_t v = src[k * 32 + c];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag = v < 0 ? -v : v;
      if ((mag & (1 << bit)) == 0) {
        continue;
      }
      const int base = 4 * c + kk;
      if (base >= 0 && base < HMX_FP16_TILE_SIZE) {
        dst[base] = 1;
      }
    }
  }
}

static void hmx_int8_pack_weight_b_full_k4(int8_t *dst, const int8_t *src, int k_base) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int kk = 0; kk < 4; ++kk) {
    const int k = k_base + kk;
    for (int c = 0; c < 32; ++c) {
      const int base = 4 * c + kk;
      if (base >= 0 && base < HMX_FP16_TILE_SIZE) {
        dst[base] = src[k * 32 + c];
      }
    }
  }
}

static void hmx_int8_pack_weight_b_full_k2(int8_t *dst, const int8_t *src, int k_base) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int kk = 0; kk < 2; ++kk) {
    const int k = k_base + kk;
    for (int c = 0; c < 32; ++c) {
      const int base = 4 * c + kk;
      if (base >= 0 && base < HMX_FP16_TILE_SIZE) {
        dst[base] = src[k * 32 + c];
      }
    }
  }
}

static void hmx_int8_accumulate_uh_tile(int32_t *acc, const uint16_t *out, int sign, int bit) {
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      const int v = hmx_gate_decode_uh_signed(out[hmx_tile_idx_32(r, c)]) << bit;
      acc[r * 32 + c] += sign * v;
    }
  }
}

static void hmx_int8_accumulate_uh_tile_unshifted(int32_t *acc, const uint16_t *out, int sign) {
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      const int v = hmx_gate_decode_uh_signed(out[hmx_tile_idx_32(r, c)]);
      acc[r * 32 + c] += sign * v;
    }
  }
}

static void __attribute__((unused)) hmx_int8_pack_activation_mag_mask(uint8_t *dst, const int8_t *src, bool negative,
                                                                      int lane_mask) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      const int8_t v = src[r * 32 + k];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag  = v < 0 ? -v : v;
      const int base = 128 * (r / 2) + 4 * k + 2 * (r & 1);
      for (int lane = 0; lane < 4; ++lane) {
        if ((lane_mask & (1 << lane)) && base + lane < HMX_FP16_TILE_SIZE) {
          dst[base + lane] = (uint8_t) (mag << 1);
        }
      }
    }
  }
}

static void __attribute__((unused)) hmx_int8_pack_activation_bit_mask(uint8_t *dst, const int8_t *src, int bit,
                                                                      bool negative, int lane_mask) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      const int8_t v = src[r * 32 + k];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag = v < 0 ? -v : v;
      if ((mag & (1 << bit)) == 0) {
        continue;
      }
      const int base = 128 * (r / 2) + 4 * k + 2 * (r & 1);
      for (int lane = 0; lane < 4; ++lane) {
        if ((lane_mask & (1 << lane)) && base + lane < HMX_FP16_TILE_SIZE) {
          dst[base + lane] = 2;
        }
      }
    }
  }
}

static void hmx_int8_pack_weight_b_bit_mask(int8_t *dst, const int8_t *src, int bit, bool negative, int lane_mask) {
  memset(dst, 0, HMX_FP16_TILE_SIZE);
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      const int8_t v = src[k * 32 + c];
      if ((v < 0) != negative || v == 0) {
        continue;
      }
      const int mag = v < 0 ? -v : v;
      if ((mag & (1 << bit)) == 0) {
        continue;
      }
      const int base = 128 * (k / 2) + 4 * c + 2 * (k & 1);
      for (int lane = 0; lane < 4; ++lane) {
        if ((lane_mask & (1 << lane)) && base + lane < HMX_FP16_TILE_SIZE) {
          dst[base + lane] = 1;
        }
      }
    }
  }
}

static void hmx_int8_tile_bitserial_compute(uint8_t *a_pos, uint8_t *a_neg, int8_t *w_pos, int8_t *w_neg,
                                            uint16_t *out, const __fp16 *scales, int32_t *acc,
                                            const int8_t *a_ref, const int8_t *w_ref, int a_mask, int w_mask) {
  for (int i = 0; i < 32 * 32; ++i) {
    acc[i] = 0;
  }

  hmx_int8_pack_activation_mag_mask(a_pos, a_ref, false, a_mask);
  hmx_int8_pack_activation_mag_mask(a_neg, a_ref, true, a_mask);

  for (int bit = 0; bit < 7; ++bit) {
    hmx_int8_pack_weight_b_bit_mask(w_pos, w_ref, bit, false, w_mask);
    hmx_int8_pack_weight_b_bit_mask(w_neg, w_ref, bit, true, w_mask);

    hmx_int8_raw_dot_tile_b(a_pos, w_pos, out, scales);
    for (int r = 0; r < 32; ++r)
      for (int c = 0; c < 32; ++c)
        acc[r * 32 + c] += hmx_gate_decode_uh_signed(out[hmx_tile_idx_32(r, c)]) << bit;

    hmx_int8_raw_dot_tile_b(a_pos, w_neg, out, scales);
    for (int r = 0; r < 32; ++r)
      for (int c = 0; c < 32; ++c)
        acc[r * 32 + c] -= hmx_gate_decode_uh_signed(out[hmx_tile_idx_32(r, c)]) << bit;

    hmx_int8_raw_dot_tile_b(a_neg, w_pos, out, scales);
    for (int r = 0; r < 32; ++r)
      for (int c = 0; c < 32; ++c)
        acc[r * 32 + c] -= hmx_gate_decode_uh_signed(out[hmx_tile_idx_32(r, c)]) << bit;

    hmx_int8_raw_dot_tile_b(a_neg, w_neg, out, scales);
    for (int r = 0; r < 32; ++r)
      for (int c = 0; c < 32; ++c)
        acc[r * 32 + c] += hmx_gate_decode_uh_signed(out[hmx_tile_idx_32(r, c)]) << bit;
  }
}

static void hmx_int8_tile_bitserial_compute_default(uint8_t *a_pos, uint8_t *a_neg, int8_t *w_pos, int8_t *w_neg,
                                                    uint16_t *out, const __fp16 *scales, int32_t *acc,
                                                    const int8_t *a_ref, const int8_t *w_ref) {
  for (int i = 0; i < 32 * 32; ++i) {
    acc[i] = 0;
  }

  for (int k_base = 0; k_base < 32; k_base += 4) {
    hmx_int8_pack_activation_mag_k4(a_pos, a_ref, k_base, false);
    hmx_int8_pack_activation_mag_k4(a_neg, a_ref, k_base, true);

    for (int bit = 0; bit < 7; ++bit) {
      hmx_int8_pack_weight_b_bit_k4(w_pos, w_ref, k_base, bit, false);
      hmx_int8_pack_weight_b_bit_k4(w_neg, w_ref, k_base, bit, true);

      hmx_int8_raw_dot_tile_b(a_pos, w_pos, out, scales);
      hmx_int8_accumulate_uh_tile(acc, out, 1, bit);

      hmx_int8_raw_dot_tile_b(a_pos, w_neg, out, scales);
      hmx_int8_accumulate_uh_tile(acc, out, -1, bit);

      hmx_int8_raw_dot_tile_b(a_neg, w_pos, out, scales);
      hmx_int8_accumulate_uh_tile(acc, out, -1, bit);

      hmx_int8_raw_dot_tile_b(a_neg, w_neg, out, scales);
      hmx_int8_accumulate_uh_tile(acc, out, 1, bit);
    }
  }
}

static void hmx_int8_tile_full_weight_compute_default(uint8_t *a_pos, uint8_t *a_neg, int8_t *w_full,
                                                      uint16_t *out, const __fp16 *scales, int32_t *acc,
                                                      const int8_t *a_ref, const int8_t *w_ref) {
  for (int i = 0; i < 32 * 32; ++i) {
    acc[i] = 0;
  }

  for (int k_base = 0; k_base < 32; k_base += 4) {
    hmx_int8_pack_activation_mag_k4(a_pos, a_ref, k_base, false);
    hmx_int8_pack_activation_mag_k4(a_neg, a_ref, k_base, true);
    hmx_int8_pack_weight_b_full_k4(w_full, w_ref, k_base);

    hmx_int8_raw_dot_tile_b(a_pos, w_full, out, scales);
    hmx_int8_accumulate_uh_tile_unshifted(acc, out, 1);

    hmx_int8_raw_dot_tile_b(a_neg, w_full, out, scales);
    hmx_int8_accumulate_uh_tile_unshifted(acc, out, -1);
  }
}

static void hmx_int8_tile_full_weight_k2_compute_default(uint8_t *a_pos, uint8_t *a_neg, int8_t *w_full,
                                                         uint16_t *out, const __fp16 *scales, int32_t *acc,
                                                         const int8_t *a_ref, const int8_t *w_ref) {
  for (int i = 0; i < 32 * 32; ++i) {
    acc[i] = 0;
  }

  for (int k_base = 0; k_base < 32; k_base += 2) {
    hmx_int8_pack_activation_mag_k2(a_pos, a_ref, k_base, false);
    hmx_int8_pack_activation_mag_k2(a_neg, a_ref, k_base, true);
    hmx_int8_pack_weight_b_full_k2(w_full, w_ref, k_base);

    hmx_int8_raw_dot_tile_b(a_pos, w_full, out, scales);
    hmx_int8_accumulate_uh_tile_unshifted(acc, out, 1);

    hmx_int8_raw_dot_tile_b(a_neg, w_full, out, scales);
    hmx_int8_accumulate_uh_tile_unshifted(acc, out, -1);
  }
}

static void hmx_int8_fill_search_refs(int8_t *a_ref, int8_t *w_ref, int32_t *cpu_ref) {
  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      a_ref[r * 32 + k] = (int8_t) (((r * 37 + k * 11 + 5) % 255) - 127);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      w_ref[k * 32 + c] = (int8_t) (((k * 19 + c * 23 + 7) % 255) - 127);
    }
  }
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      int32_t sum = 0;
      for (int k = 0; k < 32; ++k) {
        sum += (int32_t) a_ref[r * 32 + k] * (int32_t) w_ref[k * 32 + c];
      }
      cpu_ref[r * 32 + c] = sum;
    }
  }
}

static void hmx_int8_summarize_acc_error(const int32_t *acc, const int32_t *cpu_ref, int *max_abs_err, int *first_bad,
                                         double *sq_err) {
  *max_abs_err = 0;
  *first_bad   = -1;
  *sq_err      = 0.0;
  for (int i = 0; i < 32 * 32; ++i) {
    int err     = acc[i] - cpu_ref[i];
    int abs_err = err < 0 ? -err : err;
    if (abs_err > *max_abs_err) {
      *max_abs_err = abs_err;
      *first_bad   = i;
    }
    *sq_err += (double) err * (double) err;
  }
}

static inline int8_t quantize_probe_f32_to_i8(float value, float scale) {
  if (scale == 0.0f) {
    return 0;
  }
  float scaled = value / scale;
  int q = (int) (scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
  if (q > 127) {
    q = 127;
  } else if (q < -127) {
    q = -127;
  }
  return (int8_t) q;
}

static inline HVX_VectorPair probe_hvx_i8_product_ordered(HVX_Vector weights, int8_t activation) {
  HVX_VectorPair raw = Q6_Wh_vmpy_VbVb(weights, Q6_Vb_vsplat_R(activation));
  return Q6_W_vshuff_VVR(Q6_V_hi_W(raw), Q6_V_lo_W(raw), -2);
}

static void record_probe_lane_map(struct HmxInt8GateResult *res) {
  int8_t  weights[128] __attribute__((aligned(VLEN)));
  int16_t partial[128] __attribute__((aligned(VLEN)));

  for (int c = 0; c < 128; ++c) {
    weights[c] = (int8_t) (c - 64);
  }

  HVX_VectorPair vp = probe_hvx_i8_product_ordered(vmem(weights), 1);
  vmem((HVX_Vector *) partial)       = Q6_V_lo_W(vp);
  vmem(((HVX_Vector *) partial) + 1) = Q6_V_hi_W(vp);

  int first_bad = -1;
  int max_abs_err = 0;
  for (int c = 0; c < 128; ++c) {
    const int err = (int) partial[c] - (int) weights[c];
    const int abs_err = err < 0 ? -err : err;
    if (abs_err > max_abs_err) {
      max_abs_err = abs_err;
      first_bad = c;
    }
  }

  memset(res, 0, sizeof(*res));
  res->reserved = first_bad;
  res->variant = 2502;
  res->output_kind = 0;
  res->tile_bytes = 128;
  res->selector = 1;
  for (int i = 0; i < 8; ++i) {
    res->first8[i] = (float) partial[i];
  }
  res->max_value = (float) max_abs_err;
  res->rmse = (float) max_abs_err;
}

static inline int8_t probe_weight_value(int global_c, int kk) {
  return (int8_t) (((global_c * 11 + kk * 7 + 23) % 255) - 127);
}

static inline float probe_activation_value(int r, int kk) {
  return ((float) (((r * 17 + kk * 13 + 19) % 255) - 127)) / 64.0f;
}

static float w8pc_probe_ref_at(const int8_t *a_q, const float *scale_a, const uint8_t *weight, int m, int k, int n,
                               int r, int c) {
  (void) m;
  const block_q8_0 *blocks  = (const block_q8_0 *) weight;
  const int         n_tile  = c / 32;
  const int         c_tile  = c & 31;
  const float       scale_w = (float) blocks[(n_tile * (k / 32)) * 32 + c_tile].scale;
  int32_t           acc     = 0;
  for (int kk = 0; kk < k; ++kk) {
    const int          k_tile = kk / 32;
    const int          k_in   = kk & 31;
    const block_q8_0  *blk    = &blocks[(n_tile * (k / 32) + k_tile) * 32 + c_tile];
    acc += (int32_t) a_q[r * k + kk] * (int32_t) (int8_t) blk->quants[k_in];
  }
  return (float) acc * scale_a[r] * scale_w;
}

static int record_w8pc_a8pt_probe_case(struct HmxInt8GateResult *res, int selector, int m, int k, int n,
                                        bool exhaustive) {
  const size_t activation_size = (size_t) m * (size_t) k * sizeof(float);
  const size_t output_size     = (size_t) m * (size_t) n * sizeof(float);
  const size_t weight_size     = (size_t) (n / 32) * (size_t) (k / 32) * 32u * sizeof(block_q8_0);
  const size_t aq_size         = (size_t) m * (size_t) k * sizeof(int8_t);
  const size_t scale_a_size    = (size_t) m * sizeof(float);

  float   *activation = nullptr;
  float   *out        = nullptr;
  uint8_t *weight     = nullptr;
  int8_t  *a_q        = nullptr;
  float   *scale_a    = nullptr;

  memset(res, 0, sizeof(*res));
  res->selector   = selector;
  res->variant    = 1502;
  res->tile_bytes = (int32_t) weight_size;
  res->expected   = (float) ((m * 1000000) + (k * 1000) + n);

  if (posix_memalign((void **) &activation, VLEN, activation_size) ||
      posix_memalign((void **) &out, VLEN, output_size) ||
      posix_memalign((void **) &weight, VLEN, weight_size) ||
      posix_memalign((void **) &a_q, VLEN, aq_size) ||
      posix_memalign((void **) &scale_a, VLEN, scale_a_size)) {
    res->output_kind = -10;
    goto cleanup;
  }

  for (int r = 0; r < m; ++r) {
    float max_abs = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      const float v = probe_activation_value(r, kk);
      activation[(size_t) r * (size_t) k + (size_t) kk] = v;
      max_abs = fmaxf(max_abs, fabsf(v));
    }
    scale_a[r] = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
    for (int kk = 0; kk < k; ++kk) {
      a_q[(size_t) r * (size_t) k + (size_t) kk] =
        quantize_probe_f32_to_i8(activation[(size_t) r * (size_t) k + (size_t) kk], scale_a[r]);
    }
  }

  {
    block_q8_0 *blocks = (block_q8_0 *) weight;
    int         block_idx = 0;
    for (int n_tile = 0; n_tile < n / 32; ++n_tile) {
      for (int k_tile = 0; k_tile < k / 32; ++k_tile) {
        for (int c = 0; c < 32; ++c) {
          const int  global_c = n_tile * 32 + c;
          block_q8_0 *blk     = &blocks[block_idx++];
          blk->scale = (__fp16) (0.003f + 0.00001f * (float) global_c);
          for (int kk = 0; kk < 32; ++kk) {
            blk->quants[kk] = (uint8_t) probe_weight_value(global_c, k_tile * 32 + kk);
          }
        }
      }
    }
  }

  memset(out, 0, output_size);
  res->output_kind = hmx_mat_mul_permuted_w8pc_a8pt(out, activation, weight, m, k, n, 0, 0,
                                                    HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT, nullptr);
  if (res->output_kind != 0) {
    goto cleanup;
  }

  {
    int    first_bad = -1;
    int    samples   = 0;
    float  max_err   = 0.0f;
    double sq_err    = 0.0;

    auto check_one = [&](int r, int c) {
      if (r < 0 || r >= m || c < 0 || c >= n) {
        return;
      }
      const int   idx = r * n + c;
      const float ref = w8pc_probe_ref_at(a_q, scale_a, weight, m, k, n, r, c);
      const float err = fabsf(out[idx] - ref);
      if (samples == 0) {
        res->first8[0] = out[idx];
        res->first8[1] = ref;
      } else if (samples == 1) {
        res->first8[2] = out[idx];
        res->first8[3] = ref;
      } else if (samples == 2) {
        res->first8[4] = out[idx];
        res->first8[5] = ref;
      } else if (samples == 3) {
        res->first8[6] = out[idx];
        res->first8[7] = ref;
      }
      if (err > max_err) {
        max_err = err;
        first_bad = idx;
      }
      sq_err += (double) err * (double) err;
      ++samples;
    };

    if (exhaustive) {
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          check_one(r, c);
        }
      }
    } else {
      const int rows[] = { 0, m / 2, m - 1 };
      const int cols[] = { 0, 1, 31, 32, 63, 64, 127, 128, n / 2, n - 129, n - 65, n - 33, n - 32, n - 2, n - 1 };
      for (unsigned ri = 0; ri < sizeof(rows) / sizeof(rows[0]); ++ri) {
        for (unsigned ci = 0; ci < sizeof(cols) / sizeof(cols[0]); ++ci) {
          check_one(rows[ri], cols[ci]);
        }
      }
    }

    res->reserved   = first_bad;
    res->min_value  = (float) samples;
    res->max_value  = max_err;
    res->mean_value = samples > 0 ? (float) sqrt(sq_err / (double) samples) : 0.0f;
    res->rmse       = res->mean_value;
  }

cleanup:
  if (activation) free(activation);
  if (out) free(out);
  if (weight) free(weight);
  if (a_q) free(a_q);
  if (scale_a) free(scale_a);
  return res->output_kind;
}

static int w8pc_a8pt_matmul_probe_run(struct HmxInt8GateResult *results, int max_results, int mode) {
  const int n_results = mode == 20 ? 6 : (mode == 15 ? 2 : 1);
  if (!results || max_results < n_results) {
    return -1;
  }

  if (mode == 15) {
    record_w8pc_a8pt_probe_case(&results[0], 15, 2, 64, 128, true);
    record_probe_lane_map(&results[1]);
    return 0;
  }
  if (mode == 16) {
    return record_w8pc_a8pt_probe_case(&results[0], 16, 1, 1536, 1536, false);
  }
  if (mode == 17) {
    return record_w8pc_a8pt_probe_case(&results[0], 17, 1, 1536, 8960, false);
  }
  if (mode == 18) {
    return record_w8pc_a8pt_probe_case(&results[0], 18, 1, 8960, 1536, false);
  }
  if (mode == 19) {
    return record_w8pc_a8pt_probe_case(&results[0], 19, 33, 1536, 1536, false);
  }

  record_w8pc_a8pt_probe_case(&results[0], 15, 2, 64, 128, true);
  record_probe_lane_map(&results[1]);
  record_w8pc_a8pt_probe_case(&results[2], 16, 1, 1536, 1536, false);
  record_w8pc_a8pt_probe_case(&results[3], 17, 1, 1536, 8960, false);
  record_w8pc_a8pt_probe_case(&results[4], 18, 1, 8960, 1536, false);
  record_w8pc_a8pt_probe_case(&results[5], 19, 33, 1536, 1536, false);
  return 0;
}

static void hmx_int8_bitop_raw_dot_tile(const uint8_t *a, const uint8_t *b, uint16_t *out, const __fp16 *scales) {
  memset(out, 0, HMX_FP16_TILE_SIZE);
  asm volatile("mxclracc" ::: "memory");
  hmx_set_output_scales(scales);
  hmx_load_tiles_ub_bit_weight_variant(a, b, HMX_FP16_TILE_SIZE - 1, 100);
  hmx_store_accumulator_direct((uint8_t *) out, 16);
}

static void hmx_int8_raw_dot_tile_b(const uint8_t *a, const int8_t *b, uint16_t *out, const __fp16 *scales) {
  memset(out, 0, HMX_FP16_TILE_SIZE);
  asm volatile("mxclracc" ::: "memory");
  hmx_set_output_scales(scales);
  hmx_load_tiles_ub_b_variant(a, b, HMX_FP16_TILE_SIZE - 1, 8);
  hmx_store_accumulator_direct((uint8_t *) out, 16);
}

static void hmx_int8_summarize_nonzero_output(const uint16_t *out, int *first_idx, int *changed_count, int *max_value,
                                              int *out0, int *out_hmx00) {
  int first = -1;
  int count = 0;
  int maxv  = 0;
  for (int i = 0; i < HMX_INT8_TILE_N_ELMS; ++i) {
    int got = hmx_gate_decode_uh_signed(out[i]);
    if (got != 0) {
      if (first < 0) {
        first = i;
      }
      ++count;
      if (got > maxv) {
        maxv = got;
      }
    }
  }
  *first_idx     = first;
  *changed_count = count;
  *max_value     = maxv;
  *out0          = hmx_gate_decode_uh_signed(out[0]);
  *out_hmx00     = hmx_gate_decode_uh_signed(out[hmx_tile_idx_32(0, 0)]);
}

static int hmx_int8_byte_probe_run(struct HmxInt8GateResult *results, int max_results) {
  constexpr int n_offsets = HMX_FP16_TILE_SIZE;
  constexpr int n_results = n_offsets * 2;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a      = vtcm;
  uint8_t  *b      = vtcm + 0x1000;
  uint16_t *out    = (uint16_t *) (vtcm + 0x2000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x3000);
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  for (int offset = 0; offset < n_offsets; ++offset) {
    memset(a, 0, HMX_FP16_TILE_SIZE);
    memset(b, 1, HMX_FP16_TILE_SIZE);
    a[offset] = 1;
    hmx_int8_bitop_raw_dot_tile(a, b, out, scales);

    int first_idx;
    int changed_count;
    int max_value;
    int out0;
    int out_hmx00;
    hmx_int8_summarize_nonzero_output(out, &first_idx, &changed_count, &max_value, &out0, &out_hmx00);

    HmxInt8GateResult *ra = &results[result_idx++];
    memset(ra, 0, sizeof(*ra));
    ra->reserved    = 1;
    ra->variant     = 100;
    ra->output_kind = 16;
    ra->tile_bytes  = offset;
    ra->first8[0]   = (float) first_idx;
    ra->first8[1]   = (float) changed_count;
    ra->first8[2]   = (float) max_value;
    ra->first8[3]   = (float) out0;
    ra->first8[4]   = (float) out_hmx00;
    ra->min_value   = (float) first_idx;
    ra->max_value   = (float) max_value;
    ra->mean_value  = (float) changed_count;

    memset(a, 1, HMX_FP16_TILE_SIZE);
    memset(b, 0, HMX_FP16_TILE_SIZE);
    b[offset] = 1;
    hmx_int8_bitop_raw_dot_tile(a, b, out, scales);
    hmx_int8_summarize_nonzero_output(out, &first_idx, &changed_count, &max_value, &out0, &out_hmx00);

    HmxInt8GateResult *rw = &results[result_idx++];
    memset(rw, 0, sizeof(*rw));
    rw->reserved    = 0;
    rw->variant     = 100;
    rw->output_kind = 16;
    rw->tile_bytes  = offset;
    rw->first8[0]   = (float) first_idx;
    rw->first8[1]   = (float) changed_count;
    rw->first8[2]   = (float) max_value;
    rw->first8[3]   = (float) out0;
    rw->first8[4]   = (float) out_hmx00;
    rw->min_value   = (float) first_idx;
    rw->max_value   = (float) max_value;
    rw->mean_value  = (float) changed_count;
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static void hmx_int8_set_offsets(uint8_t *dst, const int *offsets, int n_offsets, uint8_t value) {
  for (int i = 0; i < n_offsets; ++i) {
    int off = offsets[i];
    if (off >= 0 && off < HMX_FP16_TILE_SIZE) {
      dst[off] = value;
    }
  }
}

static int hmx_int8_combo_probe_run(struct HmxInt8GateResult *results, int max_results) {
  struct OffsetMask {
    int n;
    int offsets[16];
  };

  const OffsetMask a_masks[] = {
    {1, {1}},
    {1, {2}},
    {2, {1, 2}},
    {1, {0}},
    {1, {3}},
    {4, {0, 1, 2, 3}},
    {2, {0, 3}},
    {2, {4, 5}},
  };
  const OffsetMask w_masks[] = {
    {1, {0}},
    {1, {1}},
    {1, {2}},
    {1, {3}},
    {2, {0, 1}},
    {2, {2, 3}},
    {2, {0, 2}},
    {2, {1, 3}},
    {4, {0, 1, 2, 3}},
    {1, {128}},
    {2, {128, 129}},
    {2, {130, 131}},
    {4, {128, 129, 130, 131}},
    {8, {0, 1, 2, 3, 128, 129, 130, 131}},
  };

  const int n_a = (int) (sizeof(a_masks) / sizeof(a_masks[0]));
  const int n_w = (int) (sizeof(w_masks) / sizeof(w_masks[0]));
  if (!results || max_results < n_a * n_w) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a      = vtcm;
  int8_t   *b      = (int8_t *) (vtcm + 0x1000);
  uint16_t *out    = (uint16_t *) (vtcm + 0x2000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x3000);
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  for (int ai = 0; ai < n_a; ++ai) {
    for (int wi = 0; wi < n_w; ++wi) {
      memset(a, 0, HMX_FP16_TILE_SIZE);
      memset(b, 0, HMX_FP16_TILE_SIZE);
      hmx_int8_set_offsets(a, a_masks[ai].offsets, a_masks[ai].n, 5);
      hmx_int8_set_offsets((uint8_t *) b, w_masks[wi].offsets, w_masks[wi].n, 1);
      hmx_int8_raw_dot_tile_b(a, b, out, scales);

      int first_idx;
      int changed_count;
      int max_value;
      int out0;
      int out_hmx00;
      hmx_int8_summarize_nonzero_output(out, &first_idx, &changed_count, &max_value, &out0, &out_hmx00);

      HmxInt8GateResult *res = &results[result_idx++];
      memset(res, 0, sizeof(*res));
      res->reserved    = ai * 100 + wi;
      res->variant     = 100;
      res->output_kind = 16;
      res->tile_bytes  = ai;
      res->selector    = wi;
      res->expected    = 5.0f;
      res->first8[0]   = (float) out0;
      res->first8[1]   = (float) out_hmx00;
      res->first8[2]   = (float) first_idx;
      res->first8[3]   = (float) changed_count;
      res->first8[4]   = (float) max_value;
      res->first8[5]   = (float) a_masks[ai].n;
      res->first8[6]   = (float) w_masks[wi].n;
      res->first8[7]   = (float) hmx_gate_decode_uh_signed(out[1]);
      res->min_value   = (float) out0;
      res->max_value   = (float) max_value;
      res->mean_value  = (float) changed_count;
      res->rmse        = fabsf((float) out0 - 5.0f);
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static void hmx_int8_collect_drop_indices(const uint16_t *out, int base_value, int *changed_count, int *indices,
                                          int n_indices) {
  int count = 0;
  for (int i = 0; i < HMX_INT8_TILE_N_ELMS; ++i) {
    int got = hmx_gate_decode_uh_signed(out[i]);
    if (got != base_value) {
      if (count < n_indices) {
        indices[count] = i;
      }
      ++count;
    }
  }
  *changed_count = count;
  for (int i = count; i < n_indices; ++i) {
    indices[i] = -1;
  }
}

static int hmx_int8_drop_probe_run(struct HmxInt8GateResult *results, int max_results) {
  constexpr int n_offsets = HMX_FP16_TILE_SIZE;
  constexpr int n_results = n_offsets * 2;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a      = vtcm;
  uint8_t  *b      = vtcm + 0x1000;
  uint16_t *out    = (uint16_t *) (vtcm + 0x2000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x3000);
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  for (int offset = 0; offset < n_offsets; ++offset) {
    int indices[6];
    int changed_count;

    memset(a, 1, HMX_FP16_TILE_SIZE);
    memset(b, 1, HMX_FP16_TILE_SIZE);
    b[offset] = 0;
    hmx_int8_bitop_raw_dot_tile(a, b, out, scales);
    hmx_int8_collect_drop_indices(out, 32, &changed_count, indices, 6);

    HmxInt8GateResult *rw = &results[result_idx++];
    memset(rw, 0, sizeof(*rw));
    rw->reserved    = 0;
    rw->variant     = 100;
    rw->output_kind = 16;
    rw->tile_bytes  = offset;
    rw->first8[0]   = (float) (32 - hmx_gate_decode_uh_signed(out[0]));
    rw->first8[1]   = (float) changed_count;
    for (int i = 0; i < 6; ++i) {
      rw->first8[i + 2] = (float) indices[i];
    }
    rw->mean_value = (float) changed_count;

    memset(a, 1, HMX_FP16_TILE_SIZE);
    memset(b, 1, HMX_FP16_TILE_SIZE);
    a[offset] = 0;
    hmx_int8_bitop_raw_dot_tile(a, b, out, scales);
    hmx_int8_collect_drop_indices(out, 32, &changed_count, indices, 6);

    HmxInt8GateResult *ra = &results[result_idx++];
    memset(ra, 0, sizeof(*ra));
    ra->reserved    = 1;
    ra->variant     = 100;
    ra->output_kind = 16;
    ra->tile_bytes  = offset;
    ra->first8[0]   = (float) (32 - hmx_gate_decode_uh_signed(out[0]));
    ra->first8[1]   = (float) changed_count;
    for (int i = 0; i < 6; ++i) {
      ra->first8[i + 2] = (float) indices[i];
    }
    ra->mean_value = (float) changed_count;
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static int hmx_int8_pack_search_run(struct HmxInt8GateResult *results, int max_results) {
  constexpr int n_masks   = 15;
  constexpr int n_results = n_masks * n_masks;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a_pos  = vtcm;
  uint8_t  *a_neg  = vtcm + 0x1000;
  int8_t   *w_pos  = (int8_t *) (vtcm + 0x2000);
  int8_t   *w_neg  = (int8_t *) (vtcm + 0x3000);
  uint16_t *out    = (uint16_t *) (vtcm + 0x4000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x5000);
  int32_t  *acc    = (int32_t *) (vtcm + 0x6000);

  int8_t  a_ref[32 * 32];
  int8_t  w_ref[32 * 32];
  int32_t cpu_ref[32 * 32];
  hmx_int8_fill_search_refs(a_ref, w_ref, cpu_ref);

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  for (int a_mask = 1; a_mask <= n_masks; ++a_mask) {
    for (int w_mask = 1; w_mask <= n_masks; ++w_mask) {
      hmx_int8_tile_bitserial_compute(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref, a_mask, w_mask);

      int    max_abs_err;
      int    first_bad;
      double sq_err;
      hmx_int8_summarize_acc_error(acc, cpu_ref, &max_abs_err, &first_bad, &sq_err);

      HmxInt8GateResult *res = &results[result_idx++];
      memset(res, 0, sizeof(*res));
      res->reserved    = a_mask;
      res->selector    = w_mask;
      res->variant     = 8;
      res->output_kind = 16;
      res->tile_bytes  = HMX_FP16_TILE_SIZE;
      res->expected    = 0.0f;
      res->first8[0]   = (float) max_abs_err;
      res->first8[1]   = (float) first_bad;
      res->first8[2]   = first_bad >= 0 ? (float) cpu_ref[first_bad] : 0.0f;
      res->first8[3]   = first_bad >= 0 ? (float) acc[first_bad] : 0.0f;
      res->first8[4]   = (float) acc[0];
      res->first8[5]   = (float) cpu_ref[0];
      res->first8[6]   = (float) acc[33];
      res->first8[7]   = (float) cpu_ref[33];
      res->min_value   = (float) max_abs_err;
      res->max_value   = (float) max_abs_err;
      res->mean_value  = (float) first_bad;
      res->rmse        = (float) sqrt(sq_err / (32.0 * 32.0));
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static int hmx_int8_sparse_map_run(struct HmxInt8GateResult *results, int max_results) {
  constexpr int n_results = 32 * 32 + 32;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a_pos  = vtcm;
  uint8_t  *a_neg  = vtcm + 0x1000;
  int8_t   *w_pos  = (int8_t *) (vtcm + 0x2000);
  int8_t   *w_neg  = (int8_t *) (vtcm + 0x3000);
  uint16_t *out    = (uint16_t *) (vtcm + 0x4000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x5000);
  int32_t  *acc    = (int32_t *) (vtcm + 0x6000);

  int8_t a_ref[32 * 32];
  int8_t w_ref[32 * 32];

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  for (int phase = 0; phase < 2; ++phase) {
    const int phase_cases = phase == 0 ? 32 * 32 : 32;
    for (int i = 0; i < phase_cases; ++i) {
      const int r = phase == 0 ? (i / 32) : 0;
      const int k = phase == 0 ? 0 : i;
      const int c = phase == 0 ? (i % 32) : 0;

      memset(a_ref, 0, sizeof(a_ref));
      memset(w_ref, 0, sizeof(w_ref));
      a_ref[r * 32 + k] = 5;
      w_ref[k * 32 + c] = 7;

      hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);

      int first_nonzero = -1;
      int first_value   = 0;
      int nonzero_count = 0;
      int max_abs       = 0;
      for (int idx = 0; idx < 32 * 32; ++idx) {
        const int v   = acc[idx];
        const int abs = v < 0 ? -v : v;
        if (v != 0) {
          if (first_nonzero < 0) {
            first_nonzero = idx;
            first_value   = v;
          }
          ++nonzero_count;
        }
        if (abs > max_abs) {
          max_abs = abs;
        }
      }

      HmxInt8GateResult *res = &results[result_idx++];
      memset(res, 0, sizeof(*res));
      res->selector    = k;
      res->variant     = 8;
      res->output_kind = 16;
      res->tile_bytes  = c;
      res->reserved    = phase * 10000 + r;
      res->expected    = 35.0f;
      res->first8[0]   = (float) r;
      res->first8[1]   = (float) k;
      res->first8[2]   = (float) c;
      res->first8[3]   = 35.0f;
      res->first8[4]   = (float) acc[r * 32 + c];
      res->first8[5]   = (float) first_nonzero;
      res->first8[6]   = (float) nonzero_count;
      res->first8[7]   = (float) first_value;
      res->min_value   = (float) max_abs;
      res->max_value   = (float) max_abs;
      res->mean_value  = (float) first_nonzero;
      res->rmse        = fabsf((float) (acc[r * 32 + c] - 35));
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static void hmx_int8_record_scalar_result(HmxInt8GateResult *res, int case_id, int expected, int actual, int r, int k,
                                          int c) {
  memset(res, 0, sizeof(*res));
  const int err    = actual - expected;
  const int abs_er = err < 0 ? -err : err;
  res->reserved    = case_id;
  res->variant     = 12;
  res->output_kind = 16;
  res->expected    = (float) expected;
  res->first8[0]   = (float) actual;
  res->first8[1]   = (float) err;
  res->first8[2]   = (float) r;
  res->first8[3]   = (float) k;
  res->first8[4]   = (float) c;
  res->min_value   = (float) abs_er;
  res->max_value   = (float) abs_er;
  res->rmse        = (float) abs_er;
}

static void hmx_int8_record_tile_result(HmxInt8GateResult *res, int case_id, const int32_t *acc,
                                        const int32_t *cpu_ref) {
  int max_abs_err = 0;
  int first_bad   = -1;
  double sq_err   = 0.0;
  hmx_int8_summarize_acc_error(acc, cpu_ref, &max_abs_err, &first_bad, &sq_err);

  memset(res, 0, sizeof(*res));
  res->reserved    = case_id;
  res->variant     = 12;
  res->output_kind = 16;
  res->first8[0]   = (float) max_abs_err;
  res->first8[1]   = (float) first_bad;
  res->first8[2]   = first_bad >= 0 ? (float) cpu_ref[first_bad] : 0.0f;
  res->first8[3]   = first_bad >= 0 ? (float) acc[first_bad] : 0.0f;
  res->first8[4]   = (float) acc[0];
  res->first8[5]   = (float) cpu_ref[0];
  res->first8[6]   = (float) acc[33];
  res->first8[7]   = (float) cpu_ref[33];
  res->min_value   = (float) max_abs_err;
  res->max_value   = (float) max_abs_err;
  res->mean_value  = (float) first_bad;
  res->rmse        = (float) sqrt(sq_err / (32.0 * 32.0));
}

static void hmx_int8_compute_cpu_ref(const int8_t *a_ref, const int8_t *w_ref, int32_t *cpu_ref) {
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      int32_t sum = 0;
      for (int k = 0; k < 32; ++k) {
        sum += (int32_t) a_ref[r * 32 + k] * (int32_t) w_ref[k * 32 + c];
      }
      cpu_ref[r * 32 + c] = sum;
    }
  }
}

static int hmx_int8_linearity_probe_run(struct HmxInt8GateResult *results, int max_results) {
  if (!results || max_results < 128) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a_pos  = vtcm;
  uint8_t  *a_neg  = vtcm + 0x1000;
  int8_t   *w_pos  = (int8_t *) (vtcm + 0x2000);
  int8_t   *w_neg  = (int8_t *) (vtcm + 0x3000);
  uint16_t *out    = (uint16_t *) (vtcm + 0x4000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x5000);
  int32_t  *acc    = (int32_t *) (vtcm + 0x6000);

  int8_t a_ref[32 * 32];
  int8_t w_ref[32 * 32];
  int32_t cpu_ref[32 * 32];

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;

  const int signs[4][2] = {{5, 7}, {-5, 7}, {5, -7}, {-5, -7}};
  for (int i = 0; i < 4; ++i) {
    memset(a_ref, 0, sizeof(a_ref));
    memset(w_ref, 0, sizeof(w_ref));
    a_ref[0] = (int8_t) signs[i][0];
    w_ref[0] = (int8_t) signs[i][1];
    hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);
    hmx_int8_record_scalar_result(&results[result_idx++], i, signs[i][0] * signs[i][1], acc[0], 0, 0, 0);
  }

  const uint8_t a_values[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255};
  for (int lane = 0; lane < 4; ++lane) {
    for (int vi = 0; vi < (int) (sizeof(a_values) / sizeof(a_values[0])); ++vi) {
      hmx_int8_pack_activation_raw_lane(a_pos, 0, 0, lane, a_values[vi]);
      memset(a_neg, 0, HMX_FP16_TILE_SIZE);
      hmx_int8_pack_weight_b_raw_pair(w_pos, 0, 0, 1);
      memset(w_neg, 0, HMX_FP16_TILE_SIZE);
      memset(out, 0, HMX_FP16_TILE_SIZE);
      asm volatile("mxclracc" ::: "memory");
      hmx_set_output_scales(scales);
      hmx_load_tiles_ub_b_variant(a_pos, w_pos, HMX_FP16_TILE_SIZE - 1, 8);
      hmx_store_accumulator_direct((uint8_t *) out, 16);
      int first_idx;
      int changed_count;
      int max_value;
      int out0;
      int out_hmx00;
      hmx_int8_summarize_nonzero_output(out, &first_idx, &changed_count, &max_value, &out0, &out_hmx00);

      HmxInt8GateResult *res = &results[result_idx++];
      memset(res, 0, sizeof(*res));
      res->reserved    = 400 + lane * 100 + vi;
      res->variant     = 12;
      res->output_kind = 16;
      res->tile_bytes  = lane;
      res->selector    = a_values[vi];
      res->first8[0]   = (float) hmx_gate_decode_uh_signed(out[hmx_tile_idx_32(0, 0)]);
      res->first8[1]   = (float) first_idx;
      res->first8[2]   = (float) changed_count;
      res->first8[3]   = (float) max_value;
      res->first8[4]   = (float) out0;
      res->first8[5]   = (float) out_hmx00;
      res->min_value   = (float) out0;
      res->max_value   = (float) max_value;
      res->mean_value  = (float) changed_count;
    }
  }

  for (int k2 = 1; k2 < 32; ++k2) {
    memset(a_ref, 0, sizeof(a_ref));
    memset(w_ref, 0, sizeof(w_ref));
    a_ref[0]      = 5;
    w_ref[0]      = 7;
    a_ref[k2]     = 5;
    w_ref[k2 * 32] = 7;
    hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);
    hmx_int8_record_scalar_result(&results[result_idx++], 100 + k2, 70, acc[0], 0, k2, 0);
  }

  memset(a_ref, 0, sizeof(a_ref));
  memset(w_ref, 0, sizeof(w_ref));
  for (int k = 0; k < 32; ++k) {
    a_ref[k]      = 5;
    w_ref[k * 32] = 7;
  }
  hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_scalar_result(&results[result_idx++], 200, 32 * 35, acc[0], 0, 31, 0);

  memset(a_ref, 0, sizeof(a_ref));
  memset(w_ref, 0, sizeof(w_ref));
  for (int k = 0; k < 32; ++k) {
    a_ref[k]      = (k & 1) ? -5 : 5;
    w_ref[k * 32] = 7;
  }
  hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_scalar_result(&results[result_idx++], 201, 0, acc[0], 0, 31, 0);

  for (int i = 0; i < 32 * 32; ++i) {
    a_ref[i] = 1;
    w_ref[i] = 1;
  }
  for (int i = 0; i < 32 * 32; ++i) {
    cpu_ref[i] = 32;
  }
  hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 300, acc, cpu_ref);

  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      a_ref[r * 32 + k] = (int8_t) (((r + 2 * k) % 3) - 1);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      w_ref[k * 32 + c] = (int8_t) (((2 * k + c) % 3) - 1);
    }
  }
  hmx_int8_compute_cpu_ref(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 301, acc, cpu_ref);

  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      a_ref[r * 32 + k] = (int8_t) ((r + k) & 3);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      w_ref[k * 32 + c] = (int8_t) ((k + 2 * c) & 3);
    }
  }
  hmx_int8_compute_cpu_ref(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 302, acc, cpu_ref);

  hmx_unit_release();
  hmx_manager_disable_execution();

  while (result_idx < max_results) {
    memset(&results[result_idx++], 0, sizeof(*results));
  }
  return 0;
}

static int hmx_int8_kalign_probe_run(struct HmxInt8GateResult *results, int max_results) {
  constexpr int n_a_offsets = 16 * 16;
  constexpr int n_w_offsets = 16 * 4;
  constexpr int n_results   = n_a_offsets * n_w_offsets;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a      = vtcm;
  int8_t   *b      = (int8_t *) (vtcm + 0x1000);
  uint16_t *out    = (uint16_t *) (vtcm + 0x2000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x3000);
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  for (int av = 0; av < 16; ++av) {
    for (int alane = 0; alane < 16; ++alane) {
      const int a_off = av * 128 + alane;
      for (int wv = 0; wv < 16; ++wv) {
        for (int wlane = 0; wlane < 4; ++wlane) {
          const int w_off = wv * 128 + wlane;
          memset(a, 0, HMX_FP16_TILE_SIZE);
          memset(b, 0, HMX_FP16_TILE_SIZE);
          a[a_off] = 2;
          b[w_off] = 1;

          hmx_int8_raw_dot_tile_b(a, b, out, scales);

          int first_idx;
          int changed_count;
          int max_value;
          int out0;
          int out_hmx00;
          hmx_int8_summarize_nonzero_output(out, &first_idx, &changed_count, &max_value, &out0, &out_hmx00);

          HmxInt8GateResult *res = &results[result_idx++];
          memset(res, 0, sizeof(*res));
          res->reserved    = a_off;
          res->variant     = 8;
          res->output_kind = 16;
          res->tile_bytes  = w_off;
          res->selector    = av * 100 + alane;
          res->expected    = 1.0f;
          res->first8[0]   = (float) out0;
          res->first8[1]   = (float) out_hmx00;
          res->first8[2]   = (float) first_idx;
          res->first8[3]   = (float) changed_count;
          res->first8[4]   = (float) max_value;
          res->first8[5]   = (float) wv;
          res->first8[6]   = (float) wlane;
          res->first8[7]   = (float) alane;
          res->min_value   = (float) out0;
          res->max_value   = (float) max_value;
          res->mean_value  = (float) changed_count;
          res->rmse        = fabsf((float) out0 - 1.0f);
        }
      }
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static int hmx_int8_tile_gate_run(struct HmxInt8GateResult *results, int max_results) {
  if (!results || max_results < 1) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a_pos  = vtcm;
  uint8_t  *a_neg  = vtcm + 0x1000;
  int8_t   *w_pos  = (int8_t *) (vtcm + 0x2000);
  int8_t   *w_neg  = (int8_t *) (vtcm + 0x3000);
  uint16_t *out    = (uint16_t *) (vtcm + 0x4000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x5000);
  int32_t  *acc    = (int32_t *) (vtcm + 0x6000);

  int8_t a_ref[32 * 32];
  int8_t w_ref[32 * 32];
  int32_t cpu_ref[32 * 32];

  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      a_ref[r * 32 + k] = (int8_t) (((r * 37 + k * 11 + 5) % 255) - 127);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      w_ref[k * 32 + c] = (int8_t) (((k * 19 + c * 23 + 7) % 255) - 127);
    }
  }
  for (int i = 0; i < 32 * 32; ++i) {
    acc[i] = 0;
  }
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      int32_t sum = 0;
      for (int k = 0; k < 32; ++k) {
        sum += (int32_t) a_ref[r * 32 + k] * (int32_t) w_ref[k * 32 + c];
      }
      cpu_ref[r * 32 + c] = sum;
    }
  }

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
  hmx_manager_enable_execution();
  hmx_unit_acquire();

  hmx_int8_tile_bitserial_compute_default(a_pos, a_neg, w_pos, w_neg, out, scales, acc, a_ref, w_ref);

  hmx_unit_release();
  hmx_manager_disable_execution();

  int max_abs_err = 0;
  int first_bad   = -1;
  double sq_err   = 0.0;
  for (int i = 0; i < 32 * 32; ++i) {
    int err = acc[i] - cpu_ref[i];
    int abs_err = err < 0 ? -err : err;
    if (abs_err > max_abs_err) {
      max_abs_err = abs_err;
      first_bad = i;
    }
    sq_err += (double) err * (double) err;
  }

  HmxInt8GateResult *res = &results[0];
  memset(res, 0, sizeof(*res));
  res->selector    = 0;
  res->variant     = 100;
  res->output_kind = 16;
  res->tile_bytes  = HMX_FP16_TILE_SIZE;
  res->reserved    = 5;
  res->expected    = 0.0f;
  res->first8[0]   = (float) max_abs_err;
  res->first8[1]   = (float) first_bad;
  res->first8[2]   = first_bad >= 0 ? (float) cpu_ref[first_bad] : 0.0f;
  res->first8[3]   = first_bad >= 0 ? (float) acc[first_bad] : 0.0f;
  res->first8[4]   = (float) acc[0];
  res->first8[5]   = (float) cpu_ref[0];
  res->first8[6]   = (float) acc[33];
  res->first8[7]   = (float) cpu_ref[33];
  res->min_value   = (float) max_abs_err;
  res->max_value   = (float) max_abs_err;
  res->mean_value  = (float) first_bad;
  res->rmse        = (float) sqrt(sq_err / (32.0 * 32.0));
  return 0;
}

static int hmx_int8_full_weight_probe_run(struct HmxInt8GateResult *results, int max_results) {
  if (!results || max_results < 4) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a_pos   = vtcm;
  uint8_t  *a_neg   = vtcm + 0x1000;
  int8_t   *w_full  = (int8_t *) (vtcm + 0x2000);
  uint16_t *out     = (uint16_t *) (vtcm + 0x3000);
  __fp16   *scales  = (__fp16 *) (vtcm + 0x4000);
  int8_t   *a_ref   = (int8_t *) (vtcm + 0x5000);
  int8_t   *w_ref   = (int8_t *) (vtcm + 0x6000);
  int32_t  *cpu_ref = (int32_t *) (vtcm + 0x7000);
  int32_t  *acc     = (int32_t *) (vtcm + 0x9000);

  hmx_gate_init_scales(scales, 0);
  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;

  hmx_int8_fill_search_refs(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_full_weight_compute_default(a_pos, a_neg, w_full, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 1300, acc, cpu_ref);

  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      a_ref[r * 32 + k] = (int8_t) (((r * 17 + k * 29 + 3) % 255) - 127);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      w_ref[k * 32 + c] = (int8_t) (((k * 43 + c * 7 + 19) % 255) - 127);
    }
  }
  hmx_int8_compute_cpu_ref(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_full_weight_compute_default(a_pos, a_neg, w_full, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 1301, acc, cpu_ref);

  memset(a_ref, 0, 32 * 32);
  memset(w_ref, 0, 32 * 32);
  a_ref[0] = 127;
  w_ref[0] = -127;
  hmx_int8_compute_cpu_ref(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_full_weight_compute_default(a_pos, a_neg, w_full, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 1302, acc, cpu_ref);

  memset(a_ref, 0, 32 * 32);
  memset(w_ref, 0, 32 * 32);
  for (int k = 0; k < 32; ++k) {
    a_ref[k]       = (k & 1) ? -13 : 17;
    w_ref[k * 32] = (k % 3 == 0) ? -11 : 9;
  }
  hmx_int8_compute_cpu_ref(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_full_weight_compute_default(a_pos, a_neg, w_full, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 1303, acc, cpu_ref);

  hmx_unit_release();
  hmx_manager_disable_execution();

  while (result_idx < max_results) {
    memset(&results[result_idx++], 0, sizeof(HmxInt8GateResult));
  }
  return 0;
}

static int hmx_int8_full_weight_k2_probe_run(struct HmxInt8GateResult *results, int max_results) {
  if (!results || max_results < 4) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a_pos   = vtcm;
  uint8_t  *a_neg   = vtcm + 0x1000;
  int8_t   *w_full  = (int8_t *) (vtcm + 0x2000);
  uint16_t *out     = (uint16_t *) (vtcm + 0x3000);
  __fp16   *scales  = (__fp16 *) (vtcm + 0x4000);
  int8_t   *a_ref   = (int8_t *) (vtcm + 0x5000);
  int8_t   *w_ref   = (int8_t *) (vtcm + 0x6000);
  int32_t  *cpu_ref = (int32_t *) (vtcm + 0x7000);
  int32_t  *acc     = (int32_t *) (vtcm + 0x9000);

  hmx_gate_init_scales(scales, 0);
  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;

  hmx_int8_fill_search_refs(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_full_weight_k2_compute_default(a_pos, a_neg, w_full, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 1400, acc, cpu_ref);

  for (int r = 0; r < 32; ++r) {
    for (int k = 0; k < 32; ++k) {
      a_ref[r * 32 + k] = (int8_t) (((r * 17 + k * 29 + 3) % 255) - 127);
    }
  }
  for (int k = 0; k < 32; ++k) {
    for (int c = 0; c < 32; ++c) {
      w_ref[k * 32 + c] = (int8_t) (((k * 43 + c * 7 + 19) % 255) - 127);
    }
  }
  hmx_int8_compute_cpu_ref(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_full_weight_k2_compute_default(a_pos, a_neg, w_full, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 1401, acc, cpu_ref);

  memset(a_ref, 0, 32 * 32);
  memset(w_ref, 0, 32 * 32);
  a_ref[0] = 127;
  a_ref[1] = 127;
  w_ref[0] = 127;
  w_ref[32] = 127;
  hmx_int8_compute_cpu_ref(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_full_weight_k2_compute_default(a_pos, a_neg, w_full, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 1402, acc, cpu_ref);

  memset(a_ref, 0, 32 * 32);
  memset(w_ref, 0, 32 * 32);
  for (int k = 0; k < 32; ++k) {
    a_ref[k]       = (k & 1) ? -13 : 17;
    w_ref[k * 32] = (k % 3 == 0) ? -11 : 9;
  }
  hmx_int8_compute_cpu_ref(a_ref, w_ref, cpu_ref);
  hmx_int8_tile_full_weight_k2_compute_default(a_pos, a_neg, w_full, out, scales, acc, a_ref, w_ref);
  hmx_int8_record_tile_result(&results[result_idx++], 1403, acc, cpu_ref);

  hmx_unit_release();
  hmx_manager_disable_execution();

  while (result_idx < max_results) {
    memset(&results[result_idx++], 0, sizeof(HmxInt8GateResult));
  }
  return 0;
}

static int hmx_int8_raw_dot_first(uint8_t *a, int8_t *b, uint16_t *out, const __fp16 *scales) {
  memset(out, 0, HMX_FP16_TILE_SIZE);

  asm volatile("mxclracc" ::: "memory");
  hmx_set_output_scales(scales);
  hmx_load_tiles_ub_b_variant(a, b, HMX_FP16_TILE_SIZE - 1, 8);
  hmx_store_accumulator_direct((uint8_t *) out, 16);
  return hmx_gate_decode_uh_signed(out[0]);
}

static int hmx_int8_bitplane_partial(uint8_t *a, int8_t *b, uint16_t *out, const __fp16 *scales, uint8_t a_bit,
                                     int8_t w_value) {
  memset(a, a_bit, HMX_FP16_TILE_SIZE);
  memset(b, w_value, HMX_FP16_TILE_SIZE);

  return hmx_int8_raw_dot_first(a, b, out, scales);
}

static int hmx_int8_bitplane_gate_run(struct HmxInt8GateResult *results, int max_results) {
  struct BitplaneCase {
    int8_t a;
    int8_t w;
  };

  const BitplaneCase cases[] = {
    {1, 1},
    {2, 3},
    {-1, 1},
    {1, -1},
    {1, 63},
    {1, 64},
    {1, 126},
    {1, 127},
    {1, -127},
    {-2, 5},
    {127, 127},
    {-127, 127},
    {64, 127},
    {-64, -127},
  };
  const int n_case_results    = (int) (sizeof(cases) / sizeof(cases[0]));
  const int n_pattern_results = 5 * 8 * 2;
  const int n_results         = n_case_results + n_pattern_results;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a   = vtcm;
  int8_t   *b   = (int8_t *) (vtcm + 0x1000);
  uint16_t *out = (uint16_t *) (vtcm + 0x2000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x3000);
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  for (int case_idx = 0; case_idx < n_case_results; ++case_idx) {
    const int8_t a_value = cases[case_idx].a;
    const int8_t w_value = cases[case_idx].w;
    const int    sign    = a_value < 0 ? -1 : 1;
    const int    mag     = a_value < 0 ? -a_value : a_value;

    int reconstructed = 0;
    int max_abs_part  = 0;
    int direct_raw    = hmx_int8_bitplane_partial(a, b, out, scales, (uint8_t) mag, w_value);
    for (int bit = 0; bit < 7; ++bit) {
      if ((mag & (1 << bit)) == 0) {
        continue;
      }
      const int part = hmx_int8_bitplane_partial(a, b, out, scales, 1, w_value);
      const int abs_part = part < 0 ? -part : part;
      max_abs_part       = max_abs_part > abs_part ? max_abs_part : abs_part;
      reconstructed += sign * (part << bit);
    }

    const int w_sign = w_value < 0 ? -1 : 1;
    const int w_mag  = w_value < 0 ? -w_value : w_value;
    int bitserial_reconstructed = 0;
    for (int abit = 0; abit < 7; ++abit) {
      if ((mag & (1 << abit)) == 0) continue;
      for (int wbit = 0; wbit < 7; ++wbit) {
        if ((w_mag & (1 << wbit)) == 0) continue;
        const int part = hmx_int8_bitplane_partial(a, b, out, scales, 1, 1);
        bitserial_reconstructed += sign * w_sign * (part << (abit + wbit));
      }
    }

    const int expected = 32 * (int) a_value * (int) w_value;

    HmxInt8GateResult *res = &results[result_idx++];
    memset(res, 0, sizeof(*res));
    res->selector    = 0;
    res->variant     = 8;
    res->output_kind = 16;
    res->tile_bytes  = HMX_FP16_TILE_SIZE;
    res->reserved    = case_idx;
    res->expected    = (float) expected;
    res->first8[0]   = (float) reconstructed;
    res->first8[1]   = (float) direct_raw;
    res->first8[2]   = (float) max_abs_part;
    res->first8[3]   = (float) a_value;
    res->first8[4]   = (float) w_value;
    res->first8[5]   = (float) bitserial_reconstructed;
    res->first8[6]   = fabsf((float) (bitserial_reconstructed - expected));
    res->min_value   = (float) reconstructed;
    res->max_value   = (float) reconstructed;
    res->mean_value  = (float) reconstructed;
    res->rmse        = fabsf((float) (reconstructed - expected));
  }

  auto fill_activation_pattern = [](uint8_t *dst, int mode, uint8_t value) {
    memset(dst, 0, HMX_FP16_TILE_SIZE);
    if (mode == 0) {
      memset(dst, value, HMX_FP16_TILE_SIZE);
    } else if (mode == 1) {
      for (int i = 1; i < 128; i += 4) dst[i] = value;
    } else if (mode == 2) {
      for (int i = 2; i < 128; i += 4) dst[i] = value;
    } else if (mode == 3) {
      for (int i = 1; i < 128; i += 4) {
        dst[i]     = value;
        dst[i + 1] = value;
      }
    } else if (mode == 4) {
      for (int i = 0; i < 128; ++i) dst[i] = value;
    }
  };

  auto fill_weight_pattern = [](int8_t *dst, int mode, int8_t value) {
    memset(dst, 0, HMX_FP16_TILE_SIZE);
    if (mode == 0) {
      memset(dst, value, HMX_FP16_TILE_SIZE);
    } else {
      for (int g = 0; g < 16; ++g) {
        int base = 128 * g;
        if (mode >= 1 && mode <= 4) {
          dst[base + mode - 1] = value;
        } else if (mode == 5) {
          dst[base + 0] = value;
          dst[base + 1] = value;
        } else if (mode == 6) {
          dst[base + 2] = value;
          dst[base + 3] = value;
        } else if (mode == 7) {
          dst[base + 0] = value;
          dst[base + 1] = value;
          dst[base + 2] = value;
          dst[base + 3] = value;
        }
      }
    }
  };

  for (int value_idx = 0; value_idx < 2; ++value_idx) {
    const int8_t w_value = value_idx == 0 ? 1 : 127;
    for (int a_mode = 0; a_mode < 5; ++a_mode) {
      for (int w_mode = 0; w_mode < 8; ++w_mode) {
        fill_activation_pattern(a, a_mode, 1);
        fill_weight_pattern(b, w_mode, w_value);
        int got = hmx_int8_raw_dot_first(a, b, out, scales);
        int expected = value_idx == 0 ? 32 : 4064;

        HmxInt8GateResult *res = &results[result_idx++];
        memset(res, 0, sizeof(*res));
        res->selector    = 0;
        res->variant     = 8;
        res->output_kind = 16;
        res->tile_bytes  = HMX_FP16_TILE_SIZE;
        res->reserved    = 1000 + value_idx * 100 + a_mode * 10 + w_mode;
        res->expected    = (float) expected;
        res->first8[0]   = (float) got;
        res->first8[1]   = (float) a_mode;
        res->first8[2]   = (float) w_mode;
        res->first8[3]   = (float) w_value;
        res->min_value   = (float) got;
        res->max_value   = (float) got;
        res->mean_value  = (float) got;
        res->rmse        = fabsf((float) (got - expected));
      }
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

static int hmx_int8_layout_gate_run(struct HmxInt8GateResult *results, int max_results) {
  constexpr int n_offsets = HMX_FP16_TILE_SIZE;
  constexpr int n_results = n_offsets * 2;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t  *a      = vtcm;
  int8_t   *b      = (int8_t *) (vtcm + 0x1000);
  uint16_t *out    = (uint16_t *) (vtcm + 0x2000);
  __fp16   *scales = (__fp16 *) (vtcm + 0x3000);
  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  auto summarize_drop_output = [](const uint16_t *out, int base0, float *max_drop, float *first_idx,
                                  float *changed_count) {
    int max_abs_drop = 0;
    int max_drop_val = 0;
    int first        = -1;
    int count        = 0;
    for (int i = 0; i < HMX_INT8_TILE_N_ELMS; ++i) {
      int got  = hmx_gate_decode_uh_signed(out[i]);
      int drop = base0 - got;
      if (drop != 0) {
        if (first < 0) {
          first = i;
        }
        ++count;
      }
      int abs_drop = drop < 0 ? -drop : drop;
      if (abs_drop > max_abs_drop) {
        max_abs_drop = abs_drop;
        max_drop_val = drop;
      }
    }
    *max_drop      = (float) max_drop_val;
    *first_idx     = (float) first;
    *changed_count = (float) count;
  };

  for (int offset = 0; offset < n_offsets; ++offset) {
    memset(a, 1, HMX_FP16_TILE_SIZE);
    memset(b, 1, HMX_FP16_TILE_SIZE);
    b[offset] = 0;
    int   weight_drop = 32 - hmx_int8_raw_dot_first(a, b, out, scales);
    float weight_max_drop;
    float weight_first_idx;
    float weight_changed_count;
    summarize_drop_output(out, 32, &weight_max_drop, &weight_first_idx, &weight_changed_count);

    HmxInt8GateResult *rw = &results[result_idx++];
    memset(rw, 0, sizeof(*rw));
    rw->reserved    = 0;
    rw->variant     = 8;
    rw->output_kind = 16;
    rw->tile_bytes  = offset;
    rw->expected    = 0.0f;
    rw->first8[0]   = (float) weight_drop;
    rw->first8[1]   = weight_max_drop;
    rw->first8[2]   = weight_first_idx;
    rw->first8[3]   = weight_changed_count;
    rw->mean_value  = (float) weight_drop;
    rw->rmse        = fabsf((float) weight_drop);

    memset(a, 1, HMX_FP16_TILE_SIZE);
    memset(b, 1, HMX_FP16_TILE_SIZE);
    a[offset] = 0;
    int   activation_drop = 32 - hmx_int8_raw_dot_first(a, b, out, scales);
    float activation_max_drop;
    float activation_first_idx;
    float activation_changed_count;
    summarize_drop_output(out, 32, &activation_max_drop, &activation_first_idx, &activation_changed_count);

    HmxInt8GateResult *ra = &results[result_idx++];
    memset(ra, 0, sizeof(*ra));
    ra->reserved    = 1;
    ra->variant     = 8;
    ra->output_kind = 16;
    ra->tile_bytes  = offset;
    ra->expected    = 0.0f;
    ra->first8[0]   = (float) activation_drop;
    ra->first8[1]   = activation_max_drop;
    ra->first8[2]   = activation_first_idx;
    ra->first8[3]   = activation_changed_count;
    ra->mean_value  = (float) activation_drop;
    ra->rmse        = fabsf((float) activation_drop);
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

extern "C" int hmx_int8_gate_run(struct HmxInt8GateResult *results, int max_results, int mode) {
  if (mode == 21) {
    return qk_os_debug_probe_run(results, max_results, mode);
  }
  if (mode >= 15 && mode <= 20) {
    return w8pc_a8pt_matmul_probe_run(results, max_results, mode);
  }
  if (mode == 14) {
    return hmx_int8_full_weight_k2_probe_run(results, max_results);
  }
  if (mode == 13) {
    return hmx_int8_full_weight_probe_run(results, max_results);
  }
  if (mode == 12) {
    return hmx_int8_linearity_probe_run(results, max_results);
  }

  if (mode == 11) {
    return hmx_int8_kalign_probe_run(results, max_results);
  }
  if (mode == 10) {
    return hmx_int8_sparse_map_run(results, max_results);
  }
  if (mode == 9) {
    return hmx_int8_pack_search_run(results, max_results);
  }
  if (mode == 8) {
    return hmx_int8_drop_probe_run(results, max_results);
  }
  if (mode == 7) {
    return hmx_int8_combo_probe_run(results, max_results);
  }
  if (mode == 6) {
    return hmx_int8_byte_probe_run(results, max_results);
  }
  if (mode == 5) {
    return hmx_int8_tile_gate_run(results, max_results);
  }
  if (mode == 4) {
    return hmx_int8_bitop_gate_run(results, max_results);
  }
  if (mode == 3) {
    return hmx_int8_layout_gate_run(results, max_results);
  }
  if (mode == 2) {
    return hmx_int8_bitplane_gate_run(results, max_results);
  }
  if (mode == 1) {
    return hmx_int8_gate_search_run(results, max_results);
  }

  constexpr int n_variants    = 12;
  constexpr int n_output_kind = 19;
  constexpr int n_results     = n_variants * 2 * n_output_kind + 6 * 2 * 5;
  if (!results || max_results < n_results) {
    return -1;
  }

  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  uint8_t *a      = vtcm;
  int8_t  *b      = (int8_t *) (vtcm + 0x1000);
  __fp16  *out    = (__fp16 *) (vtcm + 0x2000);
  __fp16  *scales = (__fp16 *) (vtcm + 0x3000);

  memset(out, 0, HMX_FP16_TILE_SIZE);
  memset(scales, 0, 256);

  hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

  hmx_manager_enable_execution();
  hmx_unit_acquire();

  int result_idx = 0;
  float expected = hmx_gate_fill_inputs(a, b, 0);
  for (int variant = 0; variant < n_variants; ++variant) {
    for (int tile_bytes_idx = 0; tile_bytes_idx < 2; ++tile_bytes_idx) {
      for (int output_kind = 0; output_kind < n_output_kind; ++output_kind) {
        const int tile_bytes = tile_bytes_idx == 0 ? HMX_INT8_TILE_SIZE : HMX_FP16_TILE_SIZE;
        hmx_int8_gate_run_one(&results[result_idx++], out, scales, a, b, 0, variant, tile_bytes, output_kind,
                              output_kind <= 3 ? output_kind : 0,
                              expected);
      }
    }
  }

  const int focused_output_kinds[5] = {0, 7, 8, 16, 18};
  for (int input_case = 1; input_case <= 6; ++input_case) {
    expected = hmx_gate_fill_inputs(a, b, input_case);
    for (int variant = 8; variant <= 9; ++variant) {
      for (int i = 0; i < 5; ++i) {
        hmx_int8_gate_run_one(&results[result_idx++], out, scales, a, b, input_case, variant, HMX_FP16_TILE_SIZE,
                              focused_output_kinds[i], 0, expected);
      }
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  return 0;
}

void test_hmx_int8_gate() {
  struct HmxInt8GateResult results[32];
  int                      ret = hmx_int8_gate_run(results, 32, 0);

  FARF(ALWAYS, "HMX_INT8_GATE start arch=%d tile_bytes=%d ret=%d", __HVX_ARCH__, HMX_INT8_TILE_SIZE, ret);
  for (int i = 0; i < 32 && ret == 0; ++i) {
    const struct HmxInt8GateResult *res = &results[i];
    FARF(ALWAYS,
         "HMX_INT8_GATE selector=%d expected=%g first8=%g,%g,%g,%g,%g,%g,%g,%g min=%g max=%g mean=%g rmse=%g "
         "nan=%d",
         res->selector, res->expected, res->first8[0], res->first8[1], res->first8[2], res->first8[3], res->first8[4],
         res->first8[5], res->first8[6], res->first8[7], res->min_value, res->max_value, res->mean_value, res->rmse,
         res->nan_count);
  }
}

void test_int16_fp16_conversion() {
#if __HVX_ARCH__ < 73
  FARF(ALWAYS, "HVX native h <-> hf conversion not supported");
  return;
#endif

  static __fp16  input[64];
  static int16_t output[64];

  for (int i = 0; i < 64; ++i) {
    float x  = i * 0.25 - 8;
    input[i] = (__fp16) x;
  }

  vmemu(output) = Q6_Vh_equals_Vhf(vmemu(input));

  for (int i = 0; i < 64; ++i) {
    FARF(ALWAYS, "%s: x=%g y=%d", __func__, (float) input[i], output[i]);
  }
}

void test_fp16_exp2() {
  int    n    = 256;
  size_t size = n * sizeof(__fp16);

  __fp16 *input = nullptr;
  posix_memalign((void **) &input, VLEN, size);

  __fp16 *output = nullptr;
  posix_memalign((void **) &output, VLEN, size);

  __fp16 *output_ref = new __fp16[n];

  for (int i = 0; i < n; ++i) {
    float x       = -0.1 * i;
    input[i]      = (__fp16) x;
    output_ref[i] = (__fp16) exp2f(x);
  }

  auto in_vecs  = (HVX_Vector *) input;
  auto out_vecs = (HVX_Vector *) output;
  for (int i = 0; i < n / 64; ++i) {
    out_vecs[i] = hvx_my_exp2_vhf(in_vecs[i]);
  }

  for (int i = 0; i < n; ++i) {
    float x  = (float) input[i];
    float y0 = (float) output_ref[i];
    float y1 = (float) output[i];
    FARF(ALWAYS, "%s: i=%d, x=%g, my: %g, ref: %g", __func__, i, x, y1, y0);
  }

  delete[] output_ref;
  free(input);
  free(output);
}

void benchmark_hmx_gemm() {
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  __fp16 *a = (__fp16 *) vtcm;
  __fp16 *b = (__fp16 *) (vtcm + 2 * 0x100000);
  __fp16 *c = (__fp16 *) (vtcm + 4 * 0x100000);
  __fp16 *s = (__fp16 *) (vtcm + 6 * 0x100000);

  int n_repeat = 1000;
  int sizes[]  = { 32, 64, 128, 256, 512, 1024 };

  hmx_manager_enable_execution();
  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hmx_mat_mul_fp16_core(c, a, b, s, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core fp16 hmx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }
  hmx_manager_disable_execution();
}

void benchmark_hvx_gemm() {
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  __fp16 *a = (__fp16 *) vtcm;
  __fp16 *b = (__fp16 *) (vtcm + 2 * 0x100000);
  __fp16 *c = (__fp16 *) (vtcm + 4 * 0x100000);

  int n_repeat = 10;
  // int sizes[]  = { 32, 64, 128, 256, 512, 1024 };

  /*
  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_fp16_core(c, a, b, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core fp16 hvx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }

  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_fp32_core((float *) c, (float *) a, (float *) b, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core fp32 hvx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }

  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_int16_core((int16_t *) c, (int16_t *) a, (int16_t *) b, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core int16 hvx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }

  for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
    int64_t n = sizes[i];

    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_int32_core((int32_t *) c, (int32_t *) a, (int32_t *) b, n, n, n);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core int32 hvx: %.2lf GFLOPS@n=%lld, %lld us", __func__, gflops, n, elapsed_us);
  }
  */

  int64_t n = 1024;
  for (int i = 1; i <= 4; i *= 2) {
    int64_t t0 = HAP_perf_get_qtimer_count();
    for (int t = 0; t < n_repeat; ++t) {
      hvx_mat_mul_fp16_core_mt(c, a, b, n, n, n, i);
    }
    int64_t t1         = HAP_perf_get_qtimer_count();
    int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

    double gflops = 1e-3 * n_repeat * (2 * n * n * n) / elapsed_us;
    FARF(ALWAYS, "%s: core fp16 hvx: %.2lf GFLOPS@%d Threads, %lld us", __func__, gflops, i, elapsed_us);
  }
}

void benchmark_vtcm_bandwidth() {
  uint8_t *vtcm = (uint8_t *) vtcm_manager_get_vtcm_base();

  HVX_Vector *a = (HVX_Vector *) vtcm;
  HVX_Vector *b = (HVX_Vector *) (vtcm + 0x400000);

  size_t size = 0x100000;

  int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < size / VLEN; ++i) {
    // HVX_Vector v0 = *a++;
    // HVX_Vector v1 = *a++;
    // HVX_Vector v2 = *a++;
    // *b++ = v0;
    // *b++ = v1;
    // *b++ = v2;

    // 4 packets
    // asm volatile (
    //   "{ v0.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v0 }\n"
    //   "{ v1.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v1 }\n"
    //   "vmem(%1++#1) = v0\n"
    //   "vmem(%1++#1) = v1\n"
    //   :"+r"(a), "+r"(b)::"v0", "v1"
    // );

    // 3 packets
    // asm volatile(
    //   "{ v0.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v0 }\n"
    //   "{ v1.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v1 }\n"
    //   "{ v2.cur = vmem(%0++#1)\n"
    //   " vmem(%1++#1) = v2 }\n"
    //   : "+r"(a), "+r"(b)::"v0", "v1", "v2");

    asm volatile(
      "{ v0.cur = vmem(%0++#1)\n"
      " vmem(%1++#1) = v0 }\n"
      "{ v1.cur = vmem(%0++#1)\n"
      " vmem(%1++#1) = v1 }\n"
      "{ v2.cur = vmem(%0++#1)\n"
      " vmem(%1++#1) = v2 }\n"
      "{ v3.cur = vmem(%0++#1)\n"
      " vmem(%1++#1) = v3 }\n"
      : "+r"(a), "+r"(b)::"v0", "v1", "v2", "v3");
  }
  int64_t t1         = HAP_perf_get_qtimer_count();
  int64_t elapsed_us = HAP_perf_qtimer_count_to_us(t1 - t0);

  const int rf       = 4;
  const int tf       = 8;
  double    read_bw  = 1e-3 * rf * size / elapsed_us;
  double    total_bw = 1e-3 * tf * size / elapsed_us;

  FARF(ALWAYS, "%s: %lld us, read bw: %.2lf GB/s, total bw: %.2lf GB/s", __func__, elapsed_us, read_bw, total_bw);
}

}  // namespace internal

extern "C" {

void internal_op_tests();

void internal_op_tests() {
  using namespace internal;

  test_hmx_int8_gate();

  // test_int16_fp16_conversion();
  // test_fp16_exp2();

  // benchmark_hmx_gemm();
  // benchmark_hvx_gemm();
  // benchmark_vtcm_bandwidth();
}
}
