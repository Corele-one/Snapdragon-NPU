#include <math.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "host/session.h"
#include "htp_ops.h"  // auto-generated
#include "message.h"
#include "op_reg.h"

static inline int64_t get_time_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000L + ts.tv_nsec / 1000;
}

static inline int align_up(size_t size, size_t align) {
  return (size + align - 1) / align * align;
}

static inline int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

static const char *figure8_component_name(int component) {
  switch (component) {
    case FIGURE8_COMP_Q_LOAD:
      return "q_load";
    case FIGURE8_COMP_K_LOAD:
      return "k_load";
    case FIGURE8_COMP_V_LOAD:
      return "v_load";
    case FIGURE8_COMP_QK_DOT:
      return "qk_dot";
    case FIGURE8_COMP_SAFE_SM:
      return "safe_sm";
    case FIGURE8_COMP_CORE_ACC:
      return "core_acc";
    case FIGURE8_COMP_O_SCALE:
      return "o_scale";
    case FIGURE8_COMP_O_STORE:
      return "o_store";
    case FIGURE8_COMP_SCNA_EXP:
      return "scna_exp";
    default:
      return "unknown";
  }
}

static inline double rand_01() {
  return ((double) rand()) / RAND_MAX;
}

int  alloc_shared_mem_buf(void **p_buf, int *p_fd, size_t size);
void free_shared_mem_buf(void *buf, int fd, size_t size);

struct Figure8AttnConfig {
  int         enabled;
  const char *mode;
  const char *scna_layout;
  const char *mask_mode;
  int         qo_len;
  int         kv_len;
  int         n_heads;
  int         n_kv_heads;
  int         head_dim;
  int         warmup;
  int         iters;
  int         print_events;
  int         compare_reference;
  const char *csv_out;
  int         hmx_int8_gate;
  int         hmx_int8_gate_search;
  int         hmx_int8_bitplane_gate;
  int         hmx_int8_layout_gate;
  int         hmx_int8_bitop_gate;
  int         hmx_int8_tile_gate;
  int         hmx_int8_byte_probe;
  int         hmx_int8_combo_probe;
  int         hmx_int8_drop_probe;
  int         hmx_int8_pack_search;
  int         hmx_int8_sparse_map;
  int         hmx_int8_kalign_probe;
  int         hmx_int8_linearity_probe;
  int         hmx_int8_full_weight_probe;
  int         hmx_int8_full_weight_k2_probe;
  int         w8pc_a8pt_matmul_probe;
  int         hmx_int8_mode;
  int         roofline_fp16_bench;
  int         roofline_bandwidth_bench;
  int         roofline_mix_precision_bench;
  int         roofline_int8_shape_bench;
  int         roofline_signed_int8_zero_overhead_bench;
  int         roofline_case;
  int         bench_bytes;
  int         scna_width;
  int         scna_exp_bench;
};

static void figure8_print_usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --figure8-attn [--mode baseline|lut-exp|scna-fp16] [--scna-layout serial|lane8]\n"
          "          [--scna-width 8|16|32] [--mask-mode full|causal|padding] [--qo-len N] [--kv-len N]\n"
          "          [--n-heads N] [--n-kv-heads N] [--head-dim N] [--warmup N] [--iters N]\n"
          "          [--no-events] [--compare-reference] [--csv-out PATH]\n"
          "       %s --hmx-int8-gate\n"
          "       %s --hmx-int8-search-gate\n"
          "       %s --hmx-int8-bitplane-gate\n"
          "       %s --hmx-int8-layout-gate\n"
          "       %s --hmx-int8-bitop-gate\n"
          "       %s --hmx-int8-tile-gate\n"
          "       %s --hmx-int8-byte-probe\n"
          "       %s --hmx-int8-combo-probe\n"
          "       %s --hmx-int8-drop-probe\n"
          "       %s --hmx-int8-pack-search\n"
          "       %s --hmx-int8-sparse-map\n"
          "       %s --hmx-int8-kalign-probe\n"
          "       %s --hmx-int8-linearity-probe\n"
          "       %s --hmx-int8-full-weight-probe\n"
          "       %s --hmx-int8-full-weight-k2-probe\n"
          "       %s --w8pc-a8pt-matmul-probe\n"
          "       %s --hmx-int8-mode N  (N=21 runs the QK output-stationary debug probe)\n"
          "       %s --roofline-fp16-bench [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --roofline-bandwidth-bench [--bench-bytes N] [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --roofline-mix-precision-bench [--roofline-case N] [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --roofline-int8-shape-bench [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --roofline-signed-int8-zero-overhead-bench [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --scna-exp-bench [--scna-layout serial|lane8] [--scna-width 8|16|32] [--warmup N] [--iters N]\n",
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog,
          prog);
}

static int parse_int_cli_value(const char *name, const char *value, int *out) {
  char *end = NULL;
  long  v   = strtol(value, &end, 10);
  if (!value[0] || *end != '\0' || v <= 0 || v > INT32_MAX) {
    fprintf(stderr, "Invalid integer for %s: %s\n", name, value);
    return -1;
  }
  *out = (int) v;
  return 0;
}

static int parse_figure8_args(int argc, char **argv, struct Figure8AttnConfig *cfg) {
  *cfg = (struct Figure8AttnConfig) {
    .enabled    = 0,
    .mode       = "baseline",
    .scna_layout = "serial",
    .mask_mode  = "full",
    .qo_len     = 4,
    .kv_len     = 4096,
    .n_heads    = 12,
    .n_kv_heads = 2,
    .head_dim   = 128,
    .warmup     = 5,
    .iters      = 20,
    .print_events = 1,
    .compare_reference = 0,
    .csv_out    = NULL,
    .hmx_int8_gate = 0,
    .hmx_int8_gate_search = 0,
    .hmx_int8_bitplane_gate = 0,
    .hmx_int8_layout_gate = 0,
    .hmx_int8_bitop_gate = 0,
    .hmx_int8_tile_gate = 0,
    .hmx_int8_byte_probe = 0,
    .hmx_int8_combo_probe = 0,
    .hmx_int8_drop_probe = 0,
    .hmx_int8_pack_search = 0,
    .hmx_int8_sparse_map = 0,
    .hmx_int8_kalign_probe = 0,
    .hmx_int8_linearity_probe = 0,
    .hmx_int8_full_weight_probe = 0,
    .hmx_int8_full_weight_k2_probe = 0,
    .w8pc_a8pt_matmul_probe = 0,
    .hmx_int8_mode = -1,
    .roofline_fp16_bench = 0,
    .roofline_bandwidth_bench = 0,
    .roofline_mix_precision_bench = 0,
    .roofline_int8_shape_bench = 0,
    .roofline_signed_int8_zero_overhead_bench = 0,
    .roofline_case = 0,
    .bench_bytes = 64 * 1024 * 1024,
    .scna_width = 8,
    .scna_exp_bench = 0,
  };

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--figure8-attn") == 0) {
      cfg->enabled = 1;
    } else if (strcmp(arg, "--scna-exp-bench") == 0 || strcmp(arg, "--scna-exp2-bench") == 0) {
      cfg->scna_exp_bench = 1;
    } else if (strcmp(arg, "--hmx-int8-gate") == 0) {
      cfg->hmx_int8_gate = 1;
    } else if (strcmp(arg, "--hmx-int8-search-gate") == 0) {
      cfg->hmx_int8_gate_search = 1;
    } else if (strcmp(arg, "--hmx-int8-bitplane-gate") == 0) {
      cfg->hmx_int8_bitplane_gate = 1;
    } else if (strcmp(arg, "--hmx-int8-layout-gate") == 0) {
      cfg->hmx_int8_layout_gate = 1;
    } else if (strcmp(arg, "--hmx-int8-bitop-gate") == 0) {
      cfg->hmx_int8_bitop_gate = 1;
    } else if (strcmp(arg, "--hmx-int8-tile-gate") == 0) {
      cfg->hmx_int8_tile_gate = 1;
    } else if (strcmp(arg, "--hmx-int8-byte-probe") == 0) {
      cfg->hmx_int8_byte_probe = 1;
    } else if (strcmp(arg, "--hmx-int8-combo-probe") == 0) {
      cfg->hmx_int8_combo_probe = 1;
    } else if (strcmp(arg, "--hmx-int8-drop-probe") == 0) {
      cfg->hmx_int8_drop_probe = 1;
    } else if (strcmp(arg, "--hmx-int8-pack-search") == 0) {
      cfg->hmx_int8_pack_search = 1;
    } else if (strcmp(arg, "--hmx-int8-sparse-map") == 0) {
      cfg->hmx_int8_sparse_map = 1;
    } else if (strcmp(arg, "--hmx-int8-kalign-probe") == 0) {
      cfg->hmx_int8_kalign_probe = 1;
    } else if (strcmp(arg, "--hmx-int8-linearity-probe") == 0) {
      cfg->hmx_int8_linearity_probe = 1;
    } else if (strcmp(arg, "--hmx-int8-full-weight-probe") == 0) {
      cfg->hmx_int8_full_weight_probe = 1;
    } else if (strcmp(arg, "--hmx-int8-full-weight-k2-probe") == 0) {
      cfg->hmx_int8_full_weight_k2_probe = 1;
    } else if (strcmp(arg, "--w8pc-a8pt-matmul-probe") == 0) {
      cfg->w8pc_a8pt_matmul_probe = 1;
    } else if (strcmp(arg, "--hmx-int8-mode") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->hmx_int8_mode)) return -1;
    } else if (strcmp(arg, "--roofline-fp16-bench") == 0) {
      cfg->roofline_fp16_bench = 1;
    } else if (strcmp(arg, "--roofline-bandwidth-bench") == 0) {
      cfg->roofline_bandwidth_bench = 1;
    } else if (strcmp(arg, "--roofline-mix-precision-bench") == 0) {
      cfg->roofline_mix_precision_bench = 1;
    } else if (strcmp(arg, "--roofline-int8-shape-bench") == 0) {
      cfg->roofline_int8_shape_bench = 1;
    } else if (strcmp(arg, "--roofline-signed-int8-zero-overhead-bench") == 0) {
      cfg->roofline_signed_int8_zero_overhead_bench = 1;
    } else if (strcmp(arg, "--roofline-case") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->roofline_case)) return -1;
    } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      figure8_print_usage(argv[0]);
      return 1;
    } else if (strcmp(arg, "--mode") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --mode\n");
        return -1;
      }
      cfg->mode = argv[i];
    } else if (strcmp(arg, "--scna-layout") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --scna-layout\n");
        return -1;
      }
      cfg->scna_layout = argv[i];
    } else if (strcmp(arg, "--mask-mode") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --mask-mode\n");
        return -1;
      }
      cfg->mask_mode = argv[i];
    } else if (strcmp(arg, "--scna-width") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->scna_width)) return -1;
    } else if (strcmp(arg, "--qo-len") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->qo_len)) return -1;
    } else if (strcmp(arg, "--kv-len") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->kv_len)) return -1;
    } else if (strcmp(arg, "--n-heads") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->n_heads)) return -1;
    } else if (strcmp(arg, "--n-kv-heads") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->n_kv_heads)) return -1;
    } else if (strcmp(arg, "--head-dim") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->head_dim)) return -1;
    } else if (strcmp(arg, "--warmup") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->warmup)) return -1;
    } else if (strcmp(arg, "--iters") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->iters)) return -1;
    } else if (strcmp(arg, "--bench-bytes") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->bench_bytes)) return -1;
    } else if (strcmp(arg, "--no-events") == 0) {
      cfg->print_events = 0;
    } else if (strcmp(arg, "--compare-reference") == 0) {
      cfg->compare_reference = 1;
    } else if (strcmp(arg, "--csv-out") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --csv-out\n");
        return -1;
      }
      cfg->csv_out = argv[i];
    } else {
      fprintf(stderr, "Unknown argument: %s\n", arg);
      return -1;
    }
  }

  if (!cfg->enabled && !cfg->roofline_fp16_bench && !cfg->roofline_bandwidth_bench &&
      !cfg->roofline_mix_precision_bench && !cfg->roofline_int8_shape_bench &&
      !cfg->roofline_signed_int8_zero_overhead_bench && !cfg->scna_exp_bench) {
    return 0;
  }
  if (!cfg->enabled && !cfg->scna_exp_bench) {
    return 0;
  }
  if (strcmp(cfg->mode, "baseline") != 0 && strcmp(cfg->mode, "lut-exp") != 0 &&
      strcmp(cfg->mode, "scna-fp16") != 0) {
    fprintf(stderr, "Unsupported --mode: %s\n", cfg->mode);
    return -1;
  }
  if (strcmp(cfg->scna_layout, "serial") != 0 && strcmp(cfg->scna_layout, "lane8") != 0) {
    fprintf(stderr, "--scna-layout must be serial or lane8\n");
    return -1;
  }
  if (strcmp(cfg->mask_mode, "full") != 0 && strcmp(cfg->mask_mode, "causal") != 0 &&
      strcmp(cfg->mask_mode, "padding") != 0) {
    fprintf(stderr, "--mask-mode must be full, causal, or padding\n");
    return -1;
  }
  if (cfg->scna_width != 8) {
    fprintf(stderr, "Only d8 is enabled before the measured d16/d32 expansion gate\n");
    return -1;
  }
  if (cfg->n_heads % cfg->n_kv_heads != 0) {
    fprintf(stderr, "n_heads must be divisible by n_kv_heads\n");
    return -1;
  }
  return 0;
}

static void figure8_fill_inputs(float *q, __fp16 *k, __fp16 *v, __fp16 *mask, int qo_len, int kv_len, int kv_pad_len,
                                int n_heads, int n_kv_heads, int head_dim, const char *mask_mode) {
  const size_t q_elems    = (size_t) qo_len * n_heads * head_dim;
  const size_t kv_elems   = (size_t) kv_len * n_kv_heads * head_dim;
  const size_t mask_elems = (size_t) qo_len * kv_pad_len;

  for (size_t i = 0; i < q_elems; ++i) {
    int val = (int) ((i * 13u + 7u) % 251u) - 125;
    q[i]    = (float) val * 0.0078125f;
  }
  for (size_t i = 0; i < kv_elems; ++i) {
    int kval = (int) ((i * 17u + 3u) % 257u) - 128;
    int vval = (int) ((i * 19u + 5u) % 263u) - 131;
    k[i]     = (__fp16) ((float) kval * 0.00390625f);
    v[i]     = (__fp16) ((float) vval * 0.00390625f);
  }
  for (size_t i = 0; i < mask_elems; ++i) {
    mask[i] = (__fp16) -65504.0f;
  }
  for (int q_idx = 0; q_idx < qo_len; ++q_idx) {
    for (int k_idx = 0; k_idx < kv_len; ++k_idx) {
      int masked = 0;
      if (strcmp(mask_mode, "causal") == 0) {
        const int absolute_q = kv_len - qo_len + q_idx;
        masked = k_idx > absolute_q;
      } else if (strcmp(mask_mode, "padding") == 0) {
        masked = k_idx >= kv_len - 3;
      }
      mask[(size_t) q_idx * kv_pad_len + k_idx] = (__fp16) (masked ? -65504.0f : 0.0f);
    }
  }
}

static uint64_t figure8_checksum(const float *data, size_t count) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    memcpy(&bits, &data[i], sizeof(bits));
    hash ^= bits;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static int figure8_compute_reference(float *output, float *scores, const float *q, const __fp16 *k,
                                     const __fp16 *v, const __fp16 *mask, int qo_len, int kv_len, int n_heads,
                                     int n_kv_heads, int head_dim) {
  if (!output || !scores || !q || !k || !v || n_heads % n_kv_heads != 0) return -1;
  const int gqa_factor = n_heads / n_kv_heads;
  const size_t qo_stride = (size_t) n_heads * head_dim;
  const size_t kv_stride = (size_t) n_kv_heads * head_dim;
  const size_t kv_pad_len = align_up((size_t) kv_len, 64);
  const float qk_scale = 1.4426950408889634f / sqrtf((float) head_dim);
  for (int q_idx = 0; q_idx < qo_len; ++q_idx) {
    for (int h = 0; h < n_heads; ++h) {
      const int kv_h = h / gqa_factor;
      const float *q_row = q + (size_t) q_idx * qo_stride + (size_t) h * head_dim;
      float row_max = -INFINITY;
      for (int c = 0; c < kv_len; ++c) {
        const __fp16 *k_row = k + (size_t) c * kv_stride + (size_t) kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_row[d] * (float) k_row[d];
        float score = dot * qk_scale + (mask ? (float) mask[(size_t) q_idx * kv_pad_len + c] : 0.0f);
        scores[c] = score;
        if (score > row_max) row_max = score;
      }
      float row_sum = 0.0f;
      for (int c = 0; c < kv_len; ++c) {
        scores[c] = exp2f(scores[c] - row_max);
        row_sum += scores[c];
      }
      if (!(row_sum > 0.0f) || !isfinite(row_sum)) return -1;
      float *o_row = output + (size_t) q_idx * qo_stride + (size_t) h * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        float value = 0.0f;
        for (int c = 0; c < kv_len; ++c) {
          const __fp16 *v_row = v + (size_t) c * kv_stride + (size_t) kv_h * head_dim;
          value += scores[c] * (float) v_row[d];
        }
        o_row[d] = value / row_sum;
      }
    }
  }
  return 0;
}

static void figure8_report_comparison(const struct Figure8AttnConfig *cfg, const float *candidate,
                                      const float *reference, size_t count) {
  double sq_error = 0.0, reference_sq = 0.0;
  float max_abs = 0.0f;
  int candidate_nonfinite = 0, reference_nonfinite = 0;
  for (size_t i = 0; i < count; ++i) {
    const float candidate_value = candidate[i];
    if (!isfinite(candidate_value)) ++candidate_nonfinite;
    if (!isfinite(reference[i])) ++reference_nonfinite;
    if (!isfinite(candidate_value) || !isfinite(reference[i])) continue;
    const float error = candidate_value - reference[i];
    sq_error += (double) error * error;
    reference_sq += (double) reference[i] * reference[i];
    if (fabsf(error) > max_abs) max_abs = fabsf(error);
  }
  const float rmse = (float) sqrt(sq_error / count);
  const float relative_l2 = reference_sq > 0.0 ? (float) sqrt(sq_error / reference_sq) : 0.0f;
  const int pass = rmse <= 0.002f && max_abs <= 0.01f && candidate_nonfinite == 0 && reference_nonfinite == 0;
  fprintf(stderr,
          "FIG8_ATTENTION_COMPARE candidate_mode=%s layout=%s scna_width=%d reference_mode=host-fp32 "
          "mask_mode=%s qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d elements=%zu "
          "rmse=%.9g relative_l2=%.9g max_abs_error=%.9g candidate_nonfinite=%d reference_nonfinite=%d "
          "rmse_limit=0.002 max_abs_limit=0.01 pass=%d\n",
          cfg->mode, cfg->scna_layout, cfg->scna_width, cfg->mask_mode, cfg->qo_len, cfg->kv_len, cfg->n_heads,
          cfg->n_kv_heads, cfg->head_dim, count, rmse, relative_l2, max_abs, candidate_nonfinite,
          reference_nonfinite, pass);
}

static int figure8_wait_channel(struct MessageHeader *msg, int64_t timeout_us) {
  const int64_t t0 = get_time_us();
  while (msg->state.v[1] != 1) {
    if (get_time_us() - t0 > timeout_us) {
      fprintf(stderr, "Timed out waiting for DSP message reply\n");
      return -1;
    }
    usleep(50);
  }
  return 0;
}

static int figure8_send_attn_request(struct MessageHeader *msg, size_t max_msg_size,
                                     const struct FlashAttnProfileParams *params) {
  struct RequestHeader req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_OP_COMPUTE,
  };
  struct OpComputeRequest compute_req = {
    .op = HTP_OPS_FLASH_ATTN_PROFILE_QO_F32_KV_F16,
  };

  msg->state.d        = 0;
  msg->n_reqs         = 1;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + sizeof(req_hdr) + sizeof(compute_req) + sizeof(*params);
  if ((size_t) msg->req_offsets[1] > max_msg_size) {
    fprintf(stderr, "Figure 8 message buffer is too small\n");
    return -1;
  }

  uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(req_hdr);
  *(struct OpComputeRequest *) p = compute_req;
  p += sizeof(compute_req);
  *(struct FlashAttnProfileParams *) p = *params;

  const int64_t t0 = get_time_us();
  msg->state.v[0]  = 1;
  if (figure8_wait_channel(msg, 30000000)) {
    return -1;
  }
  int64_t elapsed_us = get_time_us() - t0;

  int err = message_header_get_request_ptr(msg, 0)->state;
  if (err) {
    fprintf(stderr, "Figure 8 attention request failed with 0x%x after %ld us\n", err, elapsed_us);
    return err;
  }
  return 0;
}

static int figure8_release_dsp_maps(struct MessageHeader *msg, size_t max_msg_size, const int *fds, int n_fds) {
  struct RequestHeader req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_RPCMEM_MAP,
  };
  struct RpcmemMapRequest map_req = {
    .n_puts = n_fds,
    .n_gets = 0,
  };

  msg->state.d        = 0;
  msg->n_reqs         = 1;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + sizeof(req_hdr) + sizeof(map_req) + n_fds * sizeof(int);
  if ((size_t) msg->req_offsets[1] > max_msg_size) {
    fprintf(stderr, "Figure 8 map-release message buffer is too small\n");
    return -1;
  }

  uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(req_hdr);
  *(struct RpcmemMapRequest *) p = map_req;
  p += sizeof(map_req);
  for (int i = 0; i < n_fds; ++i) {
    *(int *) p = fds[i];
    p += sizeof(int);
  }

  msg->state.v[0] = 1;
  if (figure8_wait_channel(msg, 30000000)) {
    return -1;
  }
  return message_header_get_request_ptr(msg, 0)->state;
}

static int scna_exp_bench_send_request(struct MessageHeader *msg, size_t max_msg_size,
                                       const struct ScnaExp2BenchParams *params) {
  struct RequestHeader req_hdr = { .state = 0, .type = REQUEST_TYPE_OP_COMPUTE };
  struct OpComputeRequest compute_req = { .op = HTP_OPS_SCNA_EXP2_BENCH };
  msg->state.d = 0;
  msg->n_reqs = 1;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + sizeof(req_hdr) + sizeof(compute_req) + sizeof(*params);
  if ((size_t) msg->req_offsets[1] > max_msg_size) {
    return -1;
  }
  uint8_t *p = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(req_hdr);
  *(struct OpComputeRequest *) p = compute_req;
  p += sizeof(compute_req);
  *(struct ScnaExp2BenchParams *) p = *params;
  msg->state.v[0] = 1;
  if (figure8_wait_channel(msg, 30000000)) {
    return -1;
  }
  return message_header_get_request_ptr(msg, 0)->state;
}

static int run_scna_exp_benchmark(const struct Figure8AttnConfig *cfg) {
  struct ScnaExp2BenchResult *result = NULL;
  void *chan = NULL;
  int result_fd = -1, chan_fd = -1, ret = 1;
  const size_t result_size = align_up(sizeof(*result), 128);
  const size_t max_msg_size = 4096;
  if (alloc_shared_mem_buf((void **) &result, &result_fd, result_size) ||
      alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size)) {
    goto end;
  }
  memset(result, 0, result_size);
  if (create_htp_message_channel(chan_fd, max_msg_size)) {
    goto end;
  }
  struct ScnaExp2BenchParams params = {
    .output = { .fd = result_fd, .offset = 0 },
    .width = cfg->scna_width,
    .layout = strcmp(cfg->scna_layout, "lane8") == 0 ? SCNA_LAYOUT_LANE8 : SCNA_LAYOUT_SERIAL,
    .warmup = cfg->warmup,
    .iters = cfg->iters,
  };
  struct MessageHeader *msg = (struct MessageHeader *) chan;
  int err = scna_exp_bench_send_request(msg, max_msg_size, &params);
  if (err) {
    fprintf(stderr, "SCNA Exp2 bench failed: 0x%x\n", err);
    goto end;
  }
  const double single_ns = result->iters > 0 ? 1000.0 * result->elapsed_us / result->iters : 0.0;
  const double paired_ns = result->iters > 0 ? 1000.0 * result->pair_elapsed_us / result->iters : 0.0;
  fprintf(stderr,
          "SCNA_EXP_BENCH layout=%s width=%d lanes=%d warmup=%d iters=%d elapsed_us=%lld "
          "single_ns_per_64=%.6f pair_elapsed_us=%lld paired_ns_per_2x64=%.6f paired_ns_per_64=%.6f "
          "rmse=%.9g max_abs=%.9g dense_samples=%d dense_rmse=%.9g dense_max_abs=%.9g "
          "pair_max_abs_diff=%.9g monotonic_violations=%d negative_count=%d nan_count=%d "
          "lane_oracle_mismatches=%d checksum=0x%08x\n",
          cfg->scna_layout, result->width, result->lanes, cfg->warmup, result->iters,
          (long long) result->elapsed_us, single_ns, (long long) result->pair_elapsed_us, paired_ns,
          paired_ns * 0.5, result->rmse, result->max_abs_error, result->dense_samples, result->dense_rmse,
          result->dense_max_abs_error, result->pair_max_abs_diff, result->monotonic_violations,
          result->negative_count, result->nan_count, result->lane_oracle_mismatches, result->checksum_bits);
  {
    int fd = result_fd;
    (void) figure8_release_dsp_maps(msg, max_msg_size, &fd, 1);
  }
  ret = 0;
end:
  if (chan_fd >= 0) {
    htp_ops_destroy_channel(get_global_handle());
  }
  if (chan) {
    free_shared_mem_buf(chan, chan_fd, max_msg_size);
  }
  if (result) {
    free_shared_mem_buf(result, result_fd, result_size);
  }
  return ret;
}

static int hmx_int8_gate_send_request(struct MessageHeader *msg, size_t max_msg_size,
                                      const struct HmxInt8GateParams *params) {
  struct RequestHeader req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_OP_COMPUTE,
  };
  struct OpComputeRequest compute_req = {
    .op = HTP_OPS_HMX_INT8_GATE,
  };

  msg->state.d        = 0;
  msg->n_reqs         = 1;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + sizeof(req_hdr) + sizeof(compute_req) + sizeof(*params);
  if ((size_t) msg->req_offsets[1] > max_msg_size) {
    fprintf(stderr, "HMX INT8 gate message buffer is too small\n");
    return -1;
  }

  uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(req_hdr);
  *(struct OpComputeRequest *) p = compute_req;
  p += sizeof(compute_req);
  *(struct HmxInt8GateParams *) p = *params;

  msg->state.v[0] = 1;
  if (figure8_wait_channel(msg, 30000000)) {
    return -1;
  }
  return message_header_get_request_ptr(msg, 0)->state;
}

static int run_hmx_int8_gate(int mode) {
  void *chan = NULL;
  struct HmxInt8GateResult *results = NULL;
  int chan_fd = -1, results_fd = -1;
  int ret = 1;

  const int max_results = (mode == 13 || mode == 14) ? 4 : (mode == 21 ? 32 : (mode == 0 ? 516 : (mode == 5 ? 1 : (mode == 11 ? 16384 : (mode == 12 ? 128 : (mode == 10 ? 1056 : ((mode == 7 || mode == 9) ? 256 : 4096)))))));
  const size_t max_msg_size = 4096;
  const size_t results_size = align_up(max_results * sizeof(struct HmxInt8GateResult), 128);

  if (alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size)) goto end;
  if (alloc_shared_mem_buf((void **) &results, &results_fd, results_size)) goto end;
  memset(results, 0, results_size);

  int err = create_htp_message_channel(chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Create HMX INT8 gate message channel failed: 0x%x\n", err);
    goto end;
  }

  struct MessageHeader *msg = (struct MessageHeader *) chan;
  struct HmxInt8GateParams params = {
    .output = {
      .fd = results_fd,
      .offset = 0,
    },
    .max_results = max_results,
    .reserved = mode,
  };

  err = hmx_int8_gate_send_request(msg, max_msg_size, &params);
  printf("HMX_INT8_GATE_REQUEST ret=%d mode=%d max_results=%d result_size=%zu\n", err, mode, max_results,
         results_size);
  if (err) {
    goto end;
  }

  printf("input_case,variant,output_kind,tile_or_scale_mode,selector,expected,first0,first1,first2,first3,first4,first5,first6,first7,min,max,mean,rmse,nan_count\n");
  for (int i = 0; i < max_results; ++i) {
    const struct HmxInt8GateResult *r = &results[i];
    printf("%d,%d,%d,%d,%d,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%d\n",
           r->reserved, r->variant, r->output_kind, r->tile_bytes, r->selector, r->expected, r->first8[0],
           r->first8[1], r->first8[2], r->first8[3], r->first8[4], r->first8[5], r->first8[6], r->first8[7],
           r->min_value, r->max_value, r->mean_value, r->rmse, r->nan_count);
  }

  {
    int fds[] = { results_fd };
    (void) figure8_release_dsp_maps(msg, max_msg_size, fds, 1);
  }

  ret = 0;

end:
  if (chan_fd >= 0) {
    htp_ops_destroy_channel(get_global_handle());
  }
  if (results) {
    free_shared_mem_buf(results, results_fd, results_size);
  }
  if (chan) {
    free_shared_mem_buf(chan, chan_fd, max_msg_size);
  }
  return ret;
}

static const char *roofline_mode_name(int mode) {
  switch (mode) {
    case ROOFLINE_BENCH_MODE_HMX_FP16:
      return "hmx_fp16";
    case ROOFLINE_BENCH_MODE_DDR_BW:
      return "ddr_bandwidth";
    case ROOFLINE_BENCH_MODE_VTCM_BW:
      return "vtcm_bandwidth";
    case ROOFLINE_BENCH_MODE_HMX_DMA_BW:
      return "hmx_dma_bandwidth";
    case ROOFLINE_BENCH_MODE_HVX_FP16:
      return "hvx_fp16";
    case ROOFLINE_BENCH_MODE_MIX_PRECISION:
      return "mix_precision";
    case ROOFLINE_BENCH_MODE_HMX_INT8_SHAPE_SWEEP:
      return "hmx_int8_shape_sweep";
    case ROOFLINE_BENCH_MODE_SIGNED_INT8_ZERO_OVERHEAD:
      return "signed_int8_zero_overhead";
    default:
      return "unknown";
  }
}

static const char *roofline_kind_name(int kind) {
  switch (kind) {
    case ROOFLINE_BENCH_KIND_HMX_FP16_GEMM:
      return "hmx_fp16_gemm";
    case ROOFLINE_BENCH_KIND_DDR_READ:
      return "ddr_read";
    case ROOFLINE_BENCH_KIND_DDR_WRITE:
      return "ddr_write";
    case ROOFLINE_BENCH_KIND_DDR_COPY:
      return "ddr_copy";
    case ROOFLINE_BENCH_KIND_VTCM_READ:
      return "vtcm_read";
    case ROOFLINE_BENCH_KIND_VTCM_WRITE:
      return "vtcm_write";
    case ROOFLINE_BENCH_KIND_VTCM_COPY:
      return "vtcm_copy";
    case ROOFLINE_BENCH_KIND_HMX_DMA_READ:
      return "hmx_dma_read";
    case ROOFLINE_BENCH_KIND_HVX_FP16_GEMM:
      return "hvx_fp16_gemm";
    case ROOFLINE_BENCH_KIND_HVX_FP32_GEMM:
      return "hvx_fp32_gemm";
    case ROOFLINE_BENCH_KIND_HVX_INT16_GEMM:
      return "hvx_int16_gemm";
    case ROOFLINE_BENCH_KIND_HVX_INT8_AS_INT16_GEMM:
      return "hvx_int8_as_int16_gemm";
    case ROOFLINE_BENCH_KIND_HVX_INT4_AS_INT16_GEMM:
      return "hvx_int4_as_int16_gemm";
    case ROOFLINE_BENCH_KIND_HVX_INT4_INT8_AS_INT16_GEMM:
      return "hvx_int4_int8_as_int16_gemm";
    case ROOFLINE_BENCH_KIND_HVX_INT4_INT16_GEMM:
      return "hvx_int4_int16_gemm";
    case ROOFLINE_BENCH_KIND_HVX_INT8_INT16_GEMM:
      return "hvx_int8_int16_gemm";
    case ROOFLINE_BENCH_KIND_HMX_INT8_RAW_UB_B_GEMM:
      return "hmx_int8_raw_ub_b_gemm";
    case ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_K2_GEMM:
      return "hmx_int8_signed_k2_gemm";
    case ROOFLINE_BENCH_KIND_FORMAT_Q4_0_DECODE_FP16_GEMM:
      return "format_q4_0_decode_fp16_gemm";
    case ROOFLINE_BENCH_KIND_FORMAT_IQ4_NL_DECODE_FP16_GEMM:
      return "format_iq4_nl_decode_fp16_gemm";
    case ROOFLINE_BENCH_KIND_HMX_INT8_INT4_WEIGHT_N_GEMM:
      return "hmx_int8_int4_weight_n_gemm";
    case ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_ZP_CORRECTED_GEMM:
      return "hmx_int8_signed_zp_corrected_gemm";
    case ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_COPY:
      return "signed_a8_producer_copy";
    case ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_COPY_XOR:
      return "signed_a8_producer_copy_xor";
    case ROOFLINE_BENCH_KIND_SIGNED_A8_PRODUCER_XOR_INPLACE:
      return "signed_a8_producer_xor_inplace";
    case ROOFLINE_BENCH_KIND_SIGNED_A8_COLSUM_PRECOMPUTE:
      return "signed_a8_colsum_precompute";
    case ROOFLINE_BENCH_KIND_SIGNED_A8_HVX_BIAS_SCALE_STORE:
      return "signed_a8_hvx_bias_scale_store";
    case ROOFLINE_BENCH_KIND_SIGNED_A8_REQUANT_STORE_ZP0:
      return "signed_a8_requant_store_zp0";
    case ROOFLINE_BENCH_KIND_SIGNED_A8_REQUANT_STORE_ZP128:
      return "signed_a8_requant_store_zp128";
    case ROOFLINE_BENCH_KIND_HMX_FP16_INT8_WEIGHT_B_GEMM:
      return "hmx_fp16_int8_weight_b_gemm";
    case ROOFLINE_BENCH_KIND_HMX_FP8_GEMM:
      return "hmx_fp8_gemm";
    case ROOFLINE_BENCH_KIND_HVX_FP32_MULADD_PEAK:
      return "hvx_fp32_muladd_peak";
    case ROOFLINE_BENCH_KIND_HVX_FP16_MAC_PEAK:
      return "hvx_fp16_mac_peak";
    case ROOFLINE_BENCH_KIND_HVX_BF16_MAC_PEAK:
      return "hvx_bf16_mac_peak";
    case ROOFLINE_BENCH_KIND_HVX_FP8_MAC_PEAK:
      return "hvx_fp8_mac_peak";
    case ROOFLINE_BENCH_KIND_HVX_S16_MAC_PEAK:
      return "hvx_s16_mac_peak";
    case ROOFLINE_BENCH_KIND_HVX_U16_MAC_PEAK:
      return "hvx_u16_mac_peak";
    case ROOFLINE_BENCH_KIND_HVX_S16_U16_MAC_PEAK:
      return "hvx_s16_u16_mac_peak";
    case ROOFLINE_BENCH_KIND_HVX_S8_MAC_PEAK:
      return "hvx_s8_mac_peak";
    case ROOFLINE_BENCH_KIND_HVX_U8_MAC_PEAK:
      return "hvx_u8_mac_peak";
    case ROOFLINE_BENCH_KIND_HVX_U8_S8_MAC_PEAK:
      return "hvx_u8_s8_mac_peak";
    case ROOFLINE_BENCH_KIND_HMX_FP16_INT4_WEIGHT_N_GEMM:
      return "hmx_fp16_int4_weight_n_gemm";
    case ROOFLINE_BENCH_KIND_V81_HMX_MANUAL_SMOKE:
      return "v81_hmx_manual_smoke";
    case ROOFLINE_BENCH_KIND_NOT_AVAILABLE:
      return "not_available";
    default:
      return "unknown";
  }
}

static const char *roofline_metric_unit(int kind) {
  switch (kind) {
    case ROOFLINE_BENCH_KIND_HMX_FP16_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_FP16_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_FP32_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_INT16_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_INT8_AS_INT16_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_INT4_AS_INT16_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_INT4_INT8_AS_INT16_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_INT4_INT16_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_INT8_INT16_GEMM:
    case ROOFLINE_BENCH_KIND_HMX_INT8_RAW_UB_B_GEMM:
    case ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_K2_GEMM:
    case ROOFLINE_BENCH_KIND_FORMAT_Q4_0_DECODE_FP16_GEMM:
    case ROOFLINE_BENCH_KIND_FORMAT_IQ4_NL_DECODE_FP16_GEMM:
    case ROOFLINE_BENCH_KIND_HMX_INT8_INT4_WEIGHT_N_GEMM:
    case ROOFLINE_BENCH_KIND_HMX_INT8_SIGNED_ZP_CORRECTED_GEMM:
    case ROOFLINE_BENCH_KIND_HMX_FP16_INT8_WEIGHT_B_GEMM:
    case ROOFLINE_BENCH_KIND_HMX_FP8_GEMM:
    case ROOFLINE_BENCH_KIND_HVX_FP32_MULADD_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_FP16_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_BF16_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_FP8_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_S16_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_U16_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_S16_U16_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_S8_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_U8_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HVX_U8_S8_MAC_PEAK:
    case ROOFLINE_BENCH_KIND_HMX_FP16_INT4_WEIGHT_N_GEMM:
      return "TOPS";
    case ROOFLINE_BENCH_KIND_NOT_AVAILABLE:
    case ROOFLINE_BENCH_KIND_V81_HMX_MANUAL_SMOKE:
      return "N/A";
    default:
      return "GB/s";
  }
}

static const char *roofline_engine_name(int engine) {
  switch (engine) {
    case ROOFLINE_BENCH_ENGINE_HMX:
      return "HMX";
    case ROOFLINE_BENCH_ENGINE_HVX:
      return "HVX";
    case ROOFLINE_BENCH_ENGINE_FORMAT_EFFECTIVE:
      return "format_effective";
    case ROOFLINE_BENCH_ENGINE_SCALAR:
      return "scalar";
    default:
      return "unknown";
  }
}

static const char *roofline_dtype_name(int dtype) {
  switch (dtype) {
    case ROOFLINE_BENCH_DTYPE_FP32:
      return "FP32";
    case ROOFLINE_BENCH_DTYPE_FP16:
      return "FP16";
    case ROOFLINE_BENCH_DTYPE_INT16:
      return "INT16";
    case ROOFLINE_BENCH_DTYPE_INT8:
      return "INT8";
    case ROOFLINE_BENCH_DTYPE_INT4_LINEAR:
      return "INT4_LINEAR";
    case ROOFLINE_BENCH_DTYPE_Q4_0:
      return "Q4_0";
    case ROOFLINE_BENCH_DTYPE_IQ4_NL:
      return "IQ4_NL";
    case ROOFLINE_BENCH_DTYPE_INT32:
      return "INT32";
    case ROOFLINE_BENCH_DTYPE_FP8:
      return "FP8";
    case ROOFLINE_BENCH_DTYPE_BF16:
      return "BF16";
    case ROOFLINE_BENCH_DTYPE_UINT16:
      return "UINT16";
    case ROOFLINE_BENCH_DTYPE_UINT8:
      return "UINT8";
    default:
      return "unknown";
  }
}

static const char *roofline_path_name(int path) {
  switch (path) {
    case ROOFLINE_BENCH_PATH_HMX_FP16_TILE:
      return "hmx_fp16_tile";
    case ROOFLINE_BENCH_PATH_HMX_RAW_UB_B_DEEP_GEMM:
      return "hmx_raw_activation_ub_weight_b_gemm_variant";
    case ROOFLINE_BENCH_PATH_HMX_SIGNED_K2:
      return "hmx_signed_k2";
    case ROOFLINE_BENCH_PATH_HVX_NATIVE:
      return "hvx_native";
    case ROOFLINE_BENCH_PATH_HVX_SIGNEXT_I16:
      return "hvx_signextend_int16";
    case ROOFLINE_BENCH_PATH_FORMAT_DECODE_TO_FP16:
      return "format_decode_to_fp16_hmx";
    case ROOFLINE_BENCH_PATH_NOT_AVAILABLE:
      return "not_available";
    case ROOFLINE_BENCH_PATH_HMX_RAW_UB_N_DEEP_GEMM:
      return "hmx_raw_activation_ub_weight_n_gemm_variant";
    case ROOFLINE_BENCH_PATH_HMX_SIGNED_A8_VIA_UB_COLSUM:
      return "hmx_signed_a8_via_activation_ub_colsum_correction";
    case ROOFLINE_BENCH_PATH_HVX_COPY:
      return "hvx_copy";
    case ROOFLINE_BENCH_PATH_HVX_COPY_XOR_0X80:
      return "hvx_copy_xor_0x80";
    case ROOFLINE_BENCH_PATH_HVX_XOR_0X80_INPLACE:
      return "hvx_xor_0x80_inplace";
    case ROOFLINE_BENCH_PATH_OFFLINE_COLSUM:
      return "offline_colsum";
    case ROOFLINE_BENCH_PATH_HVX_BIAS_SCALE_FLOAT_STORE:
      return "hvx_bias_scale_float_store";
    case ROOFLINE_BENCH_PATH_HVX_REQUANT_STORE_ZP:
      return "hvx_requant_store_zp";
    case ROOFLINE_BENCH_PATH_HMX_HF_B_GEMM:
      return "hmx_activation_hf_weight_b_gemm_variant";
    case ROOFLINE_BENCH_PATH_HMX_F8_F8_GEMM:
      return "hmx_activation_f8_weight_f8_gemm_variant";
    case ROOFLINE_BENCH_PATH_HVX_FP32_QF32_MUL_ADD:
      return "Q6_Vqf32_vmpy_VsfVsf+Q6_Vqf32_vadd_Vqf32Vqf32";
    case ROOFLINE_BENCH_PATH_HVX_FP16_VDMPYACC:
      return "Q6_Vsf_vdmpyacc_VsfVhfVhf";
    case ROOFLINE_BENCH_PATH_HVX_BF16_VMPYACC:
      return "Q6_Wsf_vmpyacc_WsfVbfVbf";
    case ROOFLINE_BENCH_PATH_HVX_FP8_VMPYACC:
      return "Q6_Whf_vmpyacc_WhfVV";
    case ROOFLINE_BENCH_PATH_HVX_S16_VMPYACC:
      return "Q6_Ww_vmpyacc_WwVhVh";
    case ROOFLINE_BENCH_PATH_HVX_U16_VMPYACC:
      return "Q6_Wuw_vmpyacc_WuwVuhVuh";
    case ROOFLINE_BENCH_PATH_HVX_S16_U16_VMPYACC:
      return "Q6_Ww_vmpyacc_WwVhVuh";
    case ROOFLINE_BENCH_PATH_HVX_S8_VRMPYACC:
      return "Q6_Vw_vrmpyacc_VwVbVb";
    case ROOFLINE_BENCH_PATH_HVX_U8_VRMPYACC:
      return "Q6_Vuw_vrmpyacc_VuwVubVub";
    case ROOFLINE_BENCH_PATH_HVX_U8_S8_VRMPYACC:
      return "Q6_Vw_vrmpyacc_VwVubVb";
    case ROOFLINE_BENCH_PATH_HMX_HF_N_GEMM:
      return "hmx_activation_hf_weight_n_gemm_variant";
    default:
      return "unknown";
  }
}

static int roofline_bench_send_request(struct MessageHeader *msg, size_t max_msg_size,
                                       const struct RooflineBenchParams *params) {
  struct RequestHeader req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_OP_COMPUTE,
  };
  struct OpComputeRequest compute_req = {
    .op = HTP_OPS_ROOFLINE_BENCH,
  };

  msg->state.d        = 0;
  msg->n_reqs         = 1;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + sizeof(req_hdr) + sizeof(compute_req) + sizeof(*params);
  if ((size_t) msg->req_offsets[1] > max_msg_size) {
    fprintf(stderr, "roofline bench message buffer is too small\n");
    return -1;
  }

  uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(req_hdr);
  *(struct OpComputeRequest *) p = compute_req;
  p += sizeof(compute_req);
  *(struct RooflineBenchParams *) p = *params;

  msg->state.v[0] = 1;
  if (figure8_wait_channel(msg, 30000000)) {
    return -1;
  }
  return message_header_get_request_ptr(msg, 0)->state;
}

static void roofline_print_results(FILE *out, const struct RooflineBenchResult *results, int max_results) {
  for (int i = 0; i < max_results; ++i) {
    const struct RooflineBenchResult *r = &results[i];
    if (r->kind == 0 || r->elapsed_us <= 0) {
      continue;
    }
    fprintf(out, "%s,%s,%d,%d,%d,%lld,%lld,%.4f,%s\n",
            roofline_mode_name(r->mode),
            roofline_kind_name(r->kind),
            r->variant,
            r->size,
            r->iters,
            (long long) r->elapsed_us,
            (long long) r->work_items,
            ((double) r->metric_x10000) / 10000.0,
            roofline_metric_unit(r->kind));
  }
}

static void roofline_print_mix_results(FILE *out, const struct RooflineBenchResult *results, int max_results) {
  for (int i = 0; i < max_results; ++i) {
    const struct RooflineBenchResult *r = &results[i];
    if (r->kind == 0) {
      continue;
    }
    if (r->kind == ROOFLINE_BENCH_KIND_NOT_AVAILABLE || r->elapsed_us <= 0) {
      fprintf(out, "%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%lld,%lld,N/A,N/A,%d\n",
              roofline_mode_name(r->mode),
              roofline_engine_name(r->engine),
              roofline_kind_name(r->kind),
              roofline_path_name(r->path),
              roofline_dtype_name(r->lhs_dtype),
              roofline_dtype_name(r->rhs_dtype),
              roofline_dtype_name(r->acc_dtype),
              r->variant,
              r->size,
              r->iters,
              (long long) r->elapsed_us,
              (long long) r->work_items,
              r->correctness);
      continue;
    }
    fprintf(out, "%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%lld,%lld,%.4f,%s,%d\n",
            roofline_mode_name(r->mode),
            roofline_engine_name(r->engine),
            roofline_kind_name(r->kind),
            roofline_path_name(r->path),
            roofline_dtype_name(r->lhs_dtype),
            roofline_dtype_name(r->rhs_dtype),
            roofline_dtype_name(r->acc_dtype),
            r->variant,
            r->size,
            r->iters,
            (long long) r->elapsed_us,
            (long long) r->work_items,
            ((double) r->metric_x10000) / 10000.0,
            roofline_metric_unit(r->kind),
            r->correctness);
  }
}

static void roofline_print_int8_shape_results(FILE *out, const struct RooflineBenchResult *results, int max_results) {
  for (int i = 0; i < max_results; ++i) {
    const struct RooflineBenchResult *r = &results[i];
    if (r->kind == 0) {
      continue;
    }
    if (r->elapsed_us <= 0) {
      fprintf(out,
              "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%d,%lld,%lld,N/A,N/A,%d\n",
              roofline_mode_name(r->mode),
              roofline_engine_name(r->engine),
              roofline_kind_name(r->kind),
              roofline_path_name(r->path),
              r->variant,
              r->m,
              r->k,
              r->n,
              r->mt,
              r->kt,
              r->nt,
              r->tile_bytes,
              (long long) r->a_bytes,
              (long long) r->b_bytes,
              (long long) r->c_bytes,
              (long long) r->scales_bytes,
              (long long) r->allocated_bytes,
              (long long) r->total_vtcm_bytes,
              r->iters,
              (long long) r->elapsed_us,
              (long long) r->work_items,
              r->correctness);
      continue;
    }
    fprintf(out,
            "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%d,%lld,%lld,%.4f,%s,%d\n",
            roofline_mode_name(r->mode),
            roofline_engine_name(r->engine),
            roofline_kind_name(r->kind),
            roofline_path_name(r->path),
            r->variant,
            r->m,
            r->k,
            r->n,
            r->mt,
            r->kt,
            r->nt,
            r->tile_bytes,
            (long long) r->a_bytes,
            (long long) r->b_bytes,
            (long long) r->c_bytes,
            (long long) r->scales_bytes,
            (long long) r->allocated_bytes,
            (long long) r->total_vtcm_bytes,
            r->iters,
            (long long) r->elapsed_us,
            (long long) r->work_items,
            ((double) r->metric_x10000) / 10000.0,
            roofline_metric_unit(r->kind),
            r->correctness);
  }
}

static void roofline_print_signed_zero_overhead_results(FILE *out, const struct RooflineBenchResult *results,
                                                        int max_results) {
  for (int i = 0; i < max_results; ++i) {
    const struct RooflineBenchResult *r = &results[i];
    if (r->kind == 0) {
      continue;
    }
    fprintf(out,
            "%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%lld,%d,%lld,%lld,%.4f,%s,%d\n",
            roofline_mode_name(r->mode),
            roofline_engine_name(r->engine),
            roofline_kind_name(r->kind),
            roofline_path_name(r->path),
            roofline_dtype_name(r->lhs_dtype),
            roofline_dtype_name(r->rhs_dtype),
            roofline_dtype_name(r->acc_dtype),
            r->variant,
            r->size,
            r->m,
            r->k,
            r->n,
            (long long) r->a_bytes,
            r->iters,
            (long long) r->elapsed_us,
            (long long) r->work_items,
            ((double) r->metric_x10000) / 10000.0,
            roofline_metric_unit(r->kind),
            r->correctness);
  }
}

static int run_roofline_benchmark(const struct Figure8AttnConfig *cfg) {
  void *chan = NULL, *src = NULL, *dst = NULL;
  struct RooflineBenchResult *results = NULL;
  int chan_fd = -1, src_fd = -1, dst_fd = -1, results_fd = -1;
  FILE *csv = NULL;
  int ret = 1;

  const int max_results = cfg->roofline_mix_precision_bench ? 1536 :
                          (cfg->roofline_int8_shape_bench ? 64 :
                           (cfg->roofline_signed_int8_zero_overhead_bench ? 32 : 24));
  const size_t max_msg_size = 4096;
  const size_t results_size = align_up(max_results * sizeof(struct RooflineBenchResult), 128);
  const size_t bench_size = align_up((size_t) cfg->bench_bytes, 128);

  if (alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size)) goto end;
  if (alloc_shared_mem_buf((void **) &results, &results_fd, results_size)) goto end;
  memset(results, 0, results_size);

  if (cfg->roofline_bandwidth_bench) {
    if (alloc_shared_mem_buf(&src, &src_fd, bench_size)) goto end;
    if (alloc_shared_mem_buf(&dst, &dst_fd, bench_size)) goto end;
    memset(src, 0x5a, bench_size);
    memset(dst, 0, bench_size);
  }

  if (cfg->csv_out) {
    csv = fopen(cfg->csv_out, "w");
    if (!csv) {
      fprintf(stderr, "failed to open csv-out: %s\n", cfg->csv_out);
      goto end;
    }
  }

  int err = create_htp_message_channel(chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Create roofline bench message channel failed: 0x%x\n", err);
    goto end;
  }

  if (cfg->roofline_int8_shape_bench) {
    printf("mode,engine,kind,path,variant,m,k,n,mt,kt,nt,tile_bytes,a_bytes,b_bytes,c_bytes,scales_bytes,allocated_bytes,total_vtcm_bytes,iters,elapsed_us,work_items,metric,unit,correctness\n");
    if (csv) {
      fprintf(csv, "mode,engine,kind,path,variant,m,k,n,mt,kt,nt,tile_bytes,a_bytes,b_bytes,c_bytes,scales_bytes,allocated_bytes,total_vtcm_bytes,iters,elapsed_us,work_items,metric,unit,correctness\n");
    }
  } else if (cfg->roofline_signed_int8_zero_overhead_bench) {
    printf("mode,engine,kind,path,lhs_dtype,rhs_dtype,acc_dtype,variant,size,m,k,n,bytes,iters,elapsed_us,work_items,metric,unit,correctness\n");
    if (csv) {
      fprintf(csv, "mode,engine,kind,path,lhs_dtype,rhs_dtype,acc_dtype,variant,size,m,k,n,bytes,iters,elapsed_us,work_items,metric,unit,correctness\n");
    }
  } else if (cfg->roofline_mix_precision_bench) {
    printf("mode,engine,kind,path,lhs_dtype,rhs_dtype,acc_dtype,variant,size,iters,elapsed_us,work_items,metric,unit,correctness\n");
    if (csv) {
      fprintf(csv, "mode,engine,kind,path,lhs_dtype,rhs_dtype,acc_dtype,variant,size,iters,elapsed_us,work_items,metric,unit,correctness\n");
    }
  } else {
    printf("mode,kind,variant,size,iters,elapsed_us,work_items,metric,unit\n");
    if (csv) {
      fprintf(csv, "mode,kind,variant,size,iters,elapsed_us,work_items,metric,unit\n");
    }
  }

  struct MessageHeader *msg = (struct MessageHeader *) chan;
  if (cfg->roofline_fp16_bench) {
    memset(results, 0, results_size);
    struct RooflineBenchParams params = {
      .output = { .fd = results_fd, .offset = 0 },
      .src = { .fd = -1, .offset = 0 },
      .dst = { .fd = -1, .offset = 0 },
      .max_results = max_results,
      .mode = ROOFLINE_BENCH_MODE_HMX_FP16,
      .warmup = cfg->warmup,
      .iters = cfg->iters,
      .bytes = (int32_t) bench_size,
    };
    err = roofline_bench_send_request(msg, max_msg_size, &params);
    if (err) {
      fprintf(stderr, "roofline fp16 bench failed: %d\n", err);
      goto end;
    }
    roofline_print_results(stdout, results, max_results);
    if (csv) roofline_print_results(csv, results, max_results);

    memset(results, 0, results_size);
    params.mode = ROOFLINE_BENCH_MODE_HVX_FP16;
    err = roofline_bench_send_request(msg, max_msg_size, &params);
    if (err) {
      fprintf(stderr, "roofline hvx fp16 bench failed: %d\n", err);
      goto end;
    }
    roofline_print_results(stdout, results, max_results);
    if (csv) roofline_print_results(csv, results, max_results);
  }

  if (cfg->roofline_bandwidth_bench) {
    memset(results, 0, results_size);
    struct RooflineBenchParams params = {
      .output = { .fd = results_fd, .offset = 0 },
      .src = { .fd = src_fd, .offset = 0 },
      .dst = { .fd = dst_fd, .offset = 0 },
      .max_results = max_results,
      .mode = ROOFLINE_BENCH_MODE_DDR_BW,
      .warmup = cfg->warmup,
      .iters = cfg->iters,
      .bytes = (int32_t) bench_size,
    };
    err = roofline_bench_send_request(msg, max_msg_size, &params);
    if (err) {
      fprintf(stderr, "roofline DDR bandwidth bench failed: %d\n", err);
      goto end;
    }
    roofline_print_results(stdout, results, max_results);
    if (csv) roofline_print_results(csv, results, max_results);

    memset(results, 0, results_size);
    params.src.fd = -1;
    params.dst.fd = -1;
    params.mode = ROOFLINE_BENCH_MODE_VTCM_BW;
    err = roofline_bench_send_request(msg, max_msg_size, &params);
    if (err) {
      fprintf(stderr, "roofline VTCM bandwidth bench failed: %d\n", err);
      goto end;
    }
    roofline_print_results(stdout, results, max_results);
    if (csv) roofline_print_results(csv, results, max_results);

    memset(results, 0, results_size);
    params.src.fd = src_fd;
    params.dst.fd = -1;
    params.mode = ROOFLINE_BENCH_MODE_HMX_DMA_BW;
    err = roofline_bench_send_request(msg, max_msg_size, &params);
    if (err) {
      fprintf(stderr, "roofline HMX DMA bandwidth bench failed: %d\n", err);
      goto end;
    }
    roofline_print_results(stdout, results, max_results);
    if (csv) roofline_print_results(csv, results, max_results);
  }

  if (cfg->roofline_mix_precision_bench) {
    memset(results, 0, results_size);
    struct RooflineBenchParams params = {
      .output = { .fd = results_fd, .offset = 0 },
      .src = { .fd = -1, .offset = 0 },
      .dst = { .fd = -1, .offset = 0 },
      .max_results = max_results,
      .mode = ROOFLINE_BENCH_MODE_MIX_PRECISION,
      .warmup = cfg->warmup,
      .iters = cfg->iters,
      // Mixed-precision mode historically ignored bytes.  Preserve case 0 as
      // the old all-in-one sweep and use positive values for isolated dtype
      // runs so one failing HMX spelling cannot hide all other results.
      .bytes = cfg->roofline_case,
    };
    err = roofline_bench_send_request(msg, max_msg_size, &params);
    if (err) {
      fprintf(stderr, "roofline mixed precision bench failed: %d\n", err);
      goto end;
    }
    roofline_print_mix_results(stdout, results, max_results);
    if (csv) roofline_print_mix_results(csv, results, max_results);
  }

  if (cfg->roofline_int8_shape_bench) {
    memset(results, 0, results_size);
    struct RooflineBenchParams params = {
      .output = { .fd = results_fd, .offset = 0 },
      .src = { .fd = -1, .offset = 0 },
      .dst = { .fd = -1, .offset = 0 },
      .max_results = max_results,
      .mode = ROOFLINE_BENCH_MODE_HMX_INT8_SHAPE_SWEEP,
      .warmup = cfg->warmup,
      .iters = cfg->iters,
      .bytes = (int32_t) bench_size,
    };
    err = roofline_bench_send_request(msg, max_msg_size, &params);
    if (err) {
      fprintf(stderr, "roofline INT8 shape benchmark failed: %d\n", err);
      goto end;
    }
    roofline_print_int8_shape_results(stdout, results, max_results);
    if (csv) roofline_print_int8_shape_results(csv, results, max_results);
  }

  if (cfg->roofline_signed_int8_zero_overhead_bench) {
    memset(results, 0, results_size);
    struct RooflineBenchParams params = {
      .output = { .fd = results_fd, .offset = 0 },
      .src = { .fd = -1, .offset = 0 },
      .dst = { .fd = -1, .offset = 0 },
      .max_results = max_results,
      .mode = ROOFLINE_BENCH_MODE_SIGNED_INT8_ZERO_OVERHEAD,
      .warmup = cfg->warmup,
      .iters = cfg->iters,
      .bytes = (int32_t) bench_size,
    };
    err = roofline_bench_send_request(msg, max_msg_size, &params);
    if (err) {
      fprintf(stderr, "roofline signed INT8 zero-overhead benchmark failed: %d\n", err);
      goto end;
    }
    roofline_print_signed_zero_overhead_results(stdout, results, max_results);
    if (csv) roofline_print_signed_zero_overhead_results(csv, results, max_results);
  }

  {
    int fds[3];
    int n_fds = 0;
    fds[n_fds++] = results_fd;
    if (src_fd >= 0) fds[n_fds++] = src_fd;
    if (dst_fd >= 0) fds[n_fds++] = dst_fd;
    (void) figure8_release_dsp_maps(msg, max_msg_size, fds, n_fds);
  }

  ret = 0;

end:
  if (chan_fd >= 0) {
    htp_ops_destroy_channel(get_global_handle());
  }
  if (csv) {
    fclose(csv);
  }
  if (dst) {
    free_shared_mem_buf(dst, dst_fd, bench_size);
  }
  if (src) {
    free_shared_mem_buf(src, src_fd, bench_size);
  }
  if (results) {
    free_shared_mem_buf(results, results_fd, results_size);
  }
  if (chan) {
    free_shared_mem_buf(chan, chan_fd, max_msg_size);
  }
  return ret;
}

static int run_figure8_attn_benchmark(const struct Figure8AttnConfig *cfg) {
  float  *q = NULL, *o = NULL;
  float  *reference_output = NULL, *reference_scores = NULL;
  __fp16 *k = NULL, *v = NULL, *mask = NULL;
  void   *chan = NULL, *profile = NULL;
  int     q_fd = -1, k_fd = -1, v_fd = -1, o_fd = -1, mask_fd = -1, chan_fd = -1, profile_fd = -1;
  FILE   *csv = NULL;
  int     ret = 1;

  const int    profile_max_records = cfg->n_kv_heads + 8;
  const int    kv_pad_len          = ceil_div_int(cfg->kv_len, 64) * 64;
  const int    max_q_blocks        = cfg->qo_len;
  const int    max_k_blocks        = ceil_div_int(cfg->kv_len, 64);
  int          profile_max_events  = cfg->n_kv_heads * max_q_blocks * (4 + max_k_blocks * 8) + 128;
  if (profile_max_events < 1024) {
    profile_max_events = 1024;
  }
  const size_t q_size              = align_up((size_t) cfg->qo_len * cfg->n_heads * cfg->head_dim * sizeof(float), 128);
  const size_t kv_size = align_up((size_t) cfg->kv_len * cfg->n_kv_heads * cfg->head_dim * sizeof(__fp16), 128);
  const size_t mask_size = align_up((size_t) cfg->qo_len * kv_pad_len * sizeof(__fp16), 128);
  const size_t profile_size = align_up(sizeof(struct Figure8ProfileHeader) +
                                         profile_max_records * sizeof(struct Figure8ProfileRecord) +
                                         profile_max_events * sizeof(struct Figure8ProfileEvent),
                                       128);
  const size_t max_msg_size = 4096;

  if (alloc_shared_mem_buf((void **) &q, &q_fd, q_size)) goto end;
  if (alloc_shared_mem_buf((void **) &k, &k_fd, kv_size)) goto end;
  if (alloc_shared_mem_buf((void **) &v, &v_fd, kv_size)) goto end;
  if (alloc_shared_mem_buf((void **) &o, &o_fd, q_size)) goto end;
  if (alloc_shared_mem_buf((void **) &mask, &mask_fd, mask_size)) goto end;
  if (alloc_shared_mem_buf(&profile, &profile_fd, profile_size)) goto end;
  if (alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size)) goto end;
  if (cfg->compare_reference) {
    reference_output = (float *) malloc((size_t) cfg->qo_len * cfg->n_heads * cfg->head_dim * sizeof(float));
    reference_scores = (float *) malloc((size_t) cfg->kv_len * sizeof(float));
    if (!reference_output || !reference_scores) goto end;
  }

  figure8_fill_inputs(q, k, v, mask, cfg->qo_len, cfg->kv_len, kv_pad_len, cfg->n_heads, cfg->n_kv_heads,
                      cfg->head_dim, cfg->mask_mode);
  memset(o, 0, q_size);

  int err = create_htp_message_channel(chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Create Figure 8 message channel failed: 0x%x\n", err);
    goto end;
  }

  if (cfg->csv_out) {
    csv = fopen(cfg->csv_out, "w");
    if (!csv) {
      fprintf(stderr, "Failed to open csv-out path: %s\n", cfg->csv_out);
      goto end;
    }
    fprintf(csv, "mode,qo_len,kv_len,n_heads,n_kv_heads,head_dim,phase,iteration,host_elapsed_us,ret\n");
  }

  int mode_flags = strcmp(cfg->mode, "lut-exp") == 0 ? LLM_NPU_MODE_LUT_EXP : 0;
  if (strcmp(cfg->mode, "scna-fp16") == 0) {
    mode_flags |= LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_D8;
    if (strcmp(cfg->scna_layout, "lane8") == 0) {
      mode_flags |= LLM_NPU_MODE_SCNA_LANE8;
    }
  }
  struct FlashAttnProfileParams params = {
    .attn = {
      .o          = { .fd = o_fd, .offset = 0, },
      .q          = { .fd = q_fd, .offset = 0, },
      .k          = { .fd = k_fd, .offset = 0, },
      .v          = { .fd = v_fd, .offset = 0, },
      .mask       = { .fd = mask_fd, .offset = 0, },
      .qo_len     = cfg->qo_len,
      .kv_len     = cfg->kv_len,
      .n_heads    = cfg->n_heads,
      .n_kv_heads = cfg->n_kv_heads,
      .head_dim   = cfg->head_dim,
      .trace_id   = 0,
      .mode_flags = mode_flags,
    },
    .profile     = { .fd = profile_fd, .offset = 0, },
    .max_records = profile_max_records,
    .max_events  = profile_max_events,
  };

  fprintf(stderr,
          "FIG8_ATTENTION_CONFIG mode=%s layout=%s scna_width=%d mask_mode=%s qo_len=%d kv_len=%d "
          "kv_pad_len=%d n_heads=%d n_kv_heads=%d head_dim=%d warmup=%d "
          "iters=%d q_size=%zu kv_size=%zu mask_size=%zu profile_size=%zu profile_max_records=%d "
          "profile_max_events=%d mode_flags=%d print_events=%d\n",
          cfg->mode, cfg->scna_layout, cfg->scna_width, cfg->mask_mode, cfg->qo_len, cfg->kv_len, kv_pad_len,
          cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, cfg->warmup, cfg->iters,
          q_size, kv_size, mask_size, profile_size, profile_max_records, profile_max_events, mode_flags,
          cfg->print_events);

  struct MessageHeader *msg = (struct MessageHeader *) chan;
  for (int i = 0; i < cfg->warmup + cfg->iters; ++i) {
    const int measured_idx = i - cfg->warmup;
    const int is_warmup    = measured_idx < 0;
    const char *phase      = is_warmup ? "warmup" : "measure";

    memset(o, 0, q_size);
    memset(profile, 0, profile_size);
    int64_t t0      = get_time_us();
    int     req_ret = figure8_send_attn_request(msg, max_msg_size, &params);
    int64_t elapsed = get_time_us() - t0;

    fprintf(stderr,
            "FIG8_ATTENTION_HOST_TIMING mode=%s layout=%s scna_width=%d mask_mode=%s qo_len=%d kv_len=%d "
            "n_heads=%d n_kv_heads=%d head_dim=%d phase=%s "
            "iteration=%d host_elapsed_us=%ld ret=%d\n",
            cfg->mode, cfg->scna_layout, cfg->scna_width, cfg->mask_mode, cfg->qo_len, cfg->kv_len, cfg->n_heads,
            cfg->n_kv_heads, cfg->head_dim, phase,
            is_warmup ? i : measured_idx, elapsed, req_ret);
    if (csv) {
      fprintf(csv, "%s,%d,%d,%d,%d,%d,%s,%d,%ld,%d\n", cfg->mode, cfg->qo_len, cfg->kv_len, cfg->n_heads,
              cfg->n_kv_heads, cfg->head_dim, phase, is_warmup ? i : measured_idx, elapsed, req_ret);
      fflush(csv);
    }

    if (req_ret) {
      goto end;
    }

    struct Figure8ProfileHeader *profile_hdr = (struct Figure8ProfileHeader *) profile;
    if (profile_hdr->magic != FIGURE8_PROFILE_MAGIC) {
      fprintf(stderr, "Bad Figure 8 profile magic: 0x%x\n", profile_hdr->magic);
      goto end;
    }
    int record_count = profile_hdr->record_count;
    if (record_count > profile_hdr->max_records) {
      record_count = profile_hdr->max_records;
    }
    fprintf(stderr,
            "FIG8_ATTENTION_PROFILE_COUNT mode=%s qo_len=%d phase=%s iteration=%d records=%d max_records=%d\n",
            cfg->mode, cfg->qo_len, phase, is_warmup ? i : measured_idx, record_count, profile_hdr->max_records);
    struct Figure8ProfileRecord *records = figure8_profile_records(profile_hdr);
    for (int r = 0; r < record_count; ++r) {
      const struct Figure8ProfileRecord *rec = &records[r];
      fprintf(stderr,
              "FIG8_ATTENTION_TIMERS mode=%s layout=%s phase=%s iteration=%d lut_exp=%d scna_layout=%d "
              "scna_width=%d qo_len=%d kv_len=%d n_heads=%d "
              "n_kv_heads=%d head_dim=%d kv_head=%d worker=%d profiled_total=%ld q_load=%ld k_load=%ld v_load=%ld "
              "qk_dot=%ld safe_sm=%ld scna_exp=%ld core_acc=%ld o_scale=%ld o_store=%ld\n",
              cfg->mode, cfg->scna_layout, phase, is_warmup ? i : measured_idx, rec->lut_exp, rec->scna_layout,
              rec->scna_width, rec->qo_len, rec->kv_len, rec->n_heads,
              rec->n_kv_heads, rec->head_dim, rec->kv_head, rec->worker, rec->profiled_total, rec->q_load,
              rec->k_load, rec->v_load, rec->qk_dot, rec->safe_sm, rec->scna_exp, rec->core_acc, rec->o_scale,
              rec->o_store);
    }

    int event_count = profile_hdr->event_count;
    if (event_count > profile_hdr->max_events) {
      event_count = profile_hdr->max_events;
    }
    fprintf(stderr,
            "FIG8_ATTENTION_EVENT_COUNT mode=%s qo_len=%d phase=%s iteration=%d events=%d max_events=%d overflow=%d\n",
            cfg->mode, cfg->qo_len, phase, is_warmup ? i : measured_idx, event_count, profile_hdr->max_events,
            profile_hdr->event_overflow);
    const struct Figure8ProfileEvent *events = figure8_profile_events_const(profile_hdr);
    if (cfg->print_events) {
      for (int e = 0; e < event_count; ++e) {
        const struct Figure8ProfileEvent *ev = &events[e];
        fprintf(stderr,
                "FIG8_ATTENTION_EVENT mode=%s layout=%s phase=%s iteration=%d component=%s component_id=%d "
                "lut_exp=%d scna_layout=%d scna_width=%d qo_len=%d "
                "kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d kv_head=%d worker=%d block_r=%d block_c=%d "
                "t0_us=%ld t1_us=%ld dur_us=%ld\n",
                cfg->mode, cfg->scna_layout, phase, is_warmup ? i : measured_idx,
                figure8_component_name(ev->component), ev->component, ev->lut_exp, ev->scna_layout, ev->scna_width,
                ev->qo_len, ev->kv_len, ev->n_heads, ev->n_kv_heads, ev->head_dim, ev->kv_head, ev->worker,
                ev->block_r, ev->block_c, ev->t0_us, ev->t1_us, ev->dur_us);
      }
    }
    fprintf(stderr,
            "FIG8_ATTENTION_CHECKSUM mode=%s layout=%s scna_width=%d phase=%s iteration=%d checksum=0x%016llx\n",
            cfg->mode, cfg->scna_layout, cfg->scna_width, phase, is_warmup ? i : measured_idx,
            (unsigned long long) figure8_checksum(o, (size_t) cfg->qo_len * cfg->n_heads * cfg->head_dim));
  }

  if (cfg->compare_reference) {
    const int64_t t0 = get_time_us();
    if (figure8_compute_reference(reference_output, reference_scores, q, k, v, mask, cfg->qo_len, cfg->kv_len,
                                  cfg->n_heads, cfg->n_kv_heads, cfg->head_dim)) {
      fprintf(stderr, "Figure 8 host reference failed\n");
      goto end;
    }
    fprintf(stderr,
            "FIG8_ATTENTION_REFERENCE_TIMING reference_mode=host-fp32 qo_len=%d kv_len=%d n_heads=%d "
            "n_kv_heads=%d head_dim=%d elapsed_us=%ld ret=0\n",
            cfg->qo_len, cfg->kv_len, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, get_time_us() - t0);
    figure8_report_comparison(cfg, o, reference_output, (size_t) cfg->qo_len * cfg->n_heads * cfg->head_dim);
  }

  {
    int fds[] = { o_fd, q_fd, k_fd, v_fd, mask_fd, profile_fd };
    (void) figure8_release_dsp_maps(msg, max_msg_size, fds, (int) (sizeof(fds) / sizeof(fds[0])));
  }

  ret = 0;

end:
  free(reference_scores);
  free(reference_output);
  if (csv) {
    fclose(csv);
  }
  if (chan_fd >= 0) {
    htp_ops_destroy_channel(get_global_handle());
  }
  if (chan) {
    free_shared_mem_buf(chan, chan_fd, max_msg_size);
  }
  if (mask) {
    free_shared_mem_buf(mask, mask_fd, mask_size);
  }
  if (profile) {
    free_shared_mem_buf(profile, profile_fd, profile_size);
  }
  if (o) {
    free_shared_mem_buf(o, o_fd, q_size);
  }
  if (v) {
    free_shared_mem_buf(v, v_fd, kv_size);
  }
  if (k) {
    free_shared_mem_buf(k, k_fd, kv_size);
  }
  if (q) {
    free_shared_mem_buf(q, q_fd, q_size);
  }
  return ret;
}

// assert p_buf, p_fd and size are always valid
int alloc_shared_mem_buf(void **p_buf, int *p_fd, size_t size) {
  void *buf = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
  if (!buf) {
    fprintf(stderr, "alloc_shared_mem_buf: rpcmem_alloc failed\n");
    return -1;
  }

  int fd = rpcmem_to_fd(buf);
  if (fd < 0) {
    fprintf(stderr, "alloc_shared_mem_buf: rpcmem_to_fd failed\n");
    return -1;
  }

  // map buffer to the DSP
  int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf, 0, size, FASTRPC_MAP_FD);
  if (err) {
    fprintf(stderr, "alloc_shared_mem_buf: fastrpc_mmap failed, err: %d\n", err);
    return -1;
  }

  *p_buf = buf;
  *p_fd  = fd;
  return 0;
}

void free_shared_mem_buf(void *buf, int fd, size_t size) {
  fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf, size);
  rpcmem_free(buf);
}

static void rms_norm_f32_ref(float *dst, const float *src, int ne0, int ne1) {
  const float eps = 1e-5;

  for (int j = 0; j < ne1; ++j) {
    const float *x = src + j * ne0;
    float       *y = dst + j * ne0;

    float sum = 0;
    for (int i = 0; i < ne0; ++i) {
      sum += x[i] * x[i];
    }

    float mean  = sum / ne0;
    float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < ne0; ++i) {
      y[i] = x[i] * scale;
    }

    printf("%s: sum: %.5f mean: %.5f scale: %.5f\n", __func__, sum, mean, scale);
  }
}

static void test_rms_norm_f32_rpc(remote_handle64 handle, int ne0) {
  float *src, *dsp_dst, *ref_dst;
  int    fd_src, fd_dst;

  int err, passed = 0;

  src = dsp_dst = ref_dst = NULL;
  size_t size             = align_up(ne0 * sizeof(float), 128);

  if (alloc_shared_mem_buf((void **) &src, &fd_src, size)) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &dsp_dst, &fd_dst, size)) {
    goto end;
  }
  ref_dst = (float *) malloc(size);

  // fill data, [0, 20000] -> [-20, 20]
  for (int i = 0; i < ne0; ++i) {
    src[i] = (rand() % 20000) * 2e-3f - 20.0f;
  }

  int64_t t0             = get_time_us();
  err                    = htp_ops_rms_norm_f32(handle, fd_dst, 0, fd_src, 0, ne0, 1);
  int64_t rpc_elapsed_us = get_time_us() - t0;
  fprintf(stderr, "rms_norm_f32 RPC took %ld us\n", rpc_elapsed_us);

  if (err != 0) {
    fprintf(stderr, "%s: RPC failed with %x\n", __func__, err);
    goto end;
  }
  rms_norm_f32_ref(ref_dst, src, ne0, 1);

  int   n_failed = 0;
  float tol      = 1e-5;
  for (int i = 0; i < ne0; ++i) {
    if (fabs(ref_dst[i] - dsp_dst[i]) > tol) {
      n_failed++;
      if (n_failed < 16) {
        fprintf(stderr, "%s: index %d, ref val=%.5f, dsp val=%.5f\n", __func__, i, ref_dst[i], dsp_dst[i]);
      }
    }
  }
  passed = (n_failed == 0);

end:
  if (src) {
    free_shared_mem_buf(src, fd_src, size);
  }
  if (dsp_dst) {
    free_shared_mem_buf(dsp_dst, fd_dst, size);
  }
  if (ref_dst) {
    free(ref_dst);
  }

  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
  return;
}

static void test_rms_norm_f32_chan(void *chan, int ne0) {
  struct MessageHeader *msg = (struct MessageHeader *) chan;

  float *src, *dsp_dst, *ref_dst;
  int    fd_src, fd_dst;

  int err, passed = 0;

  src = dsp_dst = ref_dst = NULL;
  size_t size             = align_up(ne0 * sizeof(float), 128);

  if (alloc_shared_mem_buf((void **) &src, &fd_src, size)) {
    goto end;
  }
  if (alloc_shared_mem_buf((void **) &dsp_dst, &fd_dst, size)) {
    goto end;
  }
  ref_dst = (float *) malloc(size);

  // fill data, [0, 20000] -> [-20, 20]
  for (int i = 0; i < ne0; ++i) {
    src[i] = (rand() % 20000) * 2e-3f - 20.0f;
  }

  {
    struct RequestHeader req_hdr = {
      .state = 0,
      .type  = REQUEST_TYPE_OP_COMPUTE,
    };
    struct OpComputeRequest compute_req = {
      .op = HTP_OPS_RMS_NORM_F32,
    };
    struct RmsNormF32Params params = {
      .dst = { .fd = fd_dst, .offset = 0, },
      .src = { .fd = fd_src, .offset = 0, },
      .ne0 = ne0,
      .ne1 = 1,
    };

    size_t req_size     = sizeof(req_hdr) + sizeof(compute_req) + sizeof(params);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + req_size;

    uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
    *(struct RequestHeader *) p = req_hdr;
    p += sizeof(struct RequestHeader);
    *(struct OpComputeRequest *) p = compute_req;
    p += sizeof(struct OpComputeRequest);
    *(struct RmsNormF32Params *) p = params;
    p += sizeof(struct RmsNormF32Params);
  }

  int64_t t0      = get_time_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    // usleep(10);
  }
  int64_t chan_elapsed_us = get_time_us() - t0;
  fprintf(stderr, "rms_norm_f32 CHAN took %ld us\n", chan_elapsed_us);

  err = message_header_get_request_ptr(msg, 0)->state;
  if (err != 0) {
    fprintf(stderr, "%s: CHAN failed with %x\n", __func__, err);
    goto end;
  }
  rms_norm_f32_ref(ref_dst, src, ne0, 1);

  int   n_failed = 0;
  float tol      = 1e-5;
  for (int i = 0; i < ne0; ++i) {
    if (fabs(ref_dst[i] - dsp_dst[i]) > tol) {
      n_failed++;
      if (n_failed < 16) {
        fprintf(stderr, "%s: index %d, ref val=%.5f, dsp val=%.5f\n", __func__, i, ref_dst[i], dsp_dst[i]);
      }
    }
  }
  passed = (n_failed == 0);

  // extra test: trigger DSP-side mapping reclaimation
  // fprintf(stderr, "manually unmap fd %d, %d\n", fd_dst, fd_src);
  // fastrpc_munmap(CDSP_DOMAIN_ID, fd_dst, NULL, 0);
  // fastrpc_munmap(CDSP_DOMAIN_ID, fd_src, NULL, 0);
  {
    struct RequestHeader req_hdr = {
      .state = 0,
      .type  = REQUEST_TYPE_RPCMEM_MAP,
    };
    struct RpcmemMapRequest map_req = {
      .n_puts = 2,
      .n_gets = 0,
    };

    size_t req_size     = sizeof(req_hdr) + sizeof(map_req) + 2 * sizeof(int);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + req_size;

    uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
    *(struct RequestHeader *) p = req_hdr;
    p += sizeof(struct RequestHeader);
    *(struct RpcmemMapRequest *) p = map_req;
    p += sizeof(struct RpcmemMapRequest);

    // fill in fd data
    *(int *) p = fd_dst;
    p += sizeof(int);
    *(int *) p = fd_src;
    p += sizeof(int);
  }

  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    usleep(10);
  }

end:
  if (src) {
    free_shared_mem_buf(src, fd_src, size);
  }
  if (dsp_dst) {
    free_shared_mem_buf(dsp_dst, fd_dst, size);
  }
  if (ref_dst) {
    free(ref_dst);
  }

  fprintf(stderr, passed ? "%s passed\n" : "%s failed\n", __func__);
}

static void test_mat_mul_rpc(remote_handle64 handle) {
  float *activation, *output;
  __fp16 *weight;

  int output_fd, activation_fd, weight_fd;

  int m = 1;
  int k = 1024;
  // int n = 608; // 576 | 608
  int n = 1024;

  alloc_shared_mem_buf((void **) &output, &output_fd, m * n * sizeof(float));
  alloc_shared_mem_buf((void **) &activation, &activation_fd, m * k * sizeof(float));
  alloc_shared_mem_buf((void **) &weight, &weight_fd, k * n * sizeof(__fp16));

  float *weight_ref = (float *) malloc(n * k * sizeof(float));
  float *output_ref = (float *) malloc(m * n * sizeof(float));
  memset(output_ref, 0, m * n * sizeof(float));

  __fp16 *output_f16 = (__fp16 *) malloc(m * n * sizeof(__fp16));
  memset(output_f16, 0, m * n * sizeof(__fp16));

  float *output_mix = (float *) malloc(m * n * sizeof(float));
  memset(output_mix, 0, m * n * sizeof(float));

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      activation[i * k + j] = rand_01();
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      float x = rand_01();

      int i0 = i / 32, i1 = i % 32;
      int j0 = j / 32, j1 = j % 32;

      int tile_idx = j0 * (k / 32) + i0;
      __fp16 *tile = weight + tile_idx * 1024;
      tile[(i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = (__fp16) x;
      weight_ref[i * n + j] = x;
    }
  }

  htp_ops_mat_mul_permuted_w16a32(handle, output_fd, 0, activation_fd, 0, weight_fd, 0, m, k, n);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < k; ++l) {
        output_ref[i * n + j] += activation[i * k + l] * weight_ref[l * n + j];
        output_f16[i * n + j] += (__fp16)(((__fp16) activation[i * k + l]) * ((__fp16) weight_ref[l * n + j]));
        output_mix[i * n + j] += (float)((__fp16) activation[i * k + l] * ((__fp16) weight_ref[l * n + j]));
      }
    }
  }

  for (int i = 0; i < m * n; ++i)
    printf("#%d hmx: %g, f32: %g, f16: %g, mix: %g\n", i, output[i], output_ref[i], output_f16[i], output_mix[i]);

  free(weight_ref);
  free(output_ref);
  free(output_f16);
  free(output_mix);

  free_shared_mem_buf(output, output_fd, m * n * sizeof(float));
  free_shared_mem_buf(activation, activation_fd, m * k * sizeof(float));
  free_shared_mem_buf(weight, weight_fd, k * n * sizeof(__fp16));
}

int main(int argc, char **argv) {
  struct Figure8AttnConfig figure8_cfg;
  int                      parse_ret = parse_figure8_args(argc, argv, &figure8_cfg);
  if (parse_ret != 0) {
    return parse_ret > 0 ? 0 : 1;
  }

  int err = open_dsp_session(CDSP_DOMAIN_ID, 1);
  if (err != 0) {
    fprintf(stderr, "Open DSP session failed\n");
    return 1;
  }

  init_htp_backend();

  if (figure8_cfg.scna_exp_bench) {
    int scna_ret = run_scna_exp_benchmark(&figure8_cfg);
    close_dsp_session();
    return scna_ret;
  }

  if (figure8_cfg.enabled) {
    int figure8_ret = run_figure8_attn_benchmark(&figure8_cfg);
    close_dsp_session();
    return figure8_ret;
  }

  if (figure8_cfg.roofline_fp16_bench || figure8_cfg.roofline_bandwidth_bench ||
      figure8_cfg.roofline_mix_precision_bench || figure8_cfg.roofline_int8_shape_bench ||
      figure8_cfg.roofline_signed_int8_zero_overhead_bench) {
    int roofline_ret = run_roofline_benchmark(&figure8_cfg);
    close_dsp_session();
    return roofline_ret;
  }

  if (figure8_cfg.hmx_int8_gate || figure8_cfg.hmx_int8_gate_search || figure8_cfg.hmx_int8_bitplane_gate ||
      figure8_cfg.hmx_int8_layout_gate || figure8_cfg.hmx_int8_bitop_gate || figure8_cfg.hmx_int8_tile_gate ||
      figure8_cfg.hmx_int8_byte_probe || figure8_cfg.hmx_int8_combo_probe || figure8_cfg.hmx_int8_drop_probe ||
      figure8_cfg.hmx_int8_pack_search || figure8_cfg.hmx_int8_sparse_map || figure8_cfg.hmx_int8_kalign_probe ||
      figure8_cfg.hmx_int8_linearity_probe || figure8_cfg.hmx_int8_full_weight_probe ||
      figure8_cfg.hmx_int8_full_weight_k2_probe || figure8_cfg.w8pc_a8pt_matmul_probe ||
      figure8_cfg.hmx_int8_mode > 0) {
    int mode = figure8_cfg.hmx_int8_mode > 0 ? figure8_cfg.hmx_int8_mode :
               (figure8_cfg.w8pc_a8pt_matmul_probe ? 15 :
               (figure8_cfg.hmx_int8_full_weight_k2_probe ? 14 :
               (figure8_cfg.hmx_int8_full_weight_probe ? 13 :
               (figure8_cfg.hmx_int8_linearity_probe ? 12 :
               (figure8_cfg.hmx_int8_kalign_probe ? 11 :
               (figure8_cfg.hmx_int8_sparse_map ? 10 :
               (figure8_cfg.hmx_int8_pack_search ? 9 :
               (figure8_cfg.hmx_int8_drop_probe ? 8 :
               (figure8_cfg.hmx_int8_combo_probe ? 7 :
               (figure8_cfg.hmx_int8_byte_probe ? 6 :
               (figure8_cfg.hmx_int8_layout_gate ? 3 :
               (figure8_cfg.hmx_int8_tile_gate ? 5 :
                (figure8_cfg.hmx_int8_bitop_gate ? 4 :
                 (figure8_cfg.hmx_int8_bitplane_gate ? 2 : (figure8_cfg.hmx_int8_gate_search ? 1 : 0)))))))))))))));
    int gate_ret = run_hmx_int8_gate(mode);
    close_dsp_session();
    return gate_ret;
  }

  // test_mat_mul_rpc(get_global_handle());

  htp_ops_test_ops(get_global_handle());

  /*
  test_rms_norm_f32_rpc(get_global_handle(), 60000);

  void        *chan;
  int          chan_fd;
  const size_t max_msg_size = 4096;

  err = alloc_shared_mem_buf(&chan, &chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Cannot allocate rpcmem for message channel\n");
    goto skip1;
  }

  err = htp_ops_create_channel(get_global_handle(), chan_fd, max_msg_size);
  if (err) {
    fprintf(stderr, "Create channel failed\n");
    goto skip2;
  }

  test_rms_norm_f32_chan(chan, 60000);

  htp_ops_destroy_channel(get_global_handle());

skip2:
  free_shared_mem_buf(chan, chan_fd, max_msg_size);
  */

skip1:
  close_dsp_session();
  return 0;
}
