#include <limits.h>
#include <math.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdbool.h>
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
#include "dsp/quants.h"

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
  int         qo_len;
  int         kv_len;
  int         n_heads;
  int         n_kv_heads;
  int         head_dim;
  int         warmup;
  int         iters;
  int         print_events;
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
  int         lpbq_a8w8_matmul_probe;
  int         lpbq_real_layer_gate;
  const char *lpbq_sidecar_dir;
  const char *lpbq_layer_stem;
  const char *lpbq_r4_path;
  const char *lpbq_r4_reference_path;
  int         lpbq_r4_block;
  int         lpbq_fold_r4_input_scale;
  int         lpbq_full_v6_weight;
  int         lpbq_full_v6_group_tiles;
  int         lpbq_r4_v6_scale_1_16;
  int         lpbq_force_r4_full_u8_safe;
  int         lpbq_trace_diag;
  int         lpbq_samples;
  int         hmx_int8_mode;
  int         roofline_fp16_bench;
  int         roofline_bandwidth_bench;
  int         roofline_mix_precision_bench;
  int         roofline_int8_shape_bench;
  int         roofline_signed_int8_zero_overhead_bench;
  int         bench_bytes;
  const char *matmul_case;
  int         matmul_m;
  int         matmul_k;
  int         matmul_n;
};

static void figure8_print_usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --figure8-attn [--mode baseline|lut-exp] [--qo-len N] [--kv-len N]\n"
          "          [--n-heads N] [--n-kv-heads N] [--head-dim N] [--warmup N] [--iters N]\n"
          "          [--no-events] [--csv-out PATH]\n"
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
          "       %s --lpbq-a8w8-matmul-probe\n"
          "       %s --lpbq-real-layer-gate --lpbq-sidecar-dir DIR --lpbq-layer-stem STEM "
          "--matmul-m M --matmul-k K --matmul-n N [--lpbq-r4-path PATH] "
          "[--lpbq-r4-reference-path PATH] [--lpbq-full-v6-weight] [--lpbq-full-v6-group-tiles N] "
          "[--lpbq-r4-v6-scale-1-16] [--lpbq-force-r4-full-u8-safe]\n"
          "       %s --hmx-int8-mode N  (N=21 QK OS debug, N=30 LPBQ R4, N=31 VTCM status, N=33 LPBQ R4 timing, N=34/35 input-scale, N=36/37 LPBQ packed, N=39/40 strict LPBQ multi-row, N=41 LPBQ K4 m32 component, N=42 LPBQ m17 shape isolation, N=43 LPBQ single-pass probe, N=44 dense R4 gate, N=45/46/47/48/49/50/51/52/53/54/55/56 LPBQ writeback/full-U8 probes, N=57 LPBQ full-U8 offset probe, N=58 LPBQ uniform raw-U8 sweep, N=59 LPBQ full-U8 K32 layout probe, N=60 LPBQ full-U8 sparse map, N=61 LPBQ ub*b byte-drop map, N=62 LPBQ shifted-twin full-U8 probe, N=63 LPBQ ub*b pair map, N=68 LPBQ production m29 signed A8W8 shapes, N=69 LPBQ stream-K32 signed A8W8 probe, N=70 LPBQ stream-K32 sparse map, N=71 LPBQ stream-K32 unit-scale sparse map, N=72/73/74 LPBQ m=1 non-R4 decode gates, N=75 LPBQ full-U8 HMX variant sweep, N=76 FP16-derived signed A8W8 stream gate, N=77 FP16-derived stream byte-drop map, N=78 A0 activation + streamed K4 weight probe, N=79 stream tile-drop variant sweep, N=86 FP16 skeleton with production K32 drains, N=87 FP16-base LPBQ float-scale gate, N=95 HMX bias/drain fusion gate, N=96 HMX cvt.uh bias/drain fusion gate, N=97 HMX after.uh drain timing gate, N=98 HMX after.uh HVX epilogue gate, N=99 HMX cvt.hf bias pack sweep, N=100 HMX cvt.hf selector/retain sweep, N=101 LPBQ single-drain HVX pipeline gate, N=102 LPBQ delayed-drain overlap probe, N=103 LPBQ nibble issue/accumulate overlap probe, N=109 dense R4 production timing summary, N=110 dense R4 HMX sub-timing summary, N=111 HMX bias register storeback gate, N=112 byte-MMA cvt view sweep, N=113 cvt.uh scale/bias semantics, N=114 cvt.uh fresh selector sweep, N=115 cvt.uh vs after.uh timing, N=116 grouped V6 delayed-recover accumulate gate, N=117 dense R4 decomposition gate, N=118 broad cvt.hf selector discovery, N=119 grouped V6 production overlap gate, N=120 grouped V6 N-tile overlap gate, N=121 grouped V6 delay-recover production gate, N=122 grouped V6 weight-prestage gate, N=126 dense R4 quant/store A/B gate, N=127/128 dense R4 m32 prefill timing gates, N=129 byte-cvt feedback bridge gate, N=130 dense R4 prefill scheduling A/B, N=131 group16 HMX bucket breakdown, N=132 group16 production-loop breakdown, N=133 group16 expand/HMX overlap, N=134 group16 N-block prestage, N=135 group16 publish primitive, N=136 group16 compact one-mxmem, N=138 R4 accumulate rows gate, N=139 R4 split dequant/store gate, N=140 R4 accumulate helper A/B gate, N=141 folded after.uh epilogue rows gate, N=142 folded production-entry no-sum_w gate, N=143 FP16 cvt.hf bias semantics gate, N=144 negative FP16 cvt.hf shape gate, N=145 byte-MMA direct-HF gate, N=146 manual feedback gate, N=148 LPBQ activation HMX cache gate, N=160 exact-K64 overlap gate, N=161 after.uh pre-accumulate gate, N=162 real-HMX K-group pre-accumulate gate, N=163 full-stream K-group size gate, N=183 cvt.hf LPBQ formula closure gate, N=184 high-bit cvt.hf selector closure gate, N=185 high-bit cvt.uh bias selector closure gate, N=186 high-bit manual feedback closure gate, N=187 cvt.uh-to-HF bridge closure gate, N=188 cvt.hf retain/bank closure gate, N=189 cvt.hf sign-surface closure gate, N=215 byte-first FP-seed cvt.hf bridge closure gate, N=216 cvt.uh/cvt.hf order closure gate, N=217 clear/bank formula closure gate, N=218 bias-set index contract gate, N=219 final-clear readback gate, N=220 direct plain-HF retain gate, N=221 mxswapacc bridge gate, N=222 manual feedback layout gate, N=223 broad manual feedback layout gate, N=224 bundled cvt.hf formula gate, N=225 post-issue fence gate, N=226 full-stream cvt.hf skeleton gate, N=227 full-stream bias postload gate, N=228 FP16-skeleton selector-2 cvt.hf gate, N=229 full-stream direct-HF contrast gate, N=230 full-stream raw-UH bias gate, N=231 full-stream raw-UH input21 gate, N=232 full-stream depth cvt.hf gate, N=233 target SDK intrinsic cvt.hf gate, N=234 target SDK fresh-bank cvt.hf gate, N=235 target SDK direct-HF bridge gate, N=236 cvt.hf VTCM reentry gate, N=237 full-stream affine surface gate, N=238 target SDK bias-set/retain matrix gate, N=239 target SDK direct-UH store matrix gate, N=240 target SDK cvt-store layout gate, N=241 full-stream per-column folded-bias gate, N=242 full-stream per-column folded-bias mxmem-bias gate, N=243 full-stream mxshl accumulator bridge gate, N=244 target SDK cvt.uh per-column gate, N=245 target SDK per-column cvt-store layout gate, N=246 per-column cvt.hf selector matrix gate, N=247 per-column manual-feedback gate, N=248 mixed signed folded-bias gate, N=249 cvt.hf bank-state gate, N=250 direct-HF reentry gate, N=251 direct-UB store gate, N=252 V81 bias/cvt order gate, N=253 bundled V81 bias/cvt order gate, N=254 tile-wide bundled V81 drain gate, N=255 raw-selector bundled V81 drain gate, N=256 best-selector layout V81 drain gate, N=257 raw-UH bias quantum gate, N=258 signed V81 cvt.hf gate, N=259 signed V81 tile-wide gate, N=260 bias storeback state gate, N=261 cvt.hf spatial-Rs gate, N=262 cvt.hf wide-AR gate, N=263 cvt.hf dependency gate, N=264 adaptive row0 derive gate, N=277/278 grouped-V6 cvt.uh selector gates, N=279 selector2 folded-bias gate, N=280 selector2 raw-UH bias quantum gate, N=281 target-SDK selector2 cvt.uh gate, N=285 K-major copy geometry gate, N=286 grouped-V6 scale-drain cost gate, N=287 R4 VTCM residency gate, N=288 SDK cvt.ub gate, N=289 SDK clear/swap gate)\n"
          "       %s --roofline-fp16-bench [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --roofline-bandwidth-bench [--bench-bytes N] [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --roofline-mix-precision-bench [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --roofline-int8-shape-bench [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --roofline-signed-int8-zero-overhead-bench [--warmup N] [--iters N] [--csv-out PATH]\n"
          "       %s --matmul-case NAME --matmul-m M --matmul-k K --matmul-n N\n",
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
    .qo_len     = 4,
    .kv_len     = 4096,
    .n_heads    = 12,
    .n_kv_heads = 2,
    .head_dim   = 128,
    .warmup     = 5,
    .iters      = 20,
    .print_events = 1,
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
    .lpbq_a8w8_matmul_probe = 0,
    .lpbq_real_layer_gate = 0,
    .lpbq_sidecar_dir = NULL,
    .lpbq_layer_stem = NULL,
    .lpbq_r4_path = NULL,
    .lpbq_r4_reference_path = NULL,
    .lpbq_r4_block = 128,
    .lpbq_fold_r4_input_scale = 0,
    .lpbq_full_v6_weight = 0,
    .lpbq_full_v6_group_tiles = 8,
    .lpbq_r4_v6_scale_1_16 = 0,
    .lpbq_force_r4_full_u8_safe = 0,
    .lpbq_trace_diag = 0,
    .lpbq_samples = 128,
    .hmx_int8_mode = -1,
    .roofline_fp16_bench = 0,
    .roofline_bandwidth_bench = 0,
    .roofline_mix_precision_bench = 0,
    .roofline_int8_shape_bench = 0,
    .roofline_signed_int8_zero_overhead_bench = 0,
    .bench_bytes = 64 * 1024 * 1024,
    .matmul_case = NULL,
    .matmul_m = 0,
    .matmul_k = 0,
    .matmul_n = 0,
  };

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--figure8-attn") == 0) {
      cfg->enabled = 1;
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
    } else if (strcmp(arg, "--lpbq-a8w8-matmul-probe") == 0) {
      cfg->lpbq_a8w8_matmul_probe = 1;
    } else if (strcmp(arg, "--lpbq-real-layer-gate") == 0) {
      cfg->lpbq_real_layer_gate = 1;
    } else if (strcmp(arg, "--lpbq-sidecar-dir") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --lpbq-sidecar-dir\n");
        return -1;
      }
      cfg->lpbq_sidecar_dir = argv[i];
    } else if (strcmp(arg, "--lpbq-layer-stem") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --lpbq-layer-stem\n");
        return -1;
      }
      cfg->lpbq_layer_stem = argv[i];
    } else if (strcmp(arg, "--lpbq-r4-path") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --lpbq-r4-path\n");
        return -1;
      }
      cfg->lpbq_r4_path = argv[i];
    } else if (strcmp(arg, "--lpbq-r4-reference-path") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --lpbq-r4-reference-path\n");
        return -1;
      }
      cfg->lpbq_r4_reference_path = argv[i];
    } else if (strcmp(arg, "--lpbq-r4-block") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->lpbq_r4_block)) return -1;
    } else if (strcmp(arg, "--lpbq-fold-r4-input-scale") == 0) {
      cfg->lpbq_fold_r4_input_scale = 1;
    } else if (strcmp(arg, "--lpbq-full-v6-weight") == 0) {
      cfg->lpbq_full_v6_weight = 1;
    } else if (strcmp(arg, "--lpbq-full-v6-group-tiles") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->lpbq_full_v6_group_tiles)) return -1;
    } else if (strcmp(arg, "--lpbq-r4-v6-scale-1-16") == 0) {
      cfg->lpbq_r4_v6_scale_1_16 = 1;
    } else if (strcmp(arg, "--lpbq-force-r4-full-u8-safe") == 0) {
      cfg->lpbq_force_r4_full_u8_safe = 1;
    } else if (strcmp(arg, "--lpbq-trace-diag") == 0) {
      cfg->lpbq_trace_diag = 1;
    } else if (strcmp(arg, "--lpbq-samples") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->lpbq_samples)) return -1;
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
    } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      figure8_print_usage(argv[0]);
      return 1;
    } else if (strcmp(arg, "--mode") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --mode\n");
        return -1;
      }
      cfg->mode = argv[i];
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
    } else if (strcmp(arg, "--matmul-case") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing value for --matmul-case\n");
        return -1;
      }
      // Local pure-FP16 debugging hook: run one shape without the full
      // standalone sequence so direct-DMA state bugs can be bisected quickly.
      cfg->matmul_case = argv[i];
    } else if (strcmp(arg, "--matmul-m") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->matmul_m)) return -1;
    } else if (strcmp(arg, "--matmul-k") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->matmul_k)) return -1;
    } else if (strcmp(arg, "--matmul-n") == 0) {
      if (++i >= argc || parse_int_cli_value(arg, argv[i], &cfg->matmul_n)) return -1;
    } else if (strcmp(arg, "--no-events") == 0) {
      cfg->print_events = 0;
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
      !cfg->roofline_signed_int8_zero_overhead_bench && !cfg->matmul_case &&
      !cfg->lpbq_real_layer_gate) {
    return 0;
  }
  if (cfg->matmul_case &&
      (cfg->matmul_m <= 0 || cfg->matmul_k <= 0 || cfg->matmul_n <= 0)) {
    fprintf(stderr, "--matmul-case requires --matmul-m/--matmul-k/--matmul-n\n");
    return -1;
  }
  if (cfg->lpbq_real_layer_gate &&
      (!cfg->lpbq_sidecar_dir || !cfg->lpbq_layer_stem ||
       cfg->matmul_m <= 0 || cfg->matmul_k <= 0 || cfg->matmul_n <= 0)) {
    fprintf(stderr,
            "--lpbq-real-layer-gate requires --lpbq-sidecar-dir/--lpbq-layer-stem and "
            "--matmul-m/--matmul-k/--matmul-n\n");
    return -1;
  }
  if (!cfg->enabled) {
    return 0;
  }
  if (strcmp(cfg->mode, "baseline") != 0 && strcmp(cfg->mode, "lut-exp") != 0) {
    fprintf(stderr, "Unsupported --mode: %s\n", cfg->mode);
    return -1;
  }
  if (cfg->n_heads % cfg->n_kv_heads != 0) {
    fprintf(stderr, "n_heads must be divisible by n_kv_heads\n");
    return -1;
  }
  return 0;
}

static void figure8_fill_inputs(float *q, __fp16 *k, __fp16 *v, __fp16 *mask, int qo_len, int kv_len, int n_heads,
                                int n_kv_heads, int head_dim) {
  const size_t q_elems    = (size_t) qo_len * n_heads * head_dim;
  const size_t kv_elems   = (size_t) kv_len * n_kv_heads * head_dim;
  const size_t mask_elems = (size_t) qo_len * kv_len;

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
    mask[i] = (__fp16) 0.0f;
  }
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
  const int64_t timeout_us = params->reserved == 26 ? 120000000 : 30000000;
  if (figure8_wait_channel(msg, timeout_us)) {
    return -1;
  }
  return message_header_get_request_ptr(msg, 0)->state;
}

static int run_hmx_int8_gate(int mode) {
  void *chan = NULL;
  struct HmxInt8GateResult *results = NULL;
  int chan_fd = -1, results_fd = -1;
  int ret = 1;

  int max_results = 4096;
  if (mode == 13 || mode == 14) {
    max_results = 4;
  } else if (mode == 21) {
    max_results = 32;
  } else if (mode == 26) {
    max_results = 5;
  } else if (mode == 39) {
    max_results = 8;
  } else if (mode == 40) {
    max_results = 4;
  } else if (mode == 42) {
    max_results = 4;
  } else if (mode == 43) {
    max_results = 16;
  } else if (mode == 45) {
    max_results = 60;
  } else if (mode == 46) {
    max_results = 18;
  } else if (mode == 47) {
    max_results = 9;
  } else if (mode == 48) {
    max_results = 3;
  } else if (mode == 49) {
    max_results = 36;
  } else if (mode == 50) {
    max_results = 3;
  } else if (mode == 51) {
    max_results = 3;
  } else if (mode == 52) {
    max_results = 12;
  } else if (mode == 53) {
    max_results = 24;
  } else if (mode == 54) {
    max_results = 3;
  } else if (mode == 56) {
    max_results = 1536;
  } else if (mode == 57) {
    max_results = 16;
  } else if (mode == 58) {
    max_results = 52;
  } else if (mode == 59) {
    max_results = 18;
  } else if (mode == 60) {
    max_results = 540;
  } else if (mode == 61) {
    max_results = 4096;
  } else if (mode == 62) {
    max_results = 4096;
  } else if (mode == 63) {
    max_results = 4096;
  } else if (mode == 64) {
    max_results = 16;
  } else if (mode == 65) {
    max_results = 256;
  } else if (mode == 66) {
    max_results = 256;
  } else if (mode == 67) {
    max_results = 3;
  } else if (mode == 75) {
    max_results = 36;
  } else if (mode == 76) {
    max_results = 3;
  } else if (mode == 77) {
    max_results = 16;
  } else if (mode == 78) {
    max_results = 18;
  } else if (mode == 79) {
    max_results = 192;
  } else if (mode == 80) {
    max_results = 3;
  } else if (mode == 81) {
    max_results = 56;
  } else if (mode == 82) {
    max_results = 84;
  } else if (mode == 83) {
    max_results = 27;
  } else if (mode == 84) {
    max_results = 6;
  } else if (mode == 85) {
    max_results = 3;
  } else if (mode == 86) {
    max_results = 3;
  } else if (mode == 87) {
    max_results = 3;
  } else if (mode == 88) {
    max_results = 192;
  } else if (mode == 89) {
    max_results = 4;
  } else if (mode == 90) {
    max_results = 32;
  } else if (mode == 91) {
    max_results = 4;
  } else if (mode == 92) {
    max_results = 16;
  } else if (mode == 93) {
    max_results = 10;
  } else if (mode == 94) {
    // LPBQ deploy-v1 R4 V6 scale/recover diagnostic sweep.  Keep this bounded
    // so the host does not print thousands of zero-padded rows.
    max_results = 1440;
  } else if (mode == 95) {
    max_results = 7;
  } else if (mode == 96) {
    // LPBQ deploy-v1 cvt.uh bias-drain gate returns four uniform rows plus
    // two LPBQ formula rows and one spare/guard row; the old value 6 made the
    // standalone dispatcher reject the mode before it could report evidence.
    max_results = 7;
  } else if (mode == 97) {
    max_results = 6;
  } else if (mode == 98) {
    // LPBQ deploy-v1 mode 98 now reports both online-zp and production
    // folded-bias HVX epilogue rows; keep the old rows and add two guardrails.
    max_results = 5;
  } else if (mode == 99) {
    max_results = 14;
  } else if (mode == 100) {
    max_results = 13;
  } else if (mode == 101) {
    max_results = 4;
  } else if (mode == 102) {
    max_results = 5;
  } else if (mode == 103) {
    max_results = 4;
  } else if (mode == 104) {
    // LPBQ deploy-v1 bias-register lifetime gate: 4 initial reads, 4 VTCM
    // overwrite/no-reload reads, and 4 single-set reload reads.
    max_results = 12;
  } else if (mode == 105) {
    // LPBQ deploy-v1 bias-load accumulator-window gate: correctness rows plus
    // timing rows for bias=mxmem2 load and stream/load/drain sequencing.
    max_results = 5;
  } else if (mode == 106) {
    // LPBQ deploy-v1 two-tile bias-prestage pipeline gate: correctness, serial,
    // pipelined, bias-prep/load, and speedup summary rows.
    max_results = 6;
  } else if (mode == 107) {
    // LPBQ deploy-v1 cross-tile nibble pipeline gate: correctness, current
    // per-tile overlap, cross-tile overlap, issue/drain-only, and summary rows.
    max_results = 5;
  } else if (mode == 108) {
    // LPBQ deploy-v1 production-shaped cross-tile nibble gate: single hmx_out,
    // reusable weight scratch, correctness, timing, copy cost, and summary rows.
    max_results = 5;
  } else if (mode == 109) {
    // LPBQ deploy-v1 dense-R4 production timing gate: summary plus seven stage rows.
    max_results = 8;
  } else if (mode == 110) {
    // LPBQ deploy-v1 dense-R4 HMX sub-timing gate: summary plus seven substage rows.
    max_results = 8;
  } else if (mode == 111) {
    // LPBQ deploy-v1 HMX bias register storeback gate: 4 direct + 4 source-overwrite rows.
    max_results = 8;
  } else if (mode == 112) {
    // LPBQ deploy-v1 byte-MMA cvt visibility sweep: direct baseline plus 23 cvt view rows.
    max_results = 24;
  } else if (mode == 113) {
    // LPBQ deploy-v1 cvt.uh scale/bias semantics: baseline + 16 uniform + 4 LPBQ rows.
    max_results = 21;
  } else if (mode == 114) {
    // LPBQ deploy-v1 cvt.uh fresh selector sweep: baseline + 8 selector rows.
    max_results = 9;
  } else if (mode == 115) {
    // LPBQ deploy-v1 cvt.uh vs after.uh timing: correctness + timing summary rows.
    max_results = 4;
  } else if (mode == 116) {
    // LPBQ deploy-v1 grouped V6 delayed-recover accumulate gate: 4 rows each for K128/K256.
    max_results = 8;
  } else if (mode == 117) {
    // LPBQ deploy-v1 dense-R4 decomposition gate: six timing rows plus one equivalence row.
    max_results = 7;
  } else if (mode == 118) {
    // LPBQ deploy-v1 broad cvt.hf selector discovery: baseline + 16 candidates + summary.
    max_results = 18;
  } else if (mode == 119) {
    // LPBQ deploy-v1 production-shaped grouped V6 overlap gate: five rows each
    // for group_tiles=4, 8, and the current SOTA group_tiles=16.
    max_results = 15;
  } else if (mode == 120) {
    // LPBQ deploy-v1 production-shaped grouped V6 N-tile overlap gate: five
    // rows each for group_tiles=4, 8, and the current SOTA group_tiles=16.
    max_results = 15;
  } else if (mode == 121) {
    // LPBQ deploy-v1 production-shaped grouped V6 delayed-recover gate: five
    // rows each for group_tiles=4, 8, and the current SOTA group_tiles=16.
    max_results = 15;
  } else if (mode == 122) {
    // LPBQ deploy-v1 grouped V6 weight-prestage gate: equivalence plus
    // current, prestaged, expand-only, issue-only, accumulate-only, and
    // preexpanded/no-runtime-expand upper-bound rows. Rows 12270..12290 add
    // the zero-once publish probe without removing the original measurements.
    max_results = 10;
  } else if (mode == 123) {
    // LPBQ deploy-v1 FP16-vs-byte cvt.hf comparator: FP16 direct/cvt rows,
    // byte after.uh/cvt rows, stale-FP16 guardrail, and one summary row.
    max_results = 9;
  } else if (mode == 124) {
    // LPBQ deploy-v1 broad cvt.uh selector summary: the DSP sweeps 0x0000..0x3fff
    // internally, but only returns baseline, best-candidate, first-selector, and
    // summary rows. Keep this tight to avoid printing unused zero-padded rows.
    max_results = 18;
  } else if (mode == 125) {
    // LPBQ deploy-v1 row-pair row-mask accumulate proof: correctness plus
    // full-rowpair, row0-masked, and scalar row0 timing rows.
    max_results = 4;
  } else if (mode == 126) {
    // LPBQ deploy-v1 dense-R4 quant/store A/B gate: equivalence plus current,
    // no-zero, prebase, unrolled, and quant-only timing rows.
    max_results = 7;
  } else if (mode == 127) {
    // LPBQ deploy-v1 dense-R4 m=32 prefill-block timing summary plus
    // cross-tile R4/HMX overlap upper-bound rows.
    max_results = 10;
  } else if (mode == 128) {
    // LPBQ deploy-v1 dense-R4 m=32 prefill-block HMX sub-timing summary plus
    // HMX bucket reduction upper-bound rows.
    max_results = 11;
  } else if (mode == 129) {
    // LPBQ deploy-v1 byte-cvt feedback bridge: baseline, cvt.uh control,
    // sixteen feedback candidates, and one summary row.
    max_results = 19;
  } else if (mode == 130) {
    // LPBQ deploy-v1 dense-R4 prefill scheduling A/B: equivalence, serial,
    // current row-pair/auto, kk-group parallel, speedups, and worker metadata.
    max_results = 8;
  } else if (mode == 131) {
    // LPBQ deploy-v1 group16 HMX bucket breakdown: correctness plus scale-load,
    // cached-scale core, drain, accumulate, full bucket, and summary rows.
    max_results = 10;
  } else if (mode == 132) {
    // LPBQ deploy-v1 group16 production-loop breakdown: current compact expand
    // loop, prestaged full upper bound, core/expand/accumulate, and summary.
    max_results = 8;
  } else if (mode == 133) {
    // LPBQ deploy-v1 group16 expand/HMX overlap: correctness, current,
    // pipelined double-scratch schedule, expand-only, and summary rows.
    max_results = 6;
  } else if (mode == 134) {
    // LPBQ deploy-v1 group16 N-block prestage: current baseline, block
    // sizes 1/2/4/8/16/32, and best-block summary.
    max_results = 9;
  } else if (mode == 135) {
    // LPBQ deploy-v1 group16 publish primitive: correctness, four
    // publish-only rows, four publish+HMX rows, and one summary.
    max_results = 10;
  } else if (mode == 136) {
    // LPBQ deploy-v1 group16 compact one-mxmem: correctness, three shapes
    // times four variants, and one summary row.
    max_results = 14;
  } else if (mode == 137) {
    // LPBQ deploy-v1 cvt.uh -> mxmem=cvt layout sweep: direct baseline plus
    // two cvt forms over five store layouts.
    max_results = 11;
  } else if (mode == 138) {
    // LPBQ deploy-v1 R4/nibble accumulate row-scaling gate: rows 1/2/4/8/16/32,
    // a full-helper comparator, and one summary row.
    max_results = 8;
  } else if (mode == 139) {
    // LPBQ deploy-v1 R4 split dequant/store gate: correctness, rows 1/2/4/8/16/32,
    // and split-vs-roundhalf summary rows.
    max_results = 8;
  } else if (mode == 140) {
    // LPBQ deploy-v1 R4 accumulator helper A/B gate: six row counts,
    // speedup summary, and compile-flag row.
    max_results = 8;
  } else if (mode == 141) {
    // LPBQ deploy-v1 folded-bias after.uh epilogue rows gate: six row counts,
    // summary, and no-online-sum_w flag row.
    max_results = 8;
  } else if (mode == 142) {
    // LPBQ deploy-v1 folded production-entry gate: decode, FFN, tail rows, and
    // m32 prefill block through hmx_mat_mul_lpbq_a8w8() with sum_w=NULL.
    max_results = 4;
  } else if (mode == 143) {
    // LPBQ deploy-v1 FP16 cvt.hf bias semantics: zero-acc, FP16 acc,
    // negative/positive output bias, retain bit, and per-channel mapping rows.
    max_results = 12;
  } else if (mode == 144) {
    // LPBQ deploy-v1 negative FP16 cvt.hf shape gate: direct before/after.hf,
    // shape bits 0..7, and one summary row.
    max_results = 13;
  } else if (mode == 145) {
    // LPBQ deploy-v1 byte-MMA direct-HF gate: after.uh baseline, stale direct-HF,
    // cleared direct-HF/cvt.hf rows, and one summary row.
    max_results = 9;
  } else if (mode == 146) {
    // LPBQ deploy-v1 manual feedback gate: Chapter-5 bias/cvt feedback rows,
    // byte-MMA feedback visibility, and one summary row.
    max_results = 13;
  } else if (mode == 164) {
    // LPBQ deploy-v1 cvt.hf Rs[6] closure gate: direct FP16 sign control,
    // explicit overflow/retain selector rows, byte-MMA cvt.hf visibility, and
    // one summary row.
    max_results = 14;
  } else if (mode == 165) {
    // LPBQ deploy-v1 grouped-V6 finish-shape gate: exact compact K64 versus
    // full-U8 V6 rows=32 at fixed K=256, with stage and ratio rows.
    max_results = 12;
  } else if (mode == 166) {
    // LPBQ deploy-v1 grouped-V6 real-loop gate: rows=32, K=8960, Ntiles=48,
    // row_blocks=5, serial/overlap variants, with stage and ratio rows.
    max_results = 12;
  } else if (mode == 167) {
    // LPBQ deploy-v1 grouped-V6 drain compare gate: direct after.uh versus
    // diagnostic cvt.uh raw/recovered rows plus compile metadata.
    max_results = 8;
  } else if (mode == 168) {
    // LPBQ deploy-v1 R4 split gate: scaled-source fill, R4 dot, and
    // quant/V6 pack only. Production cached helper is isolated in mode 169.
    max_results = 8;
  } else if (mode == 169) {
    // LPBQ deploy-v1 R4 production-helper isolation gate: same tiny K32
    // shape as mode 168, with cached-helper equivalence and timing enabled.
    max_results = 8;
  } else if (mode >= 171 && mode <= 175) {
    // LPBQ deploy-v1 R4 split shape sweep: K128 rows=1/8/32, with production
    // cached helper isolated in odd-numbered modes.
    max_results = 8;
  } else if (mode == 176) {
    // LPBQ deploy-v1 R4 FP16-HMX V6-layout candidate: baseline HVX/FP32 V6
    // helper vs FP16-HMX helper, with mismatch/timing summary rows plus raw
    // FP16 output/weight tile-layout diagnostics and a prepacked-weight A/B.
    max_results = 17;
  } else if (mode == 177) {
    // LPBQ deploy-v1 R4 K128 bulk candidate: four K32 V6 activation tiles
    // generated in one helper call, compared against the accepted K32 helper.
    max_results = 8;
  } else if (mode == 178 || mode == 179) {
    // LPBQ deploy-v1 R4 K128 serial-bulk small-row gates: rows=1/8 exact
    // helper compared against the accepted K32 helper before LLM routing.
    max_results = 8;
  } else if (mode >= 180 && mode <= 182) {
    // LPBQ deploy-v1 R4 sign-FHT proof gates: synthetic Sylvester R4 rows=32/1/8.
    max_results = 8;
  } else if (mode == 265) {
    // LPBQ deploy-v1 rows=1 R4 K128 staged producer timing probe.
    max_results = 8;
  } else if (mode == 183) {
    // LPBQ deploy-v1 cvt.hf LPBQ formula closure: bias payload, signed cvt.hf,
    // byte-MMA visibility, two formula rows, and one summary row.
    max_results = 11;
  } else if (mode == 184) {
    // LPBQ deploy-v1 cvt.hf high selector closure: direct baseline, best-16
    // high selector rows, and one summary row.
    max_results = 18;
  } else if (mode == 185) {
    // LPBQ deploy-v1 cvt.uh high selector closure: direct baseline, best-32
    // high selector rows across 2x1/2x2 bias-vs-scale rankings, and summary.
    max_results = 34;
  } else if (mode == 186) {
    // LPBQ deploy-v1 manual feedback high-selector closure: baseline, fixed
    // mode-146 control, best-16 selector/clear rows, and summary.
    max_results = 19;
  } else if (mode == 187) {
    // LPBQ deploy-v1 cvt.uh-to-HF bridge closure: FP16/byte controls,
    // direct-HF/cvt-HF bridge attempts, and one summary row.
    max_results = 8;
  } else if (mode == 188) {
    // LPBQ deploy-v1 cvt.hf retain/bank closure: FP16 retain controls,
    // byte-MMA cvt.hf attempts, and one summary row.
    max_results = 7;
  } else if (mode == 189) {
    // LPBQ deploy-v1 cvt.hf sign-surface closure: zero-bias, FP16-negative,
    // byte-MMA formula attempts, and one summary row.
    max_results = 29;
  } else if (mode == 190) {
    // LPBQ deploy-v1 exact V81 sequence audit: bias payload/load, cvt.hf
    // retain/clear, byte-MMA post-bias-load lifetime, mxmem=cvt, and summary.
    max_results = 14;
  } else if (mode == 191) {
    // LPBQ deploy-v1 per-column after.uh scale audit: direct after.uh and
    // cvt.uh scale-only controls plus cvt.hf formula contrast.
    max_results = 8;
  } else if (mode == 192) {
    // LPBQ deploy-v1 UH bias-field surface audit: output-bias/input-bias/shape
    // effects for direct after.uh and cvt.uh, plus signed scale controls.
    max_results = 13;
  } else if (mode == 193) {
    // LPBQ deploy-v1 cvt VTCM reentry bridge audit: FP16 controls, byte
    // after.uh baseline, cvt.uh/cvt.ub raw-VTCM reload, and summary.
    max_results = 6;
  } else if (mode == 194) {
    // LPBQ deploy-v1 direct pos.hf surface audit: FP16 controls, byte stale
    // FP checks, byte clear-FP pos.hf variants, and summary.
    max_results = 16;
  } else if (mode == 195) {
    // LPBQ deploy-v1 cvt.hf Rs[7] data-type audit: exact bias/cvt/mxmem
    // formula controls for FP16 and byte-MMA, plus one summary row.
    max_results = 13;
  } else if (mode == 196) {
    // LPBQ deploy-v1 cvt.uh bridge formula audit: direct/cvt.uh scale-only,
    // cvt.uh bias folding, cvt.uh->cvt.hf feedback, VTCM reentry, and summary.
    max_results = 7;
  } else if (mode == 197) {
    // LPBQ deploy-v1 cvt.ub/mxmem=cvt layout audit: direct baseline, cvt.ub,
    // cvt.ub:sc0/sc1 selectors, layout-store variants, and one summary row.
    max_results = 15;
  } else if (mode == 198) {
    // LPBQ deploy-v1 raw bias payload bitfield audit: direct after.uh/cvt.uh
    // baselines, low/high non-scale field scans, scale-hi controls, and summary.
    max_results = 9;
  } else if (mode == 199) {
    // LPBQ deploy-v1 raw UH affine candidate audit: direct/cvt.uh scale
    // controls, FP16-like bias fields, raw low/high output-field scans, summary.
    max_results = 13;
  } else if (mode == 200) {
    // LPBQ deploy-v1 raw UH field map: scale control, high-nibble map,
    // shape map, high-nibble+shape map, and one summary row.
    max_results = 34;
  } else if (mode == 201) {
    // LPBQ deploy-v1 raw UH low output-bias map: single-bit, small-code,
    // FP16-neighborhood, FP16-code rows, and two summary rows.
    max_results = 78;
  } else if (mode == 202) {
    // LPBQ deploy-v1 cvt.hf selector matrix: baseline, scale-only/formula
    // rows across base selector, dtype/precision flags, bias set, and load time.
    max_results = 130;
  } else if (mode == 203) {
    // LPBQ deploy-v1 cvt.hf mxmem=cvt store-layout closure: FP16 layout
    // controls plus byte-MMA scale-only/formula rows and one summary.
    max_results = 80;
  } else if (mode == 212) {
    // LPBQ deploy-v1 cvt.hf tile-wide selector/layout closure: direct
    // baseline, FP16 layout control, best scale/formula rows, and summary.
    max_results = 5;
  } else if (mode == 204) {
    // LPBQ deploy-v1 cvt.uh -> mxmem=cvt VTCM reentry closure: direct
    // baseline, FP16 positive controls, five layout pairs, and summary.
    max_results = 14;
  } else if (mode == 205) {
    // LPBQ deploy-v1 cvt.hf bank-isolation closure: byte/FP baselines,
    // clear-order rows, byte/FP overlap rows, and summary.
    max_results = 12;
  } else if (mode == 206) {
    // LPBQ deploy-v1 cvt.uh:2x2 VTCM reentry closure: preserves mode 204's
    // original 2x1 audit while closing the 2x2 mxmem=cvt layout gap.
    max_results = 14;
  } else if (mode == 207) {
    // LPBQ deploy-v1 cvt.hf retain bridge closure: bias/FP seed controls,
    // byte-MMA no-clear bridge rows, and one summary row.
    max_results = 8;
  } else if (mode == 208) {
    // LPBQ deploy-v1 joint raw UH output-bias map: scale control, best direct
    // and cvt.uh combined low/high/shape rows, and one summary row.
    max_results = 4;
  } else if (mode == 209) {
    // LPBQ deploy-v1 zero-point folded bias closure: baseline, scale-only,
    // positive/negative bias_eff formula rows, and one summary row.
    max_results = 10;
  } else if (mode == 213) {
    // LPBQ deploy-v1 raw UH input-bias trim closure: scale control, best
    // direct/cvt.uh input21 candidates, and one summary row.
    max_results = 4;
  } else if (mode == 214) {
    // LPBQ deploy-v1 bias-set selector matrix: baseline, direct/cvt.uh/cvt.hf
    // rows for sets 0..3, and one summary row.
    max_results = 14;
  } else if (mode == 215) {
    // LPBQ deploy-v1 byte-first FP-seed bridge closure: byte-MMA first,
    // FP16 skeleton seed second, then cvt.hf retain/mxmem=cvt summary.
    max_results = 7;
  } else if (mode == 216) {
    // LPBQ deploy-v1 cvt.uh/cvt.hf order closure: scale-only cvt.uh first,
    // optional mxmem=cvt store, then cvt.hf retain/mxmem=cvt summary.
    max_results = 8;
  } else if (mode == 217) {
    // LPBQ deploy-v1 clear/bank formula matrix: byte/direct, clear controls,
    // cvt.uh scale-only rows, cvt.hf formula rows, and one summary row.
    max_results = 9;
  } else if (mode == 218) {
    // LPBQ deploy-v1 Phase 2B bias-set index contract: four retained FP16
    // set-selection rows, one final-clear row, and one summary row.
    max_results = 6;
  } else if (mode == 219) {
    // LPBQ deploy-v1 Phase 2B final-clear/readback diagnostic: baseline,
    // direct residual, second-cvt, explicit-clear rows, and one summary row.
    max_results = 7;
  } else if (mode == 220) {
    // LPBQ deploy-v1 direct plain-HF retain diagnostic: baseline, FP16
    // before/after retain controls, byte stale/clear rows, and one summary row.
    max_results = 18;
  } else if (mode == 221) {
    // LPBQ deploy-v1 mxswapacc bridge diagnostic: byte after.uh, byte cvt.hf,
    // cvt.uh, FP16 controls, and one summary row.
    max_results = 14;
  } else if (mode == 222) {
    // LPBQ deploy-v1 manual feedback layout diagnostic: byte after.uh baseline,
    // FP16 feedback layout control, best formula/manual rows, and summary.
    max_results = 5;
  } else if (mode == 223) {
    // LPBQ deploy-v1 broad manual feedback layout diagnostic: byte baseline,
    // FP16 feedback control, high-selector formula/manual rows, and summary.
    max_results = 5;
  } else if (mode == 210) {
    // LPBQ deploy-v1 bias-load timing matrix: baseline, direct/cvt.uh/cvt.hf
    // preload/postload rows, and one summary row.
    max_results = 8;
  } else if (mode == 211) {
    // LPBQ deploy-v1 post-drain retain matrix: direct after.uh baseline,
    // two-drain ordering rows, and one summary row.
    max_results = 6;
  } else if (mode == 147) {
    // LPBQ deploy-v1 dense-R4 exact-K64 path diagnostic: timing + path-count
    // rows for m=1 and m=32.
    max_results = 4;
  } else if (mode == 158) {
    // LPBQ deploy-v1 dense-R4 full-U8-safe diagnostic: timing + path-count
    // rows for m=1 and m=32 with the standalone safe bit explicitly enabled.
    max_results = 4;
  } else if (mode == 159) {
    // LPBQ deploy-v1 exact-K64 small-shape diagnostic: four compact production
    // shapes, each returning timing plus path-count rows.
    max_results = 8;
  } else if (mode == 160) {
    // LPBQ deploy-v1 exact-K64 overlap proof: rows=1/32 serial versus
    // compact-style hi4-issue-before-lo4-accumulate schedule.
    max_results = 8;
  } else if (mode == 161) {
    // LPBQ deploy-v1 after.uh pre-accumulate proof: rows=1/3/32, two-drain
    // and four-drain halfword pre-add versus repeated int32 accumulate.
    max_results = 10;
  } else if (mode == 162) {
    // LPBQ deploy-v1 real-HMX K-group pre-accumulate proof: rows=1/3/32,
    // two/four adjacent same-nibble K16 groups drained before one accumulate.
    max_results = 10;
  } else if (mode == 163) {
    // LPBQ deploy-v1 full-stream/single-drain proof: rows=1/32 over K128,
    // comparing K16 repeated drains with K32/K64/K128 exact drain groups.
    max_results = 8;
  } else if (mode == 170) {
    // LPBQ deploy-v1 exact fallback diagnostic: full-int8 K8 primitive proof
    // reuses the exact-K64 microprobe result shape.
    max_results = 10;
  } else if (mode == 148) {
    // LPBQ deploy-v1 non-R4 activation HMX cache gate: miss timing, hit timing,
    // output/reference equivalence, and compile/cap metadata.
    max_results = 4;
  } else if (mode >= 149 && mode <= 157) {
    // LPBQ deploy-v1 exact compact-K64 micro-probe: summary, eight staged
    // begin/issue/drain/accumulate rows, and one correctness/checksum row.
    max_results = 10;
  } else if (mode == 69) {
    max_results = 6;
  } else if (mode == 70) {
    max_results = 56;
  } else if (mode == 71) {
    max_results = 112;
  } else if (mode == 68) {
    max_results = 3;
  } else if (mode == 72 || mode == 73 || mode == 74) {
    max_results = 3;
  } else if (mode == 229) {
    // LPBQ deploy-v1 full-stream direct-HF contrast for mode 228: byte baseline,
    // FP16 plain-HF controls, direct-HF byte rows, and one summary row.
    max_results = 26;
  } else if (mode == 230) {
    // LPBQ deploy-v1 full-stream raw-UH bias closure: byte baseline,
    // direct/cvt.uh scale controls, near-miss rows, and one summary row.
    max_results = 6;
  } else if (mode == 231) {
    // LPBQ deploy-v1 full-stream raw-UH input21 trim closure: byte baseline,
    // scale controls, direct/cvt.uh best trim rows, and one summary row.
    max_results = 6;
  } else if (mode == 232) {
    // LPBQ deploy-v1 full-stream depth cvt.hf closure: group_tiles
    // 1/4/8/16/32 with direct/cvt.uh controls and split/bundled cvt.hf rows.
    max_results = 31;
  } else if (mode == 233) {
    // LPBQ deploy-v1 target SDK intrinsic cvt.hf closure: bias payload,
    // FP16 asm/intrinsic controls, direct byte baseline, byte asm/intrinsic
    // formula rows, and one summary row.
    max_results = 8;
  } else if (mode == 234) {
    // LPBQ deploy-v1 target SDK fresh-FP-bank cvt.hf closure: FP16 control,
    // direct byte baseline, mxclracc.hf preservation, clear-timing formula
    // rows, scale-only contrast, and one summary row.
    max_results = 9;
  } else if (mode == 235) {
    // LPBQ deploy-v1 target SDK direct-HF bridge closure: byte baseline,
    // eight official direct-HF store variants over FP/stale/fresh byte-MMA
    // views, and one summary row.
    max_results = 26;
  } else if (mode == 236) {
    // LPBQ deploy-v1 cvt.hf VTCM reentry closure: byte baseline, FP16 reentry
    // controls, five cvt store layouts for scale/formula, and one summary row.
    max_results = 14;
  } else if (mode == 237) {
    // LPBQ deploy-v1 full-stream affine surface closure: bias payload, FP16
    // control, direct after.uh, cvt.uh, cvt.hf formula rows, and one summary.
    max_results = 9;
  } else if (mode == 238) {
    // LPBQ deploy-v1 target SDK bias-set/retain matrix: byte baseline, four
    // bias sets x retain/clear formula rows, and one summary row.
    max_results = 10;
  } else if (mode == 239) {
    // LPBQ deploy-v1 target SDK direct-UH store matrix: payload row, sixteen
    // official before/after/retain/sat 2x1/2x2 variants, and one summary row.
    max_results = 18;
  } else if (mode == 240) {
    // LPBQ deploy-v1 target SDK cvt-store layout matrix: baseline, three
    // official cvt-store layouts x FP/scale/formula rows, and one summary.
    max_results = 11;
  } else if (mode == 241 || mode == 242 || mode == 243 || mode == 244 || mode == 248) {
    // LPBQ deploy-v1 full-stream per-column folded-bias or mxshl bridge
    // matrix: payload, FP16 control, byte baseline, drain rows, and summary.
    max_results = 7;
  } else if (mode == 245) {
    // LPBQ deploy-v1 target SDK per-column cvt-store layout matrix: payload,
    // byte baseline, five layouts x FP/scale/formula rows, and summary.
    max_results = 18;
  } else if (mode == 246) {
    // LPBQ deploy-v1 per-column cvt.hf selector matrix: payload, byte
    // baseline, four sets x six selectors x FP/scale/formula rows, summary.
    max_results = 75;
  } else if (mode == 247) {
    // LPBQ deploy-v1 per-column manual-feedback closure: payload, byte
    // baseline, two documented selectors x FP/feedback/formula rows, summary.
    max_results = 9;
  } else if (mode == 249) {
    // LPBQ deploy-v1 per-column cvt.hf bank-state closure: payload, FP16
    // control, byte baseline/direct rows, six bank-state rows, and summary.
    max_results = 11;
  } else if (mode == 250) {
    // LPBQ deploy-v1 per-column direct-HF reentry closure: payload, FP16
    // reentry control, direct-HF variants, two scale-kind reentries, summary.
    max_results = 28;
  } else if (mode == 251) {
    // LPBQ deploy-v1 per-column direct-UB store closure: payload, byte
    // baseline, sixteen official before/after/retain/sat/cm variants, summary.
    max_results = 19;
  } else if (mode == 252) {
    // LPBQ deploy-v1 V81 bias/cvt/mxmem order closure: payload, FP16 control,
    // byte baseline, eight exact-order variants, and summary.
    max_results = 12;
  } else if (mode == 253) {
    // LPBQ deploy-v1 bundled V81 final-drain closure: payload, FP16 control,
    // byte baseline, one split control, eight bundled variants, and summary.
    max_results = 13;
  } else if (mode == 254) {
    // LPBQ deploy-v1 tile-wide bundled V81 drain closure: payload, FP16
    // control, byte baseline, one split scan, eight bundled scans, summary.
    max_results = 13;
  } else if (mode == 255) {
    // LPBQ deploy-v1 raw-selector bundled V81 drain closure: payload, FP16
    // control, byte baseline, retain/raw-selector bundled scans, summary.
    max_results = 20;
  } else if (mode == 256) {
    // LPBQ deploy-v1 best-selector layout V81 closure: payload, byte baseline,
    // FP16 layout controls, retain/raw selector layout scans, summary.
    max_results = 18;
  } else if (mode == 257) {
    // LPBQ deploy-v1 raw-UH bias quantum closure: byte baseline, direct/cvt.uh
    // scale controls, nine bias_eff target scans, and one summary row.
    max_results = 13;
  } else if (mode == 258) {
    // LPBQ deploy-v1 signed V81 cvt.hf closure: payload controls, FP16/byte
    // signed direct controls, cvt.hf selector scans, and one summary row.
    max_results = 8;
  } else if (mode == 259) {
    // LPBQ deploy-v1 signed byte-MMA cvt.hf tile-wide closure: payload,
    // byte baseline, five mxmem=cvt layout aggregates, and one summary row.
    max_results = 8;
  } else if (mode == 260) {
    // LPBQ deploy-v1 bias storeback state gate: standalone-only probe rows plus
    // one summary row; keep explicit sizing near the fused-drain modes.
    max_results = 12;
  } else if (mode == 261) {
    // LPBQ deploy-v1 cvt.hf spatial-Rs gate: payload, byte baseline, spatial
    // aggregate rows, and one summary row.
    max_results = 27;
  } else if (mode == 262) {
    // LPBQ deploy-v1 wide-AR cvt.hf/mxmem=cvt closure: payload, byte baseline,
    // eight AR-bucket aggregates, and one summary row.
    max_results = 11;
  } else if (mode == 272) {
    // LPBQ deploy-v1 V81 spatial-mask Rt closure: payload, byte baseline,
    // four manual mask buckets, and one summary row.
    max_results = 7;
  } else if (mode == 273) {
    // LPBQ deploy-v1 Q6_bias_mxmem_A payload-layout closure: six candidate
    // payload packings plus one summary row.
    max_results = 7;
  } else if (mode == 274) {
    // LPBQ deploy-v1 rows=1 R4 reduce8 diagnostic: metadata, correctness,
    // timing, samples, and summary rows.
    max_results = 8;
  } else if (mode == 275) {
    // LPBQ deploy-v1 cvt.hf source-attribution gate: payload/FP controls,
    // byte/scale/bias and dirty-FP differentials, selector deltas, summary.
    max_results = 17;
  } else if (mode == 276) {
    // LPBQ deploy-v1 scale-drain split gate: direct/cvt.uh HMX scale-only
    // drains, HVX bias-only epilogues, full-loop timings, and summary.
    max_results = 9;
  } else if (mode == 277) {
    // LPBQ deploy-v1 grouped-V6 cvt.uh sign/layout probe: eight selector rows,
    // one direct metadata row, and one best-candidate summary row.
    max_results = 10;
  } else if (mode == 278) {
    // LPBQ deploy-v1 grouped-V6 cvt.uh selector-2 closure: raw equivalence,
    // recovered equivalence, direct/cvt timing, summary, and flags.
    max_results = 8;
  } else if (mode == 279) {
    // LPBQ deploy-v1 selector2 folded-bias closure: positive and mixed signed
    // bias_eff payloads, direct/cvt.uh/cvt.hf rows, and per-pattern summaries.
    max_results = 14;
  } else if (mode == 280) {
    // LPBQ deploy-v1 selector2 raw-UH bias quantum closure: direct plus
    // cvt.uh selector1/2/3 scale controls, target scans, and one summary row.
    max_results = 15;
  } else if (mode == 281) {
    // LPBQ deploy-v1 target-SDK selector2 cvt.uh closure: positive and mixed
    // folded-bias payloads across selector/shape/layout rows plus summaries.
    max_results = 30;
  } else if (mode == 282) {
    // LPBQ deploy-v1 cvt.hf selector-source matrix: mode-275 attribution
    // widened across representative raw selectors, deltas, and summary.
    max_results = 122;
  } else if (mode == 283) {
    // LPBQ deploy-v1 cvt.hf all-bias-set source gate: high selector bits
    // tested after preloading bias sets 0..3, with deltas and summary.
    max_results = 66;
  } else if (mode == 284) {
    // LPBQ deploy-v1 fullstream mxswap source gate: all-bias-set source
    // attribution with mxswapacc/mxswapacc.hf and bias-load order variants.
    max_results = 52;
  } else if (mode == 285) {
    // LPBQ deploy-v1 K-major copy geometry gate: eight measured/skip rows plus
    // summary and metadata, bounded so standalone logs stay readable.
    max_results = 12;
  } else if (mode == 287) {
    // LPBQ deploy-v1 R4 VTCM residency gate: metadata, K32/four-K32 timing for
    // rows=1/8/32, summary, and padding rows.
    max_results = 9;
  } else if (mode == 286) {
    // LPBQ deploy-v1 grouped-V6 scale-drain cost gate: formula rows, row-count
    // epilogue/full timings, multi-group timings, summary, and metadata.
    max_results = 10;
  } else if (mode == 288) {
    // LPBQ deploy-v1 SDK cvt.ub spelling gate: baseline, cvt.ub variants,
    // layout rows, and summary.
    max_results = 15;
  } else if (mode == 289) {
    // LPBQ deploy-v1 SDK clear/swap spelling gate: Q6_mxclracc* and
    // Q6_mxswapacc* rows plus a no-promotion summary.
    max_results = 14;
  } else if (mode == 263) {
    // LPBQ deploy-v1 cvt.hf dependency gate: payload, FP16 control, eight
    // byte-MMA dependency rows, and one summary row.
    max_results = 11;
  } else if (mode == 264) {
    // LPBQ deploy-v1 adaptive row0 derive gate: correctness, shift-count,
    // current timing, row0-even timing, and scalar timing rows.
    max_results = 5;
  } else if (mode == 266) {
    // LPBQ deploy-v1 fullstream FP-bank ownership gate: payload, FP controls,
    // full-clear byte views, no-clear byte-over-FP views, and summary.
    max_results = 10;
  } else if (mode == 267) {
    // LPBQ deploy-v1 direct-HF model gate: payload, byte/direct-HF baselines,
    // no-clear byte-over-FP matrix, intrinsic variants, cvt.hf rows, summary.
    max_results = 26;
  } else if (mode == 268) {
    // LPBQ deploy-v1 mode-266 prelude ablation: byte baseline, FP direct/cvt
    // prelude matrix, and summary.
    max_results = 14;
  } else if (mode == 269) {
    // LPBQ deploy-v1 V81 preloaded-bias + HF-clear order closure: payload,
    // FP/byte controls, four split/bundled order pairs, and summary.
    max_results = 12;
  } else if (mode == 320) {
    // LPBQ deploy-v1 R4 sidecar prelayout equivalence gate: metadata, rows
    // 1/8/32 equivalence/timing, summary, and unchanged-production flags.
    max_results = 6;
  } else if (mode == 321) {
    // LPBQ deploy-v1 R4 prelayout full-K decode-shape gate:
    // m=1,k=8960,n=1536 metadata, equivalence/timing, summary, flags.
    max_results = 4;
  } else if (mode == 322) {
    // LPBQ deploy-v1 R4 prelayout full-K prefill row-block gate:
    // rows=32,k=8960,n=1536 metadata, equivalence/timing, summary, flags.
    max_results = 4;
  } else if (mode == 323) {
    // LPBQ deploy-v1 R4 prelayout full-K active-row sweep:
    // metadata, rows 1/2/4/8/16/32 equivalence/timing, and flags.
    max_results = 8;
  } else if (mode == 324) {
    // LPBQ deploy-v1 R4 prelayout split timing:
    // metadata, rows 1/32 dot/fill/pack split rows, and flags.
    max_results = 8;
  } else if (mode == 325) {
    // LPBQ deploy-v1 R4 prelayout direct-pack gate:
    // metadata, rows 1/32 current-vs-direct rows, and flags.
    max_results = 4;
  } else if (mode == 326) {
    // LPBQ deploy-v1 R4 prelayout direct-pack order gate:
    // metadata, rows 1/32 tile-vs-row-major rows, and flags.
    max_results = 4;
  } else if (mode == 327) {
    // LPBQ deploy-v1 R4 prelayout direct-pack no-clear upper-bound gate:
    // metadata, rows 1/32 tile-vs-noclear rows, and flags.
    max_results = 4;
  } else if (mode == 328) {
    // LPBQ deploy-v1 R4 prelayout old-layout direct-pack gate:
    // metadata, rows 1/32 old-full-vs-direct rows, and flags.
    max_results = 4;
  } else if (mode == 329) {
    // LPBQ deploy-v1 R4 old-layout store8+inv-scale direct-pack gate:
    // metadata, rows 1/32 old-full-vs-store8 rows, and flags.
    max_results = 4;
  } else if (mode >= 331 && mode <= 347) {
    // PREFILL_R4_HMX_FWHT_OPTIMIZATION_README.md Stage A standalone probes.
    // Modes 331..347 return exactly eight M/K rows from the DSP A0/A1/A2/A3,
    // sidecar-consumer, activation-packet lower-bound, row-major direct
    // activation producer, production-M direct-producer sweep, and A1-safe
    // FP16-temp/A2 direct-V6 conservative-boundary/split/producer microbenches;
    // keep the old default 4096-row fallback untouched for unrelated broad sweep
    // modes so this route does not flood gate parsing. The previous 331..344
    // and 331..346 ranges are preserved by this comment for maintenance/revert
    // context.
    max_results = 8;
  } else if (mode >= 348 && mode <= 375) {
    // LPBQ deploy-v1 no-quality/performance-first standard-FHT standalone
    // probes. Modes 348..354 are preserved rejected evidence; modes 355..357
    // are the HVX tail-butterfly full-U8 probes; modes 358..360 retarget the
    // same FHT idea to default low4/high4 nibble chunks; modes 361..363 measure
    // core-only/row-major lower bounds; modes 364..366 measure row-pair direct
    // grouped-V6 stores; modes 367..369 measure int16 direct FHT; modes 370..372
    // measure full-HVX-prefix grouped-V6 FHT; modes 373..375 measure
    // full-HVX-prefix default-nibble FHT. Keep logs compact.
    max_results = 10;
  } else if ((mode >= 22 && mode <= 25) || mode == 27 || mode == 28 || mode == 29 || mode == 30 || mode == 31 ||
             mode == 32 || mode == 33 || mode == 34 || mode == 35 || mode == 36 || mode == 37 || mode == 38 ||
             mode == 44) {
    max_results = 1;
  } else if (mode == 41) {
    max_results = 1;
  } else if (mode == 0) {
    max_results = 516;
  } else if (mode == 5) {
    max_results = 1;
  } else if (mode == 11) {
    max_results = 16384;
  } else if (mode == 12) {
    max_results = 128;
  } else if (mode == 10) {
    max_results = 1056;
  } else if (mode == 7 || mode == 9) {
    max_results = 256;
  }
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

struct HostMappedBuffer {
  void  *data;
  int    fd;
  size_t size;
};

static void host_mapped_buffer_init(struct HostMappedBuffer *buf) {
  buf->data = NULL;
  buf->fd   = -1;
  buf->size = 0;
}

static int host_mapped_buffer_alloc(struct HostMappedBuffer *buf, size_t size) {
  host_mapped_buffer_init(buf);
  if (alloc_shared_mem_buf(&buf->data, &buf->fd, size)) {
    return -1;
  }
  buf->size = size;
  return 0;
}

static void host_mapped_buffer_free(struct HostMappedBuffer *buf) {
  if (buf->data) {
    free_shared_mem_buf(buf->data, buf->fd, buf->size);
  }
  host_mapped_buffer_init(buf);
}

static bool host_file_exists(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    return false;
  }
  fclose(f);
  return true;
}

static int lpbq_join_path(char *dst, size_t dst_size, const char *dir, const char *stem, const char *suffix) {
  int n = snprintf(dst, dst_size, "%s/%s%s", dir, stem, suffix);
  if (n < 0 || (size_t) n >= dst_size) {
    fprintf(stderr, "LPBQ_REAL_LAYER path too long: dir=%s stem=%s suffix=%s\n", dir, stem, suffix);
    return -1;
  }
  return 0;
}

static int lpbq_read_exact_file_to_host(const char *path, void *dst, size_t expected_bytes) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "LPBQ_REAL_LAYER missing file: %s\n", path);
    return -1;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fprintf(stderr, "LPBQ_REAL_LAYER fseek failed: %s\n", path);
    fclose(f);
    return -1;
  }
  long file_size = ftell(f);
  if (file_size < 0 || (size_t) file_size != expected_bytes) {
    fprintf(stderr, "LPBQ_REAL_LAYER size mismatch: %s got=%ld expected=%zu\n", path, file_size, expected_bytes);
    fclose(f);
    return -1;
  }
  rewind(f);
  size_t nread = fread(dst, 1, expected_bytes, f);
  fclose(f);
  if (nread != expected_bytes) {
    fprintf(stderr, "LPBQ_REAL_LAYER fread failed: %s got=%zu expected=%zu\n", path, nread, expected_bytes);
    return -1;
  }
  return 0;
}

static int lpbq_load_exact_file_to_rpcmem(const char *path, size_t expected_bytes, struct HostMappedBuffer *out) {
  host_mapped_buffer_init(out);
  if (alloc_shared_mem_buf(&out->data, &out->fd, expected_bytes)) {
    fprintf(stderr, "LPBQ_REAL_LAYER rpcmem alloc failed: %s bytes=%zu\n", path, expected_bytes);
    return -1;
  }
  out->size = expected_bytes;
  if (lpbq_read_exact_file_to_host(path, out->data, expected_bytes) != 0) {
    host_mapped_buffer_free(out);
    return -1;
  }
  return 0;
}

static int lpbq_load_optional_file_to_rpcmem(const char *path, size_t expected_bytes, struct HostMappedBuffer *out,
                                             bool *present) {
  host_mapped_buffer_init(out);
  *present = host_file_exists(path);
  if (!*present) {
    return 0;
  }
  return lpbq_load_exact_file_to_rpcmem(path, expected_bytes, out);
}

static bool lpbq_host_hmx_k_group_safe_tile(const int8_t *tile0, const int8_t *tile1,
                                            int group_chunks, int max_abs_sum_w) {
  for (int c = 0; c < 32; ++c) {
    int sum_abs = 0;
    for (int kg = 0; kg < group_chunks; ++kg) {
      const int8_t *tile = (kg < 8 || tile1 == NULL) ? tile0 : tile1;
      const int local_kg = (kg < 8 || tile1 == NULL) ? kg : (kg - 8);
      const int8_t *chunk = tile + (size_t) local_kg * 128u + (size_t) c * 4u;
      for (int kk = 0; kk < 4; ++kk) {
        int w = (int) chunk[kk];
        sum_abs += w < 0 ? -w : w;
      }
    }
    if (sum_abs > max_abs_sum_w) {
      return false;
    }
  }
  return true;
}

static bool lpbq_host_hmx_k32_safe_tile(const int8_t *tile) {
  // LPBQ deploy-v1 exact fallback correctness: K32 safe bits can be consumed by
  // the unscaled exact path, where activation nibbles are q4<<1.  The older
  // 32767/15 rule is retained here for audit, but the generated sidecar now uses
  // the strict 32767/30 bound so exact fallback cannot overflow after.uh.
  // return lpbq_host_hmx_k_group_safe_tile(tile, NULL, 8, 32767 / 15);
  return lpbq_host_hmx_k_group_safe_tile(tile, NULL, 8, 32767 / 30);
}

static bool lpbq_host_hmx_k64_safe_tile(const int8_t *tile0, const int8_t *tile1) {
  // LPBQ deploy-v1 exact-K64 correctness fix: the previous K64 sidecar used the
  // K32 bound below, but HMX activation nibbles are issued as q4<<1.  Use the
  // stricter guard before combining two K32 tiles into one after.uh drain.
  // return lpbq_host_hmx_k_group_safe_tile(tile0, tile1, 16, 32767 / 15);
  return lpbq_host_hmx_k_group_safe_tile(tile0, tile1, 16, 32767 / 30);
}

static int lpbq_load_packed_weight_k_major(const char *path, int k, int n, struct HostMappedBuffer *packed_k_major,
                                           int8_t **packed_n_major_host) {
  host_mapped_buffer_init(packed_k_major);
  *packed_n_major_host = NULL;
  if (k <= 0 || n <= 0 || (k % 32) != 0 || (n % 32) != 0) {
    fprintf(stderr, "LPBQ_REAL_LAYER invalid packed weight shape k=%d n=%d\n", k, n);
    return -1;
  }
  const size_t expected_bytes = (size_t) k * (size_t) n;
  int8_t *n_major = (int8_t *) malloc(expected_bytes);
  if (!n_major) {
    fprintf(stderr, "LPBQ_REAL_LAYER malloc failed for packed N-major copy bytes=%zu\n", expected_bytes);
    return -1;
  }
  if (lpbq_read_exact_file_to_host(path, n_major, expected_bytes) != 0) {
    free(n_major);
    return -1;
  }
  if (alloc_shared_mem_buf(&packed_k_major->data, &packed_k_major->fd, expected_bytes)) {
    fprintf(stderr, "LPBQ_REAL_LAYER rpcmem alloc failed for packed K-major bytes=%zu\n", expected_bytes);
    free(n_major);
    return -1;
  }
  packed_k_major->size = expected_bytes;

  const int k_tiles = k / 32;
  const int n_tiles = n / 32;
  const size_t chunk = 8u * 128u;
  int8_t *k_major = (int8_t *) packed_k_major->data;
  for (int kt = 0; kt < k_tiles; ++kt) {
    for (int nt = 0; nt < n_tiles; ++nt) {
      const size_t src = ((size_t) nt * (size_t) k_tiles + (size_t) kt) * chunk;
      const size_t dst = ((size_t) kt * (size_t) n_tiles + (size_t) nt) * chunk;
      memcpy(k_major + dst, n_major + src, chunk);
    }
  }
  *packed_n_major_host = n_major;
  return 0;
}

static size_t lpbq_v6_full_grouped_bytes(int k, int n, int group_tiles) {
  if (k <= 0 || n <= 0 || group_tiles <= 0 || (k % 32) != 0 || (n % 32) != 0) {
    return 0;
  }
  const int k_tiles = k / 32;
  const int n_tiles = n / 32;
  const int groups = (k_tiles + group_tiles - 1) / group_tiles;
  return (size_t) groups * (size_t) n_tiles * (size_t) group_tiles * 2048u;
}

static int lpbq_load_packed_weight_v6_full_grouped(const char *path, int k, int n, int group_tiles,
                                                   struct HostMappedBuffer *packed_v6_full,
                                                   int8_t **packed_n_major_host) {
  host_mapped_buffer_init(packed_v6_full);
  *packed_n_major_host = NULL;
  if (k <= 0 || n <= 0 || group_tiles <= 0 || (k % 32) != 0 || (n % 32) != 0 ||
      group_tiles <= 1) {
    fprintf(stderr, "LPBQ_REAL_LAYER invalid V6 full shape k=%d n=%d group_tiles=%d\n", k, n, group_tiles);
    return -1;
  }

  const int k_tiles = k / 32;
  const int n_tiles = n / 32;
  const size_t compact_chunk = 8u * 128u;
  const size_t compact_bytes = (size_t) k * (size_t) n;
  const size_t full_tile_bytes = 2048u;
  // Old strict A/B used k_tiles*n_tiles*2048 and required exact groups.  The
  // padded-tail form keeps one complete slot for the final short K group.
  const size_t full_bytes = lpbq_v6_full_grouped_bytes(k, n, group_tiles);

  int8_t *n_major = (int8_t *) malloc(compact_bytes);
  if (!n_major) {
    fprintf(stderr, "LPBQ_REAL_LAYER malloc failed for V6 full source bytes=%zu\n", compact_bytes);
    return -1;
  }
  if (lpbq_read_exact_file_to_host(path, n_major, compact_bytes) != 0) {
    free(n_major);
    return -1;
  }
  if (alloc_shared_mem_buf(&packed_v6_full->data, &packed_v6_full->fd, full_bytes)) {
    fprintf(stderr, "LPBQ_REAL_LAYER rpcmem alloc failed for V6 full bytes=%zu\n", full_bytes);
    free(n_major);
    return -1;
  }
  packed_v6_full->size = full_bytes;

  int8_t *full = (int8_t *) packed_v6_full->data;
  memset(full, 0, full_bytes);
  for (int kg = 0; kg < k_tiles; kg += group_tiles) {
    const int group_index = kg / group_tiles;
    for (int nt = 0; nt < n_tiles; ++nt) {
      int8_t *group_dst =
        full + ((size_t) group_index * (size_t) n_tiles + (size_t) nt) *
                 (size_t) group_tiles * full_tile_bytes;
      for (int sub = 0; sub < group_tiles; ++sub) {
        const int kt = kg + sub;
        if (kt >= k_tiles) {
          continue;
        }
        const int8_t *compact_src =
          n_major + ((size_t) nt * (size_t) k_tiles + (size_t) kt) * compact_chunk;
        memcpy(group_dst + (size_t) sub * full_tile_bytes, compact_src, compact_chunk);
      }
    }
  }

  *packed_n_major_host = n_major;
  return 0;
}

static int lpbq_create_k32_safe_from_k_major(const int8_t *packed_k_major, int k, int n,
                                             struct HostMappedBuffer *k32_safe) {
  host_mapped_buffer_init(k32_safe);
  const int k_tiles = k / 32;
  const int n_tiles = n / 32;
  const size_t bytes = (size_t) k_tiles * (size_t) n_tiles * sizeof(uint8_t);
  if (alloc_shared_mem_buf(&k32_safe->data, &k32_safe->fd, bytes)) {
    fprintf(stderr, "LPBQ_REAL_LAYER rpcmem alloc failed for k32_safe bytes=%zu\n", bytes);
    return -1;
  }
  k32_safe->size = bytes;
  uint8_t *dst = (uint8_t *) k32_safe->data;
  const size_t chunk = 8u * 128u;
  for (int kt = 0; kt < k_tiles; ++kt) {
    for (int nt = 0; nt < n_tiles; ++nt) {
      const size_t idx = (size_t) kt * (size_t) n_tiles + (size_t) nt;
      dst[idx] = lpbq_host_hmx_k32_safe_tile(packed_k_major + idx * chunk) ? 1u : 0u;
    }
  }
  return 0;
}

static int lpbq_create_k64_safe_from_k_major(const int8_t *packed_k_major, int k, int n,
                                             struct HostMappedBuffer *k64_safe) {
  host_mapped_buffer_init(k64_safe);
  if ((k % 64) != 0 || (n % 32) != 0) {
    return -1;
  }
  const int k32_tiles = k / 32;
  const int k64_tiles = k / 64;
  const int n_tiles = n / 32;
  const size_t bytes = (size_t) k64_tiles * (size_t) n_tiles * sizeof(uint8_t);
  if (alloc_shared_mem_buf(&k64_safe->data, &k64_safe->fd, bytes)) {
    fprintf(stderr, "LPBQ_REAL_LAYER rpcmem alloc failed for k64_safe bytes=%zu\n", bytes);
    return -1;
  }
  k64_safe->size = bytes;
  uint8_t *dst = (uint8_t *) k64_safe->data;
#if HTP_LPBQ_EXACT_K64_FORCE_UNSAFE
  // LPBQ deploy-v1 exact-K64 diagnostic: keep the exact-K branch enabled while
  // forcing every K64 group down the already-validated K32/K16 fallback path.
  // This distinguishes a bad safe-K64 drain from broader loop/indexing issues.
  memset(dst, 0, bytes);
  return 0;
#else
  const size_t chunk = 8u * 128u;
  for (int kt64 = 0; kt64 < k64_tiles; ++kt64) {
    for (int nt = 0; nt < n_tiles; ++nt) {
      const int kt0 = kt64 * 2;
      const size_t tile0 = ((size_t) kt0 * (size_t) n_tiles + (size_t) nt) * chunk;
      const size_t tile1 = ((size_t) (kt0 + 1) * (size_t) n_tiles + (size_t) nt) * chunk;
      const size_t idx = (size_t) kt64 * (size_t) n_tiles + (size_t) nt;
      dst[idx] = lpbq_host_hmx_k64_safe_tile(packed_k_major + tile0, packed_k_major + tile1) ? 1u : 0u;
    }
  }
  return 0;
#endif
}

static int lpbq_load_act_scale_value(const char *path, float *out) {
  return lpbq_read_exact_file_to_host(path, out, sizeof(float));
}

enum {
  LPBQ_R4_FWHT_HEADER_FLOATS = 16,
  LPBQ_R4_FWHT_D2_OFFSET = LPBQ_R4_FWHT_HEADER_FLOATS,
  LPBQ_R4_FWHT_D1_OFFSET = LPBQ_R4_FWHT_D2_OFFSET + 128,
};

static const float LPBQ_R4_FWHT_MAGIC0 = 314159.25f;
static const float LPBQ_R4_FWHT_MAGIC1 = -271828.25f;

static int lpbq_r4_sylvester_negative_host(int row, int col) {
  unsigned x = (unsigned) (row & col);
  x ^= x >> 16;
  x ^= x >> 8;
  x ^= x >> 4;
  x &= 0xfu;
  return (int) ((0x6996u >> x) & 1u);
}

static int lpbq_r4_fwht_companion_path(const char *path, char *dst, size_t dst_size) {
  const char *suffix = ".r4.f32";
  const size_t path_len = strlen(path);
  const size_t suffix_len = strlen(suffix);
  if (path_len >= suffix_len && strcmp(path + path_len - suffix_len, suffix) == 0) {
    const int n = snprintf(dst, dst_size, "%.*s.r4.fwht_d1d2.f32",
                           (int) (path_len - suffix_len), path);
    return (n < 0 || (size_t) n >= dst_size) ? -1 : 0;
  }
  const int n = snprintf(dst, dst_size, "%s.fwht_d1d2.f32", path);
  return (n < 0 || (size_t) n >= dst_size) ? -1 : 0;
}

static int lpbq_load_r4_col_major(const char *path, int block, int k, const float *input_scale,
                                  bool fold_input_scale, struct HostMappedBuffer *r4_col_major,
                                  float **r4_row_major_host, bool *r4_structured_fwht) {
  host_mapped_buffer_init(r4_col_major);
  *r4_row_major_host = NULL;
  if (r4_structured_fwht) {
    *r4_structured_fwht = false;
  }
  if (block <= 0 || (block % 32) != 0 || (k % block) != 0) {
    fprintf(stderr, "LPBQ_REAL_LAYER invalid R4 shape block=%d k=%d\n", block, k);
    return -1;
  }
  const size_t matrix_elems = (size_t) block * (size_t) block;
  const size_t matrix_bytes = matrix_elems * sizeof(float);

  char fwht_path[1024];
  if (block == 128 && lpbq_r4_fwht_companion_path(path, fwht_path, sizeof(fwht_path)) == 0 &&
      host_file_exists(fwht_path)) {
    float fwht_scales[256];
    if (lpbq_read_exact_file_to_host(fwht_path, fwht_scales, sizeof(fwht_scales)) != 0) {
      return -1;
    }
    float *row_major = (float *) malloc(matrix_bytes);
    if (!row_major) {
      fprintf(stderr, "LPBQ_REAL_LAYER malloc failed for structured R4 row-major bytes=%zu\n", matrix_bytes);
      return -1;
    }
    for (int row = 0; row < 128; ++row) {
      const float d2 = fwht_scales[row];
      for (int col = 0; col < 128; ++col) {
        const float sign = lpbq_r4_sylvester_negative_host(row, col) ? -1.0f : 1.0f;
        row_major[(size_t) row * 128u + (size_t) col] = sign * d2 * fwht_scales[128u + (size_t) col];
      }
    }

    const int n_blocks = fold_input_scale && input_scale ? (k / block) : 1;
    const size_t mapped_bytes = (size_t) n_blocks * matrix_bytes;
    if (alloc_shared_mem_buf(&r4_col_major->data, &r4_col_major->fd, mapped_bytes)) {
      fprintf(stderr, "LPBQ_REAL_LAYER rpcmem alloc failed for structured R4 bytes=%zu\n", mapped_bytes);
      free(row_major);
      return -1;
    }
    r4_col_major->size = mapped_bytes;
    float *payload = (float *) r4_col_major->data;
    memset(payload, 0, mapped_bytes);
    for (int blk = 0; blk < n_blocks; ++blk) {
      float *block_payload = payload + (size_t) blk * matrix_elems;
      block_payload[0] = LPBQ_R4_FWHT_MAGIC0;
      block_payload[1] = LPBQ_R4_FWHT_MAGIC1;
      block_payload[2] = (float) block;
      block_payload[3] = 1.0f;
      float *d2_payload = block_payload + LPBQ_R4_FWHT_D2_OFFSET;
      float *d1_payload = block_payload + LPBQ_R4_FWHT_D1_OFFSET;
      for (int row = 0; row < 128; ++row) {
        float d2 = fwht_scales[row];
        if (fold_input_scale && input_scale) {
          d2 *= input_scale[(size_t) blk * 128u + (size_t) row];
        }
        d2_payload[row] = d2;
      }
      for (int col = 0; col < 128; ++col) {
        d1_payload[col] = fwht_scales[128u + (size_t) col];
      }
    }
    *r4_row_major_host = row_major;
    if (r4_structured_fwht) {
      *r4_structured_fwht = true;
    }
    return 0;
  }

  float *row_major = (float *) malloc(matrix_bytes);
  if (!row_major) {
    fprintf(stderr, "LPBQ_REAL_LAYER malloc failed for R4 row-major bytes=%zu\n", matrix_bytes);
    return -1;
  }
  if (lpbq_read_exact_file_to_host(path, row_major, matrix_bytes) != 0) {
    free(row_major);
    return -1;
  }

  const int n_blocks = fold_input_scale && input_scale ? (k / block) : 1;
  const size_t mapped_bytes = (size_t) n_blocks * matrix_bytes;
  if (alloc_shared_mem_buf(&r4_col_major->data, &r4_col_major->fd, mapped_bytes)) {
    fprintf(stderr, "LPBQ_REAL_LAYER rpcmem alloc failed for R4 bytes=%zu\n", mapped_bytes);
    free(row_major);
    return -1;
  }
  r4_col_major->size = mapped_bytes;
  float *col_major = (float *) r4_col_major->data;
  for (int blk = 0; blk < n_blocks; ++blk) {
    for (int col = 0; col < block; ++col) {
      for (int row = 0; row < block; ++row) {
        float coeff = row_major[(size_t) row * (size_t) block + (size_t) col];
        if (fold_input_scale && input_scale) {
          coeff *= input_scale[(size_t) blk * (size_t) block + (size_t) row];
        }
        col_major[(size_t) blk * matrix_elems + (size_t) col * (size_t) block + (size_t) row] = coeff;
      }
    }
  }
  *r4_row_major_host = row_major;
  return 0;
}

static inline int8_t lpbq_real_quantize_f32_to_i8(float value, float scale) {
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

static inline float lpbq_real_activation_value(int r, int kk, float act_scale) {
  const int qs = (int) (((r * 17 + kk * 13 + 19) % 191) - 95);
  return (float) qs * act_scale * 0.75f;
}

static inline size_t lpbq_real_packed_weight_offset(int k, int n, int global_c, int global_k) {
  const int k_tiles = k / 32;
  const int n_tile  = global_c / 32;
  const int c_tile  = global_c & 31;
  const int k_tile  = global_k / 32;
  const int k_in    = global_k & 31;
  const int kg       = k_in / 4;
  const int kk       = k_in & 3;
  (void) n;
  // The sidecar is compact HMX K4, not row-major 32x32:
  // packed[((nt * k_tiles + kt) * 8 + kg) * 128 + 4 * c + kk].
  return (((size_t) n_tile * (size_t) k_tiles + (size_t) k_tile) * 8u + (size_t) kg) * 128u +
         (size_t) (4 * c_tile + kk);
}

static float lpbq_real_rotated_activation_at(const float *activation, const float *r4_row_major,
                                             const float *input_scale, int r, int kk, int k, int r4_block) {
  if (!r4_row_major || r4_block <= 0) {
    float v = activation[(size_t) r * (size_t) k + (size_t) kk];
    if (input_scale) {
      v *= input_scale[kk];
    }
    return v;
  }
  const int block_start = (kk / r4_block) * r4_block;
  const int col = kk - block_start;
  const float *src = activation + (size_t) r * (size_t) k + (size_t) block_start;
  float acc = 0.0f;
  for (int s = 0; s < r4_block; ++s) {
    const int src_k = block_start + s;
    float v = src[s];
    if (input_scale) {
      v *= input_scale[src_k];
    }
    acc += v * r4_row_major[(size_t) s * (size_t) r4_block + (size_t) col];
  }
  return acc;
}

static float lpbq_real_reference_at(const float *activation, const int8_t *packed_weight_n_major,
                                    const float *input_scale, const float *r4_row_major,
                                    const float *out_scale, const float *bias_eff,
                                    float act_scale, int m, int k, int n, int r, int c, int r4_block) {
  (void) m;
  int32_t raw_u = 0;
  for (int kk = 0; kk < k; ++kk) {
    const float value = lpbq_real_rotated_activation_at(activation, r4_row_major, input_scale, r, kk, k, r4_block);
    const int8_t q_s = lpbq_real_quantize_f32_to_i8(value, act_scale);
    const int q_u = (int) q_s + 128;
    const int8_t w = packed_weight_n_major[lpbq_real_packed_weight_offset(k, n, c, kk)];
    raw_u += (int32_t) q_u * (int32_t) w;
  }
  return (float) raw_u * out_scale[c] + bias_eff[c];
}

static int32_t lpbq_real_raw_u_reference_at(const float *activation, const int8_t *packed_weight_n_major,
                                            const float *input_scale, const float *r4_row_major,
                                            float act_scale, int m, int k, int n, int r, int c,
                                            int r4_block) {
  (void) m;
  int32_t raw_u = 0;
  for (int kk = 0; kk < k; ++kk) {
    const float value = lpbq_real_rotated_activation_at(activation, r4_row_major, input_scale, r, kk, k, r4_block);
    const int8_t q_s = lpbq_real_quantize_f32_to_i8(value, act_scale);
    const int q_u = (int) q_s + 128;
    const int8_t w = packed_weight_n_major[lpbq_real_packed_weight_offset(k, n, c, kk)];
    raw_u += (int32_t) q_u * (int32_t) w;
  }
  return raw_u;
}

static const char *lpbq_real_profile_stage_name(int stage) {
  switch (stage) {
    case LLM_TRACE_STAGE_R4_ROTATE: return "r4_rotate";
    case LLM_TRACE_STAGE_R4_SCALE_CACHE: return "r4_scale_cache";
    case LLM_TRACE_STAGE_R4_DOT_PACK: return "r4_dot_pack";
    case LLM_TRACE_STAGE_R4_DENSE_DOT: return "r4_dense_dot";
    case LLM_TRACE_STAGE_R4_QUANT_SCALE: return "r4_quant_scale";
    case LLM_TRACE_STAGE_R4_PACK_UB_LAYOUT: return "r4_pack_ub_layout";
    case LLM_TRACE_STAGE_R4_UNACCOUNTED: return "r4_unaccounted";
    case LLM_TRACE_STAGE_ACTIVATION_QUANTIZE: return "activation_quantize";
    case LLM_TRACE_STAGE_ACTIVATION_PACK: return "activation_pack";
    case LLM_TRACE_STAGE_WEIGHT_HVX_LOAD: return "weight_hvx_load";
    case LLM_TRACE_STAGE_HMX_MMA: return "hmx_mma";
    case LLM_TRACE_STAGE_HMX_BEGIN: return "hmx_begin";
    case LLM_TRACE_STAGE_HMX_WEIGHT_EXPAND: return "hmx_weight_expand";
    case LLM_TRACE_STAGE_HMX_ISSUE: return "hmx_issue";
    case LLM_TRACE_STAGE_HMX_FINISH: return "hmx_finish";
    case LLM_TRACE_STAGE_HMX_ACCUMULATE: return "hmx_accumulate";
    case LLM_TRACE_STAGE_HMX_CORE: return "hmx_core";
    case LLM_TRACE_STAGE_HMX_UNACCOUNTED: return "hmx_unaccounted";
    case LLM_TRACE_STAGE_HMX_ADAPTIVE_PROBE_ISSUE: return "hmx_adaptive_probe_issue";
    case LLM_TRACE_STAGE_HMX_ADAPTIVE_PROBE_FINISH: return "hmx_adaptive_probe_finish";
    case LLM_TRACE_STAGE_HMX_ADAPTIVE_SCALE_DERIVE: return "hmx_adaptive_scale_derive";
    case LLM_TRACE_STAGE_HMX_ADAPTIVE_FINAL_ISSUE: return "hmx_adaptive_final_issue";
    case LLM_TRACE_STAGE_HMX_ADAPTIVE_FINAL_FINISH: return "hmx_adaptive_final_finish";
    case LLM_TRACE_STAGE_HMX_GROUPED_V6_CONTROL_GAP: return "hmx_grouped_v6_control_gap";
    case LLM_TRACE_STAGE_ZERO_POINT_CORRECTION: return "zero_point_correction";
    case LLM_TRACE_STAGE_DEQUANT_STORE: return "dequant_store";
    case LLM_TRACE_STAGE_LPBQ_PATH_DIAG: return "lpbq_path_diag";
    case LLM_TRACE_STAGE_WEIGHT_RPCMEM_READ: return "weight_rpcmem_read";
    case LLM_TRACE_STAGE_WEIGHT_L2FETCH_OR_DMA: return "weight_l2fetch_or_dma";
    case LLM_TRACE_STAGE_WEIGHT_COMPACT_DECODE: return "weight_compact_decode";
    case LLM_TRACE_STAGE_WEIGHT_GROUP_TILE_COPY: return "weight_group_tile_copy";
    case LLM_TRACE_STAGE_WEIGHT_ROWBLOCK4_PUBLISH: return "weight_rowblock4_publish";
    case LLM_TRACE_STAGE_WEIGHT_G32_STAGING: return "weight_g32_staging";
    case LLM_TRACE_STAGE_WEIGHT_DMA_ISSUE: return "weight_dma_issue";
    case LLM_TRACE_STAGE_WEIGHT_DMA_WAIT_LPBQ: return "weight_dma_wait";
    case LLM_TRACE_STAGE_WEIGHT_VISIBILITY_SYNC: return "weight_visibility_sync";
    case LLM_TRACE_STAGE_WEIGHT_FALLBACK_HVX_PUBLISH: return "weight_fallback_hvx_publish";
    case LLM_TRACE_STAGE_WEIGHT_VTCM_COMMIT: return "weight_vtcm_commit";
    case LLM_TRACE_STAGE_WEIGHT_HVX_LOAD_ACTUAL: return "weight_hvx_load_actual";
    case LLM_TRACE_STAGE_WEIGHT_CACHE_LOOKUP: return "weight_cache_lookup";
    case LLM_TRACE_STAGE_WEIGHT_CACHE_FILL: return "weight_cache_fill";
    case LLM_TRACE_STAGE_WEIGHT_UNATTRIBUTED: return "weight_unattributed";
    case LLM_TRACE_STAGE_R4_FWHT_LOAD: return "r4_fwht_load";
    case LLM_TRACE_STAGE_R4_FWHT_BUTTERFLY: return "r4_fwht_butterfly";
    case LLM_TRACE_STAGE_R4_FWHT_SCALE: return "r4_fwht_scale";
    case LLM_TRACE_STAGE_R4_FWHT_QUANT: return "r4_fwht_quant";
    case LLM_TRACE_STAGE_R4_FWHT_V6_STORE: return "r4_fwht_v6_store";
    case LLM_TRACE_STAGE_ACT_REDUCE_MAX: return "act_reduce_max";
    case LLM_TRACE_STAGE_ACT_RECIP_SCALE: return "act_recip_scale";
    case LLM_TRACE_STAGE_ACT_QUANT: return "act_quant";
    case LLM_TRACE_STAGE_ACT_V6_STORE: return "act_v6_store";
    case LLM_TRACE_STAGE_HMX_ACQUIRE: return "hmx_acquire";
    case LLM_TRACE_STAGE_HMX_SCALE_PAYLOAD_LOAD: return "hmx_scale_payload_load";
    case LLM_TRACE_STAGE_HMX_ACC_CLEAR: return "hmx_acc_clear";
    case LLM_TRACE_STAGE_HMX_LOAD_ISSUE: return "hmx_load_issue";
    case LLM_TRACE_STAGE_HMX_ACCUMULATE_WAIT: return "hmx_accumulate_wait";
    case LLM_TRACE_STAGE_HMX_CONVERT_ISSUE: return "hmx_convert_issue";
    case LLM_TRACE_STAGE_HMX_CONVERT_WAIT: return "hmx_convert_wait";
    case LLM_TRACE_STAGE_HMX_EPILOGUE_HVX: return "hmx_epilogue_hvx";
    case LLM_TRACE_STAGE_HMX_WEIGHT_EXPAND_EXPOSED: return "hmx_weight_expand_exposed";
    case LLM_TRACE_STAGE_HMX_WEIGHT_PREPUBLISH_HIDDEN: return "hmx_weight_prepublish_hidden";
    default: return "unknown";
  }
}

static void lpbq_print_real_layer_profile(const struct HostMappedBuffer *profile) {
  if (!profile || !profile->data || profile->size < sizeof(struct LlmTraceProfileHeader)) {
    return;
  }
  const struct LlmTraceProfileHeader *hdr = (const struct LlmTraceProfileHeader *) profile->data;
  if (hdr->magic != LLM_TRACE_PROFILE_MAGIC) {
    fprintf(stderr, "LPBQ_REAL_LAYER_PROFILE bad_magic=0x%x\n", hdr->magic);
    return;
  }
  int event_count = hdr->event_count;
  if (event_count > hdr->max_events) {
    event_count = hdr->max_events;
  }
  fprintf(stderr, "LPBQ_REAL_LAYER_PROFILE events=%d max=%d overflow=%d\n", event_count, hdr->max_events,
          hdr->event_overflow);
  const struct LlmTraceProfileEvent *events = llm_trace_profile_events_const(hdr);
  for (int i = 0; i < event_count; ++i) {
    const struct LlmTraceProfileEvent *e = events + i;
    if (e->stage == LLM_TRACE_STAGE_LPBQ_PATH_DIAG) {
      fprintf(stderr,
              "LPBQ_REAL_LAYER_PATH_DIAG k64_total=%d k64_safe=%d k64_unsafe=%d fallback_k32=%d "
              "fallback_k16=%d est_drains=%lld est_k4_issues=%lld flags=%d "
              "diag0=%d diag1=%d diag2=%d diag3=%d diag4=%d diag5=%lld diag6=%lld diag7=%d\n",
              e->mr, e->nc, e->kk, e->chunk_m, e->chunk_n, (long long) e->bytes,
              (long long) e->dur_us, e->chunk_k, e->mr, e->nc, e->kk, e->chunk_m, e->chunk_n,
              (long long) e->bytes, (long long) e->dur_us, e->chunk_k);
      continue;
    }
    fprintf(stderr,
            "LPBQ_REAL_LAYER_STAGE stage=%s stage_id=%d unit=%d dur_us=%lld bytes=%lld m=%d k=%d n=%d\n",
            lpbq_real_profile_stage_name(e->stage), e->stage, e->unit, (long long) e->dur_us,
            (long long) e->bytes, e->m, e->k, e->n);
  }
}

static int lpbq_a8w8_matmul_send_request(struct MessageHeader *msg, size_t max_msg_size,
                                         const struct LpbqA8W8MatMulParams *params) {
  struct RequestHeader req_hdr = {
    .state = 0,
    .type  = REQUEST_TYPE_OP_COMPUTE,
  };
  struct OpComputeRequest compute_req = {
    .op = HTP_OPS_MAT_MUL_LPBQ_A8W8,
  };

  msg->state.d        = 0;
  msg->n_reqs         = 1;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + sizeof(req_hdr) + sizeof(compute_req) + sizeof(*params);
  if ((size_t) msg->req_offsets[1] > max_msg_size) {
    fprintf(stderr, "LPBQ real layer gate message buffer is too small\n");
    return -1;
  }

  uint8_t *p                  = (uint8_t *) message_header_get_request_ptr(msg, 0);
  *(struct RequestHeader *) p = req_hdr;
  p += sizeof(req_hdr);
  *(struct OpComputeRequest *) p = compute_req;
  p += sizeof(compute_req);
  *(struct LpbqA8W8MatMulParams *) p = *params;

  msg->state.v[0] = 1;
  if (figure8_wait_channel(msg, 120000000)) {
    return -1;
  }
  return message_header_get_request_ptr(msg, 0)->state;
}

static int run_lpbq_real_layer_gate(const struct Figure8AttnConfig *cfg) {
  const int m = cfg->matmul_m;
  const int k = cfg->matmul_k;
  const int n = cfg->matmul_n;
  const int r4_block = cfg->lpbq_r4_block;
  const bool use_r4 = cfg->lpbq_r4_path && cfg->lpbq_r4_path[0];
  const bool use_full_v6_weight = cfg->lpbq_full_v6_weight != 0;
  const int full_v6_group_tiles = cfg->lpbq_full_v6_group_tiles > 0 ? cfg->lpbq_full_v6_group_tiles : 8;
  char suffix_packed_full_v6[64];
  char path_scale2[1024];
  char path_bias[1024];
  char path_input_scale[1024];
  char path_packed[1024];
  char path_packed_k4[1024];
  char path_sum_w[1024];
  char path_act_scale[1024];

  struct HostMappedBuffer chan, output, activation, dummy_weight, packed_weight, packed_weight_ref, sum_w, k32_safe, k64_safe;
  struct HostMappedBuffer scale2, bias, input_scale, out_scale, bias_eff, r4, r4_reference, profile;
  host_mapped_buffer_init(&chan);
  host_mapped_buffer_init(&output);
  host_mapped_buffer_init(&activation);
  host_mapped_buffer_init(&dummy_weight);
  host_mapped_buffer_init(&packed_weight);
  host_mapped_buffer_init(&packed_weight_ref);
  host_mapped_buffer_init(&sum_w);
  host_mapped_buffer_init(&k32_safe);
  host_mapped_buffer_init(&k64_safe);
  host_mapped_buffer_init(&scale2);
  host_mapped_buffer_init(&bias);
  host_mapped_buffer_init(&input_scale);
  host_mapped_buffer_init(&out_scale);
  host_mapped_buffer_init(&bias_eff);
  host_mapped_buffer_init(&r4);
  host_mapped_buffer_init(&r4_reference);
  host_mapped_buffer_init(&profile);

  int8_t *packed_n_major_host = NULL;
  float *r4_row_major_host = NULL;
  float *r4_reference_row_major_host = NULL;
  bool r4_structured_fwht = false;
  int ret = 1;

  if (m <= 0 || k <= 0 || n <= 0 || (k % 32) != 0 || (n % 32) != 0) {
    fprintf(stderr, "LPBQ_REAL_LAYER invalid shape m=%d k=%d n=%d\n", m, k, n);
    return 1;
  }
  if (use_r4 && (r4_block <= 0 || (k % r4_block) != 0)) {
    fprintf(stderr, "LPBQ_REAL_LAYER invalid R4 config k=%d r4_block=%d\n", k, r4_block);
    return 1;
  }
  if (use_full_v6_weight &&
      (full_v6_group_tiles <= 1 || (k % 32) != 0)) {
    fprintf(stderr, "LPBQ_REAL_LAYER invalid full-V6 group config k=%d group_tiles=%d\n",
            k, full_v6_group_tiles);
    return 1;
  }
  snprintf(suffix_packed_full_v6, sizeof(suffix_packed_full_v6),
           ".lpbq_w_hmx_v6_full_g%d.bin", full_v6_group_tiles);
  // LPBQ deploy-v1 g8 full-V6/R4 proof: this guard used to reject R4 because
  // the first full-V6 sidecar A/B only covered non-R4 layers.  Keep that old
  // behavior as a breadcrumb while allowing ffn_down to prove the g8 route
  // through the same real-layer gate before any LLM integration.
  // if (use_full_v6_weight && use_r4) {
  //   fprintf(stderr, "LPBQ_REAL_LAYER --lpbq-full-v6-weight is currently non-R4 only\n");
  //   return 1;
  // }
  if (lpbq_join_path(path_scale2, sizeof(path_scale2), cfg->lpbq_sidecar_dir, cfg->lpbq_layer_stem,
                     ".lpbq_scale2.f32") ||
      lpbq_join_path(path_bias, sizeof(path_bias), cfg->lpbq_sidecar_dir, cfg->lpbq_layer_stem,
                     ".lpbq_bias.f32") ||
      lpbq_join_path(path_input_scale, sizeof(path_input_scale), cfg->lpbq_sidecar_dir, cfg->lpbq_layer_stem,
                     ".lpbq_input_scale.f32") ||
      lpbq_join_path(path_packed, sizeof(path_packed), cfg->lpbq_sidecar_dir, cfg->lpbq_layer_stem,
                     use_full_v6_weight ? suffix_packed_full_v6 : ".lpbq_w_hmx_k4.bin") ||
      lpbq_join_path(path_packed_k4, sizeof(path_packed_k4), cfg->lpbq_sidecar_dir, cfg->lpbq_layer_stem,
                     ".lpbq_w_hmx_k4.bin") ||
      lpbq_join_path(path_sum_w, sizeof(path_sum_w), cfg->lpbq_sidecar_dir, cfg->lpbq_layer_stem,
                     ".lpbq_sum_w.i32") ||
      lpbq_join_path(path_act_scale, sizeof(path_act_scale), cfg->lpbq_sidecar_dir, cfg->lpbq_layer_stem,
                     ".lpbq_act_scale.f32")) {
    return 1;
  }

  float act_scale = 0.0f;
  bool has_bias = false;
  bool has_input_scale = false;
  const size_t output_size = (size_t) m * (size_t) n * sizeof(float);
  const size_t activation_size = (size_t) m * (size_t) k * sizeof(float);
  const size_t q8_dummy_size = (size_t) n * (size_t) k / QK_0 * sizeof(block_q8_0);
  const size_t packed_size = use_full_v6_weight ?
                               lpbq_v6_full_grouped_bytes(k, n, full_v6_group_tiles) :
                               (size_t) k * (size_t) n;
  const size_t sum_w_size = (size_t) n * sizeof(int32_t);
  const size_t scale_size = (size_t) n * sizeof(float);
  const size_t input_scale_size = (size_t) k * sizeof(float);
  const size_t max_msg_size = 4096;
  const int profile_max_events = cfg->lpbq_trace_diag ? 64 : 0;
  const size_t profile_size =
    profile_max_events > 0 ? sizeof(struct LlmTraceProfileHeader) +
                               (size_t) profile_max_events * sizeof(struct LlmTraceProfileEvent) : 0;

  if (lpbq_load_act_scale_value(path_act_scale, &act_scale) != 0 ||
      lpbq_load_exact_file_to_rpcmem(path_scale2, scale_size, &scale2) != 0 ||
      lpbq_load_exact_file_to_rpcmem(path_sum_w, sum_w_size, &sum_w) != 0 ||
      lpbq_load_optional_file_to_rpcmem(path_bias, scale_size, &bias, &has_bias) != 0 ||
      lpbq_load_optional_file_to_rpcmem(path_input_scale, input_scale_size, &input_scale, &has_input_scale) != 0 ||
      (use_full_v6_weight ?
         lpbq_load_exact_file_to_rpcmem(path_packed, packed_size, &packed_weight) :
         lpbq_load_packed_weight_k_major(path_packed, k, n, &packed_weight, &packed_n_major_host)) != 0 ||
      (use_full_v6_weight &&
       // Reference-only K4 load: do not pass this buffer to DSP. It preserves
       // the strict host check while the online request consumes the offline
       // expanded g8 full-V6 sidecar above.
       lpbq_load_packed_weight_k_major(path_packed_k4, k, n, &packed_weight_ref, &packed_n_major_host) != 0) ||
      (!use_full_v6_weight &&
       lpbq_create_k32_safe_from_k_major((const int8_t *) packed_weight.data, k, n, &k32_safe) != 0) ||
      (!use_full_v6_weight &&
       lpbq_create_k64_safe_from_k_major((const int8_t *) packed_weight.data, k, n, &k64_safe) != 0) ||
      host_mapped_buffer_alloc(&out_scale, scale_size) ||
      host_mapped_buffer_alloc(&bias_eff, scale_size) ||
      host_mapped_buffer_alloc(&output, output_size) ||
      host_mapped_buffer_alloc(&activation, activation_size) ||
      host_mapped_buffer_alloc(&dummy_weight, q8_dummy_size) ||
      (profile_max_events > 0 && host_mapped_buffer_alloc(&profile, profile_size)) ||
      host_mapped_buffer_alloc(&chan, max_msg_size)) {
    fprintf(stderr, "LPBQ_REAL_LAYER setup failed\n");
    goto end;
  }

  memset(output.data, 0, output_size);
  memset(dummy_weight.data, 0, q8_dummy_size);
  if (profile.data) {
    memset(profile.data, 0, profile.size);
  }

  float *activation_ptr = (float *) activation.data;
  for (int r = 0; r < m; ++r) {
    for (int kk = 0; kk < k; ++kk) {
      activation_ptr[(size_t) r * (size_t) k + (size_t) kk] = lpbq_real_activation_value(r, kk, act_scale);
    }
  }

  float *out_scale_ptr = (float *) out_scale.data;
  float *bias_eff_ptr = (float *) bias_eff.data;
  const float *scale2_ptr = (const float *) scale2.data;
  const float *bias_ptr = has_bias ? (const float *) bias.data : NULL;
  const int32_t *sum_w_ptr = (const int32_t *) sum_w.data;
  for (int c = 0; c < n; ++c) {
    out_scale_ptr[c] = scale2_ptr[c] * act_scale;
    const float b = bias_ptr ? bias_ptr[c] : 0.0f;
    bias_eff_ptr[c] = b - 128.0f * (float) sum_w_ptr[c] * out_scale_ptr[c];
  }

  if (use_r4) {
    if (lpbq_load_r4_col_major(cfg->lpbq_r4_path, r4_block, k,
                               has_input_scale ? (const float *) input_scale.data : NULL,
                               cfg->lpbq_fold_r4_input_scale != 0, &r4, &r4_row_major_host,
                               &r4_structured_fwht) != 0) {
      goto end;
    }
    if (cfg->lpbq_r4_reference_path && cfg->lpbq_r4_reference_path[0]) {
      // LPBQ deploy-v1 Hadamard/FHT diagnostic: let DSP consume an approximate
      // candidate R4 while host reference stays on the trained R4. Production
      // uses the same path for both and never sets this option.
      bool r4_reference_structured_fwht = false;
      if (lpbq_load_r4_col_major(cfg->lpbq_r4_reference_path, r4_block, k,
                                 has_input_scale ? (const float *) input_scale.data : NULL,
                                 cfg->lpbq_fold_r4_input_scale != 0,
                                 &r4_reference, &r4_reference_row_major_host,
                                 &r4_reference_structured_fwht) != 0) {
        goto end;
      }
    }
  }

  if (create_htp_message_channel(chan.fd, max_msg_size)) {
    fprintf(stderr, "Create LPBQ real layer gate message channel failed\n");
    goto end;
  }

  const bool exact_non_r4_qk =
    !use_r4 && cfg->lpbq_layer_stem &&
    (strstr(cfg->lpbq_layer_stem, ".attn_q.weight") != NULL ||
     strstr(cfg->lpbq_layer_stem, ".attn_k.weight") != NULL);
  struct LpbqA8W8MatMulParams params = {
    .output = { .fd = output.fd, .offset = 0 },
    .activation = { .fd = activation.fd, .offset = 0 },
    .weight = { .fd = dummy_weight.fd, .offset = 0 },
    .packed_weight = { .fd = packed_weight.fd, .offset = 0 },
    // LPBQ deploy-v1 folded-dequant real-layer gate: sum_w is consumed on host
    // to build bias_eff, then deliberately omitted from the online DSP request.
    .sum_w = { .fd = -1, .offset = 0 },
    .k32_safe = { .fd = use_full_v6_weight ? -1 : k32_safe.fd, .offset = 0 },
    .k64_safe = { .fd = use_full_v6_weight ? -1 : k64_safe.fd, .offset = 0 },
    .scale2 = { .fd = scale2.fd, .offset = 0 },
    .bias = { .fd = has_bias ? bias.fd : -1, .offset = 0 },
    .r4 = { .fd = use_r4 ? r4.fd : -1, .offset = 0 },
    .input_scale = {
      .fd = (has_input_scale && !(use_r4 && cfg->lpbq_fold_r4_input_scale)) ? input_scale.fd : -1,
      .offset = 0,
    },
    .out_scale = { .fd = out_scale.fd, .offset = 0 },
    .bias_eff = { .fd = bias_eff.fd, .offset = 0 },
    .act_scale = act_scale,
    .r4_block = use_r4 ? r4_block : 0,
    .m = m,
    .k = k,
    .n = n,
    .trace_id = 0,
    .mode_flags = LLM_NPU_MODE_LPBQ_INT8 | LLM_NPU_MODE_LPBQ_PACKED_K_MAJOR |
                  (use_full_v6_weight ? LLM_NPU_MODE_LPBQ_PACKED_V6_FULL : 0) |
                  // Real-layer validation hook: force R4 full-U8 without
                  // requiring a full-V6 sidecar, matching the LLM backend bit.
                  ((use_full_v6_weight || cfg->lpbq_force_r4_full_u8_safe) ?
                   LLM_NPU_MODE_LPBQ_R4_FULL_U8_SAFE : 0) |
                  (use_r4 ? LLM_NPU_MODE_LPBQ_R4 : 0) |
                  (r4_structured_fwht ? LLM_NPU_MODE_LPBQ_R4_STRUCTURED_FWHT : 0) |
                  (cfg->lpbq_r4_v6_scale_1_16 ? LLM_NPU_MODE_LPBQ_R4_V6_SCALE_1_16 : 0) |
                  ((use_r4 && cfg->lpbq_fold_r4_input_scale) ? LLM_NPU_MODE_LPBQ_R4_FOLDED_INPUT_SCALE : 0) |
                  (exact_non_r4_qk ? LLM_NPU_MODE_LPBQ_EXACT_NON_R4 : 0) |
                  (cfg->lpbq_trace_diag ? (LLM_NPU_MODE_TRACE | LLM_NPU_MODE_DETAILED_TRACE |
                                           LLM_NPU_MODE_LPBQ_PATH_DIAG) : 0),
    .max_profile_events = profile_max_events,
    .profile = { .fd = profile.fd, .offset = 0 },
  };

  int64_t t0 = get_time_us();
  int req_ret = lpbq_a8w8_matmul_send_request((struct MessageHeader *) chan.data, max_msg_size, &params);
  int64_t elapsed_us = get_time_us() - t0;
  if (req_ret != 0) {
    fprintf(stderr, "LPBQ_REAL_LAYER DSP request failed ret=%d\n", req_ret);
    goto release_maps;
  }
  if (cfg->lpbq_trace_diag) {
    lpbq_print_real_layer_profile(&profile);
  }

  const float *out_ptr = (const float *) output.data;
  const float *input_scale_ptr = has_input_scale ? (const float *) input_scale.data : NULL;
  int samples = cfg->lpbq_samples > 0 ? cfg->lpbq_samples : 128;
  const int total = m * n;
  if (samples > total) {
    samples = total;
  }
  float max_abs_err = 0.0f;
  double sq_err = 0.0;
  int first_bad = -1;
  float sample_out = 0.0f;
  float sample_ref = 0.0f;
  float first_bad_out = 0.0f;
  float first_bad_ref = 0.0f;
  int first_bad_raw_ref = 0;
  int first_bad_raw_out_est = 0;
  float first_bad_raw_delta_scaled = 0.0f;
  enum { LPBQ_TOPERR_COUNT = 4 };
  float top_err[LPBQ_TOPERR_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
  int top_idx[LPBQ_TOPERR_COUNT] = {-1, -1, -1, -1};
  int top_raw_out_est[LPBQ_TOPERR_COUNT] = {0, 0, 0, 0};
  int top_raw_ref[LPBQ_TOPERR_COUNT] = {0, 0, 0, 0};
  float top_out_scale[LPBQ_TOPERR_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f};
  // LPBQ deploy-v1 R4 full-V6 diagnostic: keep a compact sampled raw-delta
  // histogram so we can distinguish constant epilogue bias from HMX residuals.
  int raw_delta_hist[13] = {0};
  int raw_delta_hist_count = 0;
  int raw_delta_hist_min = 0;
  int raw_delta_hist_max = 0;
  double raw_delta_hist_sum = 0.0;
  float raw_delta_weighted_max = 0.0f;
  int raw_delta_weighted_idx = -1;
  int raw_delta_weighted_raw_delta = 0;
  float raw_delta_weighted_out_scale = 0.0f;
  for (int s = 0; s < samples; ++s) {
    int idx = 0;
    if (s < 16) {
      const int anchor_cols[16] = {0, 1, 31, 32, 63, 64, 127, 128, n / 2,
                                   n - 129, n - 65, n - 33, n - 32, n - 2, n - 1, 17};
      int row = (s % 3 == 0) ? 0 : ((s % 3 == 1) ? (m / 2) : (m - 1));
      int col = anchor_cols[s];
      if (col < 0) col = 0;
      if (col >= n) col = n - 1;
      idx = row * n + col;
    } else {
      idx = (int) (((uint64_t) s * 2654435761ULL + 17ULL) % (uint64_t) total);
    }
    int row = idx / n;
    int col = idx % n;
    float ref = lpbq_real_reference_at(activation_ptr, packed_n_major_host, input_scale_ptr,
                                       use_r4 ? (r4_reference_row_major_host ?
                                                   r4_reference_row_major_host : r4_row_major_host) :
                                                NULL,
                                       out_scale_ptr, bias_eff_ptr,
                                       act_scale, m, k, n, row, col, r4_block);
    float err = fabsf(out_ptr[idx] - ref);
    const float col_out_scale = out_scale_ptr[col];
    int sample_raw_out_est = 0;
    int sample_raw_ref = 0;
    float sample_raw_delta_scaled = 0.0f;
    if (col_out_scale != 0.0f) {
      const float raw_est_f = (out_ptr[idx] - bias_eff_ptr[col]) / col_out_scale;
      sample_raw_out_est = (int) (raw_est_f >= 0.0f ? raw_est_f + 0.5f : raw_est_f - 0.5f);
      sample_raw_ref = lpbq_real_raw_u_reference_at(
        activation_ptr, packed_n_major_host, input_scale_ptr,
        use_r4 ? (r4_reference_row_major_host ? r4_reference_row_major_host : r4_row_major_host) : NULL,
        act_scale, m, k, n, row, col, r4_block);
      sample_raw_delta_scaled = (float) (sample_raw_out_est - sample_raw_ref) * col_out_scale;
      const int sample_raw_delta = sample_raw_out_est - sample_raw_ref;
      const int hist_slot = sample_raw_delta < -5 ? 0 :
                            (sample_raw_delta > 5 ? 12 : sample_raw_delta + 6);
      ++raw_delta_hist[hist_slot];
      if (raw_delta_hist_count == 0) {
        raw_delta_hist_min = sample_raw_delta;
        raw_delta_hist_max = sample_raw_delta;
      } else {
        if (sample_raw_delta < raw_delta_hist_min) raw_delta_hist_min = sample_raw_delta;
        if (sample_raw_delta > raw_delta_hist_max) raw_delta_hist_max = sample_raw_delta;
      }
      raw_delta_hist_sum += (double) sample_raw_delta;
      ++raw_delta_hist_count;
      const float weighted_abs = fabsf(sample_raw_delta_scaled);
      if (weighted_abs > raw_delta_weighted_max) {
        raw_delta_weighted_max = weighted_abs;
        raw_delta_weighted_idx = idx;
        raw_delta_weighted_raw_delta = sample_raw_delta;
        raw_delta_weighted_out_scale = col_out_scale;
      }
    }
    if (s == 0) {
      sample_out = out_ptr[idx];
      sample_ref = ref;
    }
    for (int rank = 0; rank < LPBQ_TOPERR_COUNT; ++rank) {
      if (err > top_err[rank]) {
        for (int mv = LPBQ_TOPERR_COUNT - 1; mv > rank; --mv) {
          top_err[mv] = top_err[mv - 1];
          top_idx[mv] = top_idx[mv - 1];
          top_raw_out_est[mv] = top_raw_out_est[mv - 1];
          top_raw_ref[mv] = top_raw_ref[mv - 1];
          top_out_scale[mv] = top_out_scale[mv - 1];
        }
        top_err[rank] = err;
        top_idx[rank] = idx;
        top_raw_out_est[rank] = sample_raw_out_est;
        top_raw_ref[rank] = sample_raw_ref;
        top_out_scale[rank] = col_out_scale;
        break;
      }
    }
    if (err > max_abs_err) {
      max_abs_err = err;
      first_bad = idx;
      first_bad_out = out_ptr[idx];
      first_bad_ref = ref;
      if (col_out_scale != 0.0f) {
        first_bad_raw_out_est = sample_raw_out_est;
        first_bad_raw_ref = sample_raw_ref;
        first_bad_raw_delta_scaled = sample_raw_delta_scaled;
      } else {
        first_bad_raw_out_est = 0;
        first_bad_raw_ref = 0;
        first_bad_raw_delta_scaled = 0.0f;
      }
    }
    sq_err += (double) err * (double) err;
  }
  // LPBQ deploy-v1 diagnostic: check whether full-V6 R4 error is a mostly
  // column-constant epilogue offset that could be folded into bias_eff, or a
  // row-varying MMA/R4 error that must be fixed before safe-listing a layer.
  float raw_col_span_max = 0.0f;
  float raw_col_mean_abs = 0.0f;
  int raw_delta_min = 0;
  int raw_delta_max = 0;
  long long raw_delta_sum = 0;
  int raw_col_count = 0;
  {
    const int diag_cols[8] = {0, 1, 31, 32, 64, 128, n / 2, n - 1};
    for (int dc = 0; dc < 8; ++dc) {
      int col = diag_cols[dc];
      if (col < 0) col = 0;
      if (col >= n) col = n - 1;
      int rows[3] = {0, m / 2, m - 1};
      int col_min_delta = 0;
      int col_max_delta = 0;
      bool have_col_delta = false;
      for (int dr = 0; dr < 3; ++dr) {
        int row = rows[dr];
        if (row < 0) row = 0;
        if (row >= m) row = m - 1;
        const int idx = row * n + col;
        const float col_out_scale = out_scale_ptr[col];
        if (col_out_scale == 0.0f) {
          continue;
        }
        const float raw_est_f = (out_ptr[idx] - bias_eff_ptr[col]) / col_out_scale;
        const int raw_out_est =
          (int) (raw_est_f >= 0.0f ? raw_est_f + 0.5f : raw_est_f - 0.5f);
        const int raw_ref = lpbq_real_raw_u_reference_at(
          activation_ptr, packed_n_major_host, input_scale_ptr,
          use_r4 ? (r4_reference_row_major_host ? r4_reference_row_major_host : r4_row_major_host) : NULL,
          act_scale, m, k, n, row, col, r4_block);
        const int raw_delta = raw_out_est - raw_ref;
        if (raw_col_count == 0) {
          raw_delta_min = raw_delta;
          raw_delta_max = raw_delta;
        } else {
          if (raw_delta < raw_delta_min) raw_delta_min = raw_delta;
          if (raw_delta > raw_delta_max) raw_delta_max = raw_delta;
        }
        if (!have_col_delta) {
          col_min_delta = raw_delta;
          col_max_delta = raw_delta;
          have_col_delta = true;
        } else {
          if (raw_delta < col_min_delta) col_min_delta = raw_delta;
          if (raw_delta > col_max_delta) col_max_delta = raw_delta;
        }
        raw_col_mean_abs += fabsf((float) raw_delta);
        raw_delta_sum += (long long) raw_delta;
        ++raw_col_count;
      }
      if (have_col_delta) {
        const float span = (float) (col_max_delta - col_min_delta);
        if (span > raw_col_span_max) {
          raw_col_span_max = span;
        }
      }
    }
    if (raw_col_count > 0) {
      raw_col_mean_abs /= (float) raw_col_count;
    }
  }
  const float raw_delta_mean =
    raw_col_count > 0 ? (float) ((double) raw_delta_sum / (double) raw_col_count) : 0.0f;
  const float rmse = samples > 0 ? (float) sqrt(sq_err / (double) samples) : 0.0f;
  const bool passed = max_abs_err <= 0.05f;
  fprintf(stderr,
          "LPBQ_REAL_LAYER_GATE %s stem=%s m=%d k=%d n=%d samples=%d elapsed_us=%lld max_abs_err=%g "
          "rmse=%g first_bad=%d first_bad_out=%g first_bad_ref=%g sample_out=%g sample_ref=%g "
          "act_scale=%g r4=%d folded_r4_scale=%d raw_out_est=%d raw_ref=%d raw_delta=%d "
          "raw_delta_scaled=%g raw_col_span_max=%g raw_col_mean_abs=%g "
          "raw_delta_min=%d raw_delta_max=%d raw_delta_mean=%g\n",
          passed ? "passed" : "failed", cfg->lpbq_layer_stem, m, k, n, samples, (long long) elapsed_us,
          max_abs_err, rmse, first_bad, first_bad_out, first_bad_ref, sample_out, sample_ref, act_scale,
          use_r4 ? 1 : 0, cfg->lpbq_fold_r4_input_scale ? 1 : 0, first_bad_raw_out_est, first_bad_raw_ref,
          first_bad_raw_out_est - first_bad_raw_ref, first_bad_raw_delta_scaled,
          raw_col_span_max, raw_col_mean_abs, raw_delta_min, raw_delta_max, raw_delta_mean);
  if (cfg->lpbq_trace_diag) {
    const double raw_delta_hist_mean =
      raw_delta_hist_count > 0 ? raw_delta_hist_sum / (double) raw_delta_hist_count : 0.0;
    fprintf(stderr,
            "LPBQ_REAL_LAYER_RAW_HIST count=%d min=%d max=%d mean=%g "
            "lt_m5=%d m5=%d m4=%d m3=%d m2=%d m1=%d z=%d p1=%d p2=%d p3=%d p4=%d p5=%d gt_p5=%d "
            "weighted_idx=%d weighted_raw_delta=%d weighted_abs=%g weighted_out_scale=%g\n",
            raw_delta_hist_count, raw_delta_hist_min, raw_delta_hist_max, raw_delta_hist_mean,
            raw_delta_hist[0], raw_delta_hist[1], raw_delta_hist[2], raw_delta_hist[3],
            raw_delta_hist[4], raw_delta_hist[5], raw_delta_hist[6], raw_delta_hist[7],
            raw_delta_hist[8], raw_delta_hist[9], raw_delta_hist[10], raw_delta_hist[11],
            raw_delta_hist[12], raw_delta_weighted_idx, raw_delta_weighted_raw_delta,
            raw_delta_weighted_max, raw_delta_weighted_out_scale);
    for (int rank = 0; rank < LPBQ_TOPERR_COUNT; ++rank) {
      if (top_idx[rank] < 0) {
        continue;
      }
      const int top_row = top_idx[rank] / n;
      const int top_col = top_idx[rank] % n;
      fprintf(stderr,
              "LPBQ_REAL_LAYER_TOPERR rank=%d idx=%d row=%d col=%d err=%g out_scale=%g raw_out_est=%d raw_ref=%d raw_delta=%d raw_delta_scaled=%g\n",
              rank, top_idx[rank], top_row, top_col, top_err[rank], top_out_scale[rank],
              top_raw_out_est[rank], top_raw_ref[rank],
              top_raw_out_est[rank] - top_raw_ref[rank],
              (float) (top_raw_out_est[rank] - top_raw_ref[rank]) * top_out_scale[rank]);
    }
  }
  ret = passed ? 0 : 1;

release_maps:
  {
    int fds[20];
    int n_fds = 0;
#define LPBQ_ADD_FD(fd_value) do { if ((fd_value) >= 0 && n_fds < (int) (sizeof(fds) / sizeof(fds[0]))) fds[n_fds++] = (fd_value); } while (0)
    LPBQ_ADD_FD(output.fd);
    LPBQ_ADD_FD(activation.fd);
    LPBQ_ADD_FD(dummy_weight.fd);
    LPBQ_ADD_FD(packed_weight.fd);
    // sum_w was not mapped into DSP for this folded no-online-sum_w request.
    // Keep the host rpcmem buffer alive until after validation, then free it
    // below without sending it through figure8_release_dsp_maps().
    LPBQ_ADD_FD(k32_safe.fd);
    LPBQ_ADD_FD(k64_safe.fd);
    LPBQ_ADD_FD(scale2.fd);
    LPBQ_ADD_FD(has_bias ? bias.fd : -1);
    LPBQ_ADD_FD(use_r4 ? r4.fd : -1);
    LPBQ_ADD_FD(has_input_scale ? input_scale.fd : -1);
    LPBQ_ADD_FD(out_scale.fd);
    LPBQ_ADD_FD(bias_eff.fd);
    LPBQ_ADD_FD(profile.fd);
#undef LPBQ_ADD_FD
    if (chan.data && n_fds > 0) {
      (void) figure8_release_dsp_maps((struct MessageHeader *) chan.data, max_msg_size, fds, n_fds);
    }
  }

end:
  if (chan.fd >= 0) {
    htp_ops_destroy_channel(get_global_handle());
  }
  if (packed_n_major_host) free(packed_n_major_host);
  if (r4_row_major_host) free(r4_row_major_host);
  if (r4_reference_row_major_host) free(r4_reference_row_major_host);
  host_mapped_buffer_free(&profile);
  host_mapped_buffer_free(&r4_reference);
  host_mapped_buffer_free(&r4);
  host_mapped_buffer_free(&bias_eff);
  host_mapped_buffer_free(&out_scale);
  host_mapped_buffer_free(&input_scale);
  host_mapped_buffer_free(&bias);
  host_mapped_buffer_free(&scale2);
  host_mapped_buffer_free(&k64_safe);
  host_mapped_buffer_free(&k32_safe);
  host_mapped_buffer_free(&sum_w);
  host_mapped_buffer_free(&packed_weight_ref);
  host_mapped_buffer_free(&packed_weight);
  host_mapped_buffer_free(&dummy_weight);
  host_mapped_buffer_free(&activation);
  host_mapped_buffer_free(&output);
  host_mapped_buffer_free(&chan);
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
      return "TOPS";
    case ROOFLINE_BENCH_KIND_NOT_AVAILABLE:
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

  const int max_results = cfg->roofline_mix_precision_bench ? 160 :
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
      .bytes = (int32_t) bench_size,
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
  __fp16 *k = NULL, *v = NULL, *mask = NULL;
  void   *chan = NULL, *profile = NULL;
  int     q_fd = -1, k_fd = -1, v_fd = -1, o_fd = -1, mask_fd = -1, chan_fd = -1, profile_fd = -1;
  FILE   *csv = NULL;
  int     ret = 1;

  const int    profile_max_records = cfg->n_kv_heads + 8;
  const int    max_q_blocks        = cfg->qo_len;
  const int    max_k_blocks        = ceil_div_int(cfg->kv_len, 64);
  int          profile_max_events  = cfg->n_kv_heads * max_q_blocks * (3 + max_k_blocks * 6) + 128;
  if (profile_max_events < 1024) {
    profile_max_events = 1024;
  }
  const size_t q_size              = align_up((size_t) cfg->qo_len * cfg->n_heads * cfg->head_dim * sizeof(float), 128);
  const size_t kv_size = align_up((size_t) cfg->kv_len * cfg->n_kv_heads * cfg->head_dim * sizeof(__fp16), 128);
  const size_t mask_size = align_up((size_t) cfg->qo_len * cfg->kv_len * sizeof(__fp16), 128);
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

  figure8_fill_inputs(q, k, v, mask, cfg->qo_len, cfg->kv_len, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim);
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

  const int mode_flags = strcmp(cfg->mode, "lut-exp") == 0 ? LLM_NPU_MODE_LUT_EXP : 0;
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
          "FIG8_ATTENTION_CONFIG mode=%s qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d warmup=%d "
          "iters=%d q_size=%zu kv_size=%zu mask_size=%zu profile_size=%zu profile_max_records=%d "
          "profile_max_events=%d mode_flags=%d print_events=%d\n",
          cfg->mode, cfg->qo_len, cfg->kv_len, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, cfg->warmup, cfg->iters,
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
            "FIG8_ATTENTION_HOST_TIMING mode=%s qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d phase=%s "
            "iteration=%d host_elapsed_us=%ld ret=%d\n",
            cfg->mode, cfg->qo_len, cfg->kv_len, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim, phase,
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
              "FIG8_ATTENTION_TIMERS mode=%s phase=%s iteration=%d lut_exp=%d qo_len=%d kv_len=%d n_heads=%d "
              "n_kv_heads=%d head_dim=%d kv_head=%d worker=%d profiled_total=%ld q_load=%ld k_load=%ld v_load=%ld "
              "qk_dot=%ld safe_sm=%ld core_acc=%ld o_scale=%ld o_store=%ld\n",
              cfg->mode, phase, is_warmup ? i : measured_idx, rec->lut_exp, rec->qo_len, rec->kv_len, rec->n_heads,
              rec->n_kv_heads, rec->head_dim, rec->kv_head, rec->worker, rec->profiled_total, rec->q_load,
              rec->k_load, rec->v_load, rec->qk_dot, rec->safe_sm, rec->core_acc, rec->o_scale, rec->o_store);
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
                "FIG8_ATTENTION_EVENT mode=%s phase=%s iteration=%d component=%s component_id=%d lut_exp=%d qo_len=%d "
                "kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d kv_head=%d worker=%d block_r=%d block_c=%d "
                "t0_us=%ld t1_us=%ld dur_us=%ld\n",
                cfg->mode, phase, is_warmup ? i : measured_idx, figure8_component_name(ev->component), ev->component,
                ev->lut_exp, ev->qo_len, ev->kv_len, ev->n_heads, ev->n_kv_heads, ev->head_dim, ev->kv_head, ev->worker,
                ev->block_r, ev->block_c, ev->t0_us, ev->t1_us, ev->dur_us);
      }
    }
  }

  {
    int fds[] = { o_fd, q_fd, k_fd, v_fd, mask_fd, profile_fd };
    (void) figure8_release_dsp_maps(msg, max_msg_size, fds, (int) (sizeof(fds) / sizeof(fds[0])));
  }

  ret = 0;

end:
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

static float test_activation_value(int i, int j) {
  return (float) (((i * 17 + j * 13) % 29) - 14) / 31.0f;
}

static float test_weight_value(int i, int j) {
  return (float) (((i * 7 + j * 11) % 23) - 11) / 37.0f;
}

static bool test_mat_mul_case_rpc(remote_handle64 handle, const char *name, int m, int k, int n) {
  float *activation, *output;
  __fp16 *weight;

  int output_fd, activation_fd, weight_fd;

  if (alloc_shared_mem_buf((void **) &output, &output_fd, m * n * sizeof(float)) ||
      alloc_shared_mem_buf((void **) &activation, &activation_fd, m * k * sizeof(float)) ||
      alloc_shared_mem_buf((void **) &weight, &weight_fd, k * n * sizeof(__fp16))) {
    fprintf(stderr, "%s failed: cannot allocate FastRPC shared buffers for %s (%d,%d,%d)\n", __func__, name, m, k, n);
    return false;
  }

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      activation[i * k + j] = test_activation_value(i, j);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      float x = test_weight_value(i, j);

      int i0 = i / 32, i1 = i % 32;
      int j0 = j / 32, j1 = j % 32;

      int tile_idx = j0 * (k / 32) + i0;
      __fp16 *tile = weight + tile_idx * 1024;
      tile[(i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = (__fp16) x;
    }
  }

  int64_t t0      = get_time_us();
  int rpc_ret     = htp_ops_mat_mul_permuted_w16a32(handle, output_fd, 0, activation_fd, 0, weight_fd, 0, m, k, n);
  int64_t elapsed = get_time_us() - t0;

  const int total = m * n;
  const int sample_count = total < 1024 ? total : 1024;
  float max_abs_err = 0.0f;
  int   max_idx     = 0;
  float max_ref_mix = 0.0f;
  for (int s = 0; s < sample_count; ++s) {
    int idx = total <= 1024 ? s : (int) (((uint64_t) s * 2654435761ULL + 17ULL) % (uint64_t) total);
    int row = idx / n;
    int col = idx % n;
    float ref_mix = 0.0f;
    for (int l = 0; l < k; ++l) {
      ref_mix += (float)((__fp16) test_activation_value(row, l) * ((__fp16) test_weight_value(l, col)));
    }
    float err = fabsf(output[idx] - ref_mix);
    if (err > max_abs_err) {
      max_abs_err = err;
      max_idx     = idx;
      max_ref_mix = ref_mix;
    }
  }
  bool passed = rpc_ret == 0 && max_abs_err < 0.25f;
  fprintf(stderr, "FP16_MATMUL_CASE_TIMING shape=%s m=%d k=%d n=%d elapsed_us=%lld ret=%d\n", name, m, k, n,
          (long long) elapsed, rpc_ret);
  fprintf(stderr, "%s %s shape=%s m=%d k=%d n=%d samples=%d ret=%d max_abs_err=%g idx=%d hmx=%g ref_mix=%g\n",
          __func__, passed ? "passed" : "failed", name, m, k, n, sample_count, rpc_ret, max_abs_err, max_idx,
          output[max_idx], max_ref_mix);
  for (int i = 0; i < 8 && i < m * n; ++i) {
    int row = i / n;
    int col = i % n;
    float ref_mix = 0.0f;
    for (int l = 0; l < k; ++l) {
      ref_mix += (float)((__fp16) test_activation_value(row, l) * ((__fp16) test_weight_value(l, col)));
    }
    fprintf(stderr, "  #%d hmx=%g ref_mix=%g\n", i, output[i], ref_mix);
  }

  free_shared_mem_buf(output, output_fd, m * n * sizeof(float));
  free_shared_mem_buf(activation, activation_fd, m * k * sizeof(float));
  free_shared_mem_buf(weight, weight_fd, k * n * sizeof(__fp16));
  return passed;
}

static bool test_mat_mul_case_rpc_repeated(remote_handle64 handle, const char *name, int m, int k, int n, int repeats) {
  float *activation, *output;
  __fp16 *weight;

  int output_fd, activation_fd, weight_fd;

  if (alloc_shared_mem_buf((void **) &output, &output_fd, m * n * sizeof(float)) ||
      alloc_shared_mem_buf((void **) &activation, &activation_fd, m * k * sizeof(float)) ||
      alloc_shared_mem_buf((void **) &weight, &weight_fd, k * n * sizeof(__fp16))) {
    fprintf(stderr, "%s failed: cannot allocate FastRPC shared buffers for %s (%d,%d,%d)\n", __func__, name, m, k, n);
    return false;
  }

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      activation[i * k + j] = test_activation_value(i, j);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      float x = test_weight_value(i, j);

      int i0 = i / 32, i1 = i % 32;
      int j0 = j / 32, j1 = j % 32;

      int tile_idx = j0 * (k / 32) + i0;
      __fp16 *tile = weight + tile_idx * 1024;
      tile[(i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = (__fp16) x;
    }
  }

  bool ok = true;
  int64_t total_elapsed = 0;
  for (int r = 0; r < repeats; ++r) {
    memset(output, 0, m * n * sizeof(float));
    int64_t t0 = get_time_us();
    int rpc_ret = htp_ops_mat_mul_permuted_w16a32(handle, output_fd, 0, activation_fd, 0, weight_fd, 0, m, k, n);
    int64_t elapsed = get_time_us() - t0;
    total_elapsed += elapsed;
    if (rpc_ret != 0) {
      fprintf(stderr, "%s failed shape=%s repeat=%d/%d ret=%d\n", __func__, name, r + 1, repeats, rpc_ret);
      ok = false;
      break;
    }
  }

  const int total = m * n;
  const int sample_count = total < 1024 ? total : 1024;
  float max_abs_err = 0.0f;
  int   max_idx     = 0;
  float max_ref_mix = 0.0f;
  for (int s = 0; s < sample_count; ++s) {
    int idx = total <= 1024 ? s : (int) (((uint64_t) s * 2654435761ULL + 17ULL) % (uint64_t) total);
    int row = idx / n;
    int col = idx % n;
    float ref_mix = 0.0f;
    for (int l = 0; l < k; ++l) {
      ref_mix += (float)((__fp16) test_activation_value(row, l) * ((__fp16) test_weight_value(l, col)));
    }
    float err = fabsf(output[idx] - ref_mix);
    if (err > max_abs_err) {
      max_abs_err = err;
      max_idx     = idx;
      max_ref_mix = ref_mix;
    }
  }
  ok = ok && max_abs_err < 0.25f;
  fprintf(stderr, "FP16_MATMUL_CASE_TIMING shape=%s m=%d k=%d n=%d repeats=%d total_elapsed_us=%lld avg_elapsed_us=%lld\n",
          name, m, k, n, repeats, (long long) total_elapsed,
          (long long) (repeats > 0 ? total_elapsed / repeats : 0));
  fprintf(stderr, "%s %s shape=%s m=%d k=%d n=%d repeats=%d samples=%d max_abs_err=%g idx=%d hmx=%g ref_mix=%g\n",
          __func__, ok ? "passed" : "failed", name, m, k, n, repeats, sample_count, max_abs_err, max_idx,
          output[max_idx], max_ref_mix);

  free_shared_mem_buf(output, output_fd, m * n * sizeof(float));
  free_shared_mem_buf(activation, activation_fd, m * k * sizeof(float));
  free_shared_mem_buf(weight, weight_fd, k * n * sizeof(__fp16));
  return ok;
}

static bool test_mat_mul_rpc(remote_handle64 handle) {
  // Cover both the small fallback path and Qwen2.5-1.5B prefill shapes. Large
  // cases use sampled CPU references so this standalone gate stays practical
  // while still exercising the exact production W16A32 pipeline.
  bool ok = true;
  ok = test_mat_mul_case_rpc(handle, "small_pipeline", 128, 128, 128) && ok;
  ok = test_mat_mul_case_rpc(handle, "qkv_o_1536", 128, 1536, 1536) && ok;
  ok = test_mat_mul_case_rpc(handle, "kv_prefill_tail_1536x256", 135, 1536, 256) && ok;
  ok = test_mat_mul_case_rpc(handle, "kv_decode_1536x256", 1, 1536, 256) && ok;
  ok = test_mat_mul_case_rpc(handle, "ffn_up_1536x8960", 128, 1536, 8960) && ok;
  ok = test_mat_mul_case_rpc(handle, "ffn_down_8960x1536", 128, 8960, 1536) && ok;
  // Sweep prompts are not always a clean multiple of 32 rows. Keep explicit
  // prefill-tail and decode-token gates so FP16 output-stationary changes do
  // not only pass the ideal m=128 tile case.
  ok = test_mat_mul_case_rpc(handle, "ffn_down_tail_m136", 136, 8960, 1536) && ok;
  ok = test_mat_mul_case_rpc(handle, "ffn_down_decode_m1", 1, 8960, 1536) && ok;
  // Real chat prompts crossed the m=128 boundary at 147 prompt tokens and then
  // exposed failures after many full-K pipeline matmuls in one graph. Keep this
  // stress case in the standalone gate so future FP16 scheduling changes cannot
  // pass only single-call ideal tile shapes.
  ok = test_mat_mul_case_rpc(handle, "qkv_o_tail_m147", 147, 1536, 1536) && ok;
  ok = test_mat_mul_case_rpc(handle, "ffn_up_tail_m147", 147, 1536, 8960) && ok;
  ok = test_mat_mul_case_rpc(handle, "ffn_down_tail_m147", 147, 8960, 1536) && ok;
  ok = test_mat_mul_case_rpc_repeated(handle, "ffn_up_tail_m147_repeat130", 147, 1536, 8960, 130) && ok;
  return ok;
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

  if (figure8_cfg.lpbq_real_layer_gate) {
    int lpbq_layer_ret = run_lpbq_real_layer_gate(&figure8_cfg);
    close_dsp_session();
    return lpbq_layer_ret;
  }

  if (figure8_cfg.matmul_case) {
    bool matmul_ok = test_mat_mul_case_rpc(get_global_handle(), figure8_cfg.matmul_case, figure8_cfg.matmul_m,
                                           figure8_cfg.matmul_k, figure8_cfg.matmul_n);
    close_dsp_session();
    return matmul_ok ? 0 : 1;
  }

  if (figure8_cfg.hmx_int8_gate || figure8_cfg.hmx_int8_gate_search || figure8_cfg.hmx_int8_bitplane_gate ||
      figure8_cfg.hmx_int8_layout_gate || figure8_cfg.hmx_int8_bitop_gate || figure8_cfg.hmx_int8_tile_gate ||
      figure8_cfg.hmx_int8_byte_probe || figure8_cfg.hmx_int8_combo_probe || figure8_cfg.hmx_int8_drop_probe ||
      figure8_cfg.hmx_int8_pack_search || figure8_cfg.hmx_int8_sparse_map || figure8_cfg.hmx_int8_kalign_probe ||
      figure8_cfg.hmx_int8_linearity_probe || figure8_cfg.hmx_int8_full_weight_probe ||
      figure8_cfg.hmx_int8_full_weight_k2_probe || figure8_cfg.w8pc_a8pt_matmul_probe ||
      figure8_cfg.lpbq_a8w8_matmul_probe ||
      figure8_cfg.hmx_int8_mode > 0) {
    int mode = 0;
    if (figure8_cfg.hmx_int8_mode > 0) {
      mode = figure8_cfg.hmx_int8_mode;
    } else if (figure8_cfg.lpbq_a8w8_matmul_probe) {
      mode = 26;
    } else if (figure8_cfg.w8pc_a8pt_matmul_probe) {
      mode = 15;
    } else if (figure8_cfg.hmx_int8_full_weight_k2_probe) {
      mode = 14;
    } else if (figure8_cfg.hmx_int8_full_weight_probe) {
      mode = 13;
    } else if (figure8_cfg.hmx_int8_linearity_probe) {
      mode = 12;
    } else if (figure8_cfg.hmx_int8_kalign_probe) {
      mode = 11;
    } else if (figure8_cfg.hmx_int8_sparse_map) {
      mode = 10;
    } else if (figure8_cfg.hmx_int8_pack_search) {
      mode = 9;
    } else if (figure8_cfg.hmx_int8_drop_probe) {
      mode = 8;
    } else if (figure8_cfg.hmx_int8_combo_probe) {
      mode = 7;
    } else if (figure8_cfg.hmx_int8_byte_probe) {
      mode = 6;
    } else if (figure8_cfg.hmx_int8_tile_gate) {
      mode = 5;
    } else if (figure8_cfg.hmx_int8_bitop_gate) {
      mode = 4;
    } else if (figure8_cfg.hmx_int8_layout_gate) {
      mode = 3;
    } else if (figure8_cfg.hmx_int8_bitplane_gate) {
      mode = 2;
    } else if (figure8_cfg.hmx_int8_gate_search) {
      mode = 1;
    }
    int gate_ret = run_hmx_int8_gate(mode);
    close_dsp_session();
    return gate_ret;
  }

  if (!test_mat_mul_rpc(get_global_handle())) {
    close_dsp_session();
    return 1;
  }

  // Keep the broader internal tests available, but the pure_fp16 workflow uses
  // the focused W16A32 regression above as the fast correctness gate.
  // htp_ops_test_ops(get_global_handle());

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
