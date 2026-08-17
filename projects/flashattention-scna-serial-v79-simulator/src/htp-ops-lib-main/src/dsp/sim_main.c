#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_types.h>
#include <math.h>
#include <qurt_hvx.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/hmx_mgr.h"
#include "dsp/ops.h"
#include "dsp/power.h"
#include "dsp/scna_exp2.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "op_reg.h"

void init_precomputed_tables(void);
int scna_sim_hmx_uses_legacy_adapter(void);

static const char *variant_name(int variant) {
  static const char *names[] = {
    "stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic",
    "pair_static_d8", "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized",
  };
  return variant >= 0 && variant < SCNA_VARIANT_COUNT ? names[variant] : "unknown";
}

static int variant_id(const char *name) {
  if (!name) return -1;
  for (int variant = 0; variant < SCNA_VARIANT_COUNT; ++variant) {
    if (strcmp(name, variant_name(variant)) == 0) return variant;
  }
  return -1;
}

static int parse_int(const char *value, int *out) {
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (!value[0] || !end || *end != '\0' || parsed < 0 || parsed > 0x7fffffffL) return -1;
  *out = (int) parsed;
  return 0;
}

static const char *find_arg(int argc, char **argv, const char *name) {
  for (int i = 2; i + 1 < argc; ++i) {
    if (strcmp(argv[i], name) == 0) return argv[i + 1];
  }
  return NULL;
}

static int int_arg(int argc, char **argv, const char *name, int fallback) {
  const char *value = find_arg(argc, argv, name);
  int parsed = fallback;
  return value && parse_int(value, &parsed) == 0 ? parsed : fallback;
}

static int run_probe(void) {
  unsigned int total = 0, available = 0;
  compute_res_vtcm_page_t total_pages, available_pages;
  const int query_ret = HAP_compute_res_query_VTCM(0, &total, &total_pages, &available, &available_pages);
  const unsigned int hvx_units = qurt_hvx_get_units();
  const int hvx128 = (int) ((hvx_units >> 8) & 0xff);
  const int hvx_lock_ret = qurt_hvx_lock(QURT_HVX_MODE_128B);
  if (hvx_lock_ret == 0) qurt_hvx_unlock();

  power_setup();
  vtcm_manager_setup();
  hmx_manager_setup();
  hmx_manager_enable_execution();

  const int vtcm_ready = vtcm_manager_get_vtcm_base() != NULL && vtcm_manager_get_total_size() > 0;
  const int hmx_api = compute_resource_hmx_lock != NULL && compute_resource_hmx_unlock != NULL;
  const int64_t q0 = HAP_perf_get_qtimer_count();
  const int64_t q1 = HAP_perf_get_qtimer_count();
  const int pass = query_ret == 0 && vtcm_ready && hmx_api && hvx128 > 0 && q1 >= q0;

  FARF(ALWAYS,
       "SIM_CAPABILITY status=%s arch=v79 model=v79na_1 query_ret=%d vtcm_total=%u vtcm_available=%u vtcm_ready=%d hvx128_contexts=%d hvx_lock_ret=%d hmx_legacy_api=%d hmx_lock2_adapter=%d qtimer_delta=%lld",
       pass ? "PASS" : "SKIP", query_ret, total, available, vtcm_ready, hvx128, hvx_lock_ret, hmx_api,
       scna_sim_hmx_uses_legacy_adapter(), (long long) (q1 - q0));

  hmx_manager_disable_execution();
  hmx_manager_reset();
  vtcm_manager_reset();
  power_reset();
  return pass ? 0 : 3;
}

static int run_micro(int argc, char **argv) {
  const int requested = int_arg(argc, argv, "--variant", -1);
  const int warmup = int_arg(argc, argv, "--warmup", 5);
  const int iters = int_arg(argc, argv, "--iters", 1000);
  const int built = scna_exp2_build_variant();
  struct ScnaExp2BenchResult result;
  memset(&result, 0, sizeof(result));

  if (requested < 0 || requested >= SCNA_VARIANT_COUNT || requested != built) {
    FARF(ALWAYS, "SCNA_SIM_RESULT status=FAIL reason=build_id_mismatch requested=%d built=%d", requested, built);
    return 2;
  }
  const int lock_ret = qurt_hvx_lock(QURT_HVX_MODE_128B);
  if (lock_ret != 0) {
    FARF(ALWAYS, "SCNA_SIM_RESULT status=SKIP reason=hvx_lock_failed ret=%d variant=%s", lock_ret,
         variant_name(requested));
    return 3;
  }
  const int ret = scna_exp2_bench_run(&result, 8, SCNA_LAYOUT_SERIAL, requested, warmup, iters);
  qurt_hvx_unlock();
  const int pass = ret == 0 && result.build_variant == requested && result.monotonic_violations == 0 &&
                   result.negative_count == 0 && result.nan_count == 0 && result.random_nonfinite_count == 0 &&
                   result.paired_single_mismatches == 0;
  FARF(ALWAYS,
       "SCNA_SIM_RESULT status=%s ret=%d variant=%s build_id=%d optimized_inline=%d width=%d layout=serial warmup=%d iters=%d elapsed_us=%lld pair_elapsed_us=%lld prepare_elapsed_us=%lld rmse=%.9g max_abs=%.9g dense_rmse=%.9g dense_max_abs=%.9g random_rmse=%.9g random_max_abs=%.9g random_nonfinite_count=%d monotonic_violations=%d negative_count=%d nan_count=%d canonical_oracle_mismatches=%d paired_single_mismatches=%d checksum_bits=%u",
       pass ? "PASS" : "FAIL", ret, variant_name(requested), result.build_variant,
       result.build_optimized_inline, result.width, warmup, iters, (long long) result.elapsed_us,
       (long long) result.pair_elapsed_us, (long long) result.prepare_elapsed_us, result.rmse,
       result.max_abs_error, result.dense_rmse, result.dense_max_abs_error, result.random_rmse,
       result.random_max_abs_error, result.random_nonfinite_count, result.monotonic_violations,
       result.negative_count, result.nan_count, result.canonical_oracle_mismatches,
       result.paired_single_mismatches, result.checksum_bits);
  return pass ? 0 : 1;
}

static uint32_t checksum_float(const float *values, size_t count) {
  uint32_t hash = UINT32_C(2166136261);
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    memcpy(&bits, &values[i], sizeof(bits));
    hash = (hash ^ bits) * UINT32_C(16777619);
  }
  return hash;
}

static void fill_inputs(float *q, __fp16 *k, __fp16 *v, __fp16 *mask, int qo, int kv, int heads,
                        int kv_heads, int dim) {
  const size_t q_count = (size_t) qo * heads * dim;
  const size_t kv_count = (size_t) kv * kv_heads * dim;
  const int kv_pad = (kv + 63) & ~63;
  for (size_t i = 0; i < q_count; ++i) q[i] = (float) ((int) ((i * 17 + 3) % 37) - 18) / 64.0f;
  for (size_t i = 0; i < kv_count; ++i) {
    k[i] = (__fp16) ((float) ((int) ((i * 13 + 5) % 31) - 15) / 64.0f);
    v[i] = (__fp16) ((float) ((int) ((i * 7 + 11) % 29) - 14) / 32.0f);
  }
  for (int r = 0; r < qo; ++r) {
    for (int c = 0; c < kv_pad; ++c) mask[(size_t) r * kv_pad + c] = c < kv ? (__fp16) 0.0f : (__fp16) -65504.0f;
  }
}

static int workers_arg(int argc, char **argv) {
  const char *value = find_arg(argc, argv, "--workers");
  if (!value || strcmp(value, "1") == 0) return 1;
  if (strcmp(value, "auto") == 0) return 0;
  int parsed = -1;
  return parse_int(value, &parsed) == 0 && parsed <= 7 ? parsed : -1;
}

static int mode_flags_for(const char *mode, int built_variant, int requested_variant, int requested_workers) {
  const int worker_flags = requested_workers << LLM_NPU_MODE_WORKER_COUNT_SHIFT;
  if (strcmp(mode, "origin") == 0) return worker_flags;
  if (strcmp(mode, "exp-lut") == 0) return worker_flags | LLM_NPU_MODE_LUT_EXP;
  if (strcmp(mode, "serial") == 0 && requested_variant == built_variant)
    return worker_flags | LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_D8 | (built_variant << 10);
  if (strcmp(mode, "stage1") == 0 && built_variant == SCNA_VARIANT_STAGE1_DYNAMIC_ROW)
    return worker_flags | LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_D8 |
           (SCNA_VARIANT_STAGE1_DYNAMIC_ROW << 10);
  if (strcmp(mode, "optimized") == 0 && built_variant == SCNA_VARIANT_OPTIMIZED)
    return worker_flags | LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_D8 |
           (SCNA_VARIANT_OPTIMIZED << 10);
  return -1;
}

static int run_attention(int argc, char **argv) {
  const char *mode = find_arg(argc, argv, "--mode");
  const char *requested_variant_name = find_arg(argc, argv, "--variant");
  const int qo = int_arg(argc, argv, "--qo", 1);
  const int kv = int_arg(argc, argv, "--kv", 64);
  const int heads = int_arg(argc, argv, "--heads", 2);
  const int kv_heads = int_arg(argc, argv, "--kv-heads", 1);
  const int dim = int_arg(argc, argv, "--head-dim", 64);
  const int warmup = int_arg(argc, argv, "--warmup", 1);
  const int iters = int_arg(argc, argv, "--iters", 5);
  const int tail_check = int_arg(argc, argv, "--tail-check", 0);
  const int built_variant = scna_exp2_build_variant();
  const int requested_variant = variant_id(requested_variant_name);
  const int requested_workers = workers_arg(argc, argv);
  const int mode_flags = mode ? mode_flags_for(mode, built_variant, requested_variant, requested_workers) : -1;
  const char *reported_variant = mode && strcmp(mode, "serial") == 0 ? variant_name(built_variant) :
                                 mode && strcmp(mode, "stage1") == 0 ? variant_name(SCNA_VARIANT_STAGE1_DYNAMIC_ROW) :
                                 mode && strcmp(mode, "optimized") == 0 ? variant_name(SCNA_VARIANT_OPTIMIZED) : "none";
  if (!mode || mode_flags < 0 || qo < 1 || kv < 1 || heads < 1 || kv_heads < 1 || dim < 1 ||
      heads % kv_heads != 0 || dim % 64 != 0 || warmup < 0 || iters < 1 || requested_workers < 0) {
    FARF(ALWAYS,
         "ATTENTION_SMOKE_RESULT status=FAIL reason=invalid_arguments mode=%s variant=%s requested_variant=%s build_id=%d requested_workers=%d qo=%d kv=%d heads=%d kv_heads=%d head_dim=%d",
         mode ? mode : "missing", reported_variant, requested_variant_name ? requested_variant_name : "none",
         built_variant, requested_workers, qo, kv, heads, kv_heads, dim);
    return 2;
  }

  const int kv_pad = (kv + 63) & ~63;
  const size_t q_count = (size_t) qo * heads * dim;
  const size_t kv_count = (size_t) kv * kv_heads * dim;
  const size_t mask_count = (size_t) qo * kv_pad;
  float *q = (float *) memalign(128, q_count * sizeof(float));
  float *out = (float *) memalign(128, q_count * sizeof(float));
  float *reference = (float *) memalign(128, q_count * sizeof(float));
  __fp16 *k = (__fp16 *) memalign(128, kv_count * sizeof(__fp16));
  __fp16 *v = (__fp16 *) memalign(128, kv_count * sizeof(__fp16));
  __fp16 *mask = (__fp16 *) memalign(128, mask_count * sizeof(__fp16));
  const int max_records = 128;
  const size_t profile_bytes = sizeof(struct Figure8ProfileHeader) +
                               (size_t) max_records * sizeof(struct Figure8ProfileRecord);
  struct Figure8ProfileHeader *profile = (struct Figure8ProfileHeader *) memalign(128, profile_bytes);
  if (!q || !out || !reference || !k || !v || !mask || !profile) {
    FARF(ALWAYS, "ATTENTION_SMOKE_RESULT status=FAIL reason=allocation");
    free(q); free(out); free(reference); free(k); free(v); free(mask); free(profile);
    return 1;
  }
  fill_inputs(q, k, v, mask, qo, kv, heads, kv_heads, dim);
  memset(out, 0, q_count * sizeof(float));
  memset(reference, 0, q_count * sizeof(float));
  const int ref_ret = naive_flash_attn(reference, q, k, v, mask, qo, kv, heads, kv_heads, dim);

  power_setup();
  vtcm_manager_setup();
  if (!vtcm_manager_get_vtcm_base() || vtcm_manager_get_total_size() < 1024 * 1024) {
    FARF(ALWAYS, "ATTENTION_SMOKE_RESULT status=SKIP reason=vtcm_unavailable vtcm_total=%u",
         (unsigned) vtcm_manager_get_total_size());
    vtcm_manager_reset(); power_reset();
    free(q); free(out); free(reference); free(k); free(v); free(mask); free(profile);
    return 3;
  }
  hmx_manager_setup();
  if (strcmp(mode, "exp-lut") == 0) init_precomputed_tables();

  int ret = 0;
  for (int iteration = -warmup; iteration < iters; ++iteration) {
    memset(profile, 0, profile_bytes);
    profile->magic = FIGURE8_PROFILE_MAGIC;
    profile->max_records = max_records;
    profile->max_events = 0;
    const int64_t t0 = HAP_perf_get_qtimer_count();
    ret = simple_flash_attn_profiled((__fp16 *) out, (const __fp16 *) q, k, v, mask, qo, kv, heads,
                                     kv_heads, dim, mode_flags, profile);
    const int64_t kernel_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
    if (ret != 0) break;
    if (iteration < 0) continue;
    int64_t profiled_total = 0, q_load = 0, k_load = 0, v_load = 0, qk_dot = 0;
    int64_t safe_sm = 0, core_acc = 0, o_scale = 0, o_store = 0, scna_exp = 0, prepare = 0;
    int tail_nonzero = 0, masked_nonzero = 0;
    const int records = profile->record_count < max_records ? profile->record_count : max_records;
    const struct Figure8ProfileRecord *rows = figure8_profile_records_const(profile);
    for (int i = 0; i < records; ++i) {
      profiled_total += rows[i].profiled_total; q_load += rows[i].q_load; k_load += rows[i].k_load;
      v_load += rows[i].v_load; qk_dot += rows[i].qk_dot; safe_sm += rows[i].safe_sm;
      core_acc += rows[i].core_acc; o_scale += rows[i].o_scale; o_store += rows[i].o_store;
      scna_exp += rows[i].scna_exp; prepare += rows[i].param_prepare;
      tail_nonzero += rows[i].debug_tail_p_nonzero_count;
      masked_nonzero += rows[i].debug_masked_p_nonzero_count;
    }
    FARF(ALWAYS,
         "ATTENTION_TIMER status=PASS mode=%s variant=%s build_id=%d iteration=%d qo=%d kv=%d heads=%d kv_heads=%d head_dim=%d requested_workers=%d active_workers=%d workers=%d records=%d kernel_us=%lld profiled_total_us=%lld q_load_us=%lld k_load_us=%lld v_load_us=%lld qk_dot_us=%lld safe_sm_us=%lld core_acc_us=%lld o_scale_us=%lld o_store_us=%lld scna_exp_us=%lld param_prepare_us=%lld tail_nonzero=%d masked_nonzero=%d checksum_bits=%u",
         mode, reported_variant, built_variant, iteration, qo, kv, heads, kv_heads, dim, requested_workers,
         profile->active_workers, profile->active_workers, records,
         (long long) kernel_us, (long long) profiled_total, (long long) q_load, (long long) k_load,
         (long long) v_load, (long long) qk_dot, (long long) safe_sm, (long long) core_acc,
         (long long) o_scale, (long long) o_store, (long long) scna_exp, (long long) prepare,
         tail_nonzero, masked_nonzero, checksum_float(out, q_count));
  }

  double sum_sq = 0.0;
  float max_abs = 0.0f;
  int candidate_nonfinite = 0, reference_nonfinite = 0;
  for (size_t i = 0; i < q_count; ++i) {
    const float diff = out[i] - reference[i];
    sum_sq += (double) diff * diff;
    if (fabsf(diff) > max_abs) max_abs = fabsf(diff);
    if (!isfinite(out[i])) ++candidate_nonfinite;
    if (!isfinite(reference[i])) ++reference_nonfinite;
  }
  const float rmse = q_count ? (float) sqrt(sum_sq / q_count) : INFINITY;
  const int pass = ret == 0 && ref_ret == 0 && rmse <= 0.002f && max_abs <= 0.01f &&
                   candidate_nonfinite == 0 && reference_nonfinite == 0;
  FARF(ALWAYS,
       "ATTENTION_VERIFY status=%s mode=%s variant=%s build_id=%d requested_workers=%d ret=%d ref_ret=%d qo=%d kv=%d heads=%d kv_heads=%d head_dim=%d tail_check=%d elements=%u rmse=%.9g max_abs=%.9g candidate_nonfinite=%d reference_nonfinite=%d checksum_bits=%u",
       pass ? "PASS" : "FAIL", mode, reported_variant, built_variant, requested_workers, ret, ref_ret,
       qo, kv, heads, kv_heads, dim,
       tail_check, (unsigned) q_count, rmse, max_abs, candidate_nonfinite, reference_nonfinite,
       checksum_float(out, q_count));
  FARF(ALWAYS, "ATTENTION_SMOKE_RESULT status=%s mode=%s variant=%s qo=%d kv=%d build_id=%d requested_workers=%d",
       pass ? "PASS" : "FAIL", mode, reported_variant, qo, kv, built_variant, requested_workers);

  hmx_manager_reset();
  vtcm_manager_reset();
  power_reset();
  free(q); free(out); free(reference); free(k); free(v); free(mask); free(profile);
  return pass ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    FARF(ALWAYS, "SCNA_SIM_USAGE commands=probe|micro|attention");
    return 2;
  }
  FARF(ALWAYS, "SCNA_SIM_START command=%s argc=%d", argv[1], argc);
  if (strcmp(argv[1], "probe") == 0) return run_probe();
  if (strcmp(argv[1], "micro") == 0) return run_micro(argc, argv);
  if (strcmp(argv[1], "attention") == 0) return run_attention(argc, argv);
  FARF(ALWAYS, "SCNA_SIM_USAGE error=unknown_command command=%s", argv[1]);
  return 2;
}
