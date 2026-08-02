#include "htp-ops.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <string>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml.h"

////// Special headers intended for CPU-NPU communication. Keep them in sync with ops backend.
#include "dsprpc_interface.h"
#include "message.h"
#include "op_reg.h"

namespace {

auto get_all_rpcmem_mappings(const ggml_tensor * dst) {
    const auto & mapper = ggml_backend_htp_context::instance()->mapper;

    std::vector<std::pair<int, ssize_t>> mappings;
    if (ggml_backend_buft_is_rpcmem(dst->buffer->buft)) {
        mappings.push_back(mapper.get_tensor_mapping(dst));
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src && ggml_backend_buft_is_rpcmem(src->buffer->buft)) {
            mappings.push_back(mapper.get_tensor_mapping(src));
        }
    }
    return mappings;
}

template <typename T> void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

void write_buf(uint8_t *& p, void * src, size_t size) {
    std::memcpy((void *) p, src, size);
    p += size;
}

uint8_t param_buf[4096];  // TODO(hzx): better implementation

std::atomic<int64_t> g_llm_trace_id{1};
std::atomic<int> g_w8pc_host_debug_count{0};
std::atomic<int> g_flash_host_debug_count{0};

void * g_llm_trace_profile = nullptr;
int    g_llm_trace_profile_fd = -1;
size_t g_llm_trace_profile_size = 0;
int    g_llm_trace_profile_max_events = 0;

const char * get_npu_mode() {
    const char * mode = std::getenv("LLAMA_NPU_MODE");
    return (mode && mode[0]) ? mode : "baseline";
}

bool env_truthy(const char * name) {
    const char * value = std::getenv(name);
    return value && (std::strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 ||
                     strcasecmp(value, "yes") == 0 || strcasecmp(value, "on") == 0);
}

int get_mode_flags(const char * mode) {
    int flags = 0;
    if (std::strcmp(mode, "lut_exp") == 0 || std::strcmp(mode, "lut-exp") == 0) {
        flags |= LLM_NPU_MODE_LUT_EXP;
    }
    if (env_truthy("LLAMA_NPU_TRACE")) {
        flags |= LLM_NPU_MODE_TRACE;
    }
    if (env_truthy("LLAMA_NPU_DETAILED_TRACE")) {
        flags |= LLM_NPU_MODE_TRACE | LLM_NPU_MODE_DETAILED_TRACE;
    }
    return flags;
}

int env_int_or_default(const char * name, int default_value) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > 1000000) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

bool ensure_llm_trace_profile_buffer(int max_events) {
    if (max_events <= 0) {
        return false;
    }
    const size_t required_size = sizeof(LlmTraceProfileHeader) +
            static_cast<size_t>(max_events) * sizeof(LlmTraceProfileEvent);
    if (g_llm_trace_profile && g_llm_trace_profile_size >= required_size) {
        g_llm_trace_profile_max_events = max_events;
        std::memset(g_llm_trace_profile, 0, g_llm_trace_profile_size);
        return true;
    }

    if (g_llm_trace_profile) {
        fastrpc_munmap(CDSP_DOMAIN_ID, g_llm_trace_profile_fd, g_llm_trace_profile, g_llm_trace_profile_size);
        rpcmem_free(g_llm_trace_profile);
        g_llm_trace_profile = nullptr;
        g_llm_trace_profile_fd = -1;
        g_llm_trace_profile_size = 0;
        g_llm_trace_profile_max_events = 0;
    }

    g_llm_trace_profile = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, static_cast<int>(required_size));
    if (!g_llm_trace_profile) {
        return false;
    }
    std::memset(g_llm_trace_profile, 0, required_size);
    g_llm_trace_profile_fd = rpcmem_to_fd(g_llm_trace_profile);
    if (g_llm_trace_profile_fd < 0 ||
        fastrpc_mmap(CDSP_DOMAIN_ID, g_llm_trace_profile_fd, g_llm_trace_profile, 0, required_size,
                     FASTRPC_MAP_FD)) {
        if (g_llm_trace_profile_fd >= 0) {
            fastrpc_munmap(CDSP_DOMAIN_ID, g_llm_trace_profile_fd, g_llm_trace_profile, required_size);
        }
        rpcmem_free(g_llm_trace_profile);
        g_llm_trace_profile = nullptr;
        g_llm_trace_profile_fd = -1;
        return false;
    }
    g_llm_trace_profile_size = required_size;
    g_llm_trace_profile_max_events = max_events;
    return true;
}

bool is_w8pc_a8pt_mode(const char * mode) {
    return std::strcmp(mode, "int8pc") == 0 || std::strcmp(mode, "w8pc_a8pt") == 0 ||
           std::strcmp(mode, "w8pc-a8pt") == 0;
}

const char * htp_op_name(int op_index) {
    switch (op_index) {
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
        default:
            return "unknown";
    }
}

const char * llm_trace_stage_name(int stage) {
    switch (stage) {
        case LLM_TRACE_STAGE_VALIDATE_IN: return "validate_in";
        case LLM_TRACE_STAGE_VALIDATE_OUT: return "validate_out";
        case LLM_TRACE_STAGE_ACTIVATION_HVX_LOAD: return "activation_hvx_load";
        case LLM_TRACE_STAGE_ACTIVATION_DMA_INFLIGHT: return "activation_dma_inflight";
        case LLM_TRACE_STAGE_ACTIVATION_DMA_WAIT: return "activation_dma_wait";
        case LLM_TRACE_STAGE_WEIGHT_DMA_INFLIGHT: return "weight_dma_inflight";
        case LLM_TRACE_STAGE_WEIGHT_DMA_WAIT: return "weight_dma_wait";
        case LLM_TRACE_STAGE_WEIGHT_HVX_DEQUANT: return "weight_hvx_dequant";
        case LLM_TRACE_STAGE_WEIGHT_HVX_LOAD: return "weight_hvx_load";
        case LLM_TRACE_STAGE_HMX_MMA: return "hmx_mma";
        case LLM_TRACE_STAGE_HVX_COMPUTE: return "hvx_compute";
        case LLM_TRACE_STAGE_OUTPUT_STORE: return "output_store";
        case LLM_TRACE_STAGE_ACTIVATION_QUANTIZE: return "activation_quantize";
        case LLM_TRACE_STAGE_ACTIVATION_PACK: return "activation_pack";
        case LLM_TRACE_STAGE_FLASH_Q_LOAD: return "flash_q_load";
        case LLM_TRACE_STAGE_FLASH_K_LOAD: return "flash_k_load";
        case LLM_TRACE_STAGE_FLASH_V_LOAD: return "flash_v_load";
        case LLM_TRACE_STAGE_FLASH_QK_DOT: return "flash_qk_dot";
        case LLM_TRACE_STAGE_FLASH_SAFE_SM: return "flash_safe_sm";
        case LLM_TRACE_STAGE_FLASH_CORE_ACC: return "flash_core_acc";
        case LLM_TRACE_STAGE_FLASH_O_SCALE: return "flash_o_scale";
        case LLM_TRACE_STAGE_FLASH_O_STORE: return "flash_o_store";
        default: return "unknown";
    }
}

const char * llm_trace_unit_name(int unit) {
    switch (unit) {
        case LLM_TRACE_UNIT_DMA: return "dma";
        case LLM_TRACE_UNIT_HVX: return "hvx";
        case LLM_TRACE_UNIT_HMX: return "hmx";
        case LLM_TRACE_UNIT_STORE: return "store";
        case LLM_TRACE_UNIT_MEMORY: return "mem";
        case LLM_TRACE_UNIT_SCALAR: return "scalar";
        default: return "other";
    }
}

const char * category_from_weight_name(const char * name) {
    if (!name) {
        return "other";
    }
    if (std::strstr(name, ".attn_q.weight")) {
        return "q_matrix";
    }
    if (std::strstr(name, ".attn_k.weight")) {
        return "k_matrix";
    }
    if (std::strstr(name, ".attn_v.weight")) {
        return "v_matrix";
    }
    if (std::strstr(name, ".attn_output.weight")) {
        return "o_matrix";
    }
    if (std::strstr(name, ".ffn_gate.weight")) {
        return "ffn_gate";
    }
    if (std::strstr(name, ".ffn_up.weight")) {
        return "ffn_up";
    }
    if (std::strstr(name, ".ffn_down.weight")) {
        return "ffn_down";
    }
    return "matmul_other";
}

const char * phase_from_rows(int rows) {
    return rows > 1 ? "prefill" : "decode";
}

void tensor_stats_f32(const ggml_tensor * t, float & min_v, float & max_v, int & bad_count) {
    const int64_t n_el = ggml_nelements(t);
    const float * vals = reinterpret_cast<const float *>(t->data);
    min_v = n_el > 0 ? vals[0] : 0.0f;
    max_v = min_v;
    bad_count = 0;
    for (int64_t i = 0; i < n_el; ++i) {
        const float v = vals[i];
        if (!std::isfinite(v)) {
            ++bad_count;
            continue;
        }
        if (v < min_v) {
            min_v = v;
        }
        if (v > max_v) {
            max_v = v;
        }
    }
}

void print_llm_trace_profile(const LlmTraceProfileHeader * profile, int64_t trace_id, const char * mode,
                             const char * phase, const char * category, const char * tensor, const char * weight) {
    if (!profile) {
        return;
    }
    if (profile->magic != LLM_TRACE_PROFILE_MAGIC) {
        fprintf(stderr, "LLMTRACE_DSP_STAGE_BAD_PROFILE trace_id=%lld magic=0x%x\n",
                (long long) trace_id, profile->magic);
        return;
    }

    int count = profile->event_count;
    if (count < 0) {
        count = 0;
    }
    if (count > profile->max_events) {
        count = profile->max_events;
    }

    fprintf(stderr,
            "LLMTRACE_DSP_STAGE_EVENT_COUNT trace_id=%lld mode=%s phase=%s category=%s tensor=%s weight=%s "
            "events=%d max_events=%d overflow=%d\n",
            (long long) trace_id, mode, phase, category, tensor ? tensor : "", weight ? weight : "", count,
            profile->max_events, profile->event_overflow);

    const LlmTraceProfileEvent * events = llm_trace_profile_events_const(profile);
    for (int i = 0; i < count; ++i) {
        const auto & e = events[i];
        fprintf(stderr,
                "LLMTRACE_DSP_STAGE_EVENT trace_id=%lld mode=%s flags=%d phase=%s op=%s op_index=%d category=%s "
                "tensor=%s weight=%s stage=%s stage_id=%d unit=%s unit_id=%d worker=%d "
                "m=%d k=%d n=%d qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d "
                "mr=%d nc=%d kk=%d chunk_m=%d chunk_n=%d chunk_k=%d bytes=%lld "
                "raw_t0_us=%lld raw_t1_us=%lld t0_us=%lld t1_us=%lld dur_us=%lld\n",
                (long long) e.trace_id, mode, e.flags, phase, htp_op_name(e.op_index), e.op_index, category,
                tensor ? tensor : "", weight ? weight : "", llm_trace_stage_name(e.stage), e.stage,
                llm_trace_unit_name(e.unit), e.unit, e.worker, e.m, e.k, e.n, e.qo_len, e.kv_len, e.n_heads,
                e.n_kv_heads, e.head_dim, e.mr, e.nc, e.kk, e.chunk_m, e.chunk_n, e.chunk_k,
                (long long) e.bytes, (long long) e.t0_us, (long long) e.t1_us, (long long) e.t0_us,
                (long long) e.t1_us, (long long) e.dur_us);
    }
}

}  // namespace

extern "C" {

bool htp_ops_support_op(const struct ggml_tensor * dst) {
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->skip_htp_ops) {
        return false;
    }
    if (!ctx->ops_backend_initialized) {
        return false;
    }

    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            return false;

            if (dst->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32) {
                // NOTE: RPC version is mainly for testing
                return dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32") != nullptr;
            }
            return false;
        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                size_t k = weight->ne[0];
                size_t n = weight->ne[1];

                bool shape_ok = k % 32 == 0 && n % 32 == 0 && ggml_nrows(dst) == dst->ne[1] &&
                                ggml_nrows(activation) == activation->ne[1];
                const char * mode = get_npu_mode();
                const char * category = category_from_weight_name(weight->name);
                bool w8pc_category_ok = std::strcmp(category, "q_matrix") == 0 ||
                                        std::strcmp(category, "k_matrix") == 0 ||
                                        std::strcmp(category, "v_matrix") == 0 ||
                                        std::strcmp(category, "o_matrix") == 0 ||
                                        std::strcmp(category, "ffn_gate") == 0 ||
                                        std::strcmp(category, "ffn_up") == 0 ||
                                        std::strcmp(category, "ffn_down") == 0;

                // FP16 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) Q4_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) Q8_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 && activation->type == GGML_TYPE_F32) {
                    if (is_w8pc_a8pt_mode(mode)) {
                        return shape_ok && w8pc_category_ok;
                    }
                    return shape_ok;
                }
                // (repacked) IQ4_NL weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                    activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                fprintf(stderr, "unsupported matmul: dst: %s weight: %s act: %s\n", ggml_type_name(dst->type),
                        ggml_type_name(weight->type), ggml_type_name(activation->type));
                return false;
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                float scale         = *reinterpret_cast<const float *>(&dst->op_params[0]);
                float max_bias      = *reinterpret_cast<const float *>(&dst->op_params[1]);
                float logit_softcap = *reinterpret_cast<const float *>(&dst->op_params[2]);

                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];

                auto print_tensor_info = [](const ggml_tensor * t) {
                    printf("%s: shape [%ld,%ld,%ld,%ld] type %s\n", t->name, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                           ggml_type_name(t->type));
                };
                // print_tensor_info(dst);
                // print_tensor_info(q);
                // print_tensor_info(k);
                // print_tensor_info(v);
                // print_tensor_info(mask);

                return dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                       v->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 && max_bias == 0 && logit_softcap == 0;
            }
        default:
            return false;
    }
}

int htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return 0;
    }

    const int64_t trace_id   = g_llm_trace_id.fetch_add(1, std::memory_order_relaxed);
    const char *  npu_mode   = get_npu_mode();
    const int      mode_flags = get_mode_flags(npu_mode);
    const bool     trace_on   = (mode_flags & LLM_NPU_MODE_TRACE) != 0;
    const bool     detailed_trace_on = (mode_flags & LLM_NPU_MODE_DETAILED_TRACE) != 0;
    const int      trace_profile_max_events = detailed_trace_on ? env_int_or_default("LLAMA_NPU_DETAILED_TRACE_MAX_EVENTS", 4096) : 0;
    void *         trace_profile = nullptr;
    int            trace_profile_fd = -1;
    const int64_t t_begin_us = ggml_time_us();
    prepare_tensor_rpcmem_mapping(dst);
    const bool trace_profile_ready = detailed_trace_on && ensure_llm_trace_profile_buffer(trace_profile_max_events);
    if (trace_profile_ready) {
        trace_profile = g_llm_trace_profile;
        trace_profile_fd = g_llm_trace_profile_fd;
    } else if (detailed_trace_on) {
        const size_t requested_size = trace_profile_max_events > 0 ?
                sizeof(LlmTraceProfileHeader) +
                static_cast<size_t>(trace_profile_max_events) * sizeof(LlmTraceProfileEvent) : 0;
        fprintf(stderr, "LLMTRACE_DSP_STAGE_PROFILE_ALLOC_FAILED trace_id=%lld fd=%d size=%zu\n",
                (long long) trace_id, g_llm_trace_profile_fd, requested_size);
    }
    const RpcmemBufAddr active_trace_profile_addr{ trace_profile_fd, 0 };
    const int64_t t_prepare_done_us = ggml_time_us();

    auto * ctx           = ggml_backend_htp_context::instance();
    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    constexpr bool prefer_rpc = false;

    int op_index  = -1;
    int args_size = 0;  // strictly 32 bits
    int m = 0, k = 0, n = 0;
    int qo_len = 0, kv_len = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0;
    const char * category    = "other";
    const char * phase       = "unknown";
    const char * weight_name = "";

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            {
                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 2);

                auto [dst_fd, dst_offset] = mappings[0];
                auto [src_fd, src_offset] = mappings[1];

                if (prefer_rpc) {
                    using fn_type = int(int, int, int, int, int, int);

                    auto op_fn = reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32"));
                    GGML_ASSERT(op_fn);

                    return op_fn(dst_fd, dst_offset, src_fd, src_offset, dst->ne[0], ggml_nrows(dst));
                }

                RmsNormF32Params params{
                    .dst = { dst_fd, (int32_t) dst_offset },
                    .src = { src_fd, (int32_t) src_offset },
                    .ne0 = (int32_t) dst->ne[0],
                    .ne1 = (int32_t) ggml_nrows(dst),
                    .trace_id = trace_id,
                    .mode_flags = mode_flags,
                    .max_profile_events = trace_profile_fd >= 0 ? trace_profile_max_events : 0,
                    .profile = active_trace_profile_addr,
                };
                *reinterpret_cast<RmsNormF32Params *>(param_buf) = params;

                op_index  = HTP_OPS_RMS_NORM_F32;
                args_size = sizeof(RmsNormF32Params);
                m = ggml_nrows(dst);
                k = dst->ne[0];
                n = 1;
                phase = phase_from_rows(m);
                category = "rms_norm";
            }
            break;

        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 3);

                auto [output_fd, output_offset]         = mappings[0];
                auto [weight_fd, weight_offset]         = mappings[1];
                auto [activation_fd, activation_offset] = mappings[2];

                m = ggml_nrows(activation);
                k = weight->ne[0];
                n = weight->ne[1];
                phase = phase_from_rows(m);
                weight_name = weight->name;
                category = category_from_weight_name(weight_name);

                MatMulParams params{
                    .output     = { output_fd,     (int32_t) output_offset     },
                    .activation = { activation_fd, (int32_t) activation_offset },
                    .weight     = { weight_fd,     (int32_t) weight_offset     },
                    .m          = m,
                    .k          = k,
                    .n          = n,
                    .trace_id   = trace_id,
                    .mode_flags = mode_flags,
                    .max_profile_events = trace_profile_fd >= 0 ? trace_profile_max_events : 0,
                    .profile    = active_trace_profile_addr,
                };
                *reinterpret_cast<MatMulParams *>(param_buf) = params;

                args_size = sizeof(MatMulParams);

                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    if (prefer_rpc) {
                        using fn_type = int(int, int, int, int, int, int, int, int, int);

                        auto op_fn =
                            reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_permuted_w16a32"));
                        GGML_ASSERT(op_fn);

                        return op_fn(output_fd, output_offset, activation_fd, activation_offset, weight_fd,
                                     weight_offset, m, k, n);
                    }

                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = is_w8pc_a8pt_mode(npu_mode) ? HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT :
                                                             HTP_OPS_MAT_MUL_PERMUTED_W8D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL;
                } else {
                    GGML_ASSERT(false && "not implemented");
                }
            }
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            {
                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 5);

                auto [o_fd, o_offset]       = mappings[0];
                auto [q_fd, q_offset]       = mappings[1];
                auto [k_fd, k_offset]       = mappings[2];
                auto [v_fd, v_offset]       = mappings[3];
                auto [mask_fd, mask_offset] = mappings[4];

                head_dim   = q->ne[0];
                qo_len     = q->ne[1];
                kv_len     = k->ne[1];
                n_heads    = q->ne[2];
                n_kv_heads = k->ne[2];
                m = qo_len;
                phase = phase_from_rows(qo_len);
                category = "attention";

                FlashAttnParams params{
                    .o          = { o_fd,    (int32_t) o_offset    },
                    .q          = { q_fd,    (int32_t) q_offset    },
                    .k          = { k_fd,    (int32_t) k_offset    },
                    .v          = { v_fd,    (int32_t) v_offset    },
                    .mask       = { mask_fd, (int32_t) mask_offset },
                    .qo_len     = qo_len,
                    .kv_len     = kv_len,
                    .n_heads    = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim   = head_dim,
                    .trace_id   = trace_id,
                    .mode_flags = mode_flags,
                    .max_profile_events = trace_profile_fd >= 0 ? trace_profile_max_events : 0,
                    .profile    = active_trace_profile_addr,
                };
                *reinterpret_cast<FlashAttnParams *>(param_buf) = params;

                op_index  = HTP_OPS_FLASH_ATTN_QO_F32_KV_F16;
                args_size = sizeof(FlashAttnParams);
            }
            break;

        default:
            break;
    }

    // TODO(hzx): make sure only one thread can arrive here
    int  n_reqs                 = 1;
    int  n_unmap_fds            = ctx->mapper.get_pending_unmap_reqs().size();
    bool has_profile_dsp_unmap  = trace_profile_fd >= 0;
    int  n_map_put_fds          = n_unmap_fds + (has_profile_dsp_unmap ? 1 : 0);
    bool has_unmap_reqs         = n_map_put_fds > 0;
    if (has_unmap_reqs) {
        ++n_reqs;
    }

    size_t op_req_size = sizeof(RequestHeader) + sizeof(OpComputeRequest) + args_size;

    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);

    // FIXME: this is very ugly
    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    // std::atomic_store_explicit(d_ptr, 0, std::memory_order_release);

    // The memory order here is not very important
    std::atomic_store(d_ptr, uint64_t{0});

    msg_hdr->n_reqs         = n_reqs;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);
    msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + op_req_size;

    {
        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_OP_COMPUTE,
        };
        OpComputeRequest op_req{
            .op = (uint32_t) op_index,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
        write_buf(p, req_hdr);
        write_buf(p, op_req);
        write_buf(p, param_buf, args_size);
    }

    if (has_unmap_reqs) {
        size_t map_req_size     = sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + n_map_put_fds * sizeof(int32_t);
        msg_hdr->req_offsets[2] = msg_hdr->req_offsets[1] + map_req_size;

        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_RPCMEM_MAP,
        };
        RpcmemMapRequest map_req{
            .n_puts = n_map_put_fds,
            .n_gets = 0,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 1));
        write_buf(p, req_hdr);
        write_buf(p, map_req);
        for (const auto & [fd, _base, _len] : ctx->mapper.get_pending_unmap_reqs()) {
            write_buf(p, fd);
        }
        if (has_profile_dsp_unmap) {
            write_buf(p, trace_profile_fd);
        }
    }

    // compute checksum
    if (1) {
        uint32_t   sum   = 0;
        uint32_t * begin = ((uint32_t *) msg_hdr) + 3;  // skip state & checksum
        uint32_t * end   = ((uint32_t *) msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;

        for (auto * p = begin; p < end; ++p) {
            sum += *p;
        }
        sum += 0x00000001 + 0x00000000;  // value of `state`

        msg_hdr->checksum = -sum;
    } else {
#ifdef __aarch64__
        asm volatile("dmb sy" ::: "memory");
#endif
    }

    const int64_t t_request_ready_us = ggml_time_us();

    // issue request
    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));

    // NOTE(hzx): make sure memory_order_release is used here to ensure all previous writes are valid
    const int64_t t_issue_us = ggml_time_us();
    std::atomic_store_explicit(v0_ptr, uint8_t{1}, std::memory_order_release);

    // poll for response
    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        // TODO(hzx): use cpu_relax here
        usleep(1);
    }
    const int64_t t_response_us = ggml_time_us();
    d_ptr->store(0, std::memory_order_relaxed);

    if (has_unmap_reqs) {
        ctx->mapper.unmap_all_pending_buffers();
    }
    const int64_t t_unmap_done_us = ggml_time_us();

    std::atomic_thread_fence(std::memory_order_acquire);
    const int ret = message_header_get_request_ptr(msg_hdr, 0)->state;

    if (detailed_trace_on && trace_profile) {
        print_llm_trace_profile(reinterpret_cast<const LlmTraceProfileHeader *>(trace_profile), trace_id, npu_mode,
                                phase, category, dst->name, weight_name);
    }

    if (trace_on && op_index == HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT) {
        const int debug_idx = g_w8pc_host_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 16) {
            const auto n_el = ggml_nelements(dst);
            const float * vals = reinterpret_cast<const float *>(dst->data);
            float min_v = 0.0f;
            float max_v = 0.0f;
            int bad_count = 0;
            tensor_stats_f32(dst, min_v, max_v, bad_count);

            const ggml_tensor * act = dst->src[1];
            const float * act_vals = reinterpret_cast<const float *>(act->data);
            float act_min = 0.0f;
            float act_max = 0.0f;
            int act_bad = 0;
            tensor_stats_f32(act, act_min, act_max, act_bad);
            fprintf(stderr,
                    "W8PC_A8PT_HOST_DEBUG trace_id=%lld idx=%d tensor=%s weight=%s m=%d k=%d n=%d "
                    "first8=%g,%g,%g,%g,%g,%g,%g,%g min=%g max=%g bad=%d "
                    "act_first8=%g,%g,%g,%g,%g,%g,%g,%g act_min=%g act_max=%g act_bad=%d "
                    "act_ne=%lld,%lld,%lld,%lld act_nb=%zu,%zu,%zu,%zu\n",
                    (long long) trace_id, debug_idx, dst->name, weight_name, m, k, n,
                    n_el > 0 ? vals[0] : 0.0f,
                    n_el > 1 ? vals[1] : 0.0f,
                    n_el > 2 ? vals[2] : 0.0f,
                    n_el > 3 ? vals[3] : 0.0f,
                    n_el > 4 ? vals[4] : 0.0f,
                    n_el > 5 ? vals[5] : 0.0f,
                    n_el > 6 ? vals[6] : 0.0f,
                    n_el > 7 ? vals[7] : 0.0f,
                    min_v, max_v, bad_count,
                    ggml_nelements(act) > 0 ? act_vals[0] : 0.0f,
                    ggml_nelements(act) > 1 ? act_vals[1] : 0.0f,
                    ggml_nelements(act) > 2 ? act_vals[2] : 0.0f,
                    ggml_nelements(act) > 3 ? act_vals[3] : 0.0f,
                    ggml_nelements(act) > 4 ? act_vals[4] : 0.0f,
                    ggml_nelements(act) > 5 ? act_vals[5] : 0.0f,
                    ggml_nelements(act) > 6 ? act_vals[6] : 0.0f,
                    ggml_nelements(act) > 7 ? act_vals[7] : 0.0f,
                    act_min, act_max, act_bad,
                    (long long) act->ne[0], (long long) act->ne[1], (long long) act->ne[2], (long long) act->ne[3],
                    act->nb[0], act->nb[1], act->nb[2], act->nb[3]);
        }
    }

    if (trace_on && op_index == HTP_OPS_FLASH_ATTN_QO_F32_KV_F16) {
        const int debug_idx = g_flash_host_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 16) {
            const int64_t n_el = ggml_nelements(dst);
            const float * vals = reinterpret_cast<const float *>(dst->data);
            float min_v = 0.0f;
            float max_v = 0.0f;
            int bad_count = 0;
            tensor_stats_f32(dst, min_v, max_v, bad_count);
            fprintf(stderr,
                    "FLASH_ATTN_HOST_DEBUG trace_id=%lld idx=%d tensor=%s first8=%g,%g,%g,%g,%g,%g,%g,%g "
                    "min=%g max=%g bad=%d dst_ne=%lld,%lld,%lld,%lld dst_nb=%zu,%zu,%zu,%zu "
                    "q_ne=%lld,%lld,%lld,%lld q_nb=%zu,%zu,%zu,%zu k_ne=%lld,%lld,%lld,%lld k_nb=%zu,%zu,%zu,%zu "
                    "v_ne=%lld,%lld,%lld,%lld v_nb=%zu,%zu,%zu,%zu mask_ne=%lld,%lld,%lld,%lld mask_nb=%zu,%zu,%zu,%zu\n",
                    (long long) trace_id, debug_idx, dst->name,
                    n_el > 0 ? vals[0] : 0.0f,
                    n_el > 1 ? vals[1] : 0.0f,
                    n_el > 2 ? vals[2] : 0.0f,
                    n_el > 3 ? vals[3] : 0.0f,
                    n_el > 4 ? vals[4] : 0.0f,
                    n_el > 5 ? vals[5] : 0.0f,
                    n_el > 6 ? vals[6] : 0.0f,
                    n_el > 7 ? vals[7] : 0.0f,
                    min_v, max_v, bad_count,
                    (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    (long long) dst->src[0]->ne[0], (long long) dst->src[0]->ne[1], (long long) dst->src[0]->ne[2], (long long) dst->src[0]->ne[3],
                    dst->src[0]->nb[0], dst->src[0]->nb[1], dst->src[0]->nb[2], dst->src[0]->nb[3],
                    (long long) dst->src[1]->ne[0], (long long) dst->src[1]->ne[1], (long long) dst->src[1]->ne[2], (long long) dst->src[1]->ne[3],
                    dst->src[1]->nb[0], dst->src[1]->nb[1], dst->src[1]->nb[2], dst->src[1]->nb[3],
                    (long long) dst->src[2]->ne[0], (long long) dst->src[2]->ne[1], (long long) dst->src[2]->ne[2], (long long) dst->src[2]->ne[3],
                    dst->src[2]->nb[0], dst->src[2]->nb[1], dst->src[2]->nb[2], dst->src[2]->nb[3],
                    (long long) dst->src[3]->ne[0], (long long) dst->src[3]->ne[1], (long long) dst->src[3]->ne[2], (long long) dst->src[3]->ne[3],
                    dst->src[3]->nb[0], dst->src[3]->nb[1], dst->src[3]->nb[2], dst->src[3]->nb[3]);
        }
    }

    if (trace_on) {
        fprintf(stderr,
                "LLMTRACE_HOST_EVENT trace_id=%lld mode=%s flags=%d phase=%s op=%s op_index=%d category=%s "
                "tensor=%s weight=%s m=%d k=%d n=%d qo_len=%d kv_len=%d n_heads=%d n_kv_heads=%d head_dim=%d "
                "t0_us=%lld t1_us=%lld dur_us=%lld prepare_us=%lld build_us=%lld wait_us=%lld unmap_us=%lld ret=%d\n",
                (long long) trace_id, npu_mode, mode_flags, phase, htp_op_name(op_index), op_index, category,
                dst->name, weight_name, m, k, n, qo_len, kv_len, n_heads, n_kv_heads, head_dim,
                (long long) t_begin_us, (long long) t_unmap_done_us, (long long) (t_unmap_done_us - t_begin_us),
                (long long) (t_prepare_done_us - t_begin_us),
                (long long) (t_request_ready_us - t_prepare_done_us),
                (long long) (t_response_us - t_issue_us),
                (long long) (t_unmap_done_us - t_response_us), ret);
    }

    return ret;
}
}
