#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)
struct fixture_header {
    char magic[8];
    uint32_t version;
    uint32_t qo_len;
    uint32_t kv_len;
    uint32_t kv_pad;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t mask_mode;
    uint32_t seed;
    uint32_t reserved;
    uint64_t q_bytes;
    uint64_t k_bytes;
    uint64_t v_bytes;
    uint64_t mask_bytes;
};
#pragma pack(pop)

static_assert(sizeof(fixture_header) == 80, "fixture header ABI");

struct fixture {
    fixture_header h{};
    std::vector<float> q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<ggml_fp16_t> mask;
};

struct options {
    std::string fixture_path;
    std::string backend_contains = "Hexagon";
    int warmups = 5;
    int iterations = 20;
    bool check = false;
};

std::string captured_log;

void log_callback(enum ggml_log_level, const char * text, void *) {
    if (text != nullptr) {
        captured_log += text;
    }
}

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

template <typename T>
void read_exact(std::ifstream & in, std::vector<T> & dst, uint64_t bytes, const char * name) {
    if (bytes % sizeof(T) != 0) {
        fail(std::string(name) + " byte count is not element aligned");
    }
    dst.resize(bytes / sizeof(T));
    in.read(reinterpret_cast<char *>(dst.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        fail(std::string("short fixture read for ") + name);
    }
}

fixture load_fixture(const std::string & path) {
    fixture f;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("cannot open fixture: " + path);
    }
    in.read(reinterpret_cast<char *>(&f.h), sizeof(f.h));
    if (!in || std::memcmp(f.h.magic, "S25FIX01", 8) != 0 || f.h.version != 1) {
        fail("invalid Stage2.5 fixture header");
    }
    if (f.h.n_heads % f.h.n_kv_heads != 0 || f.h.mask_mode > 2) {
        fail("unsupported fixture contract");
    }
    const uint64_t q_expected = uint64_t(f.h.qo_len) * f.h.n_heads * f.h.head_dim * sizeof(float);
    const uint64_t kv_expected = uint64_t(f.h.kv_len) * f.h.n_kv_heads * f.h.head_dim * sizeof(ggml_fp16_t);
    const uint64_t mask_expected = uint64_t(f.h.qo_len) * f.h.kv_pad * sizeof(ggml_fp16_t);
    if (f.h.q_bytes != q_expected || f.h.k_bytes != kv_expected || f.h.v_bytes != kv_expected ||
        f.h.mask_bytes != mask_expected) {
        fail("fixture sizes do not match shape metadata");
    }
    read_exact(in, f.q, f.h.q_bytes, "Q");
    read_exact(in, f.k, f.h.k_bytes, "K");
    read_exact(in, f.v, f.h.v_bytes, "V");
    read_exact(in, f.mask, f.h.mask_bytes, "mask");
    if (in.peek() != std::ifstream::traits_type::eof()) {
        fail("fixture has trailing data");
    }
    return f;
}

options parse_options(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) fail("missing value after " + arg);
            return argv[i];
        };
        if (arg == "--fixture") o.fixture_path = value();
        else if (arg == "--backend-contains") o.backend_contains = value();
        else if (arg == "--warmups") o.warmups = std::stoi(value());
        else if (arg == "--iterations") o.iterations = std::stoi(value());
        else if (arg == "--check") o.check = true;
        else fail("unknown argument: " + arg);
    }
    if (o.fixture_path.empty() || o.warmups < 0 || o.iterations < 1) {
        fail("usage: test-stage25-fa --fixture PATH [--warmups N] [--iterations N] [--check]");
    }
    return o;
}

ggml_backend_dev_t find_device(const std::string & needle) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        if ((name && std::string(name).find(needle) != std::string::npos) ||
            (desc && std::string(desc).find(needle) != std::string::npos)) {
            return dev;
        }
    }
    return nullptr;
}

std::string extract_field(const std::string & log, const std::string & begin, const std::string & end) {
    const size_t p = log.rfind(begin);
    if (p == std::string::npos) return "N/A";
    const size_t start = p + begin.size();
    const size_t stop = log.find(end, start);
    return log.substr(start, stop == std::string::npos ? std::string::npos : stop - start);
}

std::string json_escape(const std::string & value) {
    std::string out;
    for (char c : value) {
        if (c == '\\' || c == '"') out.push_back('\\');
        if (c == '\n') out += "\\n";
        else if (c != '\r') out.push_back(c);
    }
    return out;
}

struct accuracy_result {
    double rmse = 0.0;
    double max_abs = 0.0;
    size_t nonfinite = 0;
};

accuracy_result check_reference(const fixture & f, const std::vector<float> & got) {
    const uint32_t D = f.h.head_dim;
    const uint32_t Q = f.h.qo_len;
    const uint32_t KV = f.h.kv_len;
    const uint32_t H = f.h.n_heads;
    const uint32_t HKV = f.h.n_kv_heads;
    const uint32_t G = H / HKV;
    const float scale = 1.0f / std::sqrt(float(D));
    long double sum_sq = 0.0;
    double max_abs = 0.0;
    size_t count = 0;
    size_t nonfinite = 0;
    std::vector<double> scores(KV);
    std::vector<double> ref(D);
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t hk = h / G;
        for (uint32_t q = 0; q < Q; ++q) {
            double max_score = -std::numeric_limits<double>::infinity();
            for (uint32_t k = 0; k < KV; ++k) {
                double dot = 0.0;
                for (uint32_t d = 0; d < D; ++d) {
                    const float qv = f.q[(uint64_t(q) * H + h) * D + d];
                    const float kv = ggml_fp16_to_fp32(f.k[(uint64_t(k) * HKV + hk) * D + d]);
                    dot += double(qv) * kv;
                }
                const float mv = ggml_fp16_to_fp32(f.mask[uint64_t(q) * f.h.kv_pad + k]);
                scores[k] = dot * scale + mv;
                max_score = std::max(max_score, scores[k]);
            }
            double denom = 0.0;
            std::fill(ref.begin(), ref.end(), 0.0);
            for (uint32_t k = 0; k < KV; ++k) {
                const double p = std::exp(scores[k] - max_score);
                denom += p;
                for (uint32_t d = 0; d < D; ++d) {
                    ref[d] += p * ggml_fp16_to_fp32(f.v[(uint64_t(k) * HKV + hk) * D + d]);
                }
            }
            for (uint32_t d = 0; d < D; ++d) {
                const double expected = ref[d] / denom;
                // FLASH_ATTN_EXT returns [D, H, Q, batch], while Q is
                // [D, Q, H, batch].  Keep this transform explicit.
                const double actual = got[(uint64_t(q) * H + h) * D + d];
                if (!std::isfinite(actual)) ++nonfinite;
                const double error = actual - expected;
                sum_sq += error * error;
                max_abs = std::max(max_abs, std::abs(error));
                ++count;
            }
        }
    }
    return {std::sqrt(double(sum_sq / count)), max_abs, nonfinite};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        // Keep each JSON record intact when adb merges remote stdout/stderr.
        // Backend logs are captured in memory and included in the record.
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        const options opt = parse_options(argc, argv);
        const fixture f = load_fixture(opt.fixture_path);
        ggml_log_set(log_callback, nullptr);
        ggml_backend_load_all();
        ggml_backend_dev_t dev = find_device(opt.backend_contains);
        if (!dev) fail("Hexagon backend device not found");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        if (!backend) fail("failed to initialize Hexagon backend");

        ggml_init_params ip = { 16U * 1024U * 1024U, nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        if (!ctx) fail("ggml_init failed");
        ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, f.h.head_dim, f.h.qo_len, f.h.n_heads, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, f.h.head_dim, f.h.kv_len, f.h.n_kv_heads, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, f.h.head_dim, f.h.kv_len, f.h.n_kv_heads, 1);
        ggml_tensor * m = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, f.h.kv_len, f.h.qo_len, 1, 1);
        ggml_set_name(q, "q"); ggml_set_name(k, "k"); ggml_set_name(v, "v"); ggml_set_name(m, "mask");
        ggml_tensor * out = ggml_flash_attn_ext(ctx.get(), q, k, v, m, 1.0f / std::sqrt(float(f.h.head_dim)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_set_name(out, "out");
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(graph, out);
        if (!ggml_backend_supports_op(backend.get(), out)) fail("Hexagon backend does not support requested FlashAttention op");
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) fail("backend tensor allocation failed");

        std::vector<float> q_ggml(f.q.size());
        std::vector<ggml_fp16_t> k_ggml(f.k.size()), v_ggml(f.v.size());
        std::vector<ggml_fp16_t> m_ggml(uint64_t(f.h.qo_len) * f.h.kv_len);
        for (uint32_t h = 0; h < f.h.n_heads; ++h)
            for (uint32_t qi = 0; qi < f.h.qo_len; ++qi)
                std::copy_n(&f.q[(uint64_t(qi) * f.h.n_heads + h) * f.h.head_dim], f.h.head_dim,
                            &q_ggml[(uint64_t(h) * f.h.qo_len + qi) * f.h.head_dim]);
        for (uint32_t h = 0; h < f.h.n_kv_heads; ++h)
            for (uint32_t ki = 0; ki < f.h.kv_len; ++ki) {
                std::copy_n(&f.k[(uint64_t(ki) * f.h.n_kv_heads + h) * f.h.head_dim], f.h.head_dim,
                            &k_ggml[(uint64_t(h) * f.h.kv_len + ki) * f.h.head_dim]);
                std::copy_n(&f.v[(uint64_t(ki) * f.h.n_kv_heads + h) * f.h.head_dim], f.h.head_dim,
                            &v_ggml[(uint64_t(h) * f.h.kv_len + ki) * f.h.head_dim]);
            }
        for (uint32_t qi = 0; qi < f.h.qo_len; ++qi)
            std::copy_n(&f.mask[uint64_t(qi) * f.h.kv_pad], f.h.kv_len, &m_ggml[uint64_t(qi) * f.h.kv_len]);
        ggml_backend_tensor_set(q, q_ggml.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k, k_ggml.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(v, v_ggml.data(), 0, ggml_nbytes(v));
        ggml_backend_tensor_set(m, m_ggml.data(), 0, ggml_nbytes(m));

        for (int i = 0; i < opt.warmups; ++i) {
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) fail("warmup compute failed");
        }
        const char * selector = std::getenv("GGML_HEXAGON_FA_SELECT");
        const char * nhvx = std::getenv("GGML_HEXAGON_NHVX");
        for (int i = 0; i < opt.iterations; ++i) {
            captured_log.clear();
            const auto start = std::chrono::steady_clock::now();
            const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
            const auto stop = std::chrono::steady_clock::now();
            if (status != GGML_STATUS_SUCCESS) fail("measured compute failed");
            const double host_us = std::chrono::duration<double, std::micro>(stop - start).count();
            const std::string profile = extract_field(captured_log, "profile-op ", "\n");
            const std::string kernel = profile.find("hmx-pipe") != std::string::npos ? "hmx-pipe" :
                                       profile.find("hmx-seq") != std::string::npos ? "hmx-seq" :
                                       profile.find("|hvx Br") != std::string::npos ? "hvx" : "N/A";
            const std::string dsp = extract_field(profile, "|usec ", " ");
            std::printf("{\"record\":\"measurement\",\"iteration\":%d,\"q\":%u,\"kv\":%u,\"heads\":%u,"
                        "\"kv_heads\":%u,\"head_dim\":%u,\"requested_selector\":\"%s\",\"nhvx\":\"%s\","
                        "\"effective_kernel\":\"%s\",\"profile\":\"%s\",\"host_us\":%.3f,\"dsp_op_us\":\"%s\"}\n",
                        i, f.h.qo_len, f.h.kv_len, f.h.n_heads, f.h.n_kv_heads, f.h.head_dim,
                        selector ? selector : "unset", nhvx ? nhvx : "unset", kernel.c_str(),
                        json_escape(profile).c_str(), host_us, dsp.c_str());
        }
        if (opt.check) {
            std::vector<float> got(ggml_nelements(out));
            ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
            const accuracy_result a = check_reference(f, got);
            const bool pass = a.nonfinite == 0 && a.rmse <= 0.002 && a.max_abs <= 0.01;
            std::printf("{\"record\":\"correctness\",\"rmse\":%.9g,\"max_abs\":%.9g,\"nonfinite\":%zu,\"pass\":%s}\n",
                        a.rmse, a.max_abs, a.nonfinite, pass ? "true" : "false");
            if (!pass) return 2;
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-stage25-fa: %s\n", e.what());
        return 1;
    }
}
