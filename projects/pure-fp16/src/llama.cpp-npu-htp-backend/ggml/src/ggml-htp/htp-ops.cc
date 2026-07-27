#include "htp-ops.h"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

struct LpbqSidecarBuffer {
    void * data = nullptr;
    int    fd = -1;
    size_t bytes = 0;
    size_t offset = 0;
};

struct LpbqSidecar {
    bool loaded = false;
    bool ok = false;
    float act_scale = 0.0f;
    LpbqSidecarBuffer scale2;
    LpbqSidecarBuffer bias;
    LpbqSidecarBuffer r4;
    LpbqSidecarBuffer r4_structured_fwht_buffer;
    LpbqSidecarBuffer r4_hmx_dense_fp16;
    LpbqSidecarBuffer input_scale;
    LpbqSidecarBuffer packed_weight;
    LpbqSidecarBuffer packed_weight_v6_full_buffer;
    LpbqSidecarBuffer sum_w;
    LpbqSidecarBuffer k32_safe;
    LpbqSidecarBuffer k64_safe;
    LpbqSidecarBuffer out_scale;
    LpbqSidecarBuffer bias_eff;
    bool has_bias = false;
    bool has_r4 = false;
    bool has_input_scale = false;
    bool has_packed_weight = false;
    bool has_packed_weight_v6_full = false;
    bool has_k32_safe = false;
    bool has_k64_safe = false;
    bool has_folded_dequant = false;
    bool packed_weight_k_major = false;
    bool packed_weight_v6_full = false;
    bool r4_v6_scale_1_16 = false;
    bool exact_non_r4 = false;
    bool r4_structured_fwht = false;
    bool has_r4_hmx_dense_fp16 = false;
    int packed_weight_v6_full_group_tiles = 0;
    bool r4_input_scale_folded = false;
    int k32_safe_tiles = 0;
    int k32_total_tiles = 0;
    int k64_safe_tiles = 0;
    int k64_total_tiles = 0;
    int r4_block = 0;
};

std::unordered_map<std::string, LpbqSidecar> g_lpbq_sidecar_cache;

struct LpbqV6FullSafeList {
    bool loaded = false;
    bool present = false;
    std::unordered_set<std::string> stems;
    std::unordered_set<std::string> r4_v6_scale_1_16_stems;
};

std::unordered_map<std::string, LpbqV6FullSafeList> g_lpbq_v6_full_safe_lists;
std::unordered_map<std::string, LpbqV6FullSafeList> g_lpbq_r4_full_u8_safe_lists;
std::unordered_map<std::string, LpbqV6FullSafeList> g_lpbq_v6_full_non_r4_allow_lists;

bool env_truthy(const char * name);
int env_int_or_default(const char * name, int default_value);

std::string lpbq_sanitize_tensor_name(const char * name) {
    std::string out;
    for (const unsigned char ch : std::string(name ? name : "")) {
        if (std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "unnamed" : out;
}

std::string lpbq_join_path(const char * dir, const std::string & file) {
    std::string base = dir ? dir : "";
    if (!base.empty() && base.back() != '/' && base.back() != '\\') {
        base.push_back('/');
    }
    return base + file;
}

std::string lpbq_join_path_str(const std::string & dir, const std::string & file) {
    std::string base = dir;
    if (!base.empty() && base.back() != '/' && base.back() != '\\') {
        base.push_back('/');
    }
    return base + file;
}

std::string lpbq_dirname_copy(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(0, pos);
}

bool lpbq_file_exists(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    std::fclose(f);
    return true;
}

constexpr float LPBQ_R4_FWHT_MAGIC0 = 314159.25f;
constexpr float LPBQ_R4_FWHT_MAGIC1 = -271828.25f;
constexpr int   LPBQ_R4_FWHT_HEADER_FLOATS = 16;
constexpr int   LPBQ_R4_FWHT_D2_OFFSET = LPBQ_R4_FWHT_HEADER_FLOATS;
constexpr int   LPBQ_R4_FWHT_D1_OFFSET = LPBQ_R4_FWHT_D2_OFFSET + 128;

std::string lpbq_r4_fwht_sidecar_path(const std::string & dense_path) {
    const std::string suffix = ".r4.f32";
    if (dense_path.size() >= suffix.size() &&
        dense_path.compare(dense_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return dense_path.substr(0, dense_path.size() - suffix.size()) + ".r4.fwht_d1d2.f32";
    }
    return dense_path + ".fwht_d1d2.f32";
}

int lpbq_r4_layer_id_from_path(const std::string & r4_path) {
    const std::string marker = "blk.";
    const std::string suffix = ".mlp.r4.f32";
    const size_t begin = r4_path.find(marker);
    if (begin == std::string::npos) {
        return -1;
    }
    const size_t layer_begin = begin + marker.size();
    const size_t layer_end = r4_path.find(suffix, layer_begin);
    if (layer_end == std::string::npos || layer_end <= layer_begin) {
        return -1;
    }
    const std::string layer_text = r4_path.substr(layer_begin, layer_end - layer_begin);
    char * end = nullptr;
    const long value = std::strtol(layer_text.c_str(), &end, 10);
    if (!end || *end != '\0' || value < 0 || value > INT_MAX) {
        return -1;
    }
    return static_cast<int>(value);
}

std::string lpbq_r4_fwht_v2_layer_prefix(int layer) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "layer_%02d_mlp", layer);
    return std::string(buf);
}

std::vector<std::string> lpbq_r4_fwht_v2_candidate_dirs(const std::string & r4_path) {
    std::vector<std::string> dirs;
    const std::string r4_dir = lpbq_dirname_copy(r4_path);
    const std::string sidecar_dir = lpbq_dirname_copy(r4_dir);
    if (!sidecar_dir.empty()) {
        // Guide v2 keeps r4_fwht_v2 next to the active lpbq_g16_a8w8 sidecar dir.
        const std::string sidecar_parent = lpbq_dirname_copy(sidecar_dir);
        if (!sidecar_parent.empty()) {
            dirs.push_back(lpbq_join_path_str(sidecar_parent, "r4_fwht_v2"));
        }
        // Fallback for packaged deployments that choose to nest the v2 schema.
        dirs.push_back(lpbq_join_path_str(sidecar_dir, "r4_fwht_v2"));
    }
    return dirs;
}

std::vector<std::string> lpbq_r4_hmx_dense_fp16_candidate_dirs(const std::string & r4_path) {
    std::vector<std::string> dirs;
    const std::string r4_dir = lpbq_dirname_copy(r4_path);
    const std::string sidecar_dir = lpbq_dirname_copy(r4_dir);
    if (!sidecar_dir.empty()) {
        // Stage-A HMX Dense R4 uses a sibling schema next to lpbq_g16_a8w8,
        // matching the deployment layout used by r4_fwht_v2.
        const std::string sidecar_parent = lpbq_dirname_copy(sidecar_dir);
        if (!sidecar_parent.empty()) {
            dirs.push_back(lpbq_join_path_str(sidecar_parent, "r4_hmx_dense_fp16_v1"));
        }
        dirs.push_back(lpbq_join_path_str(sidecar_dir, "r4_hmx_dense_fp16_v1"));
    }
    return dirs;
}

bool lpbq_read_file_exact(const std::string & path, size_t expected_bytes, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    const long file_size = std::ftell(f);
    if (file_size < 0 || static_cast<size_t>(file_size) != expected_bytes) {
        std::fprintf(stderr, "LPBQ_SIDECAR size mismatch: %s got=%ld expected=%zu\n",
                     path.c_str(), file_size, expected_bytes);
        std::fclose(f);
        return false;
    }
    std::rewind(f);
    out.resize(expected_bytes);
    const size_t nread = std::fread(out.data(), 1, expected_bytes, f);
    std::fclose(f);
    if (nread != expected_bytes) {
        std::fprintf(stderr, "LPBQ_SIDECAR fread failed: %s got=%zu expected=%zu\n",
                     path.c_str(), nread, expected_bytes);
        return false;
    }
    return true;
}

float lpbq_fp16_bits_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1u;
    const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1fu;
    const uint32_t mant = static_cast<uint32_t>(h) & 0x3ffu;
    float value = 0.0f;
    if (exp == 0u) {
        value = mant == 0u ? 0.0f : std::ldexp(static_cast<float>(mant), -24);
    } else if (exp == 31u) {
        value = mant == 0u ? INFINITY : NAN;
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f, static_cast<int>(exp) - 15);
    }
    return sign ? -value : value;
}

bool lpbq_load_fp16_vector_128(const std::string & path, std::vector<float> & out) {
    std::vector<uint8_t> bytes;
    if (!lpbq_read_file_exact(path, 128u * sizeof(uint16_t), bytes)) {
        return false;
    }
    out.resize(128);
    for (int i = 0; i < 128; ++i) {
        const uint16_t bits = static_cast<uint16_t>(bytes[2 * i]) |
                              (static_cast<uint16_t>(bytes[2 * i + 1]) << 8);
        out[i] = lpbq_fp16_bits_to_float(bits);
    }
    return true;
}

bool lpbq_bitpack_is_identity(const std::string & path, int block) {
    std::vector<uint8_t> bytes;
    if (!lpbq_read_file_exact(path, static_cast<size_t>((block + 7) / 8), bytes)) {
        return false;
    }
    for (uint8_t b : bytes) {
        if (b != 0u) {
            std::fprintf(stderr,
                         "LPBQ_SIDECAR r4_fwht_v2 sign bitpack is not identity, unsupported by the legacy bridge: %s\n",
                         path.c_str());
            return false;
        }
    }
    return true;
}

std::string lpbq_trim_copy(const std::string & in) {
    size_t begin = 0;
    while (begin < in.size() && std::isspace(static_cast<unsigned char>(in[begin]))) {
        ++begin;
    }
    size_t end = in.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(in[end - 1]))) {
        --end;
    }
    return in.substr(begin, end - begin);
}

std::string lpbq_normalize_v6_safe_stem(std::string line) {
    line = lpbq_trim_copy(line);
    const size_t comment = line.find('#');
    if (comment != std::string::npos) {
        line = lpbq_trim_copy(line.substr(0, comment));
    }
    const size_t first_space = line.find_first_of(" \t\r\n");
    if (first_space != std::string::npos) {
        line = line.substr(0, first_space);
    }
    const size_t sidecar_suffix = line.find(".lpbq_");
    if (sidecar_suffix != std::string::npos) {
        line = line.substr(0, sidecar_suffix);
    }
    return line;
}

bool lpbq_parse_v6_safe_line(const char * raw_line, std::string & stem, bool & r4_v6_scale_1_16) {
    stem.clear();
    r4_v6_scale_1_16 = false;
    std::string line = lpbq_trim_copy(raw_line ? raw_line : "");
    const size_t comment = line.find('#');
    if (comment != std::string::npos) {
        line = lpbq_trim_copy(line.substr(0, comment));
    }
    if (line.empty()) {
        return false;
    }

    bool first = true;
    size_t pos = 0;
    while (pos < line.size()) {
        while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        const size_t begin = pos;
        while (pos < line.size() && !std::isspace(static_cast<unsigned char>(line[pos]))) {
            ++pos;
        }
        if (begin == pos) {
            break;
        }
        const std::string token = line.substr(begin, pos - begin);
        if (first) {
            stem = lpbq_normalize_v6_safe_stem(token);
            first = false;
        } else if (token == "r4_v6_scale_1_16") {
            r4_v6_scale_1_16 = true;
        }
    }
    return !stem.empty();
}

const LpbqV6FullSafeList & lpbq_get_v6_full_safe_list(const char * dir) {
    const char * override_path = std::getenv("LLAMA_NPU_LPBQ_V6_FULL_SAFE_LIST");
    const bool override_set = override_path && override_path[0];
    const std::string key = override_set ? std::string(override_path) : std::string(dir ? dir : "");
    auto & list = g_lpbq_v6_full_safe_lists[key];
    if (list.loaded) {
        return list;
    }
    list.loaded = true;

    const std::string path = override_set ? std::string(override_path) :
                                           lpbq_join_path(dir, "lpbq_v6_full_safe_layers.txt");
    FILE * f = std::fopen(path.c_str(), "r");
    if (!f) {
        // LPBQ deploy-v1 safety-list behavior: an absent default file preserves
        // the old all-enabled A/B, but an explicit override path is treated as
        // an empty allowlist so a typo cannot silently enable unsafe full-V6.
        list.present = override_set;
        if (override_set) {
            std::fprintf(stderr, "LPBQ_SIDECAR full-V6 safe-list missing: %s\n", path.c_str());
        }
        return list;
    }

    list.present = true;
    char line_buf[1024];
    while (std::fgets(line_buf, sizeof(line_buf), f)) {
        std::string stem;
        bool r4_v6_scale_1_16 = false;
        if (lpbq_parse_v6_safe_line(line_buf, stem, r4_v6_scale_1_16)) {
            list.stems.insert(stem);
            if (r4_v6_scale_1_16) {
                list.r4_v6_scale_1_16_stems.insert(stem);
            }
        }
    }
    std::fclose(f);
    return list;
}

bool lpbq_v6_full_allowed_for_stem(const char * dir, const std::string & stem) {
    if (env_truthy("LLAMA_NPU_LPBQ_V6_FULL_IGNORE_SAFE_LIST")) {
        return true;
    }
    const LpbqV6FullSafeList & list = lpbq_get_v6_full_safe_list(dir);
    if (!list.present) {
        return true;
    }
    return list.stems.find(stem) != list.stems.end();
}

bool lpbq_v6_full_r4_scale_1_16_for_stem(const char * dir, const std::string & stem) {
    const LpbqV6FullSafeList & list = lpbq_get_v6_full_safe_list(dir);
    return list.r4_v6_scale_1_16_stems.find(stem) != list.r4_v6_scale_1_16_stems.end();
}

const LpbqV6FullSafeList & lpbq_get_v6_full_non_r4_allow_list() {
    const char * override_path = std::getenv("LLAMA_NPU_LPBQ_V6_FULL_NON_R4_ALLOW_LIST");
    const bool override_set = override_path && override_path[0];
    const std::string key = override_set ? std::string(override_path) : std::string("<unset>");
    auto & list = g_lpbq_v6_full_non_r4_allow_lists[key];
    if (list.loaded) {
        return list;
    }
    list.loaded = true;

    if (!override_set) {
        // Preserve the historical all-non-R4 A/B behavior unless a caller
        // supplies an explicit allowlist to cap FastRPC mmap pressure.
        list.present = false;
        return list;
    }

    FILE * f = std::fopen(override_path, "r");
    list.present = true;
    if (!f) {
        std::fprintf(stderr, "LPBQ_SIDECAR non-R4 full-V6 allowlist missing: %s\n", override_path);
        return list;
    }

    char line_buf[1024];
    while (std::fgets(line_buf, sizeof(line_buf), f)) {
        std::string stem;
        bool ignored_scale_flag = false;
        if (lpbq_parse_v6_safe_line(line_buf, stem, ignored_scale_flag)) {
            list.stems.insert(stem);
        }
    }
    std::fclose(f);
    return list;
}

bool lpbq_v6_full_non_r4_allowed_for_stem(const std::string & stem) {
    const LpbqV6FullSafeList & list = lpbq_get_v6_full_non_r4_allow_list();
    if (!list.present) {
        return true;
    }
    return list.stems.find(stem) != list.stems.end();
}

const LpbqV6FullSafeList & lpbq_get_r4_full_u8_safe_list(const char * dir) {
    const char * override_path = std::getenv("LLAMA_NPU_LPBQ_R4_FULL_U8_SAFE_LIST");
    const bool override_set = override_path && override_path[0];
    const std::string key = override_set ? std::string(override_path) : std::string(dir ? dir : "");
    auto & list = g_lpbq_r4_full_u8_safe_lists[key];
    if (list.loaded) {
        return list;
    }
    list.loaded = true;

    const std::string path = override_set ? std::string(override_path) :
                                           lpbq_join_path(dir, "lpbq_r4_full_u8_safe_layers.txt");
    FILE * f = std::fopen(path.c_str(), "r");
    if (!f) {
        // LPBQ deploy-v1 force-R4 guard: a missing default file preserves the
        // old global A/B behavior, while an explicit override typo becomes an
        // empty allowlist instead of silently enabling unsafe layers.
        list.present = override_set;
        if (override_set) {
            std::fprintf(stderr, "LPBQ_SIDECAR R4 full-U8-safe list missing: %s\n", path.c_str());
        }
        return list;
    }

    list.present = true;
    char line_buf[1024];
    while (std::fgets(line_buf, sizeof(line_buf), f)) {
        std::string stem;
        bool ignored_scale_flag = false;
        if (lpbq_parse_v6_safe_line(line_buf, stem, ignored_scale_flag)) {
            list.stems.insert(stem);
        }
    }
    std::fclose(f);
    return list;
}

bool lpbq_r4_full_u8_safe_allowed_for_stem(const char * dir, const std::string & stem) {
    if (env_truthy("LLAMA_NPU_LPBQ_R4_FULL_U8_SAFE_IGNORE_LIST")) {
        return true;
    }
    const LpbqV6FullSafeList & list = lpbq_get_r4_full_u8_safe_list(dir);
    if (!list.present) {
        return true;
    }
    return list.stems.find(stem) != list.stems.end();
}

bool lpbq_exact_non_r4_allowed_for_stem(const std::string & stem) {
    if (env_truthy("LLAMA_NPU_LPBQ_DISABLE_EXACT_QK")) {
        return false;
    }
    if (env_truthy("LLAMA_NPU_LPBQ_EXACT_NON_R4_ALL")) {
        return true;
    }
    // LPBQ deploy-v1 correctness-first route: real-layer gates showed
    // row-varying grouped-V6 recover error in Q/K, while V/MLP can keep the
    // faster grouped path. Keep this stem-based until a faster exact Q/K drain
    // replaces the temporary fallback.
    return stem.find(".attn_q.weight") != std::string::npos ||
           stem.find(".attn_k.weight") != std::string::npos;
}

size_t lpbq_v6_full_grouped_bytes(int k, int n, int group_tiles) {
    if (k <= 0 || n <= 0 || group_tiles <= 0 || (k % 32) != 0 || (n % 32) != 0) {
        return 0;
    }
    const int k_tiles = k / 32;
    const int n_tiles = n / 32;
    const int groups = (k_tiles + group_tiles - 1) / group_tiles;
    return static_cast<size_t>(groups) * static_cast<size_t>(n_tiles) *
           static_cast<size_t>(group_tiles) * 2048u;
}

bool lpbq_r4_path_for_weight(const char * dir, const std::string & stem, std::string & path, int & block) {
    const std::string marker = ".ffn_down.weight";
    if (stem.rfind("blk.", 0) != 0 || stem.size() <= marker.size() ||
        stem.compare(stem.size() - marker.size(), marker.size(), marker) != 0) {
        return false;
    }

    const std::string layer = stem.substr(4, stem.size() - 4 - marker.size());
    if (layer.empty()) {
        return false;
    }

    // OSTQuant keeps only the per-block R4 rotation online; the current export
    // writes one dense 128x128 matrix per MLP block for down_proj inputs.
    block = 128;
    path = lpbq_join_path(dir, "r4/blk." + layer + ".mlp.r4.f32");
    return true;
}

bool lpbq_load_rpcmem_file(const std::string & path, size_t expected_bytes, LpbqSidecarBuffer & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "LPBQ_SIDECAR missing file: %s\n", path.c_str());
        return false;
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    const long file_size = std::ftell(f);
    if (file_size < 0 || static_cast<size_t>(file_size) != expected_bytes) {
        std::fprintf(stderr, "LPBQ_SIDECAR size mismatch: %s got=%ld expected=%zu\n", path.c_str(), file_size,
                     expected_bytes);
        std::fclose(f);
        return false;
    }
    std::rewind(f);
    void * ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, static_cast<int>(expected_bytes));
    if (!ptr) {
        std::fprintf(stderr, "LPBQ_SIDECAR rpcmem_alloc failed: %s bytes=%zu\n", path.c_str(), expected_bytes);
        std::fclose(f);
        return false;
    }
    const size_t nread = std::fread(ptr, 1, expected_bytes, f);
    std::fclose(f);
    if (nread != expected_bytes) {
        std::fprintf(stderr, "LPBQ_SIDECAR fread failed: %s got=%zu expected=%zu\n", path.c_str(), nread,
                     expected_bytes);
        rpcmem_free(ptr);
        return false;
    }
    const int fd = rpcmem_to_fd(ptr);
    if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, expected_bytes, FASTRPC_MAP_FD)) {
        std::fprintf(stderr, "LPBQ_SIDECAR fastrpc_mmap failed: %s fd=%d bytes=%zu\n", path.c_str(), fd,
                     expected_bytes);
        if (fd >= 0) {
            fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, expected_bytes);
        }
        rpcmem_free(ptr);
        return false;
    }
    out.data = ptr;
    out.fd = fd;
    out.bytes = expected_bytes;
    return true;
}

bool lpbq_create_rpcmem_buffer_from_data(const std::string & label, const void * src, size_t bytes,
                                         LpbqSidecarBuffer & out) {
    if (!src || bytes == 0 || bytes > static_cast<size_t>(INT32_MAX)) {
        std::fprintf(stderr, "LPBQ_SIDECAR invalid generated buffer: %s bytes=%zu\n", label.c_str(), bytes);
        return false;
    }
    void * ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, static_cast<int>(bytes));
    if (!ptr) {
        std::fprintf(stderr, "LPBQ_SIDECAR rpcmem_alloc failed: %s bytes=%zu\n", label.c_str(), bytes);
        return false;
    }
    std::memcpy(ptr, src, bytes);
    const int fd = rpcmem_to_fd(ptr);
    if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, bytes, FASTRPC_MAP_FD)) {
        std::fprintf(stderr, "LPBQ_SIDECAR fastrpc_mmap failed: %s fd=%d bytes=%zu\n", label.c_str(), fd, bytes);
        if (fd >= 0) {
            fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, bytes);
        }
        rpcmem_free(ptr);
        return false;
    }
    out.data = ptr;
    out.fd = fd;
    out.bytes = bytes;
    return true;
}

bool lpbq_create_folded_dequant_sidecars(const std::string & stem, int n, const LpbqSidecarBuffer & scale2,
                                         const LpbqSidecarBuffer & sum_w, const LpbqSidecarBuffer & bias,
                                         bool has_bias, float act_scale, LpbqSidecarBuffer & out_scale_buf,
                                         LpbqSidecarBuffer & bias_eff_buf) {
    if (n <= 0 || !scale2.data || !sum_w.data || scale2.bytes != static_cast<size_t>(n) * sizeof(float) ||
        sum_w.bytes != static_cast<size_t>(n) * sizeof(int32_t) || act_scale <= 0.0f ||
        !std::isfinite(act_scale)) {
        std::fprintf(stderr, "LPBQ_SIDECAR folded dequant invalid input: %s n=%d\n", stem.c_str(), n);
        return false;
    }
    if (has_bias && (!bias.data || bias.bytes != static_cast<size_t>(n) * sizeof(float))) {
        std::fprintf(stderr, "LPBQ_SIDECAR folded dequant invalid bias: %s n=%d\n", stem.c_str(), n);
        return false;
    }

    const float * scale2_ptr = reinterpret_cast<const float *>(scale2.data);
    const int32_t * sum_w_ptr = reinterpret_cast<const int32_t *>(sum_w.data);
    const float * bias_ptr = has_bias ? reinterpret_cast<const float *>(bias.data) : nullptr;
    std::vector<float> out_scale(static_cast<size_t>(n));
    std::vector<float> bias_eff(static_cast<size_t>(n));
    for (int c = 0; c < n; ++c) {
        const float os = scale2_ptr[c] * act_scale;
        const float b = bias_ptr ? bias_ptr[c] : 0.0f;
        out_scale[static_cast<size_t>(c)] = os;
        bias_eff[static_cast<size_t>(c)] = b - 128.0f * static_cast<float>(sum_w_ptr[c]) * os;
    }

#if 0
    // LPBQ deploy-v1 rollback path: the original folded-dequant loader used two
    // generated RPCMEM mappings per tensor.  It is kept here for maintenance
    // reference, but the full LLM can exhaust FastRPC mmap resources near the
    // final blocks.  The active path below packs both arrays into one fd and
    // passes bias_eff through an offset.
    if (!lpbq_create_rpcmem_buffer_from_data(stem + ".lpbq_out_scale.generated",
                                             out_scale.data(), out_scale.size() * sizeof(float), out_scale_buf)) {
        return false;
    }
    if (!lpbq_create_rpcmem_buffer_from_data(stem + ".lpbq_bias_eff.generated",
                                             bias_eff.data(), bias_eff.size() * sizeof(float), bias_eff_buf)) {
        return false;
    }
    return true;
#endif

    const size_t scale_bytes = out_scale.size() * sizeof(float);
    std::vector<uint8_t> folded(scale_bytes * 2u);
    std::memcpy(folded.data(), out_scale.data(), scale_bytes);
    std::memcpy(folded.data() + scale_bytes, bias_eff.data(), scale_bytes);

    LpbqSidecarBuffer folded_buf;
    if (!lpbq_create_rpcmem_buffer_from_data(stem + ".lpbq_folded_dequant.generated",
                                             folded.data(), folded.size(), folded_buf)) {
        return false;
    }
    out_scale_buf = folded_buf;
    out_scale_buf.bytes = scale_bytes;
    out_scale_buf.offset = 0;
    bias_eff_buf = folded_buf;
    bias_eff_buf.data = static_cast<uint8_t *>(folded_buf.data) + scale_bytes;
    bias_eff_buf.bytes = scale_bytes;
    bias_eff_buf.offset = scale_bytes;
    return true;
}

bool lpbq_load_packed_weight_k_major_rpcmem_file(const std::string & path, int k, int n, LpbqSidecarBuffer & out) {
    if (k <= 0 || n <= 0 || (k % 32) != 0 || (n % 32) != 0) {
        std::fprintf(stderr, "LPBQ_SIDECAR invalid packed K-major shape: %s k=%d n=%d\n", path.c_str(), k, n);
        return false;
    }
    const size_t expected_bytes = static_cast<size_t>(k) * static_cast<size_t>(n);
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "LPBQ_SIDECAR missing file: %s\n", path.c_str());
        return false;
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    const long file_size = std::ftell(f);
    if (file_size < 0 || static_cast<size_t>(file_size) != expected_bytes) {
        std::fprintf(stderr, "LPBQ_SIDECAR size mismatch: %s got=%ld expected=%zu\n", path.c_str(), file_size,
                     expected_bytes);
        std::fclose(f);
        return false;
    }
    std::rewind(f);

    std::vector<int8_t> n_major(expected_bytes);
    const size_t nread = std::fread(n_major.data(), 1, expected_bytes, f);
    std::fclose(f);
    if (nread != expected_bytes) {
        std::fprintf(stderr, "LPBQ_SIDECAR fread failed: %s got=%zu expected=%zu\n", path.c_str(), nread,
                     expected_bytes);
        return false;
    }

    void * ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, static_cast<int>(expected_bytes));
    if (!ptr) {
        std::fprintf(stderr, "LPBQ_SIDECAR rpcmem_alloc failed: %s bytes=%zu\n", path.c_str(), expected_bytes);
        return false;
    }

    int8_t * k_major = reinterpret_cast<int8_t *>(ptr);
    const int k_tiles = k / 32;
    const int n_tiles = n / 32;
    const size_t chunk = 8u * 128u;
    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            const size_t src = (static_cast<size_t>(nt) * static_cast<size_t>(k_tiles) +
                                static_cast<size_t>(kt)) * chunk;
            const size_t dst = (static_cast<size_t>(kt) * static_cast<size_t>(n_tiles) +
                                static_cast<size_t>(nt)) * chunk;
            std::memcpy(k_major + dst, n_major.data() + src, chunk);
        }
    }

    const int fd = rpcmem_to_fd(ptr);
    if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, expected_bytes, FASTRPC_MAP_FD)) {
        std::fprintf(stderr, "LPBQ_SIDECAR fastrpc_mmap failed: %s fd=%d bytes=%zu\n", path.c_str(), fd,
                     expected_bytes);
        if (fd >= 0) {
            fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, expected_bytes);
        }
        rpcmem_free(ptr);
        return false;
    }
    out.data = ptr;
    out.fd = fd;
    out.bytes = expected_bytes;
    return true;
}

bool lpbq_hmx_k_group_safe_tile_host(const int8_t * tile0, const int8_t * tile1, int group_chunks,
                                     int max_abs_sum_w) {
    for (int c = 0; c < 32; ++c) {
        int sum_abs = 0;
        for (int kg = 0; kg < group_chunks; ++kg) {
            const int8_t * tile = (kg < 8 || tile1 == nullptr) ? tile0 : tile1;
            const int local_kg = (kg < 8 || tile1 == nullptr) ? kg : (kg - 8);
            const int8_t * chunk = tile + static_cast<size_t>(local_kg) * 128u + static_cast<size_t>(c) * 4u;
            for (int kk = 0; kk < 4; ++kk) {
                int w = static_cast<int>(chunk[kk]);
                if (w < 0) {
                    w = -w;
                }
                sum_abs += w;
            }
        }
        if (sum_abs > max_abs_sum_w) {
            return false;
        }
    }
    return true;
}

bool lpbq_hmx_k32_safe_tile_host(const int8_t * tile) {
    // after.uh is a signed 16-bit view.  This bound guarantees one K32 HMX
    // writeback is exact for any uint4 activation nibble, without a DSP scan.
    return lpbq_hmx_k_group_safe_tile_host(tile, nullptr, 8, 32767 / 15);
}

bool lpbq_hmx_k64_safe_tile_host(const int8_t * tile0, const int8_t * tile1) {
    // Exact K64 grouping is valid only when the combined two-K32 column bound
    // is still inside the signed after.uh range.  This mirrors the DSP default
    // HTP_LPBQ_EXACT_K64_SAFE_ABS_LIMIT=2184.
    return lpbq_hmx_k_group_safe_tile_host(tile0, tile1, 16, 32767 / 15);
}

bool lpbq_create_k32_safe_rpcmem_from_packed(const std::string & label, const LpbqSidecarBuffer & packed,
                                             int k, int n, bool k_major, LpbqSidecarBuffer & out,
                                             int & safe_tiles, int & total_tiles) {
    if (!packed.data || k <= 0 || n <= 0 || (k % 32) != 0 || (n % 32) != 0 ||
        packed.bytes != static_cast<size_t>(k) * static_cast<size_t>(n)) {
        std::fprintf(stderr, "LPBQ_SIDECAR invalid K32-safe source: %s k=%d n=%d bytes=%zu\n",
                     label.c_str(), k, n, packed.bytes);
        return false;
    }

    const int k_tiles = k / 32;
    const int n_tiles = n / 32;
    const size_t chunk = 8u * 128u;
    total_tiles = k_tiles * n_tiles;
    safe_tiles = 0;

    void * ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, total_tiles);
    if (!ptr) {
        std::fprintf(stderr, "LPBQ_SIDECAR rpcmem_alloc failed: %s.k32_safe bytes=%d\n",
                     label.c_str(), total_tiles);
        return false;
    }

    uint8_t * flags = reinterpret_cast<uint8_t *>(ptr);
    const int8_t * packed_weight = reinterpret_cast<const int8_t *>(packed.data);
    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            const size_t tile_index = k_major ?
                (static_cast<size_t>(kt) * static_cast<size_t>(n_tiles) + static_cast<size_t>(nt)) :
                (static_cast<size_t>(nt) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt));
            const bool safe = lpbq_hmx_k32_safe_tile_host(packed_weight + tile_index * chunk);
            flags[tile_index] = safe ? 1u : 0u;
            safe_tiles += safe ? 1 : 0;
        }
    }

    const int fd = rpcmem_to_fd(ptr);
    if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, static_cast<size_t>(total_tiles), FASTRPC_MAP_FD)) {
        std::fprintf(stderr, "LPBQ_SIDECAR fastrpc_mmap failed: %s.k32_safe fd=%d bytes=%d\n",
                     label.c_str(), fd, total_tiles);
        if (fd >= 0) {
            fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, static_cast<size_t>(total_tiles));
        }
        rpcmem_free(ptr);
        return false;
    }

    out.data = ptr;
    out.fd = fd;
    out.bytes = static_cast<size_t>(total_tiles);
    return true;
}

bool lpbq_create_k64_safe_rpcmem_from_packed(const std::string & label, const LpbqSidecarBuffer & packed,
                                             int k, int n, bool k_major, LpbqSidecarBuffer & out,
                                             int & safe_tiles, int & total_tiles) {
    if (!packed.data || k <= 0 || n <= 0 || (k % 64) != 0 || (n % 32) != 0 ||
        packed.bytes != static_cast<size_t>(k) * static_cast<size_t>(n)) {
        std::fprintf(stderr, "LPBQ_SIDECAR invalid K64-safe source: %s k=%d n=%d bytes=%zu\n",
                     label.c_str(), k, n, packed.bytes);
        return false;
    }

    const int k32_tiles = k / 32;
    const int k64_tiles = k / 64;
    const int n_tiles = n / 32;
    const size_t chunk = 8u * 128u;
    total_tiles = k64_tiles * n_tiles;
    safe_tiles = 0;

    void * ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, total_tiles);
    if (!ptr) {
        std::fprintf(stderr, "LPBQ_SIDECAR rpcmem_alloc failed: %s.k64_safe bytes=%d\n",
                     label.c_str(), total_tiles);
        return false;
    }

    uint8_t * flags = reinterpret_cast<uint8_t *>(ptr);
    const int8_t * packed_weight = reinterpret_cast<const int8_t *>(packed.data);
    for (int kt64 = 0; kt64 < k64_tiles; ++kt64) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            const int kt0 = kt64 * 2;
            const size_t tile0_index = k_major ?
                (static_cast<size_t>(kt0) * static_cast<size_t>(n_tiles) + static_cast<size_t>(nt)) :
                (static_cast<size_t>(nt) * static_cast<size_t>(k32_tiles) + static_cast<size_t>(kt0));
            const size_t tile1_index = k_major ?
                (static_cast<size_t>(kt0 + 1) * static_cast<size_t>(n_tiles) + static_cast<size_t>(nt)) :
                (static_cast<size_t>(nt) * static_cast<size_t>(k32_tiles) + static_cast<size_t>(kt0 + 1));
            const size_t out_index = k_major ?
                (static_cast<size_t>(kt64) * static_cast<size_t>(n_tiles) + static_cast<size_t>(nt)) :
                (static_cast<size_t>(nt) * static_cast<size_t>(k64_tiles) + static_cast<size_t>(kt64));
            const bool safe = lpbq_hmx_k64_safe_tile_host(packed_weight + tile0_index * chunk,
                                                          packed_weight + tile1_index * chunk);
            flags[out_index] = safe ? 1u : 0u;
            safe_tiles += safe ? 1 : 0;
        }
    }

    const int fd = rpcmem_to_fd(ptr);
    if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, static_cast<size_t>(total_tiles), FASTRPC_MAP_FD)) {
        std::fprintf(stderr, "LPBQ_SIDECAR fastrpc_mmap failed: %s.k64_safe fd=%d bytes=%d\n",
                     label.c_str(), fd, total_tiles);
        if (fd >= 0) {
            fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, static_cast<size_t>(total_tiles));
        }
        rpcmem_free(ptr);
        return false;
    }

    out.data = ptr;
    out.fd = fd;
    out.bytes = static_cast<size_t>(total_tiles);
    return true;
}

bool lpbq_load_r4_fwht_v2_rpcmem_payload(const std::string & r4_path, int block, const float * input_scale, int k,
                                         LpbqSidecarBuffer & out, bool & folded_input_scale,
                                         bool & structured_fwht) {
    folded_input_scale = false;
    structured_fwht = false;
    if (block != 128) {
        return false;
    }

    const int layer = lpbq_r4_layer_id_from_path(r4_path);
    if (layer < 0) {
        return false;
    }

    std::string v2_dir;
    for (const std::string & candidate : lpbq_r4_fwht_v2_candidate_dirs(r4_path)) {
        if (lpbq_file_exists(lpbq_join_path_str(candidate, "manifest.json"))) {
            v2_dir = candidate;
            break;
        }
    }
    if (v2_dir.empty()) {
        return false;
    }

    const std::string prefix = lpbq_r4_fwht_v2_layer_prefix(layer);
    const std::string scale_in_path = lpbq_join_path_str(v2_dir, prefix + "_scale_in.fp16");
    const std::string scale_out_path = lpbq_join_path_str(v2_dir, prefix + "_scale_out.fp16");
    const std::string sign_in_path = lpbq_join_path_str(v2_dir, prefix + "_sign_in.bitpack");
    const std::string sign_out_path = lpbq_join_path_str(v2_dir, prefix + "_sign_out.bitpack");
    const std::string perm_in_path = lpbq_join_path_str(v2_dir, prefix + "_perm_in.u16");
    const std::string perm_out_path = lpbq_join_path_str(v2_dir, prefix + "_perm_out.u16");

    if (lpbq_file_exists(perm_in_path) || lpbq_file_exists(perm_out_path)) {
        std::fprintf(stderr,
                     "LPBQ_SIDECAR r4_fwht_v2 perm files are unsupported by the legacy bridge: %s\n",
                     v2_dir.c_str());
        return false;
    }
    if (!lpbq_bitpack_is_identity(sign_in_path, block) || !lpbq_bitpack_is_identity(sign_out_path, block)) {
        return false;
    }

    std::vector<float> scale_in;
    std::vector<float> scale_out;
    if (!lpbq_load_fp16_vector_128(scale_in_path, scale_in) ||
        !lpbq_load_fp16_vector_128(scale_out_path, scale_out)) {
        return false;
    }

    constexpr float inv_sqrt128 = 0.08838834764831845f;
    const size_t matrix_elems = static_cast<size_t>(block) * static_cast<size_t>(block);
    const int r4_blocks = (input_scale && k > 0 && (k % block) == 0) ? (k / block) : 1;
    const size_t mapped_bytes = matrix_elems * sizeof(float) * static_cast<size_t>(r4_blocks);
    folded_input_scale = input_scale && r4_blocks > 1;

    void * ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, static_cast<int>(mapped_bytes));
    if (!ptr) {
        std::fprintf(stderr, "LPBQ_SIDECAR rpcmem_alloc failed for r4_fwht_v2: %s bytes=%zu\n",
                     v2_dir.c_str(), mapped_bytes);
        return false;
    }

    float * payload = reinterpret_cast<float *>(ptr);
    std::fill(payload, payload + matrix_elems * static_cast<size_t>(r4_blocks), 0.0f);
    for (int blk = 0; blk < r4_blocks; ++blk) {
        float * block_payload = payload + static_cast<size_t>(blk) * matrix_elems;
        block_payload[0] = LPBQ_R4_FWHT_MAGIC0;
        block_payload[1] = LPBQ_R4_FWHT_MAGIC1;
        block_payload[2] = static_cast<float>(block);
        block_payload[3] = 2.0f;
        float * d2 = block_payload + LPBQ_R4_FWHT_D2_OFFSET;
        float * d1 = block_payload + LPBQ_R4_FWHT_D1_OFFSET;
        for (int row = 0; row < 128; ++row) {
            // Guide v2 stores scales for normalized H_128.  The current DSP FWHT
            // producer is the legacy unnormalized butterfly, so fold 1/sqrt(128)
            // back into D2 while keeping the dense R4 fallback untouched.
            float scale = scale_in[static_cast<size_t>(row)] * inv_sqrt128;
            if (folded_input_scale) {
                scale *= input_scale[static_cast<size_t>(blk) * 128u + static_cast<size_t>(row)];
            }
            d2[row] = scale;
        }
        for (int col = 0; col < 128; ++col) {
            d1[col] = scale_out[static_cast<size_t>(col)];
        }
    }

    const int fd = rpcmem_to_fd(ptr);
    if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, mapped_bytes, FASTRPC_MAP_FD)) {
        std::fprintf(stderr, "LPBQ_SIDECAR fastrpc_mmap failed for r4_fwht_v2: %s fd=%d bytes=%zu\n",
                     v2_dir.c_str(), fd, mapped_bytes);
        if (fd >= 0) {
            fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, mapped_bytes);
        }
        rpcmem_free(ptr);
        return false;
    }

    out.data = ptr;
    out.fd = fd;
    out.bytes = mapped_bytes;
    structured_fwht = true;
    std::fprintf(stderr, "LPBQ_SIDECAR loaded r4_fwht_v2 layer=%d dir=%s fd=%d bytes=%zu folded_input_scale=%d\n",
                 layer, v2_dir.c_str(), out.fd, out.bytes, folded_input_scale ? 1 : 0);
    return true;
}

bool lpbq_load_r4_hmx_dense_fp16_rpcmem_payload(const std::string & r4_path, int block, int k,
                                                LpbqSidecarBuffer & out) {
    if (block != 128 || k <= 0 || (k % block) != 0) {
        return false;
    }

    const int layer = lpbq_r4_layer_id_from_path(r4_path);
    if (layer < 0) {
        return false;
    }

    std::string sidecar_dir;
    for (const std::string & candidate : lpbq_r4_hmx_dense_fp16_candidate_dirs(r4_path)) {
        if (lpbq_file_exists(lpbq_join_path_str(candidate, "manifest.json"))) {
            sidecar_dir = candidate;
            break;
        }
    }
    if (sidecar_dir.empty()) {
        return false;
    }

    const std::string prefix = lpbq_r4_fwht_v2_layer_prefix(layer);
    const std::string tiles_path = lpbq_join_path_str(sidecar_dir, prefix + "_r4_hmx_tiles.bin");
    const int r4_blocks = k / block;
    const int tiles_per_block = (block / 32) * (block / 32);
    const size_t expected_bytes =
        static_cast<size_t>(r4_blocks) * static_cast<size_t>(tiles_per_block) * 2048u;
    if (!lpbq_load_rpcmem_file(tiles_path, expected_bytes, out)) {
        return false;
    }

    std::fprintf(stderr,
                 "LPBQ_SIDECAR loaded no-quality r4_hmx_dense_fp16 layer=%d dir=%s fd=%d bytes=%zu\n",
                 layer, sidecar_dir.c_str(), out.fd, out.bytes);
    return true;
}

bool lpbq_load_r4_rpcmem_file_transposed(const std::string & path, int block, const float * input_scale, int k,
                                         LpbqSidecarBuffer & out, bool & folded_input_scale,
                                         bool & structured_fwht, bool prefer_structured_fwht = true) {
    const size_t matrix_elems = static_cast<size_t>(block) * static_cast<size_t>(block);
    const size_t expected_bytes = matrix_elems * sizeof(float);
    const int r4_blocks = (input_scale && k > 0 && (k % block) == 0) ? (k / block) : 1;
    const size_t mapped_bytes = expected_bytes * static_cast<size_t>(r4_blocks);
    folded_input_scale = input_scale && r4_blocks > 1;
    structured_fwht = false;

    const std::string fwht_path = lpbq_r4_fwht_sidecar_path(path);
    if (prefer_structured_fwht && block == 128 &&
        lpbq_load_r4_fwht_v2_rpcmem_payload(path, block, input_scale, k, out,
                                            folded_input_scale, structured_fwht)) {
        return true;
    }
    if (prefer_structured_fwht && block == 128 && lpbq_file_exists(fwht_path)) {
        constexpr size_t fwht_scale_count = 256;
        constexpr size_t fwht_scale_bytes = fwht_scale_count * sizeof(float);
        std::vector<float> fwht_scales(fwht_scale_count);
        FILE * f_fwht = std::fopen(fwht_path.c_str(), "rb");
        if (!f_fwht) {
            std::fprintf(stderr, "LPBQ_SIDECAR missing structured R4 file after exists check: %s\n",
                         fwht_path.c_str());
            return false;
        }
        const size_t nread_fwht = std::fread(fwht_scales.data(), 1, fwht_scale_bytes, f_fwht);
        std::fclose(f_fwht);
        if (nread_fwht != fwht_scale_bytes) {
            std::fprintf(stderr, "LPBQ_SIDECAR structured R4 fread failed: %s got=%zu expected=%zu\n",
                         fwht_path.c_str(), nread_fwht, fwht_scale_bytes);
            return false;
        }

        void * ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, static_cast<int>(mapped_bytes));
        if (!ptr) {
            std::fprintf(stderr, "LPBQ_SIDECAR rpcmem_alloc failed for structured R4: %s bytes=%zu\n",
                         fwht_path.c_str(), mapped_bytes);
            return false;
        }
        float * payload = reinterpret_cast<float *>(ptr);
        std::fill(payload, payload + matrix_elems * static_cast<size_t>(r4_blocks), 0.0f);
        for (int blk = 0; blk < r4_blocks; ++blk) {
            float * block_payload = payload + static_cast<size_t>(blk) * matrix_elems;
            block_payload[0] = LPBQ_R4_FWHT_MAGIC0;
            block_payload[1] = LPBQ_R4_FWHT_MAGIC1;
            block_payload[2] = static_cast<float>(block);
            block_payload[3] = 1.0f;
            float * d2 = block_payload + LPBQ_R4_FWHT_D2_OFFSET;
            float * d1 = block_payload + LPBQ_R4_FWHT_D1_OFFSET;
            for (int row = 0; row < 128; ++row) {
                float scale = fwht_scales[static_cast<size_t>(row)];
                if (folded_input_scale) {
                    scale *= input_scale[static_cast<size_t>(blk) * 128u + static_cast<size_t>(row)];
                }
                d2[row] = scale;
            }
            for (int col = 0; col < 128; ++col) {
                d1[col] = fwht_scales[128u + static_cast<size_t>(col)];
            }
        }

        const int fd = rpcmem_to_fd(ptr);
        if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, mapped_bytes, FASTRPC_MAP_FD)) {
            std::fprintf(stderr, "LPBQ_SIDECAR fastrpc_mmap failed for structured R4: %s fd=%d bytes=%zu\n",
                         fwht_path.c_str(), fd, mapped_bytes);
            if (fd >= 0) {
                fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, mapped_bytes);
            }
            rpcmem_free(ptr);
            return false;
        }
        out.data = ptr;
        out.fd = fd;
        out.bytes = mapped_bytes;
        structured_fwht = true;
        return true;
    }

    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "LPBQ_SIDECAR missing R4 file: %s\n", path.c_str());
        return false;
    }
    std::vector<float> row_major(static_cast<size_t>(block) * static_cast<size_t>(block));
    const size_t nread = std::fread(row_major.data(), 1, expected_bytes, f);
    std::fclose(f);
    if (nread != expected_bytes) {
        std::fprintf(stderr, "LPBQ_SIDECAR R4 fread failed: %s got=%zu expected=%zu\n", path.c_str(), nread,
                     expected_bytes);
        return false;
    }

    void * ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, static_cast<int>(mapped_bytes));
    if (!ptr) {
        std::fprintf(stderr, "LPBQ_SIDECAR rpcmem_alloc failed for R4: %s bytes=%zu\n", path.c_str(), mapped_bytes);
        return false;
    }
    float * col_major = reinterpret_cast<float *>(ptr);
    for (int blk = 0; blk < r4_blocks; ++blk) {
        for (int col = 0; col < block; ++col) {
            for (int row = 0; row < block; ++row) {
                float coeff =
                    row_major[static_cast<size_t>(row) * static_cast<size_t>(block) + static_cast<size_t>(col)];
                if (folded_input_scale) {
                    coeff *= input_scale[static_cast<size_t>(blk) * static_cast<size_t>(block) +
                                         static_cast<size_t>(row)];
                }
                col_major[static_cast<size_t>(blk) * matrix_elems +
                          static_cast<size_t>(col) * static_cast<size_t>(block) + static_cast<size_t>(row)] = coeff;
            }
        }
    }

    const int fd = rpcmem_to_fd(ptr);
    if (fd < 0 || fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, mapped_bytes, FASTRPC_MAP_FD)) {
        std::fprintf(stderr, "LPBQ_SIDECAR fastrpc_mmap failed for R4: %s fd=%d bytes=%zu\n", path.c_str(), fd,
                     mapped_bytes);
        if (fd >= 0) {
            fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, mapped_bytes);
        }
        rpcmem_free(ptr);
        return false;
    }
    out.data = ptr;
    out.fd = fd;
    out.bytes = mapped_bytes;
    return true;
}

bool lpbq_load_act_scale(const std::string & bin_path, const std::string & text_path, float & out) {
    FILE * f = std::fopen(bin_path.c_str(), "rb");
    if (f) {
        const size_t nread = std::fread(&out, 1, sizeof(float), f);
        std::fclose(f);
        return nread == sizeof(float) && out > 0.0f && std::isfinite(out);
    }
    f = std::fopen(text_path.c_str(), "r");
    if (!f) {
        std::fprintf(stderr, "LPBQ_SIDECAR missing act_scale: %s or %s\n", bin_path.c_str(), text_path.c_str());
        return false;
    }
    const int nscan = std::fscanf(f, "%f", &out);
    std::fclose(f);
    return nscan == 1 && out > 0.0f && std::isfinite(out);
}

const LpbqSidecar * get_lpbq_sidecar(const char * weight_name, int k, int n) {
    const std::string key = weight_name ? weight_name : "";
    auto & sidecar = g_lpbq_sidecar_cache[key];
    if (sidecar.loaded) {
        return sidecar.ok ? &sidecar : nullptr;
    }
    sidecar.loaded = true;

    const char * dir = std::getenv("LLAMA_NPU_LPBQ_SIDECAR_DIR");
    if (!dir || !dir[0]) {
        std::fprintf(stderr, "LPBQ_SIDECAR LLAMA_NPU_LPBQ_SIDECAR_DIR is not set for weight=%s\n",
                     weight_name ? weight_name : "");
        return nullptr;
    }

    const std::string stem = lpbq_sanitize_tensor_name(weight_name);
    const std::string scale2_path = lpbq_join_path(dir, stem + ".lpbq_scale2.f32");
    const std::string bias_path = lpbq_join_path(dir, stem + ".lpbq_bias.f32");
    const std::string input_scale_path = lpbq_join_path(dir, stem + ".lpbq_input_scale.f32");
    const std::string packed_weight_path = lpbq_join_path(dir, stem + ".lpbq_w_hmx_k4.bin");
    const std::string sum_w_path = lpbq_join_path(dir, stem + ".lpbq_sum_w.i32");
    const std::string act_scale_bin_path = lpbq_join_path(dir, stem + ".lpbq_act_scale.f32");
    const std::string act_scale_txt_path = lpbq_join_path(dir, stem + ".lpbq_act_scale.txt");
    std::string r4_path;
    int r4_block = 0;
    const bool r4_sidecar_available = lpbq_r4_path_for_weight(dir, stem, r4_path, r4_block);
    // 2026-07-03 no-quality/performance-first decision: R4/FWHT is a
    // structural HVX bottleneck for ffn_down (and at best marginal for
    // gate/up), so the runtime now treats LPBQ weights as non-R4 even when R4
    // sidecars or R4 env flags are present.  Keep the old sidecar/env route as
    // a rollback note, but do not select it in this performance track:
    // const bool needs_r4 = lpbq_r4_path_for_weight(dir, stem, r4_path, r4_block);
    // const bool needs_r4 = r4_sidecar_available && env_truthy("LLAMA_NPU_LPBQ_ENABLE_R4_PATH");
    const bool needs_r4 = false;
    const bool exact_non_r4 = !needs_r4 && lpbq_exact_non_r4_allowed_for_stem(stem);
    sidecar.exact_non_r4 = exact_non_r4;
    const int v6_full_group_tiles = env_int_or_default("LLAMA_NPU_LPBQ_V6_FULL_GROUP_TILES", 16);
    const bool r4_use_full_v6_weight_fd = env_truthy("LLAMA_NPU_LPBQ_R4_USE_FULL_V6_WEIGHT_FD");
    // LPBQ deploy-v1 bottleneck reset: enabling full-V6 for every Q8 tensor
    // exhausts FastRPC mmap/RPCMEM resources before the request reaches the
    // real bottleneck.  For R4, real-layer timing on 2026-07-03 showed the
    // offline full-V6 fd still caused repeated runtime publish/expand, while
    // compact K-major let the grouped-V6 DSP path consume the same logical
    // weight shape with hmx_weight_expand near zero. Keep the old full-V6 fd
    // route only behind LLAMA_NPU_LPBQ_R4_USE_FULL_V6_WEIGHT_FD=1.
    // const bool want_v6_full_weight =
    //     env_truthy("LLAMA_NPU_LPBQ_ENABLE_V6_FULL_WEIGHT") && !needs_r4;
    const bool want_v6_full_non_r4 = env_truthy("LLAMA_NPU_LPBQ_V6_FULL_NON_R4");
    const bool want_v6_full_non_r4_for_layer =
        !needs_r4 && want_v6_full_non_r4 && lpbq_v6_full_non_r4_allowed_for_stem(stem);
    // Old R4 condition, kept for rollback reference:
    //     env_truthy("LLAMA_NPU_LPBQ_ENABLE_V6_FULL_WEIGHT") &&
    //     (needs_r4 || env_truthy("LLAMA_NPU_LPBQ_V6_FULL_ALL") ||
    //      want_v6_full_non_r4_for_layer);
    const bool want_v6_full_weight =
        env_truthy("LLAMA_NPU_LPBQ_ENABLE_V6_FULL_WEIGHT") &&
        ((needs_r4 && r4_use_full_v6_weight_fd) ||
         (!needs_r4 && (env_truthy("LLAMA_NPU_LPBQ_V6_FULL_ALL") ||
                        want_v6_full_non_r4_for_layer)));
    const std::string packed_weight_v6_full_path =
        lpbq_join_path(dir, stem + ".lpbq_w_hmx_v6_full_g" + std::to_string(v6_full_group_tiles) + ".bin");

    if (!lpbq_load_rpcmem_file(scale2_path, static_cast<size_t>(n) * sizeof(float), sidecar.scale2)) {
        return nullptr;
    }
    if (!lpbq_load_act_scale(act_scale_bin_path, act_scale_txt_path, sidecar.act_scale)) {
        return nullptr;
    }
    if (lpbq_file_exists(bias_path)) {
        if (!lpbq_load_rpcmem_file(bias_path, static_cast<size_t>(n) * sizeof(float), sidecar.bias)) {
            return nullptr;
        }
        sidecar.has_bias = true;
    }
    if (lpbq_file_exists(input_scale_path)) {
        // LPBQ deploy-v1 compatibility note: early exports assumed LET/input
        // scale had been fully folded and therefore omitted this sidecar.  When
        // present, it carries inverse per-input scale and is fused into the
        // activation quant/R4 stage on DSP.
        if (!lpbq_load_rpcmem_file(input_scale_path, static_cast<size_t>(k) * sizeof(float), sidecar.input_scale)) {
            return nullptr;
        }
        sidecar.has_input_scale = true;
    }
    const bool has_packed_file = lpbq_file_exists(packed_weight_path);
    const bool has_sum_w_file = lpbq_file_exists(sum_w_path);
    if (has_packed_file || has_sum_w_file) {
        if (!has_packed_file || !has_sum_w_file) {
            std::fprintf(stderr, "LPBQ_SIDECAR packed sidecar mismatch for %s: packed=%d sum_w=%d\n",
                         stem.c_str(), has_packed_file ? 1 : 0, has_sum_w_file ? 1 : 0);
            return nullptr;
        }
        // Optional fast path: prepacked K4/HMX weight tiles plus output-column
        // sums remove the online Q8_0 container unpack/repack from every op.
        // Deploy-v1 defaults to a host-side K-major transpose so the DSP can
        // stream all N tiles for one K tile sequentially.  The old N-major
        // mapping remains available for rollback and standalone parity.
        bool use_v6_full_weight_for_layer = want_v6_full_weight;
        if (exact_non_r4) {
            // Q/K exact fallback consumes compact K4 chunks plus safe tables.
            // Keep full-V6 disabled for these layers even if a broader
            // full-V6 experiment is requested through the environment.
            use_v6_full_weight_for_layer = false;
        }
        if (want_v6_full_weight) {
            if (use_v6_full_weight_for_layer && needs_r4 && !lpbq_v6_full_allowed_for_stem(dir, stem)) {
                std::fprintf(stderr,
                             "LPBQ_SIDECAR full-V6 safe-list fallback to compact for %s\n",
                             stem.c_str());
                use_v6_full_weight_for_layer = false;
            }
            const int group_k = 32 * v6_full_group_tiles;
            if (group_k <= 32 || (k % 32) != 0) {
                std::fprintf(stderr,
                             "LPBQ_SIDECAR full-V6 fallback to compact for %s: unsupported shape k=%d group_tiles=%d\n",
                             stem.c_str(), k, v6_full_group_tiles);
                // Old strict A/B behavior returned nullptr here.  Keep that
                // policy documented.  The earlier guard also required
                // (k % group_k) == 0; padded tail groups now let ffn_down use
                // g16 main groups without falling back to compact.
                // return nullptr;
                use_v6_full_weight_for_layer = false;
            }
            if (use_v6_full_weight_for_layer && !lpbq_file_exists(packed_weight_v6_full_path)) {
                std::fprintf(stderr,
                             "LPBQ_SIDECAR full-V6 fallback to compact for %s: missing sidecar %s\n",
                             stem.c_str(), packed_weight_v6_full_path.c_str());
                // Old strict A/B behavior returned nullptr here.  Keep the
                // compact packed sidecar as the per-layer fallback when only a
                // subset of shapes has offline full-V6 materialization.
                // return nullptr;
                use_v6_full_weight_for_layer = false;
            }
        }
        if (use_v6_full_weight_for_layer && !needs_r4) {
            // LPBQ deploy-v1 full-V6 A/B: keep the compact sidecar as the
            // rollback source on disk, but when explicitly enabled pass the
            // grouped full-stride HMX layout through the existing packed_weight
            // fd and mark it with LLM_NPU_MODE_LPBQ_PACKED_V6_FULL.
            if (!lpbq_load_rpcmem_file(packed_weight_v6_full_path,
                                       lpbq_v6_full_grouped_bytes(k, n, v6_full_group_tiles),
                                       sidecar.packed_weight)) {
                return nullptr;
            }
            sidecar.packed_weight_k_major = true;
            sidecar.packed_weight_v6_full = true;
            sidecar.packed_weight_v6_full_group_tiles = v6_full_group_tiles;
        } else if (!env_truthy("LLAMA_NPU_LPBQ_DISABLE_K_MAJOR_PACKED")) {
            if (!lpbq_load_packed_weight_k_major_rpcmem_file(packed_weight_path, k, n, sidecar.packed_weight)) {
                return nullptr;
            }
            sidecar.packed_weight_k_major = true;
        } else {
            if (!lpbq_load_rpcmem_file(packed_weight_path, static_cast<size_t>(k) * static_cast<size_t>(n),
                                       sidecar.packed_weight)) {
                return nullptr;
            }
            sidecar.packed_weight_k_major = false;
        }
        if (use_v6_full_weight_for_layer && needs_r4) {
            // LPBQ deploy-v1 2026-06-10 correction: the DSP full-V6 R4 path is
            // compiled as a small-M/decode path (currently m<=4). The old host
            // code replaced the compact packed sidecar with full-V6 at load
            // time, so prefill m>4 hit the DSP guard and returned -16012. Keep
            // compact as the default online fd, and carry full-V6 separately
            // for call-time small-M selection.
            // Old behavior kept above in the non-R4 branch: full-V6 was stored
            // directly in sidecar.packed_weight and marked PACKED_V6_FULL.
            if (!lpbq_load_rpcmem_file(packed_weight_v6_full_path,
                                       lpbq_v6_full_grouped_bytes(k, n, v6_full_group_tiles),
                                       sidecar.packed_weight_v6_full_buffer)) {
                return nullptr;
            }
            sidecar.has_packed_weight_v6_full = true;
            sidecar.packed_weight_v6_full_group_tiles = v6_full_group_tiles;
            sidecar.r4_v6_scale_1_16 = lpbq_v6_full_r4_scale_1_16_for_stem(dir, stem);
        }
        if (!lpbq_load_rpcmem_file(sum_w_path, static_cast<size_t>(n) * sizeof(int32_t), sidecar.sum_w)) {
            return nullptr;
        }
        if (!env_truthy("LLAMA_NPU_LPBQ_DISABLE_FOLDED_DEQUANT")) {
            if (!lpbq_create_folded_dequant_sidecars(stem, n, sidecar.scale2, sidecar.sum_w, sidecar.bias,
                                                     sidecar.has_bias, sidecar.act_scale, sidecar.out_scale,
                                                     sidecar.bias_eff)) {
                return nullptr;
            }
            sidecar.has_folded_dequant = true;
        }
        if ((needs_r4 || exact_non_r4) && !sidecar.packed_weight_v6_full &&
            !env_truthy("LLAMA_NPU_LPBQ_DISABLE_K32_SAFE_SIDECAR")) {
            if (lpbq_create_k32_safe_rpcmem_from_packed(stem, sidecar.packed_weight, k, n,
                                                        sidecar.packed_weight_k_major, sidecar.k32_safe,
                                                        sidecar.k32_safe_tiles, sidecar.k32_total_tiles)) {
                sidecar.has_k32_safe = true;
            } else {
                // LPBQ deploy-v1 FP16-base retune: k32_safe is only a
                // performance hint for the R4/nibble fallback.  If FastRPC runs
                // out of tiny mappings, keep the old K16 route alive instead
                // of failing the whole LLM request.
                std::fprintf(stderr,
                             "LPBQ_SIDECAR warning: k32_safe disabled for %s after generation/mmap failure\n",
                             stem.c_str());
            }
        }
        if ((needs_r4 || exact_non_r4) && !sidecar.packed_weight_v6_full && (k % 64) == 0 &&
            !env_truthy("LLAMA_NPU_LPBQ_DISABLE_K64_SAFE_SIDECAR")) {
            if (lpbq_create_k64_safe_rpcmem_from_packed(stem, sidecar.packed_weight, k, n,
                                                        sidecar.packed_weight_k_major, sidecar.k64_safe,
                                                        sidecar.k64_safe_tiles, sidecar.k64_total_tiles)) {
                sidecar.has_k64_safe = true;
            } else {
                std::fprintf(stderr,
                             "LPBQ_SIDECAR warning: k64_safe disabled for %s after generation/mmap failure\n",
                             stem.c_str());
            }
        }
        sidecar.has_packed_weight = true;
    }
    if (needs_r4) {
        const bool enable_r4_scale_fold = env_truthy("LLAMA_NPU_LPBQ_ENABLE_R4_SCALE_FOLD");
        bool folded_input_scale = false;
        const float * input_scale_data =
            (enable_r4_scale_fold && sidecar.has_input_scale) ?
                reinterpret_cast<const float *>(sidecar.input_scale.data) : nullptr;
        bool dense_loader_structured_fwht = false;
        if (!lpbq_load_r4_rpcmem_file_transposed(r4_path, r4_block, input_scale_data, k, sidecar.r4,
                                                 folded_input_scale, dense_loader_structured_fwht,
                                                 /*prefer_structured_fwht=*/false)) {
            return nullptr;
        }
        sidecar.has_r4 = true;
        sidecar.r4_input_scale_folded = folded_input_scale;
        sidecar.r4_structured_fwht = false;
        if (env_truthy("LLAMA_NPU_LPBQ_R4_HMX_DENSE_FP16_SIDECAR")) {
            if (!lpbq_load_r4_hmx_dense_fp16_rpcmem_payload(r4_path, r4_block, k,
                                                            sidecar.r4_hmx_dense_fp16)) {
                std::fprintf(stderr,
                             "LPBQ_SIDECAR requested r4_hmx_dense_fp16 but failed for %s; "
                             "this is a no-quality/performance-first experiment only\n",
                             stem.c_str());
                return nullptr;
            }
            sidecar.has_r4_hmx_dense_fp16 = true;
        }
        if (r4_block == 128 && lpbq_file_exists(lpbq_r4_fwht_sidecar_path(r4_path))) {
            bool structured_folded_input_scale = false;
            bool structured_fwht = false;
            if (!lpbq_load_r4_rpcmem_file_transposed(r4_path, r4_block, input_scale_data, k,
                                                     sidecar.r4_structured_fwht_buffer,
                                                     structured_folded_input_scale, structured_fwht,
                                                     /*prefer_structured_fwht=*/true)) {
                return nullptr;
            }
            // The structured fd is selected only for small-M requests below.
            // Keep dense R4 as the default fd so prefill can stay on the
            // accepted exact producer when scalar/parallel FWHT is slower.
            sidecar.r4_structured_fwht = structured_fwht;
            if (structured_folded_input_scale != folded_input_scale) {
                std::fprintf(stderr,
                             "LPBQ_SIDECAR warning: structured R4 fold mismatch for %s dense=%d structured=%d\n",
                             stem.c_str(), folded_input_scale ? 1 : 0,
                             structured_folded_input_scale ? 1 : 0);
            }
        }
        sidecar.r4_block = r4_block;
    }

    sidecar.ok = true;
    if (env_truthy("LLAMA_NPU_LPBQ_SIDECAR_LOG")) {
        std::fprintf(stderr,
                     "LPBQ_SIDECAR loaded weight=%s stem=%s k=%d n=%d scale2_fd=%d bias_fd=%d out_scale_fd=%d bias_eff_fd=%d folded_dequant=%d r4_fd=%d r4_structured_fd=%d r4_hmx_dense_fp16_fd=%d input_scale_fd=%d packed_fd=%d packed_k_major=%d sum_w_fd=%d k32_safe_fd=%d k32_safe_tiles=%d/%d k64_safe_fd=%d k64_safe_tiles=%d/%d r4_block=%d r4_folded_input_scale=%d r4_v6_scale_1_16=%d r4_structured_fwht=%d r4_hmx_dense_fp16=%d act_scale=%g\n",
                      weight_name ? weight_name : "", stem.c_str(), k, n, sidecar.scale2.fd,
                     sidecar.has_bias ? sidecar.bias.fd : -1,
                     sidecar.has_folded_dequant ? sidecar.out_scale.fd : -1,
                     sidecar.has_folded_dequant ? sidecar.bias_eff.fd : -1,
                     sidecar.has_folded_dequant ? 1 : 0,
                     sidecar.has_r4 ? sidecar.r4.fd : -1,
                     sidecar.r4_structured_fwht ? sidecar.r4_structured_fwht_buffer.fd : -1,
                      sidecar.has_r4_hmx_dense_fp16 ? sidecar.r4_hmx_dense_fp16.fd : -1,
                      sidecar.has_input_scale ? sidecar.input_scale.fd : -1,
                      sidecar.has_packed_weight ? sidecar.packed_weight.fd : -1,
                      sidecar.packed_weight_k_major ? 1 : 0,
                      sidecar.has_packed_weight ? sidecar.sum_w.fd : -1,
                      sidecar.has_k32_safe ? sidecar.k32_safe.fd : -1,
                      sidecar.k32_safe_tiles, sidecar.k32_total_tiles,
                       sidecar.has_k64_safe ? sidecar.k64_safe.fd : -1,
                       sidecar.k64_safe_tiles, sidecar.k64_total_tiles, sidecar.r4_block,
                      sidecar.r4_input_scale_folded ? 1 : 0,
                      sidecar.r4_v6_scale_1_16 ? 1 : 0,
                      sidecar.r4_structured_fwht ? 1 : 0,
                      sidecar.has_r4_hmx_dense_fp16 ? 1 : 0, sidecar.act_scale);
        std::fprintf(stderr,
                     "LPBQ_SIDECAR packed_v6_full=%d v6_full_buffer_fd=%d v6_full_group_tiles=%d exact_non_r4=%d r4_available=%d r4_enabled=%d\n",
                     sidecar.packed_weight_v6_full ? 1 : 0,
                     sidecar.has_packed_weight_v6_full ? sidecar.packed_weight_v6_full_buffer.fd : -1,
                     sidecar.packed_weight_v6_full_group_tiles,
                     sidecar.exact_non_r4 ? 1 : 0,
                     r4_sidecar_available ? 1 : 0,
                     needs_r4 ? 1 : 0);
    }
    return &sidecar;
}

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
    if (std::strcmp(mode, "pure_fp16") == 0 || std::strcmp(mode, "pure-fp16") == 0 ||
        std::strcmp(mode, "fp16") == 0) {
        // The pure-FP16 route is selected by tensor dtype below. This flag is a
        // trace/deployment label so scripts can prove the run did not enter an
        // INT4/INT8 weight-dequant stage.
        flags |= LLM_NPU_MODE_PURE_FP16;
    }
    if (std::strcmp(mode, "lpbq_int8") == 0 || std::strcmp(mode, "lpbq-int8") == 0 ||
        std::strcmp(mode, "lpbq-w4a8-int8") == 0 || std::strcmp(mode, "w4a8_lpbq") == 0) {
        flags |= LLM_NPU_MODE_LPBQ_INT8;
    }
    if (env_truthy("LLAMA_NPU_TRACE")) {
        flags |= LLM_NPU_MODE_TRACE;
    }
    if (env_truthy("LLAMA_NPU_DETAILED_TRACE")) {
        flags |= LLM_NPU_MODE_TRACE | LLM_NPU_MODE_DETAILED_TRACE;
    }
    if (env_truthy("LLAMA_NPU_LPBQ_TILE_TRACE")) {
        flags |= LLM_NPU_MODE_LPBQ_TILE_TRACE;
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

int env_int_range_or_default(const char * name, int default_value, int min_value, int max_value) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < min_value || parsed > max_value) {
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

bool is_lpbq_int8_mode(const char * mode) {
    return std::strcmp(mode, "lpbq_int8") == 0 || std::strcmp(mode, "lpbq-int8") == 0 ||
           std::strcmp(mode, "lpbq-w4a8-int8") == 0 || std::strcmp(mode, "w4a8_lpbq") == 0;
}

bool is_hmx_layout_matmul_category(const char * category) {
    return std::strcmp(category, "q_matrix") == 0 ||
           std::strcmp(category, "k_matrix") == 0 ||
           std::strcmp(category, "v_matrix") == 0 ||
           std::strcmp(category, "o_matrix") == 0 ||
           std::strcmp(category, "ffn_gate") == 0 ||
           std::strcmp(category, "ffn_up") == 0 ||
           std::strcmp(category, "ffn_down") == 0;
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
        case HTP_OPS_MAT_MUL_LPBQ_A8W8:
            return "matmul_lpbq_a8w8";
        case HTP_OPS_FLASH_ATTN_QO_F32_KV_F16:
            return "flash_attn";
        default:
            return "unknown";
    }
}

const char * htp_op_trace_name(int op_index, int mode_flags) {
    if (op_index == HTP_OPS_MAT_MUL_PERMUTED_W16A32 && (mode_flags & LLM_NPU_MODE_PURE_FP16)) {
        return "matmul_f16";
    }
    return htp_op_name(op_index);
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
        case LLM_TRACE_STAGE_ZERO_POINT_CORRECTION: return "zero_point_correction";
        case LLM_TRACE_STAGE_DEQUANT_STORE: return "dequant_store";
        case LLM_TRACE_STAGE_R4_ROTATE: return "r4_rotate";
        case LLM_TRACE_STAGE_R4_SCALE_CACHE: return "r4_scale_cache";
        case LLM_TRACE_STAGE_R4_DOT_PACK: return "r4_dot_pack";
        case LLM_TRACE_STAGE_R4_DENSE_DOT: return "r4_dense_dot";
        case LLM_TRACE_STAGE_R4_QUANT_SCALE: return "r4_quant_scale";
        case LLM_TRACE_STAGE_R4_PACK_UB_LAYOUT: return "r4_pack_ub_layout";
        case LLM_TRACE_STAGE_R4_UNACCOUNTED: return "r4_unaccounted";
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
        // LPBQ deploy-v1 trace-attribution follow-up: keep the grouped-V6
        // scheduling/control residual visible by name instead of falling
        // through to "unknown" in NTFF/CSV summaries.
        case LLM_TRACE_STAGE_HMX_GROUPED_V6_CONTROL_GAP: return "hmx_grouped_v6_control_gap";
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
                (long long) e.trace_id, mode, e.flags, phase, htp_op_trace_name(e.op_index, e.flags), e.op_index, category,
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
                bool hmx_layout_category_ok = is_hmx_layout_matmul_category(category);

                // FP16 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    // The HTP FP16 converter only writes HMX tile layout for
                    // Q/K/V/O and FFN matrices. Other F16 matmuls, especially
                    // the output head, remain regular row-major GGUF tensors and
                    // must stay on the CPU or logits become corrupted.
                    return shape_ok && hmx_layout_category_ok;
                }
                // (repacked) Q4_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) Q8_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 && activation->type == GGML_TYPE_F32) {
                    if (is_lpbq_int8_mode(mode)) {
                        return shape_ok && hmx_layout_category_ok;
                    }
                    if (is_w8pc_a8pt_mode(mode)) {
                        return shape_ok && hmx_layout_category_ok;
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
    int request_mode_flags = mode_flags;
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
                    if (!is_hmx_layout_matmul_category(category)) {
                        GGML_ASSERT(false && "F16 HMX matmul requested for a non-HMX-layout weight");
                    }
                    // pure_fp16: FP32 activation stays in GGML, is converted to FP16 only
                    // when loading into VTCM; the weight is already HMX-layout FP16.
                    // Do not route this case through any INT4/INT8 per-group
                    // dequantization kernel.
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
                    if (is_lpbq_int8_mode(npu_mode)) {
                        if (!is_hmx_layout_matmul_category(category)) {
                            GGML_ASSERT(false && "LPBQ A8W8 matmul requested for a non-HMX-layout weight");
                        }
                        const LpbqSidecar * sidecar = get_lpbq_sidecar(weight_name, k, n);
                        GGML_ASSERT(sidecar && sidecar->ok && "LPBQ sidecar missing or invalid");
                        const char * lpbq_sidecar_dir = std::getenv("LLAMA_NPU_LPBQ_SIDECAR_DIR");
                        const std::string lpbq_stem = lpbq_sanitize_tensor_name(weight_name);
                        const bool force_r4_full_u8_safe =
                            env_truthy("LLAMA_NPU_LPBQ_FORCE_R4_FULL_U8_SAFE") &&
                            lpbq_r4_full_u8_safe_allowed_for_stem(lpbq_sidecar_dir, lpbq_stem);
                        const bool r4_compact_full_u8_safe_ab =
                            env_truthy("LLAMA_NPU_LPBQ_R4_COMPACT_FULL_U8_SAFE_AB");
                        // LPBQ deploy-v1 direct-final DMA probe note:
                        // prefill m=136 was tested with a 256 default after the
                        // DSP grouped full-V6 path gained the same compile-time
                        // limit. Real-layer gates on 2026-06-25 rejected
                        // blk.3/4/5 at the strict R4 max_abs threshold, so keep
                        // the default decode-only and require an explicit env
                        // override for broader prefill experiments.
                        // const int r4_full_v6_small_m_max =
                        //     env_int_or_default("LLAMA_NPU_LPBQ_R4_FULL_V6_SMALL_M_MAX", 4);
                        // 2026-06-25 performance-only probe: the user allowed
                        // looser numerical error, so widen the host R4 full-V6
                        // route to prefill-sized M while keeping the old
                        // decode-only default above as the rollback note.
                        const int r4_full_v6_min_m =
                            env_int_range_or_default("LLAMA_NPU_LPBQ_R4_FULL_V6_MIN_M", 32, 0, 1000000);
                        const int r4_full_v6_small_m_max =
                            env_int_range_or_default("LLAMA_NPU_LPBQ_R4_FULL_V6_SMALL_M_MAX", 256, -1, 1000000);
                        const bool r4_full_v6_m_allowed =
                            sidecar->has_r4 && m >= r4_full_v6_min_m &&
                            (r4_full_v6_small_m_max < 0 || m <= r4_full_v6_small_m_max);
                        const bool r4_use_full_v6_weight_fd =
                            env_truthy("LLAMA_NPU_LPBQ_R4_USE_FULL_V6_WEIGHT_FD");
                        // Old selection picked the full-V6 fd whenever it was
                        // present:
                        //     sidecar->has_r4 && sidecar->has_packed_weight_v6_full &&
                        //     r4_full_v6_m_allowed;
                        // LPBQ per-group scale is already offline-folded, so
                        // the R4/full-U8 path should default to the compact
                        // K-major fd and keep HMX computation at the logical
                        // layer/group shape. The full-V6 fd remains an explicit
                        // performance A/B only.
                        const bool use_r4_decode_full_v6 =
                            r4_use_full_v6_weight_fd &&
                            sidecar->has_r4 && sidecar->has_packed_weight_v6_full &&
                            r4_full_v6_m_allowed;
                        const bool request_packed_v6_full =
                            sidecar->packed_weight_v6_full || use_r4_decode_full_v6;
                        const LpbqSidecarBuffer & request_packed_weight =
                            use_r4_decode_full_v6 ? sidecar->packed_weight_v6_full_buffer :
                                                     sidecar->packed_weight;
                        const bool request_packed_k_major =
                            sidecar->packed_weight_k_major || request_packed_v6_full;
                        const bool request_r4_v6_scale_1_16 =
                            use_r4_decode_full_v6 && sidecar->r4_v6_scale_1_16;
                        const bool request_exact_non_r4 =
                            sidecar->exact_non_r4 && sidecar->has_k64_safe && !request_packed_v6_full;
                        // 2026-07-03 no-quality/performance-first routing:
                        // keep the older global force expression as the rollback idea, but do
                        // not let ordinary smoke inherit the compact R4 full-U8 branch. That
                        // branch fixes repeated full-V6 fd publish cost, but real-layer gates
                        // still mark it no-quality; require an explicit A/B env before routing
                        // compact K-major weights through R4_FULL_U8_SAFE.
                        // const bool use_r4_full_u8_safe =
                        //     request_packed_v6_full || (force_r4_full_u8_safe && sidecar->has_r4);
                        const bool force_r4_full_u8_safe_for_m =
                            r4_compact_full_u8_safe_ab &&
                            force_r4_full_u8_safe && sidecar->has_r4 && r4_full_v6_m_allowed;
                        const bool use_r4_full_u8_safe =
                            request_packed_v6_full ||
                            force_r4_full_u8_safe_for_m;
                        // LPBQ deploy-v1 structured R4 selection: keep FWHT on
                        // decode/small-M, where it is a measured win, while
                        // prefill keeps the dense R4 fd to avoid the current
                        // prefill regression. Set -1 to force structured FWHT
                        // for all M in diagnostic runs.
                        const int r4_structured_fwht_small_m_max =
                            // LPBQ deploy-v1 2026-06-26 all-M FWHT A/B:
                            // m=32 real-layer and LLM smoke passed, but P64/D16
                            // prefill regressed to 132.2089 tok/s while decode
                            // only nudged to 10.3636. Keep the FWHT selector on
                            // decode/small-M by default; the all-M probe remains
                            // reproducible through the env override below.
                            // Rejected all-M default:
                            // env_int_or_default("LLAMA_NPU_LPBQ_R4_STRUCTURED_FWHT_SMALL_M_MAX", -1);
                            // Use the range parser here because the legacy
                            // positive-only env_int_or_default intentionally
                            // rejects -1, while this diagnostic selector uses
                            // -1 to mean force all M.
                            env_int_range_or_default("LLAMA_NPU_LPBQ_R4_STRUCTURED_FWHT_SMALL_M_MAX", 4, -1, 1000000);
                        const bool use_r4_structured_fwht =
                            sidecar->r4_structured_fwht &&
                            (r4_structured_fwht_small_m_max < 0 || m <= r4_structured_fwht_small_m_max);
                        const LpbqSidecarBuffer & request_r4 =
                            use_r4_structured_fwht ? sidecar->r4_structured_fwht_buffer : sidecar->r4;
                        const bool use_r4_hmx_dense_fp16_sidecar =
                            sidecar->has_r4_hmx_dense_fp16 && !use_r4_structured_fwht && m >= 32;
                        request_mode_flags = mode_flags | (sidecar->has_r4 ? LLM_NPU_MODE_LPBQ_R4 : 0) |
                                             (sidecar->r4_input_scale_folded ?
                                                  LLM_NPU_MODE_LPBQ_R4_FOLDED_INPUT_SCALE : 0) |
                                             (request_packed_k_major ?
                                                  LLM_NPU_MODE_LPBQ_PACKED_K_MAJOR : 0) |
                                               (request_packed_v6_full ?
                                                    LLM_NPU_MODE_LPBQ_PACKED_V6_FULL : 0) |
                                               // LPBQ deploy-v1 bottleneck reset: forcing every R4 compact
                                               // layer into the full-U8-safe DSP path made LLM smoke emit
                                               // repeated wrong tokens.  Keep that experiment available via
                                               // LLAMA_NPU_LPBQ_FORCE_R4_FULL_U8_SAFE, but default the real
                                               // LLM path back to the mature compact skeleton unless a layer
                                               // actually loaded a full-V6 sidecar.
                                               // Old experiment:
                                               // ((sidecar->packed_weight_v6_full || sidecar->has_r4) ?
                                               (use_r4_full_u8_safe ?
                                                    LLM_NPU_MODE_LPBQ_R4_FULL_U8_SAFE : 0) |
                                              (request_r4_v6_scale_1_16 ?
                                                   LLM_NPU_MODE_LPBQ_R4_V6_SCALE_1_16 : 0) |
                                              (request_exact_non_r4 ?
                                                   LLM_NPU_MODE_LPBQ_EXACT_NON_R4 : 0) |
                                              (use_r4_structured_fwht ?
                                                   LLM_NPU_MODE_LPBQ_R4_STRUCTURED_FWHT : 0) |
                                              (use_r4_hmx_dense_fp16_sidecar ?
                                                   LLM_NPU_MODE_LPBQ_R4_HMX_DENSE_FP16_SIDECAR : 0);

                        const bool send_sum_w_online =
                            sidecar->has_packed_weight && !sidecar->has_folded_dequant;
                        LpbqA8W8MatMulParams lpbq_params{
                            .output     = { output_fd,     (int32_t) output_offset     },
                            .activation = { activation_fd, (int32_t) activation_offset },
                            .weight     = { weight_fd,     (int32_t) weight_offset     },
                            .packed_weight = { sidecar->has_packed_weight ? request_packed_weight.fd : -1, 0 },
                            // LPBQ deploy-v1 folded-dequant path: sum_w is used
                            // while loading the sidecar to build bias_eff, then
                            // intentionally omitted from the online RPC request.
                            .sum_w      = { send_sum_w_online ? sidecar->sum_w.fd : -1, 0 },
                            .k32_safe   = { sidecar->has_k32_safe ? sidecar->k32_safe.fd : -1, 0 },
                            .k64_safe   = { sidecar->has_k64_safe ? sidecar->k64_safe.fd : -1, 0 },
                            .scale2     = { sidecar->scale2.fd, 0 },
                            .bias       = { sidecar->has_bias ? sidecar->bias.fd : -1, 0 },
                            .r4         = { sidecar->has_r4 ? request_r4.fd : -1, 0 },
                            .r4_hmx_dense_fp16 = {
                                use_r4_hmx_dense_fp16_sidecar ? sidecar->r4_hmx_dense_fp16.fd : -1, 0
                            },
                            .input_scale = { (sidecar->has_input_scale && !sidecar->r4_input_scale_folded) ?
                                                 sidecar->input_scale.fd : -1, 0 },
                            .out_scale  = { sidecar->has_folded_dequant ? sidecar->out_scale.fd : -1,
                                            sidecar->has_folded_dequant ?
                                                static_cast<int32_t>(sidecar->out_scale.offset) : 0 },
                            .bias_eff   = { sidecar->has_folded_dequant ? sidecar->bias_eff.fd : -1,
                                            sidecar->has_folded_dequant ?
                                                static_cast<int32_t>(sidecar->bias_eff.offset) : 0 },
                            .act_scale  = sidecar->act_scale,
                            .r4_block   = sidecar->has_r4 ? sidecar->r4_block : 0,
                            .m          = m,
                            .k          = k,
                            .n          = n,
                            .trace_id   = trace_id,
                            .mode_flags = request_mode_flags,
                            .max_profile_events = trace_profile_fd >= 0 ? trace_profile_max_events : 0,
                            .profile    = active_trace_profile_addr,
                        };
                        *reinterpret_cast<LpbqA8W8MatMulParams *>(param_buf) = lpbq_params;
                        args_size = sizeof(LpbqA8W8MatMulParams);
                        op_index = HTP_OPS_MAT_MUL_LPBQ_A8W8;
                    } else {
                        op_index = is_w8pc_a8pt_mode(npu_mode) ? HTP_OPS_MAT_MUL_PERMUTED_W8PC_A8PT :
                                                                 HTP_OPS_MAT_MUL_PERMUTED_W8D16A32;
                    }
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
    if (trace_on && op_index == HTP_OPS_MAT_MUL_LPBQ_A8W8) {
        fprintf(stderr,
                "LPBQ_HOST_ISSUE trace_id=%lld mode=%s flags=%d phase=%s category=%s tensor=%s weight=%s "
                "m=%d k=%d n=%d trace=%d detailed=%d\n",
                (long long) trace_id, npu_mode, request_mode_flags, phase, category, dst->name, weight_name,
                m, k, n, trace_on ? 1 : 0, detailed_trace_on ? 1 : 0);
        fflush(stderr);
    }
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
                (long long) trace_id, npu_mode, request_mode_flags, phase,
                htp_op_trace_name(op_index, request_mode_flags), op_index, category,
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
