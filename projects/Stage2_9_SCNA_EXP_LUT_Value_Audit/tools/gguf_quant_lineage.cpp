#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

struct model_file {
    int fd = -1;
    gguf_context * gguf = nullptr;
    ggml_context * tensors = nullptr;
};

static bool ends_with(const std::string & value, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return value.size() >= n && value.compare(value.size() - n, n, suffix) == 0;
}

static bool is_hmx_matmul(const std::string & name) {
    static const char * suffixes[] = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    };
    for (const char * suffix : suffixes) {
        if (ends_with(name, suffix)) return true;
    }
    return false;
}

static bool open_model(const char * path, model_file & model) {
    gguf_init_params params = {/*no_alloc=*/true, /*ctx=*/&model.tensors};
    model.gguf = gguf_init_from_file(path, params);
    model.fd = open(path, O_RDONLY | O_CLOEXEC);
    return model.gguf != nullptr && model.tensors != nullptr && model.fd >= 0;
}

static void close_model(model_file & model) {
    if (model.fd >= 0) close(model.fd);
    if (model.gguf != nullptr) gguf_free(model.gguf);
    if (model.tensors != nullptr) ggml_free(model.tensors);
}

static bool read_exact(int fd, void * data, size_t size, off_t offset) {
    uint8_t * cursor = static_cast<uint8_t *>(data);
    while (size != 0) {
        const ssize_t n = pread(fd, cursor, size, offset);
        if (n <= 0) return false;
        cursor += n;
        offset += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

static std::vector<uint8_t> tensor_bytes(const model_file & model, int index) {
    const char * name = gguf_get_tensor_name(model.gguf, index);
    const ggml_tensor * tensor = ggml_get_tensor(model.tensors, name);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    const off_t offset = static_cast<off_t>(gguf_get_data_offset(model.gguf) +
                                             gguf_get_tensor_offset(model.gguf, index));
    if (!read_exact(model.fd, bytes.data(), bytes.size(), offset)) return {};
    return bytes;
}

static std::vector<uint8_t> unpack_q8_hvx(const std::vector<uint8_t> & input) {
    if (input.size() % 272 != 0) return {};
    std::vector<uint8_t> output(input.size());
    for (size_t base = 0; base < input.size(); base += 272) {
        for (size_t block = 0; block < 8; ++block) {
            const size_t out = base + block * 34;
            output[out] = input[base + block * 2];
            output[out + 1] = input[base + block * 2 + 1];
            std::memcpy(output.data() + out + 2, input.data() + base + 16 + block * 32, 32);
        }
    }
    return output;
}

static std::vector<uint8_t> unpack_iq4_hvx(const std::vector<uint8_t> & input) {
    if (input.size() % 144 != 0) return {};
    std::vector<uint8_t> output(input.size());
    uint8_t values[256];
    for (size_t base = 0; base < input.size(); base += 144) {
        for (size_t j = 0; j < 64; ++j) {
            values[j] = input[base + 16 + j * 2] & 15;
            values[64 + j] = input[base + 16 + j * 2 + 1] & 15;
            values[128 + j] = input[base + 16 + j * 2] >> 4;
            values[192 + j] = input[base + 16 + j * 2 + 1] >> 4;
        }
        for (size_t block = 0; block < 8; ++block) {
            const size_t out = base + block * 18;
            output[out] = input[base + block * 2];
            output[out + 1] = input[base + block * 2 + 1];
            for (size_t j = 0; j < 16; ++j) {
                output[out + 2 + j] = values[block * 32 + j] |
                                      (values[block * 32 + 16 + j] << 4);
            }
        }
    }
    return output;
}

static std::vector<float> f16_reference(const model_file & model, int index) {
    const char * name = gguf_get_tensor_name(model.gguf, index);
    const ggml_tensor * tensor = ggml_get_tensor(model.tensors, name);
    const size_t count = ggml_nelements(tensor);
    std::vector<uint8_t> bytes = tensor_bytes(model, index);
    std::vector<float> output(count);
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(bytes.data()), output.data(), count);
    } else if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(output.data(), bytes.data(), count * sizeof(float));
    } else {
        output.clear();
    }
    return output;
}

static std::vector<float> dequantize(const model_file & model, int index, bool hmx_repacked) {
    const char * name = gguf_get_tensor_name(model.gguf, index);
    const ggml_tensor * tensor = ggml_get_tensor(model.tensors, name);
    std::vector<uint8_t> bytes = tensor_bytes(model, index);
    if (hmx_repacked && tensor->type == GGML_TYPE_Q8_0) bytes = unpack_q8_hvx(bytes);
    if (hmx_repacked && tensor->type == GGML_TYPE_IQ4_NL) bytes = unpack_iq4_hvx(bytes);
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    std::vector<float> output(ggml_nelements(tensor));
    if (bytes.empty() || traits == nullptr || traits->to_float == nullptr) return {};
    traits->to_float(bytes.data(), output.data(), output.size());
    return output;
}

static std::vector<float> hmx_inverse(const std::vector<float> & physical, int64_t n, int64_t k) {
    std::vector<float> logical(physical.size());
    for (int64_t nt = 0; nt < n / 32; ++nt) {
        for (int64_t kt = 0; kt < k / 32; ++kt) {
            const size_t tile = static_cast<size_t>(nt * (k / 32) + kt) * 1024;
            for (int64_t pair = 0; pair < 16; ++pair) {
                for (int64_t row = 0; row < 32; ++row) {
                    for (int64_t lane = 0; lane < 2; ++lane) {
                        const size_t dst = static_cast<size_t>(nt * 32 + row) * k + kt * 32 + pair * 2 + lane;
                        const size_t src = tile + static_cast<size_t>((pair * 32 + row) * 2 + lane);
                        logical[dst] = physical[src];
                    }
                }
            }
        }
    }
    return logical;
}

int main(int argc, char ** argv) {
    if (argc != 5 || (std::strcmp(argv[4], "cpu") != 0 && std::strcmp(argv[4], "hmx") != 0)) {
        std::fprintf(stderr, "usage: %s CPU_F16 BRANCH_F16 QUANT cpu|hmx\n", argv[0]);
        return 2;
    }
    const bool hmx = std::strcmp(argv[4], "hmx") == 0;
    model_file cpu_f16, branch_f16, quant;
    if (!open_model(argv[1], cpu_f16) || !open_model(argv[2], branch_f16) || !open_model(argv[3], quant)) {
        std::fprintf(stderr, "cannot open model triplet\n");
        return 3;
    }
    const int count = gguf_get_n_tensors(cpu_f16.gguf);
    bool all_pass = count == gguf_get_n_tensors(branch_f16.gguf) && count == gguf_get_n_tensors(quant.gguf);
    int failed = 0;
    for (int i = 0; i < count; ++i) {
        const std::string name = gguf_get_tensor_name(cpu_f16.gguf, i);
        const int branch_index = gguf_find_tensor(branch_f16.gguf, name.c_str());
        const int quant_index = gguf_find_tensor(quant.gguf, name.c_str());
        const ggml_tensor * ref_tensor = ggml_get_tensor(cpu_f16.tensors, name.c_str());
        const ggml_tensor * branch_tensor = branch_index >= 0 ? ggml_get_tensor(branch_f16.tensors, name.c_str()) : nullptr;
        const ggml_tensor * quant_tensor = quant_index >= 0 ? ggml_get_tensor(quant.tensors, name.c_str()) : nullptr;
        bool pass = branch_tensor != nullptr && quant_tensor != nullptr &&
                    ggml_nelements(ref_tensor) == ggml_nelements(branch_tensor) &&
                    ggml_nelements(ref_tensor) == ggml_nelements(quant_tensor);
        if (!pass) {
            std::printf("{\"record_type\":\"device_quant_tensor\",\"branch\":\"%s\",\"tensor\":\"%s\",\"pass\":false,\"error\":\"missing_or_shape_mismatch\"}\n", argv[4], name.c_str());
            ++failed;
            all_pass = false;
            continue;
        }
        if (quant_tensor->type == branch_tensor->type) {
            const std::vector<uint8_t> source = tensor_bytes(branch_f16, branch_index);
            const std::vector<uint8_t> actual = tensor_bytes(quant, quant_index);
            pass = !source.empty() && source == actual;
            std::printf("{\"record_type\":\"device_quant_tensor\",\"branch\":\"%s\",\"tensor\":\"%s\",\"quant_type\":\"%s\",\"bitwise_equal_to_branch_f16\":%s,\"pass\":%s}\n",
                        argv[4], name.c_str(), ggml_type_name(quant_tensor->type), pass ? "true" : "false", pass ? "true" : "false");
        } else {
            std::vector<float> reference = f16_reference(cpu_f16, i);
            std::vector<float> actual = dequantize(quant, quant_index, hmx && is_hmx_matmul(name));
            if (hmx && is_hmx_matmul(name) && actual.size() == reference.size()) {
                actual = hmx_inverse(actual, ref_tensor->ne[1], ref_tensor->ne[0]);
            }
            double error2 = 0.0, ref2 = 0.0, dot = 0.0, actual2 = 0.0, max_abs = 0.0;
            size_t nonfinite = 0;
            pass = !reference.empty() && actual.size() == reference.size();
            if (pass) {
                for (size_t j = 0; j < reference.size(); ++j) {
                    if (!std::isfinite(actual[j])) {
                        ++nonfinite;
                        continue;
                    }
                    const double delta = static_cast<double>(actual[j]) - reference[j];
                    error2 += delta * delta;
                    ref2 += static_cast<double>(reference[j]) * reference[j];
                    dot += static_cast<double>(reference[j]) * actual[j];
                    actual2 += static_cast<double>(actual[j]) * actual[j];
                    max_abs = std::max(max_abs, std::abs(delta));
                }
            }
            const double rmse = pass ? std::sqrt(error2 / reference.size()) : INFINITY;
            const double relative_l2 = pass ? std::sqrt(error2 / std::max(ref2, 1e-30)) : INFINITY;
            const double cosine = pass ? dot / std::sqrt(std::max(ref2 * actual2, 1e-30)) : -1.0;
            pass = pass && nonfinite == 0 && cosine >= 0.90;
            std::printf("{\"record_type\":\"device_quant_tensor\",\"branch\":\"%s\",\"tensor\":\"%s\",\"quant_type\":\"%s\",\"elements\":%zu,\"rmse\":%.12g,\"relative_l2\":%.12g,\"max_abs\":%.12g,\"cosine\":%.12g,\"nonfinite_count\":%zu,\"pass\":%s}\n",
                        argv[4], name.c_str(), ggml_type_name(quant_tensor->type), reference.size(), rmse,
                        relative_l2, max_abs, cosine, nonfinite, pass ? "true" : "false");
        }
        if (!pass) {
            ++failed;
            all_pass = false;
        }
    }
    std::printf("{\"record_type\":\"device_quant_lineage_summary\",\"branch\":\"%s\",\"pass\":%s,\"tensor_count\":%d,\"failed_count\":%d}\n",
                argv[4], all_pass ? "true" : "false", count, failed);
    close_model(cpu_f16);
    close_model(branch_f16);
    close_model(quant);
    return all_pass ? 0 : 4;
}
