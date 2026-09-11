#include "ggml.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

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
        if (ends_with(name, suffix)) {
            return true;
        }
    }
    return false;
}

static bool read_exact(int fd, void * data, size_t size, off_t offset) {
    uint8_t * cursor = static_cast<uint8_t *>(data);
    while (size != 0) {
        const ssize_t n = pread(fd, cursor, size, offset);
        if (n <= 0) {
            return false;
        }
        cursor += n;
        offset += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

static bool write_exact(int fd, const void * data, size_t size, off_t offset) {
    const uint8_t * cursor = static_cast<const uint8_t *>(data);
    while (size != 0) {
        const ssize_t n = pwrite(fd, cursor, size, offset);
        if (n <= 0) {
            return false;
        }
        cursor += n;
        offset += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s IN_PLACE_F16_GGUF\n", argv[0]);
        return 2;
    }

    ggml_context * tensor_ctx = nullptr;
    gguf_init_params params = {/*no_alloc=*/true, /*ctx=*/&tensor_ctx};
    gguf_context * gguf = gguf_init_from_file(argv[1], params);
    if (gguf == nullptr || tensor_ctx == nullptr) {
        std::fprintf(stderr, "cannot parse GGUF: %s\n", argv[1]);
        return 3;
    }
    const int fd = open(argv[1], O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "cannot open output: %s\n", std::strerror(errno));
        gguf_free(gguf);
        ggml_free(tensor_ctx);
        return 3;
    }

    const size_t data_offset = gguf_get_data_offset(gguf);
    int converted = 0;
    uint64_t converted_bytes = 0;
    const int tensor_count = gguf_get_n_tensors(gguf);
    for (int i = 0; i < tensor_count; ++i) {
        const std::string name = gguf_get_tensor_name(gguf, i);
        if (!is_hmx_matmul(name)) {
            continue;
        }
        if (gguf_get_tensor_type(gguf, i) != GGML_TYPE_F16) {
            std::fprintf(stderr, "target tensor is not F16: %s\n", name.c_str());
            close(fd);
            gguf_free(gguf);
            ggml_free(tensor_ctx);
            return 4;
        }
        const ggml_tensor * tensor = ggml_get_tensor(tensor_ctx, name.c_str());
        if (tensor == nullptr || ggml_n_dims(tensor) != 2) {
            std::fprintf(stderr, "target tensor is not rank 2: %s\n", name.c_str());
            close(fd);
            gguf_free(gguf);
            ggml_free(tensor_ctx);
            return 4;
        }
        const int64_t k = tensor->ne[0];
        const int64_t n = tensor->ne[1];
        if (k <= 0 || n <= 0 || k % 32 != 0 || n % 32 != 0) {
            std::fprintf(stderr, "target tensor is not 32-aligned: %s [%lld,%lld]\n",
                         name.c_str(), static_cast<long long>(n), static_cast<long long>(k));
            close(fd);
            gguf_free(gguf);
            ggml_free(tensor_ctx);
            return 4;
        }

        const size_t elements = static_cast<size_t>(n) * static_cast<size_t>(k);
        std::vector<uint16_t> logical(elements);
        std::vector<uint16_t> permuted(elements);
        const off_t offset = static_cast<off_t>(data_offset + gguf_get_tensor_offset(gguf, i));
        if (!read_exact(fd, logical.data(), elements * sizeof(uint16_t), offset)) {
            std::fprintf(stderr, "read failed: %s\n", name.c_str());
            close(fd);
            gguf_free(gguf);
            ggml_free(tensor_ctx);
            return 5;
        }

        for (int64_t nt = 0; nt < n / 32; ++nt) {
            for (int64_t kt = 0; kt < k / 32; ++kt) {
                const size_t tile = static_cast<size_t>(nt * (k / 32) + kt) * 1024;
                for (int64_t pair = 0; pair < 16; ++pair) {
                    for (int64_t row = 0; row < 32; ++row) {
                        for (int64_t lane = 0; lane < 2; ++lane) {
                            const size_t src = static_cast<size_t>(nt * 32 + row) * k + kt * 32 + pair * 2 + lane;
                            const size_t dst = tile + static_cast<size_t>((pair * 32 + row) * 2 + lane);
                            permuted[dst] = logical[src];
                        }
                    }
                }
            }
        }
        if (!write_exact(fd, permuted.data(), elements * sizeof(uint16_t), offset)) {
            std::fprintf(stderr, "write failed: %s\n", name.c_str());
            close(fd);
            gguf_free(gguf);
            ggml_free(tensor_ctx);
            return 5;
        }
        ++converted;
        converted_bytes += elements * sizeof(uint16_t);
        std::printf("{\"record_type\":\"device_hmx_permute_tensor\",\"tensor\":\"%s\",\"n\":%lld,\"k\":%lld,\"bytes\":%zu}\n",
                    name.c_str(), static_cast<long long>(n), static_cast<long long>(k), elements * sizeof(uint16_t));
    }
    if (fsync(fd) != 0) {
        std::fprintf(stderr, "fsync failed: %s\n", std::strerror(errno));
        close(fd);
        gguf_free(gguf);
        ggml_free(tensor_ctx);
        return 5;
    }
    close(fd);
    gguf_free(gguf);
    ggml_free(tensor_ctx);
    if (converted == 0) {
        std::fprintf(stderr, "no HMX matmul tensors found\n");
        return 4;
    }
    std::printf("{\"record_type\":\"device_hmx_permute_summary\",\"pass\":true,\"tensors\":%d,\"bytes\":%llu}\n",
                converted, static_cast<unsigned long long>(converted_bytes));
    return 0;
}
