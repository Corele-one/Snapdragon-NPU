static inline size_t hmx_permuted_weight_index(int ki, int nj, int k) {
    const int i0 = ki / 32, i1 = ki % 32;
    const int j0 = nj / 32, j1 = nj % 32;
    return (static_cast<size_t>(j0) * (k / 32) + i0) * 1024 +
           (i1 & ~1) * 32 + j1 * 2 + (i1 & 1);
}

static float hmx_model_weight_value(const ggml_tensor * weight, size_t index) {
    if (weight->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(weight->data)[index]);
    }
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(weight->data);
    const size_t sb = index / 256;
    const int x = index % 256;
    if (weight->type == GGML_TYPE_Q8_0) {
        const uint8_t * block = bytes + sb * 272;
        const float scale = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(block + 2 * (x / 32)));
        return scale * static_cast<float>(reinterpret_cast<const int8_t *>(block + 16)[x]);
    }
    if (weight->type == GGML_TYPE_IQ4_NL) {
        static const int8_t values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                           1,   13,  25,  38,  53,  69,  89, 113};
        const uint8_t * block = bytes + sb * 144;
        const float scale = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(block + 2 * (x / 32)));
        const uint8_t * q = block + 16;
        uint8_t code;
        if (x < 64) code = q[2 * x] & 15;
        else if (x < 128) code = q[2 * (x - 64) + 1] & 15;
        else if (x < 192) code = q[2 * (x - 128)] >> 4;
        else code = q[2 * (x - 192) + 1] >> 4;
        return scale * values[code];
    }
    return NAN;
}

static void audit_model_matmul(const ggml_tensor * dst, const ggml_tensor * weight,
                               const ggml_tensor * activation, int m, int k, int n) {
    const char * selector = std::getenv("LLAMA_NPU_MATMUL_AUDIT_TENSOR");
    if (!selector || !selector[0] || (std::strcmp(selector, "*") != 0 && !std::strstr(weight->name, selector))) {
        return;
    }
    if (weight->type != GGML_TYPE_F16 && weight->type != GGML_TYPE_IQ4_NL && weight->type != GGML_TYPE_Q8_0) {
        std::fprintf(stderr, "MODEL_MATMUL_AUDIT_JSON {\"tensor\":\"%s\",\"supported\":false}\n", weight->name);
        return;
    }
    const float * a = reinterpret_cast<const float *>(activation->data);
    const float * got = reinterpret_cast<const float *>(dst->data);
    double sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    const int row_candidates[] = {0, 1, 31, 32, m / 2, m - 1};
    const int col_candidates[] = {0, 15, 31, 32, 63, 64, n / 2, n - 33, n - 1};
    int sample_count = 0;
    for (size_t ri = 0; ri < sizeof(row_candidates) / sizeof(row_candidates[0]); ++ri) {
        const int r = row_candidates[ri];
        if (r < 0 || r >= m || (ri > 0 && r == row_candidates[ri - 1])) continue;
        for (size_t ji = 0; ji < sizeof(col_candidates) / sizeof(col_candidates[0]); ++ji) {
            const int j = col_candidates[ji];
            if (j < 0 || j >= n || (ji > 0 && j == col_candidates[ji - 1])) continue;
            double ref = 0.0;
            for (int ki = 0; ki < k; ++ki) {
                ref += static_cast<double>(a[static_cast<size_t>(r) * k + ki]) *
                       hmx_model_weight_value(weight, hmx_permuted_weight_index(ki, j, k));
            }
            const float value = got[static_cast<size_t>(r) * n + j];
            if (!std::isfinite(value)) ++nonfinite;
            const double delta = std::isfinite(value) ? value - ref : 0.0;
            sq += delta * delta;
            ref_sq += ref * ref;
            max_abs = std::max(max_abs, static_cast<float>(std::fabs(delta)));
            ++sample_count;
        }
    }
    const double relative_l2 = std::sqrt(sq / std::max(ref_sq, 1e-30));
    const bool pass = nonfinite == 0 && relative_l2 <= 0.01;
    if (!pass) g_matmul_model_audit_divergences.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr,
        "MODEL_MATMUL_AUDIT_JSON {\"tensor\":\"%s\",\"output\":\"%s\",\"dtype\":\"%s\","
        "\"m\":%d,\"k\":%d,\"n\":%d,\"sample_count\":%d,"
        "\"relative_l2\":%.9g,\"max_abs\":%.9g,\"nonfinite_count\":%d,\"pass\":%s,"
        "\"divergence_ordinal\":%d}\n",
        weight->name, dst->name, ggml_type_name(weight->type), m, k, n, sample_count,
        relative_l2, max_abs, nonfinite, pass ? "true" : "false",
        g_matmul_model_audit_divergences.load(std::memory_order_relaxed));
}
