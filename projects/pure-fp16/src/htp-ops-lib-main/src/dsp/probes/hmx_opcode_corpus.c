#include <stdint.h>

// LPBQ deploy-v1 Track A diagnostic only.  These functions intentionally use
// HMX spellings that already exist elsewhere in this artifact-local source tree
// so a save-temps/disassembly pass can map the accepted assembler surface
// without guessing new proprietary syntax.

__attribute__((noinline, used))
void hmx_corpus_fp16_load_cvt_store(const void *a, const void *w, void *out,
                                    uint32_t limit, uint32_t spatial) {
  asm volatile(
    "{ activation.hf = mxmem(%0, %1):deep\n"
    "  weight.hf = mxmem(%2, %3) }\n"
    "cvt.hf = acc(%4)\n"
    "mxmem(%5, %6) = cvt\n"
    :: "r"(a), "r"(limit), "r"(w), "r"(limit), "r"(2), "r"(out), "r"(spatial)
    : "memory");
}

__attribute__((noinline, used))
void hmx_corpus_byte_mma_after_uh(const void *a, const void *w, void *out,
                                  uint32_t a_limit, uint32_t w_limit) {
  asm volatile(
    "mxclracc\n"
    "{ activation.ub = mxmem(%0, %1):above:cm\n"
    "  weight.b = mxmem(%2, %3) }\n"
    "mxmem(%4, %5):after.uh = acc:2x1\n"
    :: "r"(a), "r"(a_limit), "r"(w), "r"(w_limit), "r"(out), "r"(0)
    : "memory");
}

__attribute__((noinline, used))
void hmx_corpus_byte_mma_deep_after_uh(const void *a, const void *w, void *out,
                                       uint32_t a_limit, uint32_t w_limit) {
  asm volatile(
    "mxclracc\n"
    "{ activation.ub = mxmem(%0, %1):deep\n"
    "  weight.b = mxmem(%2, %3):deep }\n"
    "mxmem(%4, %5):after.uh = acc:2x1\n"
    :: "r"(a), "r"(a_limit), "r"(w), "r"(w_limit), "r"(out), "r"(0)
    : "memory");
}

__attribute__((noinline, used))
void hmx_corpus_cvt_uh_store(void *out, uint32_t selector, uint32_t spatial) {
  asm volatile(
    "cvt.uh = acc(%0):2x1\n"
    "mxmem(%1, %2) = cvt\n"
    :: "r"(selector), "r"(out), "r"(spatial)
    : "memory");
}

__attribute__((noinline, used))
void hmx_corpus_bias_and_acc_control(const void *bias_tile, void *bias_out) {
  asm volatile(
    "bias = mxmem2(%0)\n"
    "mxmem2(%1) = bias\n"
    "mxswapacc\n"
    "mxswapacc.hf\n"
    "mxclracc.hf\n"
    :: "r"(bias_tile), "r"(bias_out)
    : "memory");
}
