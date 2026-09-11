#!/usr/bin/env python3
"""Apply the minimal Stage2.75 ABI/mode bridge to an isolated llama.cpp copy."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


MODE_FUNCTION = r'''int get_mode_flags(const char * mode) {
    int flags = LLM_NPU_MODE_Q_TASK_ROWS_AUTO;
    if (std::strcmp(mode, "baseline") == 0) {
        // Native Stage2.75 evaluator, retained only as an integration control.
    } else if (std::strcmp(mode, "lut_exp") == 0 || std::strcmp(mode, "lut-exp") == 0) {
        flags |= LLM_NPU_MODE_LUT_EXP;
    } else if (std::strcmp(mode, "scna_fp16") == 0 || std::strcmp(mode, "scna-fp16") == 0) {
        flags |= LLM_NPU_MODE_SCNA_FP16 | LLM_NPU_MODE_SCNA_D8;
        flags |= SCNA_VARIANT_PAIR_STATIC_D8 << 10;
    } else {
        std::fprintf(stderr, "STAGE29_INVALID_MODE mode=%s\n", mode);
        std::abort();
    }
    if (env_truthy("LLAMA_NPU_NUMERIC_DEBUG")) {
        flags |= LLM_NPU_MODE_NUMERIC_DEBUG;
    }
    if (env_truthy("LLAMA_NPU_TRACE")) {
        flags |= LLM_NPU_MODE_TRACE;
    }
    if (env_truthy("LLAMA_NPU_DETAILED_TRACE")) {
        flags |= LLM_NPU_MODE_TRACE | LLM_NPU_MODE_DETAILED_TRACE;
    }
    static std::atomic<bool> reported{false};
    if (!reported.exchange(true)) {
        std::fprintf(stderr,
                     "STAGE29_BACKEND_CONFIRM backend=my-htp mode=%s flags=%d "
                     "scheduler=stage2_75_adaptive kernel=d7_pairret_noinline softmax=fused_state_update\n",
                     mode, flags);
    }
    return flags;
}

'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--audit-op-reg", type=Path, required=True)
    args = parser.parse_args()
    backend = args.tree / "ggml/src/ggml-htp"
    target = backend / "htp-ops.cc"
    text = target.read_text()
    pattern = re.compile(r"int get_mode_flags\(const char \* mode\) \{.*?\n\}\n\n(?=int env_int_or_default)", re.S)
    patched, count = pattern.subn(lambda _: MODE_FUNCTION, text)
    if count != 1:
        raise SystemExit(f"expected one get_mode_flags block, found {count}")
    if "#include <cstdio>" not in patched:
        patched = patched.replace("#include <cstdlib>\n", "#include <cstdlib>\n#include <cstdio>\n")
    if "MODEL_MATMUL_AUDIT_JSON" not in patched:
        if "#include <algorithm>" not in patched:
            patched = patched.replace("#include <atomic>\n", "#include <atomic>\n#include <algorithm>\n")
        global_anchor = "std::atomic<int> g_flash_host_debug_count{0};\n"
        if patched.count(global_anchor) != 1:
            raise SystemExit("model audit global insertion anchor is not unique")
        patched = patched.replace(global_anchor, global_anchor +
                                  "std::atomic<int> g_matmul_model_audit_divergences{0};\n")
        audit_block = (Path(__file__).resolve().parents[1] /
                       "templates/model_matmul_audit.inc").read_text()
        mode_anchor = "int get_mode_flags(const char * mode) {\n"
        if patched.count(mode_anchor) != 1:
            raise SystemExit("model audit function insertion anchor is not unique")
        patched = patched.replace(mode_anchor, audit_block + mode_anchor)
        call_anchor = ("    std::atomic_thread_fence(std::memory_order_acquire);\n"
                       "    const int ret = message_header_get_request_ptr(msg_hdr, 0)->state;\n\n")
        if patched.count(call_anchor) != 1:
            raise SystemExit("model audit call insertion anchor is not unique")
        patched = patched.replace(call_anchor, call_anchor +
                                  "    if (ret == 0 && dst->op == GGML_OP_MUL_MAT) {\n"
                                  "        audit_model_matmul(dst, dst->src[0], dst->src[1], m, k, n);\n"
                                  "    }\n\n")
    target.write_text(patched)
    shutil.copy2(args.audit_op_reg, backend / "op_reg.h")
    print(target)
    print(backend / "op_reg.h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
