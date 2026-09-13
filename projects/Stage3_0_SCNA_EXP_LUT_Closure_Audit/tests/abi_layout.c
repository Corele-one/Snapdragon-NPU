#include <stddef.h>
#include "op_reg.h"

_Static_assert(sizeof(struct RpcmemBufAddr) == 8, "RpcmemBufAddr ABI");
_Static_assert(sizeof(struct FlashAttnParams) == 84, "FlashAttnParams ABI");
_Static_assert(offsetof(struct FlashAttnParams, mode_flags) == 68, "FlashAttnParams mode offset");
_Static_assert(sizeof(struct NonlinearValueAuditParams) == 44, "Nonlinear params ABI");
_Static_assert(sizeof(struct NonlinearValueAuditResult) == 120, "Nonlinear result ABI");
_Static_assert(sizeof(struct MatMulValueAuditParams) == 64, "Matmul audit params ABI");
_Static_assert(sizeof(struct MatMulValueAuditResult) == 72, "Matmul audit result ABI");
_Static_assert(offsetof(struct MatMulValueAuditParams, dtype) == 44, "Matmul dtype ABI offset");
_Static_assert(offsetof(struct MatMulValueAuditParams, weight_bytes) == 56, "Matmul weight bytes ABI offset");
_Static_assert(offsetof(struct MatMulValueAuditResult, activation_checksum) == 48, "Matmul checksum ABI offset");
_Static_assert(sizeof(struct Figure8ProfileHeader) == 88, "Figure8 resource header ABI");
_Static_assert(offsetof(struct NonlinearValueAuditParams, evaluator) == 28, "evaluator ABI offset");
_Static_assert(offsetof(struct NonlinearValueAuditResult, elapsed_ticks) == 40, "result ticks ABI offset");
_Static_assert(offsetof(struct Figure8ProfileHeader, vtcm_total_bytes) == 48, "resource ABI offset");
_Static_assert(LLM_NPU_MODE_LUT_EXP == 1, "LUT mode bit");
_Static_assert(LLM_NPU_MODE_SCNA_FP16 == 4, "SCNA mode bit");
_Static_assert(LLM_NPU_MODE_SCNA_D8 == 16, "SCNA D8 mode bit");
_Static_assert(LLM_NPU_MODE_Q_TASK_ROWS_AUTO == (1 << 22), "scheduler mode bit");
_Static_assert(SCNA_VARIANT_PAIR_STATIC_D8 == 3, "frozen SCNA variant encoding");

int main(void) { return 0; }
