# FlashAttention Trace Semantics Audit 2026-05-21

本次审计目标是确认 `artifacts/flashattention/` 是否也出现了 LLM profiling 里曾经的 naive/software fallback 问题，以及 trace lane 是否错误。

## 结论

- `artifacts/flashattention/` 没有走 LLM server 的 `GGML_OP_FLASH_ATTN_EXT` naive fallback。它是 standalone Figure 8 path：`HTP_OPS_FLASH_ATTN_PROFILE_QO_F32_KV_F16 -> simple_flash_attn_profiled(...)`。
- 原始日志包含 `FIG8_ATTENTION_EVENT`，所以 trace 的时间边界来自 DSP qtimer event，不是估计值。
- 旧生成器把 `q_load/k_load/v_load/o_store` 放到 `HVX Unit`、把 `core_acc` 放到 `HMX Unit`，这个 lane mapping 已修正并重新生成全部四组 V73/V79 baseline/LUT-exp trace。
- 旧的 thread-only / before-event trace archive 已删除，避免误用旧结果。

## 当前 Lane Mapping

| Lane | Component |
|---|---|
| `Memory/L2 Load-Pack` | `q_load`, `k_load`, `v_load` |
| `HMX Unit` | `qk_dot` |
| `HVX Unit` | `safe_sm` |
| `Mixed HVX/HMX Section` | `core_acc`, `o_scale` |
| `Store/Writeback` | `o_store` |

`core_acc/o_scale` 暂时保留为 mixed，因为 standalone Figure 8 event 还没有把 HVX state/setup 和 HMX dot 子阶段拆成更细的 DSP event。
