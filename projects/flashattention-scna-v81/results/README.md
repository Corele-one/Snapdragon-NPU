# FlashAttention Figure 8 results

该目录保存 `htp_ops_test --figure8-attn` standalone profile op 的紧凑结果，不是完整 LLM server 的 `GGML_OP_FLASH_ATTN_EXT` 端到端结果。

| 路径 | 内容 |
|---|---|
| `v73/` | `DSP_ARCH=v73` 的 baseline 与 LUT-exp |
| `v79/` | `DSP_ARCH=v79` 的 baseline、LUT-exp 和静态 v73/v79 对比 |
| `trace_semantics_audit_20260521.*` | qtimer event 与 lane 语义审计 |

每个 mode 目录主要包含：

- `raw_q*.log`：设备端 qtimer 原始输出；
- `attention_timers.csv`：每次 measured iteration 的 component timer；
- `attention_timers_summary.json`：按 mode/shape 聚合的 summary；
- `figure8_*_percentages.md`：各 component 占比；
- `baseline_vs_lut_exp.*`：同一架构、同一 shape 的 runtime mode 对比。

这些 trace 来自 `HTP_OPS_FLASH_ATTN_PROFILE_QO_F32_KV_F16 -> simple_flash_attn_profiled(...)`。lane 语义：

| Lane | Component |
|---|---|
| `Memory/L2 Load-Pack` | `q_load`, `k_load`, `v_load` |
| `HMX Unit` | `qk_dot` |
| `HVX Unit` | `safe_sm` |
| `Mixed HVX/HMX Section` | `core_acc`, `o_scale` |
| `Store/Writeback` | `o_store` |

它们是 DSP qtimer 软件事件，不是 Qualcomm PMU hardware utilization。

完整构建、运行、解析与正确性验证命令见上一级 `README.md`。大型 `.ntff`/`.perfetto.json` 没有提交，可从 `raw_q*.log` 重新生成。
