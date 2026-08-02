# V79 LUT-exp Attention Kernel

这个目录是 V79 LUT-exp standalone FlashAttention prefill kernel 的 Figure 8 结果。LUT-exp 只替换 safe-softmax 里部分 exp 路径，其他 tile layout 与 baseline 保持同一套 HMX-aware 结构。

## 文件说明

| 文件/目录 | 功能 |
|---|---|
| `attention_timers_raw.log` | 合并后的设备端原始输出，包含 `FIG8_ATTENTION_TIMERS`、`FIG8_ATTENTION_EVENT_COUNT` 和 `FIG8_ATTENTION_EVENT`。 |
| `attention_timers.csv` | 解析后的 iteration-level component timer。 |
| `attention_timers_summary.json` | 按 shape 聚合后的 summary。 |
| `figure8_lut_exp_percentages.md` | LUT-exp component 时间占比。 |
| `baseline_vs_lut_exp.csv` | 与 V79 baseline 的逐 component 差异表。 |
| `baseline_vs_lut_exp.md` | 与 V79 baseline 的人类可读差异总结。 |
| `baseline_vs_lut_exp_summary.json` | 对比 summary 的结构化版本。 |
| `raw_q4.log`, `raw_q8.log`, `raw_q16.log`, `raw_q32.log` | 不同 `qo_len` / batch 配置的原始日志切片。 |
| `ntff/` | 当前可用的 Perfetto/NTFF trace。 |
| `sanity_q4.log` | V79 LUT-exp q4 sanity run 日志。 |

## Trace 语义

本目录 trace 来自 `simple_flash_attn_profiled(...)` 的 DSP qtimer event，不是 `naive_flash_attn_profiled(...)`。当前 lane mapping 为：

```text
q_load/k_load/v_load -> Memory/L2 Load-Pack
qk_dot               -> HMX Unit
safe_sm              -> HVX Unit
core_acc/o_scale     -> Mixed HVX/HMX Section
o_store              -> Store/Writeback
```

## Reparse

```powershell
python tools\parse_figure8_attention_timers.py --input artifacts\flashattention\v79\lut_exp\attention_timers_raw.log --out-dir artifacts\flashattention\v79\lut_exp
python tools\generate_figure8_perfetto_trace.py --input-dir artifacts\flashattention\v79\lut_exp --out-dir artifacts\flashattention\v79\lut_exp\ntff
python tools\compare_figure8_lut_exp.py --baseline artifacts\flashattention\v79\baseline\attention_timers_summary.json --lut artifacts\flashattention\v79\lut_exp\attention_timers_summary.json --out-dir artifacts\flashattention\v79\lut_exp
```
