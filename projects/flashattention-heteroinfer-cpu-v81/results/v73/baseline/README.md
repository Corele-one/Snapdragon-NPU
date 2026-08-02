# V73 Baseline Attention Kernel

这个目录是 V73 baseline standalone FlashAttention prefill kernel 的 Figure 8 结果。baseline 使用 HMX-aware tile layout 和原始 exp safe-softmax。

## 文件说明

| 文件/目录 | 功能 |
|---|---|
| `attention_timers_raw.log` | 合并后的设备端原始输出，包含 `FIG8_ATTENTION_TIMERS`、`FIG8_ATTENTION_EVENT_COUNT` 和 `FIG8_ATTENTION_EVENT`。 |
| `attention_timers.csv` | 解析后的 iteration-level component timer。 |
| `attention_timers_summary.json` | 按 shape 聚合后的 summary。 |
| `figure8_baseline_percentages.md` | component 时间占比。 |
| `raw_q4.log`, `raw_q8.log`, `raw_q16.log`, `raw_q32.log` | 不同 `qo_len` / batch 配置的原始日志切片。 |
| `event_level_results/` | 事件级插桩后生成的一份解析结果副本。 |
| `ntff/` | 当前可用的 Perfetto/NTFF trace。 |

## Trace 语义

本目录 trace 来自 `simple_flash_attn_profiled(...)` 的 DSP qtimer event，不是 `naive_flash_attn_profiled(...)`。当前 lane mapping 为：

```text
q_load/k_load/v_load -> Memory/L2 Load-Pack
qk_dot               -> HMX Unit
safe_sm              -> HVX Unit
core_acc/o_scale     -> Mixed HVX/HMX Section
o_store              -> Store/Writeback
```

`core_acc/o_scale` 仍标为 mixed，因为 Figure 8 standalone event 尚未把其中的 HVX state/setup 和 HMX dot 子阶段拆成独立 event。

## Reparse

```powershell
python tools\parse_figure8_attention_timers.py --input artifacts\flashattention\v73\baseline\attention_timers_raw.log --out-dir artifacts\flashattention\v73\baseline
python tools\generate_figure8_perfetto_trace.py --input-dir artifacts\flashattention\v73\baseline --out-dir artifacts\flashattention\v73\baseline\ntff
```
