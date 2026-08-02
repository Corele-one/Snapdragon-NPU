# V79 Baseline NTFF

这个目录保存 V79 baseline attention kernel 的 Perfetto/NTFF trace。当前文件由 `FIG8_ATTENTION_EVENT` DSP qtimer start/end rows 生成。

## 文件说明

| 文件 | 功能 |
|---|---|
| `baseline_all_q.ntff` | q4/q8/q16/q32 合并后的 NTFF trace，推荐优先在 Perfetto UI 打开。 |
| `baseline_all_q.perfetto.json` | 合并 trace 的 JSON 版本。 |
| `baseline_q4.ntff`, `baseline_q8.ntff`, `baseline_q16.ntff`, `baseline_q32.ntff` | 单个 `qo_len` 配置的 NTFF trace。 |
| `baseline_q4.perfetto.json`, `baseline_q8.perfetto.json`, `baseline_q16.perfetto.json`, `baseline_q32.perfetto.json` | 单个配置的 Perfetto JSON。 |
| `event_trace_summary.json` | 事件数量、持续时间、overflow 等 trace summary。 |
| `derived_trace_notes.md` | trace 生成方式、lane mapping 和限制说明。 |

## Lane Mapping

| Lane | Component |
|---|---|
| `Memory/L2 Load-Pack` | `q_load`, `k_load`, `v_load` |
| `HMX Unit` | `qk_dot` |
| `HVX Unit` | `safe_sm` |
| `Mixed HVX/HMX Section` | `core_acc`, `o_scale` |
| `Store/Writeback` | `o_store` |

## Open In Perfetto

打开 `https://ui.perfetto.dev`，拖入 `baseline_all_q.ntff`。
