# Pure-FP16 profiling traces

该目录保留 pure-FP16 路径的最小 trace 索引和历史摘要，用于判断实际 operator/stage、是否进入 dequant，以及 matmul 内部时间构成。

常见产物：

| 文件 | 作用 |
|---|---|
| `llm_trace_events.csv` | operator event 顺序和耗时 |
| `llm_trace_stage_events.csv` | `weight_load`、`hmx_mma`、`weight_hvx_dequant` 等 stage |
| `llm_trace_summary.json` | 结构化汇总 |
| `llm_trace_breakdown.md` | 人可读 breakdown |
| `traces/*.ntff` | Snapdragon/HTP profiler trace |
| `traces/*.perfetto.json` | Perfetto trace |

历史大型 trace 未保留。最小索引位于 `final_direct_noinv_8case_20260527/`。

重新生成单个 case：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_trace_pure_fp16.ps1 `
  -Serial <device-serial> `
  -CaseName prefill_64_decode_16 `
  -TargetPrefill 64 `
  -Decode 16 `
  -Batch 512 `
  -Ubatch 512 `
  -Threads 4
```

验收时检查：

- 出现 `matmul_f16` 或 `matmul_w16a32`；
- `matmul_weight_dequant_pct = 0`；
- 不出现 `weight_hvx_dequant`；
- 运行 mode 明确为 `pure_fp16`。

trace 插桩会影响耗时，不能用 trace wall time 替代 no-trace 性能。
