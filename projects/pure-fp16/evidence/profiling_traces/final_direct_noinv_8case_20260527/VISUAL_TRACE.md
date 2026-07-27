# Pure-FP16 visual trace entry

历史代表 case 为 `prefill_64_decode_16`。原始大型 `.perfetto.json` 和 `.ntff` 未进入仓库；重新生成：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_trace_pure_fp16.ps1 `
  -Serial <device-serial> `
  -Port 8080 `
  -CaseName manual_prefill64_visual `
  -TargetPrefill 64 `
  -Decode 16 `
  -Batch 512 `
  -Ubatch 512 `
  -Threads 4
```

输出位于：

```text
profiling_traces/manual_prefill64_visual_<timestamp>/
```

验证：

```powershell
Select-String `
  -Path .\profiling_traces\manual_prefill64_visual_*\llm_trace_stage_events.csv `
  -Pattern 'weight_hvx_dequant'
```

pure-FP16 路径不应命中 `weight_hvx_dequant`，并应包含 `matmul_f16` 或 `matmul_w16a32`。
