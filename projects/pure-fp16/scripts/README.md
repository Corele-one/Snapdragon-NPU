# Pure-FP16 automation scripts

本目录包含 Windows PowerShell 自动化脚本。所有路径均相对于 `projects/pure-fp16/` 解析；模型通过 `-ModelPath`、`SNAPDRAGON_NPU_FP16_MODEL` 或本地 `models/` 提供。

| 脚本 | 功能 |
|---|---|
| `common.ps1` | ADB、模型定位、JSON、HTTP 请求和日志采集公共函数 |
| `build_pure_fp16.ps1` | 构建 HTP ops 与 llama.cpp HTP backend，并生成 `deploy/` |
| `switch_llm_inference_route.ps1` | 部署并切换 pure-FP16、IQ4 或 LPBQ runtime |
| `run_smoke_pure_fp16.ps1` | 端到端内容 smoke |
| `run_trace_pure_fp16.ps1` | 单 case detailed trace |
| `run_pd_sweep_pure_fp16.ps1` | prefill/decode sweep |
| `tune_batch_pure_fp16.ps1` | batch/ubatch/thread 搜索 |
| `rebuild_pd_sweep_summary.ps1` | 从已有 CSV/trace 重建 summary，不访问设备 |

最小构建：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_pure_fp16.ps1 `
  -DspArch v73 `
  -Jobs 8 `
  -ModelPath C:\path\to\model.f16-hmx.gguf
```

部署：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\switch_llm_inference_route.ps1 `
  -Mode pure_fp16 `
  -ModelPath C:\path\to\model.f16-hmx.gguf `
  -Serial <device-serial>
```

Smoke：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_smoke_pure_fp16.ps1 `
  -Mode pure_fp16 `
  -Serial <device-serial>
```

速度 sweep：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_pd_sweep_pure_fp16.ps1 `
  -Serial <device-serial> `
  -Batch 512 `
  -Ubatch 512 `
  -Threads 4 `
  -Repeats 3 `
  -SpeedOnly
```

单 case trace：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_trace_pure_fp16.ps1 `
  -Serial <device-serial> `
  -CaseName prefill_512_decode_16 `
  -TargetPrefill 512 `
  -Decode 16
```

脚本不重启设备，也不执行 `adb kill-server`。详细环境、模型转换、输出目录和验证口径见上一级 `README.md`。
