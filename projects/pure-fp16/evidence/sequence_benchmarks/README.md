# Pure-FP16 sequence benchmarks

该目录保留 pure-FP16 的 prefill/decode shape、batch/ubatch 调优和少量历史结果。历史数据只说明当时设备、SDK、模型和 build flags 下的测量，不自动代表当前 clone。

代表 case：

```text
prefill_64_decode_16
prefill_256_decode_16
prefill_768_decode_16
prefill_256_decode_64
long_p1024_d16
long_p1024_d128
long_p1536_d64
long_p1792_d256
```

速度复现：

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

已存在 raw CSV 时可重建 summary：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\rebuild_pd_sweep_summary.ps1 `
  -OutRoot .\evidence\sequence_benchmarks\<result-directory>
```

推荐记录：SoC、DSP arch、SDK、模型 SHA-256、commit、build flags、server 参数、warmup、repeat 数和 no-trace/trace 状态。
