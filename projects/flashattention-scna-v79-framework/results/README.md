# 性能结果目录

`results/local/` 由 `scripts/run_figure8_baselines.sh` 在本机采集，已被 Git 忽略。

标准目录结构如下：

```text
results/local/v79/
├── baseline/
│   ├── raw_q*.log
│   ├── attention_timers.csv
│   └── attention_timers_summary.json
└── lut_exp/
    ├── raw_q*.log
    ├── attention_timers.csv
    ├── attention_timers_summary.json
    └── baseline_vs_lut_exp.{csv,md,json}
```

只比较相同设备、相同 DSP 架构、同一 shape 与同一采集会话的结果。`baseline` 和 `lut_exp` 是 runtime mode，不应通过不同编译产物或跨 v79/v81 数据混合比较。
