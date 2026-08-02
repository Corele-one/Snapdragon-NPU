# FlashAttention v73 results

该目录保存同一 v73 skel 的两个 runtime mode：

- `baseline/`：原始 safe-softmax exp 路径；
- `lut_exp/`：LUT/VTCM gather exp 路径。

两者无需分别编译源码。使用 `--mode baseline` 与 `--mode lut-exp` 切换，并保持 `qo_len`、`kv_len`、head shape、warmup 和 iteration 数一致。

推荐阅读：

1. `baseline/figure8_baseline_percentages.md`
2. `lut_exp/figure8_lut_exp_percentages.md`
3. `lut_exp/baseline_vs_lut_exp.md`
4. 对应目录的 `attention_timers_summary.json`

重建命令见 `projects/flashattention/README.md`。
