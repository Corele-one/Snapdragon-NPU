# FlashAttention v79 results

该目录保存同一 v79 skel 的两个 runtime mode：

- `baseline/`：原始 safe-softmax exp 路径；
- `lut_exp/`：LUT/VTCM gather exp 路径。

`v73_vs_v79.csv`、`v73_vs_v79.md` 和 `v73_vs_v79_summary.json` 是随快照保留的静态历史对比；当前工具目录没有独立的 v73/v79 comparator。

两种 mode 无需分别编译。运行时使用 `--mode baseline` 与 `--mode lut-exp`，并保持 shape、warmup 和 iteration 数一致。重建命令见 `projects/flashattention/README.md`。
