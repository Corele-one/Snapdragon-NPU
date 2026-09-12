# v79 三方实测结果

下表是 Host wall latency；baseline 与 LUT-exp 来自原版 `flashattention`，SCNA 来自最终优化系统。
因此它是系统级比较，不是只替换 exp evaluator 的单变量实验。

| Q | baseline (us) | LUT-exp (us) | SCNA (us) | LUT/base | SCNA/base |
|---:|---:|---:|---:|---:|---:|
| 4 | 561.5 | 548.0 | 498.0 | 0.9607 | 0.8768 |
| 8 | 658.0 | 549.0 | 526.5 | 0.8369 | 0.7951 |
| 16 | 877.0 | 659.0 | 657.0 | 0.7520 | 0.7292 |
| 32 | 1434.0 | 1020.0 | 833.5 | 0.7190 | 0.5780 |

比值为同 session、同 iteration 配对后取中位数；JSON/CSV 中包含 10,000 次 bootstrap 95% CI。
