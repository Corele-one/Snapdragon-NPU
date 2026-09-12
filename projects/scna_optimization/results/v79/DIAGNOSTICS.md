# q32 组件诊断

以下是带 timer/event 的独立诊断样本中位数，单位为跨 task 的 work-sum us；不参与正式排名，也不能与 Host wall latency 混用。

| mode | profiled total | K load | V load | QK | safe softmax | SCNA exp | PV/core | workers/tasks/q rows |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 2147.0 | 175.0 | 112.0 | 31.0 | 1730.0 | 0.0 | 56.0 | original scheduler |
| lut_exp | 1389.0 | 190.0 | 118.0 | 32.0 | 951.0 | 0.0 | 61.0 | original scheduler |
| scna | 2302.0 | 375.0 | 222.0 | 38.0 | 1562.0 | 1005.0 | 46.0 | 4/4/16 |

SCNA 的 `safe softmax` 包含其中的 `SCNA exp` 子区间，两列不可相加。
