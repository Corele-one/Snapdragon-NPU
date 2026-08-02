# Figure 8 baseline Attention Percentages

Percentages use median component time divided by the sum of median profiled components.

Trace source note: these component times come from DSP qtimer records emitted as `FIG8_ATTENTION_TIMERS` / `FIG8_ATTENTION_EVENT`; they are not reconstructed from host wait time. The `Q batch` / `qo_len` rows mean the number of query positions processed by one standalone attention-kernel invocation.

| qo_len | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.39% | 36.13% | 31.54% | 0.59% | 30.57% | 0.39% | 0.00% | 0.39% |
| 8 | 0.56% | 23.12% | 27.99% | 1.25% | 45.13% | 0.97% | 0.14% | 0.84% |
| 16 | 0.74% | 16.90% | 12.40% | 1.21% | 66.30% | 1.25% | 0.09% | 1.11% |
| 32 | 0.84% | 8.79% | 6.29% | 1.21% | 79.62% | 1.94% | 0.16% | 1.15% |

## Median Component Time (us)

| qo_len | profiled_total | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 512.0 | 2.0 | 185.0 | 161.5 | 3.0 | 156.5 | 2.0 | 0.0 | 2.0 |
| 8 | 718.0 | 4.0 | 166.0 | 201.0 | 9.0 | 324.0 | 7.0 | 1.0 | 6.0 |
| 16 | 1077.0 | 8.0 | 182.0 | 133.5 | 13.0 | 714.0 | 13.5 | 1.0 | 12.0 |
| 32 | 1906.5 | 16.0 | 167.5 | 120.0 | 23.0 | 1518.0 | 37.0 | 3.0 | 22.0 |
