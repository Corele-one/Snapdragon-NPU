# Figure 8 baseline Attention Percentages

Percentages use median component time divided by the sum of median profiled components.

| qo_len | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.37% | 33.30% | 32.84% | 0.55% | 31.64% | 0.37% | 0.18% | 0.74% |
| 8 | 0.51% | 21.87% | 26.98% | 0.90% | 47.83% | 0.90% | 0.13% | 0.90% |
| 16 | 0.69% | 14.84% | 12.54% | 1.13% | 68.55% | 1.13% | 0.09% | 1.04% |
| 32 | 0.73% | 8.01% | 5.89% | 1.10% | 81.31% | 1.78% | 0.14% | 1.05% |

## Median Component Time (us)

| qo_len | profiled_total | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 542.0 | 2.0 | 180.5 | 178.0 | 3.0 | 171.5 | 2.0 | 1.0 | 4.0 |
| 8 | 782.0 | 4.0 | 171.0 | 211.0 | 7.0 | 374.0 | 7.0 | 1.0 | 7.0 |
| 16 | 1152.5 | 8.0 | 171.0 | 144.5 | 13.0 | 790.0 | 13.0 | 1.0 | 12.0 |
| 32 | 2191.0 | 16.0 | 175.5 | 129.0 | 24.0 | 1781.5 | 39.0 | 3.0 | 23.0 |
