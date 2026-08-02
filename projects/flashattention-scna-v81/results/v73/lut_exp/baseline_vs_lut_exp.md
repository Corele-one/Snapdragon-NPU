# Figure 8 Baseline vs LUT-exp

Comparison uses median per-iteration profiled component time. Negative delta means LUT-exp is faster.

| qo_len | baseline total us | LUT-exp total us | total delta | baseline safe_sm us | LUT-exp safe_sm us | safe_sm delta |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 512.0 | 480.5 | -6.15% | 156.5 | 105.0 | -32.91% |
| 8 | 718.0 | 625.5 | -12.88% | 324.0 | 217.0 | -33.02% |
| 16 | 1077.0 | 808.0 | -24.98% | 714.0 | 445.0 | -37.68% |
| 32 | 1906.5 | 1339.0 | -29.77% | 1518.0 | 931.0 | -38.67% |

## Percentage Point Change

| qo_len | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.03 | 1.74 | 6.23 | -0.17 | -8.71 | 0.44 | 0.00 | 0.44 |
| 8 | 0.08 | 5.26 | 4.62 | -0.13 | -10.43 | 0.30 | 0.02 | 0.28 |
| 16 | 0.25 | 6.43 | 3.20 | 0.34 | -11.22 | 0.60 | 0.03 | 0.37 |
| 32 | 0.32 | 5.07 | 2.29 | 0.59 | -10.09 | 1.27 | 0.07 | 0.49 |
