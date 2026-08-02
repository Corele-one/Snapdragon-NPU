# Figure 8 Baseline vs LUT-exp

Comparison uses median per-iteration profiled component time. Negative delta means LUT-exp is faster.

| qo_len | baseline total us | LUT-exp total us | total delta | baseline safe_sm us | LUT-exp safe_sm us | safe_sm delta |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 542.0 | 483.5 | -10.79% | 171.5 | 103.0 | -39.94% |
| 8 | 782.0 | 621.0 | -20.59% | 374.0 | 212.0 | -43.32% |
| 16 | 1152.5 | 810.0 | -29.72% | 790.0 | 438.5 | -44.49% |
| 32 | 2191.0 | 1341.0 | -38.80% | 1781.5 | 928.5 | -47.88% |

## Percentage Point Change

| qo_len | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.04 | 4.55 | 4.90 | 0.07 | -10.34 | 0.46 | 0.23 | 0.09 |
| 8 | 0.13 | 6.55 | 6.11 | 0.39 | -13.69 | 0.39 | 0.03 | 0.07 |
| 16 | 0.29 | 8.13 | 4.44 | 0.35 | -14.41 | 0.72 | 0.04 | 0.44 |
| 32 | 0.46 | 5.19 | 3.40 | 0.77 | -12.07 | 1.43 | 0.09 | 0.74 |
