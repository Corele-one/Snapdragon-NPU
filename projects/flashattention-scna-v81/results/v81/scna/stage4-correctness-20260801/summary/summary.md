# V81 SCNA Correctness Matrix

- FP32 comparisons: 100
- Direct/tree comparisons: 48
- Mask/tail probes: 100
- Gate failures: 0

| Mode | Function | Kernel | Cases | Max RMSE | Max absolute error | Nonfinite |
|---|---|---|---:|---:|---:|---:|
| baseline | exp2 | direct | 4 | 0.00137799 | 0.00621134 | 0 |
| scna-fp16 | exp | direct | 12 | 0.000700928 | 0.00145626 | 0 |
| scna-fp16 | exp | tree | 12 | 0.000701124 | 0.00146008 | 0 |
| scna-fp16 | exp2 | direct | 12 | 0.000705324 | 0.00149202 | 0 |
| scna-fp16 | exp2 | tree | 12 | 0.000707168 | 0.00146771 | 0 |
| scna-int8 | exp | direct | 12 | 0.000886363 | 0.00291472 | 0 |
| scna-int8 | exp | tree | 12 | 0.000886363 | 0.00291472 | 0 |
| scna-int8 | exp2 | direct | 12 | 0.000850713 | 0.00239311 | 0 |
| scna-int8 | exp2 | tree | 12 | 0.000850713 | 0.00239311 | 0 |
