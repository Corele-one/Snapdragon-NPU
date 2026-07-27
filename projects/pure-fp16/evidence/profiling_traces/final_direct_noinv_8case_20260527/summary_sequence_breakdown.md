# Pure FP16 Sequence Length Sweep

Runtime: isolated pure_fp16, HMX-layout FP16 weights, no INT4/INT8 per-group weight dequantization.

| case | P:D | runs | actual prompt | actual decode | prefill tok/s | decode tok/s | HMX MMA % | weight load % | dequant % | trace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| prefill_64_decode_16 | 64:16 | 3 | 136 | 16 | 766.2914 | 10.3419 | 3.4538 | 28.6822 | 0 | `.\artifacts\llm_inference_pure_fp16\sequence_benchmarks\final_8case_direct_noinv_3run_20260527_1205\profiling\trace_prefill_64_decode_16` |
| prefill_256_decode_16 | 256:16 | 3 | 456 | 16 | 1112.6315 | 10.0629 | 5.8587 | 26.8991 | 0 | `.\artifacts\llm_inference_pure_fp16\sequence_benchmarks\final_8case_direct_noinv_3run_20260527_1205\profiling\trace_prefill_256_decode_16` |
| prefill_768_decode_16 | 768:16 | 3 | 1316 | 16 | 1040.0678 | 9.8248 | 10.2047 | 23.8124 | 0 | `.\artifacts\llm_inference_pure_fp16\sequence_benchmarks\final_8case_direct_noinv_3run_20260527_1205\profiling\trace_prefill_768_decode_16` |
| prefill_256_decode_64 | 256:64 | 3 | 456 | 64 | 1103.8031 | 9.3695 | 3.2443 | 28.8978 | 0 | `.\artifacts\llm_inference_pure_fp16\sequence_benchmarks\final_8case_direct_noinv_3run_20260527_1205\profiling\trace_prefill_256_decode_64` |
| long_p1024_d16 | 1024:16 | 3 | 1736 | 16 | 981.6202 | 9.2834 | 11.7249 | 22.6864 | 0 | `.\artifacts\llm_inference_pure_fp16\sequence_benchmarks\final_8case_direct_noinv_3run_20260527_1205\profiling\trace_long_p1024_d16` |
| long_p1024_d128 | 1024:128 | 3 | 1737 | 128 | 962.6409 | 8.784 | 4.0646 | 28.7616 | 0 | `.\artifacts\llm_inference_pure_fp16\sequence_benchmarks\final_8case_direct_noinv_3run_20260527_1205\profiling\trace_long_p1024_d128` |
| long_p1536_d64 | 1536:64 | 3 | 2596 | 38 | 877.3924 | 8.5603 | 8.9125 | 24.7582 | 0 | `.\artifacts\llm_inference_pure_fp16\sequence_benchmarks\final_8case_direct_noinv_3run_20260527_1205\profiling\trace_long_p1536_d64` |
| long_p1792_d256 | 1792:256 | 3 | 3017 | 256 | 876.8066 | 8.3709 | 3.8991 | 28.9295 | 0 | `.\artifacts\llm_inference_pure_fp16\sequence_benchmarks\final_8case_direct_noinv_3run_20260527_1205\profiling\trace_long_p1792_d256` |

Artifacts:
- Raw no-trace runs: `raw_no_trace_runs.csv`
- One detailed trace per case: `profiling/trace_<case>/`
- Each trace directory contains `llm_trace_events.csv`, `llm_trace_stage_events.csv`, `llm_trace_summary.json`, `llm_trace_breakdown.md`, `.ntff`, and `.perfetto.json`.
- Acceptance check: `matmul_weight_dequant_pct` must remain `0`, proving this route did not enter `weight_hvx_dequant`.
