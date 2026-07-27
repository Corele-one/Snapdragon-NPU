# Pure FP16 Sequence Length Sweep (Speed Only)

Runtime: isolated pure_fp16, no trace, tuned default batch settings from run_server.sh.

| case | P:D | runs | actual prompt | actual decode | prefill tok/s | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| prefill_64_decode_16 | 64:16 | 3 | 136 | 2 | 126.3984 | 1.9332 |
| prefill_256_decode_16 | 256:16 | 3 | 456 | 16 | 311.6362 | 1.0134 |
| prefill_768_decode_16 | 768:16 | 3 | 1316 | 2 | 289.1798 | 1.8474 |
| prefill_256_decode_64 | 256:64 | 3 | 456 | 2 | 294.0637 | 1.8516 |
| long_p1024_d16 | 1024:16 | 3 | 1736 | 2 | 289.2076 | 2.0906 |
| long_p1024_d128 | 1024:128 | 3 | 1737 | 2 | 280.0433 | 1.9173 |
| long_p1536_d64 | 1536:64 | 3 | 2596 | 2 | 276.9291 | 1.8386 |
| long_p1792_d256 | 1792:256 | 3 | 3017 | 51 | 295.2158 | 0.9596 |

Detailed no-dequant trace validation is kept in profiling_traces/smoke_trace_fixed_20260526_221324 and can be regenerated with un_trace_pure_fp16.ps1.
