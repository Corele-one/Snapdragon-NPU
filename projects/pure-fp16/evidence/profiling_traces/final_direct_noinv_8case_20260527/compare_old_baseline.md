# Pure FP16 Direct-DMA vs Old INT4/INT8 Baseline

| case | P:D | pure FP16 prefill tok/s | old INT4/INT8 tok/s | delta | delta % | dequant % | status |
|---|---:|---:|---:|---:|---:|---:|---|
| prefill_64_decode_16 | 64:16 | 766.2914 | 685.204 | 81.0874 | 11.834% | 0 | win |
| prefill_256_decode_16 | 256:16 | 1112.6315 | 849.96 | 262.6715 | 30.904% | 0 | win |
| prefill_768_decode_16 | 768:16 | 1040.0678 | 792.101 | 247.9668 | 31.305% | 0 | win |
| prefill_256_decode_64 | 256:64 | 1103.8031 | 836.495 | 267.3081 | 31.956% | 0 | win |
| long_p1024_d16 | 1024:16 | 981.6202 | 759.361 | 222.2592 | 29.269% | 0 | win |
| long_p1024_d128 | 1024:128 | 962.6409 | 745.916 | 216.7249 | 29.055% | 0 | win |
| long_p1536_d64 | 1536:64 | 877.3924 | 711.644 | 165.7484 | 23.291% | 0 | win |
| long_p1792_d256 | 1792:256 | 876.8066 | 675.44 | 201.3666 | 29.813% | 0 | win |

Source pure-FP16 sweep: historical `final_8case_direct_noinv_3run_20260527_1205` snapshot
Old baseline source: `artifacts\llm_inference_modes_v73_qwen15b\sequence_benchmarks\sequence_length_sweep_20260520\summary_sequence_breakdown.md`
