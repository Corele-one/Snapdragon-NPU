cmake_minimum_required(VERSION 3.14.3)
project(htp_ops C CXX ASM)
# cmake_policy(SET CMP0115 OLD)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(FIGURE8_ENABLE_LUT_EXP "Use LUT/vgather exp in the Figure 8 profiled flash attention path" OFF)
option(FIGURE8_ENABLE_PROFILE_TIMERS "Enable Figure 8 flash attention profiling timers/logging" ON)
option(HTP_ENABLE_OUT_STATIONARY_PREFILL "Enable the repaired output-stationary prefill matmul path" OFF)
option(HTP_OS_DEBUG_COMPARE "Enable output-stationary debug accumulator checks" OFF)
set(HTP_MATMUL_PIPELINE_MODE "0" CACHE STRING "Matmul qk pipeline mode: 0=auto original 4-stage path, 1=force sequential safe path")
set(HTP_MATMUL_WEIGHT_AREA_KB "1024" CACHE STRING "VTCM KiB reserved for matmul weights")
set(HTP_MATMUL_ACTIVATION_AREA_KB "2048" CACHE STRING "VTCM KiB reserved for matmul activations")
set(HTP_MATMUL_OUTPUT_AREA_KB "1536" CACHE STRING "VTCM KiB reserved for matmul outputs")
set(HTP_MATMUL_SCRATCH_AREA_KB "1024" CACHE STRING "VTCM KiB reserved for each matmul scratch buffer")
set(HTP_MATMUL_PIPELINE_OUTPUT_KB "512" CACHE STRING "VTCM KiB reserved for each K<=N FP16 pipeline output buffer")
set(HTP_FP16_WEIGHT_LOAD_MODE "2" CACHE STRING "Pure-FP16 W16A32 weight load: 0=HVX DDR copy, 1=DMA direct to HMX weight VTCM, 2=DMA to VTCM scratch then HVX VTCM copy")
set(HTP_FP16_DMA_DST_BYPASS "0" CACHE STRING "Pure-FP16 weight DMA dst_bypass bit; 1 writes through to VTCM for direct-HMX buffers")
set(HTP_FP16_DIRECT_FINAL_DMA "0" CACHE STRING "Experimental hybrid: keep safe scratch mode globally, but use direct DMA for K<=N final HMX weight tiles")
set(HTP_FP16_DIRECT_FINAL_PIPELINE "0" CACHE STRING "Experimental pure-FP16 pipeline: DMA directly into a 3-entry final HMX weight-buffer ring")
set(HTP_FP16_DIRECT_FINAL_CACHE_INVALIDATE "0" CACHE STRING "Experimental pure-FP16 direct DMA: invalidate final VTCM weight buffer before HMX consume")
set(HTP_FP16_DIRECT_FINAL_TOUCH "1" CACHE STRING "Experimental pure-FP16 direct DMA: keep an HVX read/write producer touch before HMX consume")
set(HTP_FP16_DECODE_DIRECT_PIPELINE "0" CACHE STRING "Pure-FP16 decode: allow direct-final DMA pipeline for small-M K<=N matmuls")
set(HTP_FP16_DECODE_PIPELINE_MAX_M "1" CACHE STRING "Pure-FP16 decode direct pipeline max M rows")
set(HTP_FP16_PARALLEL_WEIGHT_PUBLISH "1" CACHE STRING "Pure-FP16 safe publish: split DMA-scratch to final-HMX VTCM copy over HVX workers")
set(HTP_FP16_PUBLISH_USE_MEMCPY "0" CACHE STRING "Experimental pure-FP16 publish backend: use platform memcpy instead of explicit HVX vector loop")
set(HTP_FP16_PARALLEL_OUTPUT_STORE "1" CACHE STRING "Pure-FP16 output store: split prefill-sized FP16 tile to FP32 row-major conversion over HVX workers")
set(HTP_FP16_SMALL_M_VTCM_ENABLE "0" CACHE STRING "Pure-FP16 short-M profile: rebalance VTCM toward larger weight tiles")
set(HTP_FP16_SMALL_M_THRESHOLD "192" CACHE STRING "Pure-FP16 short-M profile threshold in rows")
set(HTP_FP16_SMALL_M_WEIGHT_AREA_KB "2048" CACHE STRING "Pure-FP16 short-M VTCM KiB reserved for weights")
set(HTP_FP16_SMALL_M_ACTIVATION_AREA_KB "512" CACHE STRING "Pure-FP16 short-M VTCM KiB reserved for activations")
set(HTP_FP16_SMALL_M_OUTPUT_AREA_KB "768" CACHE STRING "Pure-FP16 short-M VTCM KiB reserved for outputs")
set(HTP_FP16_SMALL_M_SCRATCH_AREA_KB "2048" CACHE STRING "Pure-FP16 short-M VTCM KiB reserved for scratch")
set(HTP_FP16_SMALL_M_PIPELINE_OUTPUT_KB "384" CACHE STRING "Pure-FP16 short-M VTCM KiB reserved for each pipeline output buffer")
set(HTP_FP16_OS_M_BLOCK_SIZE "512" CACHE STRING "Pure-FP16 output-stationary max M block")
set(HTP_FP16_OS_N_BLOCK_SIZE "1024" CACHE STRING "Pure-FP16 output-stationary max N block")
set(HTP_FP16_OS_K_BLOCK_SIZE "1024" CACHE STRING "Pure-FP16 output-stationary K block")
set(HTP_FP16_OS_WEIGHT_GATHER_DMA "0" CACHE STRING "Pure-FP16 output-stationary K-slice gather: direct DMA into final HMX weight buffer")
set(HTP_FP16_OS_WEIGHT_GATHER_DMA_SCRATCH "1" CACHE STRING "Pure-FP16 output-stationary K-slice gather: DMA into scratch then publish with HVX")
set(HTP_FP16_OS_WEIGHT_GATHER_DMA_PREFILL_ONLY "1" CACHE STRING "Limit direct OS gather DMA to prefill-sized M so decode keeps the conservative scratch path")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS "0" CACHE STRING "LPBQ deploy-v1 R4 A8W8 candidate: route R4/down_proj into the full-U8 single-pass path")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6 "0" CACHE STRING "LPBQ deploy-v1 R4 A8W8 candidate: use FP16-style V6 activation.ub/weight.b streaming for R4")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_TILES "4" CACHE STRING "LPBQ deploy-v1 R4 A8W8 candidate: V6 grouped K32 tile count; 4 means K128")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_SMALL_M_MAX "4" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 guard: max M rows allowed; -1 disables the guard for diagnostics")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_MIN_M "0" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 guard: min M rows allowed; default 0 preserves historical small-M routing")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_VALUE "-1" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 recover round override; -1 keeps half-LSB")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_ALT "0" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 recover round alternates +1 on odd groups")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_ALT_EXTRA "0" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 recover round adds +1 to first N groups")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_ROUND_VALUE "-1" CACHE STRING "LPBQ deploy-v1 R4 K16/K4 full-U8 exact recover round override; -1 keeps half-LSB")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_ROUND_VALUE_ROWS_GT1 "-2" CACHE STRING "LPBQ deploy-v1 R4 K16/K4 full-U8 exact recover rows>1 round override; -2 uses scalar setting, -1 keeps half-LSB")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_APPLY_257 "1" CACHE STRING "LPBQ deploy-v1 R4 K16/K4 full-U8 exact recover: apply historical 256/257 compensation")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_NEG_EXTRA "0" CACHE STRING "LPBQ deploy-v1 R4 K16/K4 full-U8 exact recover: extra raw unit for negative partial accumulators")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_RSHIFT "0" CACHE STRING "LPBQ deploy-v1 R4 K16/K4 full-U8 exact recover: sign-aware magnitude correction shift, 0 disables")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_RSHIFT_ROWS_GT1 "-1" CACHE STRING "LPBQ deploy-v1 R4 K16/K4 full-U8 exact recover: rows>1 sign-aware correction shift override; -1 uses scalar setting")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_EXTRA_RSHIFT "0" CACHE STRING "LPBQ deploy-v1 R4 K16/K4 full-U8 exact recover: optional extra fractional sign-aware correction shift, 0 disables")
set(HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_EXTRA_RSHIFT_ROWS_GT1 "0" CACHE STRING "LPBQ deploy-v1 R4 K16/K4 full-U8 exact recover: rows>1 extra fractional correction shift override, 0 uses scalar setting")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_SIGN_COMPENSATE "0" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 recover A/B: compensate the HMX 257/256 scale law before dequant")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_EXACT_RECOVER "0" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 recover A/B: scalar exact 256/257 recovery for correctness diagnosis")
set(HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_EXACT_RECOVER "0" CACHE STRING "LPBQ deploy-v1 non-R4 grouped V6 recover A/B: scalar exact 256/257 recovery for Q/K correctness diagnosis")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_SCALE "0" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 A/B: 1=bound scan, 2=safe probe + final issue, 3=safe probe only")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_RAW_ADJUST "0" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 adaptive recover A/B: add signed raw units after shift recovery")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_ROUND_BIAS_BITS "0x3800u" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 adaptive HMX scale-register round-bias bits; 0x3800u keeps fp16 0.5, 0 tests truncation")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_SCALE_BITS_OVERRIDE "0" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 scale-bits override; 0 keeps historical scale law")
set(HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_RECOVER_SHIFT_OVERRIDE "-1" CACHE STRING "LPBQ deploy-v1 R4 grouped V6 recover-shift override; -1 keeps historical scale law")
set(HTP_LPBQ_R4_FULL_U8_K16_GROUPS "0" CACHE STRING "LPBQ deploy-v1 R4 A8W8 candidate: enable K16 full-U8 groups")
set(HTP_LPBQ_R4_FULL_U8_K32_GROUPS "0" CACHE STRING "LPBQ deploy-v1 R4 A8W8 candidate: enable K32/grouped full-U8 groups")
set(HTP_LPBQ_R4_FULL_U8_K32_SCALE_BITS_OVERRIDE "0" CACHE STRING "LPBQ deploy-v1 R4 K32 full-U8 HMX scale bits override; 0 keeps normal K32 scale")
set(HTP_LPBQ_R4_FULL_U8_K32_RECOVER_SHIFT_OVERRIDE "-1" CACHE STRING "LPBQ deploy-v1 R4 K32 full-U8 recover shift override; -1 keeps normal K32 shift")
set(HTP_LPBQ_FULL_U8_PRECOMPENSATE_HMX_SCALE "0" CACHE STRING "LPBQ deploy-v1 A8W8 candidate: fold 256/257 scale-law compensation into HMX output scale")
set(HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_TILES "16" CACHE STRING "LPBQ deploy-v1 non-R4 A8W8 candidate: V6 grouped K32 tile count; 16 means K512")
set(HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_DELAY_RECOVER "0" CACHE STRING "LPBQ deploy-v1 A8W8 candidate: delay grouped V6 recover to a fused HVX epilogue")
set(HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_COMPACT_WEIGHT "0" CACHE STRING "LPBQ deploy-v1 bottleneck-first A/B: have grouped V6 HMX consume compact K32 weight stream instead of publishing full V6 tiles")
set(HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP "0" CACHE STRING "LPBQ deploy-v1 A8W8 candidate: overlap current N-tile HMX issue with previous N-tile HVX after.uh accumulate")
set(HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_SMALL_M_ONLY "0" CACHE STRING "LPBQ deploy-v1 A8W8 candidate: restrict grouped-V6 N-tile overlap to small-M/decode rows")
set(HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_PREFILL_ONLY "0" CACHE STRING "LPBQ deploy-v1 rowblock4 A8W8 candidate: restrict rowblock adaptive N-tile overlap to prefill rows")
set(HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_ACC_FINAL_ISSUE "0" CACHE STRING "LPBQ deploy-v1 rowblock4 A/B: overlap pending after.uh accumulate with the next final issue instead of probe issue")
set(HTP_LPBQ_FULL_U8_V6_COPY_1024_LOOP8 "0" CACHE STRING "LPBQ deploy-v1 A8W8 candidate: use loop8 HVX copy for aligned 1024B compact-to-full V6 publish")
set(HTP_LPBQ_FULL_U8_V6_COPY_BULK_LOOP8 "0" CACHE STRING "LPBQ deploy-v1 A8W8 candidate: use loop8 HVX copy chunks for larger aligned full-V6 publishes")
set(HTP_LPBQ_SKIP_UNUSED_FULL_U8_HI_ACT_CLEAR "0" CACHE STRING "LPBQ deploy-v1 A8W8 candidate: skip unused hi activation HMX clear for full-U8 singlepass paths")
set(HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH "1" CACHE STRING "LPBQ deploy-v1 weight-staging A/B: keep or disable l2fetch in the aligned HVX weight copy helper")
set(HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH_BYTES "4096" CACHE STRING "LPBQ deploy-v1 weight-staging A/B: l2fetch lookahead bytes in the aligned HVX weight copy helper")
set(HTP_LPBQ_KMAJOR_WEIGHT_GROUP_COPY "0" CACHE STRING "LPBQ deploy-v1 weight-staging A/B: copy a grouped-V6 K-major weight window with one contiguous HVX copy")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_LOAD "0" CACHE STRING "LPBQ deploy-v1 weight-staging A/B: DMA a compact K-major weight window directly into the HMX-consumed VTCM stream")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_DST_BYPASS "1" CACHE STRING "LPBQ deploy-v1 K-major DMA publish dst_bypass bit for HMX-facing VTCM buffers")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_SRC_BYPASS "0" CACHE STRING "LPBQ deploy-v1 K-major DMA publish src_bypass bit for compact weight sidecars; old bypass=1 is unsafe on cached sidecars")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_PREISSUE_WAIT "1" CACHE STRING "LPBQ deploy-v1 K-major DMA A/B: keep the historical pre-issue idle wait before submit")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_VISIBILITY_SYNC "0" CACHE STRING "LPBQ deploy-v1 K-major DMA A/B: old post-DMA invalidate rollback, default off after standalone and real-layer validation")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_OVERLAP_ACT "0" CACHE STRING "LPBQ deploy-v1 K-major DMA A/B: issue grouped weight DMA before activation/R4 producer and wait before HMX consumption")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_DOUBLE_BUFFER "0" CACHE STRING "LPBQ deploy-v1 K-major DMA A/B: prefetch the next logical grouped-V6 weight window into a second VTCM stream while HMX consumes the current window")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_N_BLOCK_TILES "0" CACHE STRING "LPBQ deploy-v1 K-major DMA A/B: gather a large N-column block per grouped-V6 K window instead of staging the full N stream")
set(HTP_LPBQ_KMAJOR_WEIGHT_DMA_OVERLAP_ROWBLOCK "0" CACHE STRING "LPBQ deploy-v1 K-major DMA A/B: rowblock-reuse variant of activation-overlapped grouped weight DMA")
set(HTP_LPBQ_NONR4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE "0" CACHE STRING "LPBQ deploy-v1 non-R4 grouped-V6 A/B: stage one K-major weight group for several 32-row activation blocks")
set(HTP_LPBQ_NONR4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE_BLOCKS "2" CACHE STRING "LPBQ deploy-v1 non-R4 grouped-V6 A/B: number of 32-row blocks sharing one staged K-major weight group")
set(HTP_LPBQ_NONR4_GROUPED_V6_NBLOCK_ROWBLOCK_REUSE "0" CACHE STRING "LPBQ deploy-v1 non-R4 grouped-V6 A/B: reuse one limited N-block K-major weight group across several 32-row blocks")
set(HTP_LPBQ_NONR4_GROUPED_V6_NBLOCK_ROWBLOCK_REUSE_TILES "8" CACHE STRING "LPBQ deploy-v1 non-R4 grouped-V6 A/B: N tiles per rowblock-reuse weight chunk")
set(HTP_LPBQ_FFNDOWN_DECODE_G32_STAGING_AB "0" CACHE STRING "LPBQ deploy-v1 B1 A/B: narrow ffn_down decode rowblock4 grouped K-major staging for m=1,k=8960,n=1536")
set(HTP_LPBQ_FFNGATEUP_DECODE_G32_STAGING_AB "0" CACHE STRING "LPBQ deploy-v1 B2 A/B: narrow non-R4 ffn_gate/up decode grouped K-major staging for m=1,k=1536,n=8960")
set(HTP_LPBQ_WEIGHT_PUBLISH_TRACE_SPLIT "0" CACHE STRING "LPBQ deploy-v1 diagnostic: legacy alias for weight-path trace split")
set(HTP_LPBQ_WEIGHT_PATH_TRACE_SPLIT "0" CACHE STRING "LPBQ deploy-v1 diagnostic: split decode ffn_down weight_hvx_load into coarse non-overlapping substages")
set(HTP_LPBQ_TRACE_W0_EXPERT_BUCKETS "0" CACHE STRING "LPBQ deploy-v1 diagnostic: reserve expert-guided W0 trace bucket names without changing the data path")
set(HTP_LPBQ_R4_FWHT_V2 "0" CACHE STRING "LPBQ deploy-v1 expert plan R4-S0/S1/S2: enable structured R4 FWHT v2 sidecar/runtime path")
set(HTP_LPBQ_R4_FWHT_DIRECT_V6_STORE "0" CACHE STRING "LPBQ deploy-v1 expert plan R4-S2: FWHT producer emits HMX grouped-V6 activation layout directly")
set(HTP_LPBQ_R4_FWHT_BLOCK "128" CACHE STRING "LPBQ deploy-v1 expert plan R4-S*: structured FWHT block size")
set(HTP_LPBQ_R4_FWHT_SCALE_GRANULARITY "16" CACHE STRING "LPBQ deploy-v1 expert plan R4-S*: structured FWHT scale group size")
set(HTP_LPBQ_ACT_ROWS1_DIRECT_V6 "0" CACHE STRING "LPBQ deploy-v1 expert plan ACT-S1: rows=1 direct A8/V6 activation producer")
set(HTP_LPBQ_ACT_PACKET_REUSE "0" CACHE STRING "LPBQ deploy-v1 expert plan ACT-S2: adjacent-linear zero-copy activation packet reuse")
set(HTP_LPBQ_ACT_STATIC_SCALE_NONR4 "0" CACHE STRING "LPBQ deploy-v1 expert plan ACT-S3: static/semi-static non-R4 activation scale with dynamic fallback")
set(HTP_LPBQ_HMX_CONVERT_AFFINE_BENCH "0" CACHE STRING "LPBQ deploy-v1 expert plan HMX-S1: HMX convert affine microbench")
set(HTP_LPBQ_HMX_REUSE_SCALE_PAYLOAD_PER_NTILE "0" CACHE STRING "LPBQ deploy-v1 expert plan HMX-S2: reuse scale/bias payload per N tile")
set(HTP_LPBQ_HMX_HOLD_ACQUIRE_PER_ROWBLOCK "0" CACHE STRING "LPBQ deploy-v1 expert plan HMX-S2: hold HMX acquire across a rowblock")
set(HTP_LPBQ_HMX_BATCH_ISSUE_KGROUPS "0" CACHE STRING "LPBQ deploy-v1 expert plan HMX-S2/S3: batch grouped-V6 K-group issue")
set(HTP_LPBQ_HMX_DOUBLE_BUFFER_DRAIN "0" CACHE STRING "LPBQ deploy-v1 expert plan HMX-S3: double-buffer HMX drain/epilogue")
set(HTP_LPBQ_HMX_EPILOGUE_LAG_ONE_TILE "0" CACHE STRING "LPBQ deploy-v1 expert plan HMX-S3: lag HVX epilogue one tile behind HMX issue")
set(HTP_LPBQ_HMX_OPCODE_CORPUS "0" CACHE STRING "LPBQ deploy-v1 diagnostic: link a small HMX opcode spelling corpus for disassembly mining")
set(HTP_LPBQ_HMX_AFTER_ISSUE_HVX_PROBE "0" CACHE STRING "LPBQ deploy-v1 macro-pipeline microprobe: insert fixed HVX scratch work after logical grouped-V6 HMX issue and before finish")
set(HTP_LPBQ_HMX_AFTER_ISSUE_HVX_PROBE_ITERS "8" CACHE STRING "LPBQ deploy-v1 macro-pipeline microprobe: number of private VTCM HVX touch passes")
set(HTP_LPBQ_FULL_V6_WEIGHT_DMA_LOAD "0" CACHE STRING "LPBQ deploy-v1 A8W8 candidate: publish full-V6 weight sidecars to VTCM with DMA instead of HVX copy")
set(HTP_LPBQ_FULL_V6_WEIGHT_DMA_SCRATCH "1" CACHE STRING "LPBQ deploy-v1 full-V6 DMA publish: 1 keeps DMA scratch plus HVX publish, 0 probes direct-final DMA")
set(HTP_LPBQ_FULL_V6_WEIGHT_DMA_DST_BYPASS "1" CACHE STRING "LPBQ deploy-v1 full-V6 DMA publish dst_bypass bit for HMX-facing VTCM buffers")
set(HTP_LPBQ_FULL_V6_WEIGHT_DMA_OVERLAP_ACT "0" CACHE STRING "LPBQ deploy-v1 macro-pipeline A/B: issue the first full-V6 weight DMA before activation/R4 producer, then finish before HMX")
set(HTP_LPBQ_FULL_V6_WEIGHT_DIRECT_MXMEM_LOAD "0" CACHE STRING "LPBQ deploy-v1 A/B: let grouped V6 HMX load full-V6 weight sidecar directly instead of republishing to VTCM")
set(HTP_LPBQ_FULL_V6_WEIGHT_PUBLISH_PINGPONG "0" CACHE STRING "LPBQ deploy-v1 macro-pipeline A/B: ping-pong full-V6 VTCM weight publish so HVX prepares tile t+1 while HMX computes tile t")
set(HTP_LPBQ_FULL_V6_WEIGHT_PUBLISH_PINGPONG_SMALL_M_MAX "0" CACHE STRING "LPBQ deploy-v1 macro-pipeline A/B: optional M cap for full-V6 ping-pong publish; 0 keeps the legacy unrestricted ping-pong behavior")
set(HTP_LPBQ_FULL_V6_WEIGHT_PARALLEL_PUBLISH "0" CACHE STRING "LPBQ deploy-v1 macro-pipeline A/B: parallelize HVX full-V6 weight publish without changing HMX logical grouped-V6 issue granularity")
set(HTP_LPBQ_FULL_V6_WEIGHT_ASYNC_PREPUBLISH_ACT "0" CACHE STRING "LPBQ deploy-v1 macro-pipeline A/B: publish first full-V6 weight tile on one HVX worker while activation/R4 producer runs")
set(HTP_LPBQ_FULL_V6_MACRO_KGROUP_PIPELINE "0" CACHE STRING "LPBQ deploy-v1 macro-pipeline A/B: ping-pong complete grouped-V6 K windows so HVX activation/FHT group t+1 overlaps HMX group t")
set(HTP_LPBQ_ACTIVATION_HMX_CACHE "0" CACHE STRING "LPBQ deploy-v1 bottleneck-first: cache non-R4 grouped-V6 HMX activation tiles across adjacent LPBQ matmuls")
set(HTP_LPBQ_ACTIVATION_HMX_CACHE_MAX_KB "1536" CACHE STRING "LPBQ deploy-v1 non-R4 HMX activation cache VTCM cap in KiB")
set(HTP_LPBQ_ACTIVATION_HVX_QUANT "1" CACHE STRING "LPBQ deploy-v1 accepted 2026-06-28: use guarded HVX FP32-to-U8 producer for non-R4 grouped-V6 activation tiles")
set(HTP_LPBQ_ACTIVATION_FP16_STAGED_QUANT "0" CACHE STRING "LPBQ deploy-v1 A/B: stage non-R4 grouped-V6 activation through FP16 HMX layout before U8 quant; can preserve online input_scale via a combined K32 scale vector")
set(HTP_LPBQ_DISABLE_ONLINE_INPUT_SCALE "0" CACHE STRING "LPBQ deploy-v1 performance-first A/B: ignore input_scale in activation/R4 producers; default off because non-R4 real-layer gate/up needs sidecar/reference input_scale math")
set(HTP_LPBQ_ACTIVATION_HVX_VECTOR_ROUND_STORE "0" CACHE STRING "LPBQ deploy-v1 gate/up A/B: vectorize correctness-preserving HVX activation round/clamp/V6-store tail")
set(HTP_LPBQ_ACTIVATION_HVX_FLOOR_ROUND_STORE "0" CACHE STRING "LPBQ deploy-v1 gate/up A/B: use floor(x+0.5) in the vector activation round/store tail")
set(HTP_LPBQ_R4_PARALLEL_PREFILL "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: parallelize prefill-sized dense-R4 quant/pack row pairs")
set(HTP_LPBQ_R4_CACHED_X4_KK_OUTER "1" CACHE STRING "LPBQ deploy-v1 R4 A/B: reuse dense R4 column groups across rows in cached-x4 activation pack")
set(HTP_LPBQ_R4_HVX_DOT "1" CACHE STRING "LPBQ deploy-v1 R4 A/B: use HVX dot-based dense R4 rotate/quant producer; 0 selects scalar reference producer for correctness isolation")
set(HTP_LPBQ_R4_HVX_DOT16 "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: use dot16 paired-column HVX rotate/quant for folded-input-scale R4")
set(HTP_LPBQ_R4_DOT4_ROWPAIR "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: compute two active rows over four dense R4 columns per HVX dot")
set(HTP_LPBQ_R4_FP16_HMX_ROTATE "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: compute exact R4 rotate with FP16 HMX before A8 V6 quant/pack")
set(HTP_LPBQ_R4_PREFILL_HMX_DENSE "0" CACHE STRING "LPBQ deploy-v1 no-quality Stage-A probe: route prefill dense R4 through FP16 HMX only for M>=min")
set(HTP_LPBQ_R4_PREFILL_HMX_MIN_M "32" CACHE STRING "LPBQ deploy-v1 no-quality Stage-A probe: minimum M for prefill-only HMX dense R4")
set(HTP_LPBQ_R4_HMX_DENSE_BLOCK "128" CACHE STRING "LPBQ deploy-v1 no-quality Stage-A probe: dense R4 HMX block size")
set(HTP_LPBQ_R4_HMX_HIGHBYTE_DIRECT_V6_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality Stage-A2 probe: map FP16 HMX R4 output high bytes directly into grouped-V6")
set(HTP_LPBQ_R4_HMX_HFSCALE_DIRECT_V6_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality Stage-A2 probe: use HVX FP16 scale direct-V6 producer for HMX dense R4")
set(HTP_LPBQ_R4_HMX_HFSCALE_VTCM_WEIGHT_CACHE_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality Stage-A2 probe: cache published FP16 HMX R4 weights in VTCM across rowblocks")
set(HTP_LPBQ_R4_HMX_HFSCALE_VTCM_WEIGHT_CACHE_GROUPS "8" CACHE STRING "LPBQ deploy-v1 no-quality Stage-A2 probe: number of K128 R4 weight groups cached in VTCM")
set(HTP_LPBQ_R4_V6_K128_BULK "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: generate four V6 K32 activation tiles for one exact 128x128 R4 block in one helper")
set(HTP_LPBQ_R4_V6_K128_PARALLEL_ROWS "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: parallelize only the K128 grouped-V6 R4 producer rows, leaving cached_x4 scheduling unchanged")
set(HTP_LPBQ_R4_V6_K128_SKIP_INNER_CLEAR "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: skip the K128 helper-local activation tile clear when the rowblock outer zero-once clear is active")
set(HTP_LPBQ_R4_V6_K128_BULK_SMALL_M "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: allow the exact R4 K128 bulk helper for decode/small-M rows")
set(HTP_LPBQ_R4_V6_K128_BULK_ROWPAIR "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: use rowpair dot4x2 inside the serial K128 bulk helper without enabling the global rowpair path")
set(HTP_LPBQ_R4_SIGN_FHT_FAST "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: use sign-only Sylvester FHT for R4 producer")
set(HTP_LPBQ_R4_STANDARD_FHT_FORCE "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: force standard Hadamard/FHT producer without reading learned R4 values")
set(HTP_LPBQ_R4_STANDARD_FHT_HMX_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: standalone HMX FP16 constant-Hadamard producer probe")
set(HTP_LPBQ_R4_STANDARD_FHT_HMX_INTEGRATED "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: route grouped-V6 activation producer through HMX standard Hadamard/FHT")
set(HTP_LPBQ_R4_STANDARD_FHT_HVX_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: standalone HVX FP32 tail-butterfly standard Hadamard/FHT producer probe")
set(HTP_LPBQ_R4_STANDARD_FHT_HVX_INTEGRATED "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: route grouped-V6 activation producer through the standalone-accepted HVX standard Hadamard/FHT producer")
set(HTP_LPBQ_R4_STANDARD_FHT_HVX_NIBBLE_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: standalone HVX standard Hadamard/FHT producer into default low4/high4 HMX K4 chunks")
set(HTP_LPBQ_R4_STANDARD_FHT_HVX_NIBBLE_INTEGRATED "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: route default low4/high4 R4 activation chunks through the HVX standard Hadamard/FHT producer")
set(HTP_LPBQ_R4_STANDARD_FHT_CORE_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: standalone standard-FHT core and row-major U8 lower-bound probe")
set(HTP_LPBQ_R4_STANDARD_FHT_ROWPAIR_DIRECT "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: standalone online standard-FHT row-pair direct grouped-V6 producer probe")
set(HTP_LPBQ_R4_STANDARD_FHT_I16_DIRECT "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: standalone online standard-FHT int16 direct grouped-V6 producer probe")
set(HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: standalone online standard-FHT full-HVX-prefix grouped-V6 producer probe")
set(HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_INTEGRATED "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: runtime online standard-FHT full-HVX-prefix grouped-V6 producer experiment")
set(HTP_LPBQ_R4_STANDARD_FHT_FP16_DIRECT "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: runtime online FP16-staged standard-FHT direct grouped-V6 producer experiment")
set(HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_NIBBLE_PROBE "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: standalone online standard-FHT full-HVX-prefix default-nibble producer probe")
set(HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_NIBBLE_INTEGRATED "0" CACHE STRING "LPBQ deploy-v1 no-quality R4 pivot: runtime online standard-FHT full-HVX-prefix K128 default-nibble producer experiment")
set(HTP_LPBQ_R4_SIGN_FHT_APPROX "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: allow approximate sign-only FHT when learned R4 magnitudes are not constant")
set(HTP_LPBQ_R4_SIGN_FHT_COLSCALE_APPROX "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: use Sylvester FHT with cached learned per-column magnitude scales")
set(HTP_LPBQ_R4_PREFETCH_COLS8 "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: prefetch dense R4 columns in cached-x4 activation pack")
set(HTP_LPBQ_R4_PREFETCH_K128_ROWPAR_COLS8 "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: prefetch dense R4 columns in the K128 row-parallel producer")
set(HTP_LPBQ_R4_QUANT_STORE8_FAST "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: unroll 8-lane R4 quant/store in cached activation producers")
set(HTP_LPBQ_R4_U8_V6_QUANT_STORE8_FAST "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: unroll only the grouped-V6 U8 cached activation producer store8 path")
set(HTP_LPBQ_R4_ROWS1_REDUCE8_FAST "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: use batched reduce8 only for rows=1 grouped-V6 R4 producer calls")
set(HTP_LPBQ_R4_ROWS1_DOT4_SPLIT "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: split rows=1 grouped-V6 dot8 into two dot4 noscale calls")
set(HTP_LPBQ_R4_DOT8_R128_UNROLL "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: explicitly unroll folded-scale dot8 when r4_block is 128")
set(HTP_LPBQ_R4_DOT_PACK_PRECOMPUTE_ROW_STORE "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: precompute grouped-V6 U8 row-store base for R4 dot-pack lane writes")
set(HTP_LPBQ_R4_K128_DOT_PACK_PRECOMPUTE_ROW_STORE "0" CACHE STRING "LPBQ deploy-v1 R4 K128 A/B: precompute grouped-V6 U8 row-store base only in the K128 producer")
set(HTP_LPBQ_R4_V6_K128_ROWGROUP4_DOT2 "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: K128 prefill producer computes four rows by two R4 columns to trade source loads for fewer R4 column loads")
set(HTP_LPBQ_R4_V6_K128_COLGROUP_PARALLEL "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: K128 prefill producer splits worker tasks by 8-column groups instead of row pairs")
set(HTP_LPBQ_R4_V6_K128_WORKERS_MAX "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: cap K128 row-parallel producer workers; 0 keeps all HVX128 contexts")
set(HTP_LPBQ_R4_V6_K128_SKIP_ACC_ZERO "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: skip redundant stack accumulator zeroing in the exact K128 rowpair producer")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_R128_UNROLL "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: unroll the K128 rowpair dot4x2 hot path when r4_block is 128")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_DOT8 "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: K128 rowpair producer computes eight R4 columns in one dot8x2 helper while preserving the dot4x2 fallback")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_REDUCE8 "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: reduce eight K128 rowpair dot4 accumulators together while preserving the dot4x2 issue pattern")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_STORE2_FAST "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: K128 rowpair producer writes paired V6 row lanes through one store helper")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_FUSED_STORE4 "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: K128 rowpair producer fuses dot4x2 reduction with paired V6 row-lane store")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_FUSED_STORE4_R128_UNROLL "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: unroll the accepted fused-store4 K128 rowpair producer when r4_block is 128")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_ALIGNED_LOADS "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: use aligned HVX loads in the guarded K128 fused rowpair producer when runtime addresses are VLEN-aligned")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_PREBASE_STORE8 "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: K128 rowpair producer precomputes V6 store bases while preserving the accepted per-row store8 helper")
set(HTP_LPBQ_R4_V6_K128_ROWPAIR_CONST_LANES "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: K128 rowpair producer uses constant V6 row lanes for pair*2 rows")
set(HTP_LPBQ_R4_SCALE_CACHE_R128_UNROLL "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: unroll the HVX scaled-src cache fill for the dominant r4_block=128 path")
set(HTP_LPBQ_R4_TRACE_DOT_PACK_SPLIT "0" CACHE STRING "LPBQ deploy-v1 diagnostic: split R4 dot-pack into dense-dot, quant, and UB-layout pack substages")
set(HTP_LPBQ_R4_NIBBLE_ISSUE_ACC_OVERLAP "1" CACHE STRING "LPBQ deploy-v1 R4 A/B: issue hi4 HMX while accumulating lo4 after.uh")
set(HTP_LPBQ_R4_NIBBLE_ACTIVE_ROW_ACCUM "1" CACHE STRING "LPBQ deploy-v1 R4/source-parity A/B: keep D30-era full-row accumulator available by setting 0")
set(HTP_LPBQ_R4_NIBBLE_RAW_ACCUM "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: accumulate hi4 shifted into the lo4 raw accumulator")
set(HTP_LPBQ_R4_NIBBLE_RAW_ACCUM_OVERLAP "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: raw-accumulate hi4 while preserving the lo/hi overlap schedule")
set(HTP_LPBQ_R4_ACCUM_DECODE_UNROLL "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: use standalone-proven unrolled row accumulator only for rows<=4 decode callsites")
set(HTP_LPBQ_R4_STRUCTURED_FWHT "1" CACHE STRING "LPBQ deploy-v1 R4 Stage-B guard: keep existing structured-FWHT sidecar path compiled by default; set 0 for Stage-A-only D30/source parity probes")
set(HTP_LPBQ_R4_STRUCTURED_FWHT_STAGE_UNROLL "0" CACHE STRING "LPBQ deploy-v1 R4 structured FWHT A/B: unroll the seven butterfly stages in the exact producer; rejected as default after P64/D16 decode regression")
set(HTP_LPBQ_R4_STRUCTURED_FWHT_DIRECT_K128 "0" CACHE STRING "LPBQ deploy-v1 no-quality Stage-B probe: structured-FWHT-only K128 four-tile producer without enabling dense K128 bulk")
set(HTP_LPBQ_ADAPTIVE_BUCKET_FAST_0_3 "0" CACHE STRING "LPBQ deploy-v1 A/B: specialize adaptive grouped-V6 HVX accumulate when recover shifts are only 0..3")
set(HTP_LPBQ_ADAPTIVE_BUCKET_FAST_0_3_ROWS1 "0" CACHE STRING "LPBQ deploy-v1 A/B: specialize adaptive grouped-V6 HVX accumulate for rows=1 when recover shifts are only 0..3")
set(HTP_LPBQ_ADAPTIVE_ACCUM_VARSHIFT_ROWS1 "0" CACHE STRING "LPBQ deploy-v1 A/B: use HVX per-lane variable shift for rows=1 adaptive grouped-V6 accumulate")
set(HTP_LPBQ_ADAPTIVE_SHIFT0_FAST_PATH "0" CACHE STRING "LPBQ deploy-v1 A/B: narrow fast path for adaptive grouped-V6 all-zero recover shifts")
set(HTP_LPBQ_ADAPTIVE_UNIFORM_SHIFT0_NOSHIFT "0" CACHE STRING "LPBQ deploy-v1 A/B: use no-shift accumulator for uniform adaptive recover_shift=0 after the existing all-same scan")
set(HTP_LPBQ_ADAPTIVE_SHIFT0_SINGLEPASS "0" CACHE STRING "LPBQ deploy-v1 A/B: assume adaptive grouped-V6 shift0 is safe and skip the probe/final retry drain")
set(HTP_LPBQ_ADAPTIVE_PROBE_SCALE_CACHE "0" CACHE STRING "LPBQ deploy-v1 A/B: prepack the fixed adaptive retry safe-probe mxmem2 scale payload in VTCM")
set(HTP_LPBQ_ADAPTIVE_PROBE_ONLY_SHIFT5_FAST "0" CACHE STRING "LPBQ deploy-v1 A/B: preinit AdaptiveScale=3 shift5 payload once and bypass generic adaptive recover scan")
set(HTP_LPBQ_ADAPTIVE_PROBE_SCALAR_SMALL_ROWS "0" CACHE STRING "LPBQ deploy-v1 A/B: use exact scalar adaptive derive only for rows<=2 decode tiles")
set(HTP_LPBQ_ADAPTIVE_RETRY_SKIP_INIT "0" CACHE STRING "LPBQ deploy-v1 A/B: skip the overwritten shift0 scale/recover init in adaptive retry")
set(HTP_LPBQ_ADAPTIVE_SCALE_LOW32_ONLY "0" CACHE STRING "LPBQ deploy-v1 A/B: only rewrite the low 32 adaptive scale words after retry/probe has zeroed the high half")
set(HTP_LPBQ_ADAPTIVE_SCALE_SKIP_UNCHANGED "0" CACHE STRING "LPBQ deploy-v1 A/B: skip final adaptive scale word stores that already match the probe payload")
set(HTP_LPBQ_ADAPTIVE_DERIVE_I32_MAX "0" CACHE STRING "LPBQ deploy-v1 A/B: use int32 scratch for adaptive safe-probe max_abs derive")
set(HTP_LPBQ_ADAPTIVE_SCALE_WORD_LUT "0" CACHE STRING "LPBQ deploy-v1 A/B: use a tiny local LUT for final adaptive scale-word formation")
set(HTP_LPBQ_ADAPTIVE_DERIVE_ROW0_EXACT_ROWS1 "0" CACHE STRING "LPBQ deploy-v1 A/B: derive rows=1 adaptive scales from row0 even lanes instead of full row-pair max")
set(HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE "0" CACHE STRING "LPBQ deploy-v1 R4 grouped-V6 A/B: stage each K-major weight group once across multiple 32-row blocks")
set(HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE_BLOCKS "2" CACHE STRING "LPBQ deploy-v1 R4 grouped-V6 row-block weight-reuse window; valid values are 1,2,4,8")
set(HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_HMX_ACQUIRE_PER_KGROUP "0" CACHE STRING "LPBQ deploy-v1 rowblock4 A/B: hold HMX across one K-group rowblock reuse window")
set(HTP_LPBQ_R4_ROWBLOCK_PROBE_ONLY_SHIFT5_FAST "0" CACHE STRING "LPBQ deploy-v1 rowblock HMX drain A/B: reuse the adaptive probe-only shift5 fast path inside rowblock reuse")
set(HTP_LPBQ_R4_ROWBLOCK_SKIP_INITIAL_SCALE_LOAD "0" CACHE STRING "LPBQ deploy-v1 HMX control-gap A/B: skip rowblock-entry scale preload for adaptive R4 grouped-V6")
set(HTP_LPBQ_HMX_FINE_SUBTRACE "0" CACHE STRING "LPBQ deploy-v1 diagnostic: attribute full-U8 HMX begin/expand/issue/finish; too expensive for production")
set(HTP_LPBQ_FULL_U8_ROWS1_ACCUM_PAIR_UNROLL "0" CACHE STRING "LPBQ deploy-v1 HMX drain A/B: unroll full row-pair accumulate helpers for rows=1")
set(HTP_LPBQ_SCALED_K128_GROUPS "0" CACHE STRING "LPBQ deploy-v1 A/B: enable scaled after.uh K-group nibble drains")
set(HTP_LPBQ_SCALED_K_GROUP_TILES "1" CACHE STRING "LPBQ deploy-v1 A/B: scaled-K K32 tile count, e.g. 2=K64, 4=K128")
set(HTP_LPBQ_R4_SCALED_K_GROUPS "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: allow R4/down_proj to use scaled-K nibble drains")
set(HTP_LPBQ_EXACT_K64_SAFE_GROUPS "0" CACHE STRING "LPBQ deploy-v1 A/B: exact unscaled K64 drains when packed weights are after.uh-safe")
set(HTP_LPBQ_EXACT_K_SAFE_GROUP_TILES "2" CACHE STRING "LPBQ deploy-v1 A/B: exact safe group tile count; 2 preserves K64, 4 tests dynamic K128")
set(HTP_LPBQ_R4_EXACT_K64_SAFE_GROUPS "0" CACHE STRING "LPBQ deploy-v1 R4 A/B: allow R4/down_proj to use exact unscaled K64-safe drains")
set(HTP_LPBQ_EXACT_K64_NON_R4_GROUPS "0" CACHE STRING "LPBQ deploy-v1 Q/K correctness A/B: allow non-R4 layers to use exact unscaled K64 drains")
# LPBQ deploy-v1 exact fallback correctness: old default 2184 was 32767/15.
# Exact drains issue q4<<1 at HMX scale=1.0, so the safe bound is 32767/30.
set(HTP_LPBQ_EXACT_K64_SAFE_ABS_LIMIT "1092" CACHE STRING "LPBQ deploy-v1 A/B: max sum(abs(w)) per output column for exact K64 grouping")
set(HTP_LPBQ_EXACT_K64_DEBUG_STOP_STAGE "0" CACHE STRING "LPBQ deploy-v1 standalone-only exact-K64 fail-fast stage for timeout isolation")
set(HTP_LPBQ_MODE166_STOP_AFTER "0" CACHE STRING "LPBQ deploy-v1 standalone-only mode166 timeout isolator: 0=full probe, 1..7 return after that substage")
set(HTP_LPBQ_MODE166_V6_ONE_STOP_STAGE "0" CACHE STRING "LPBQ deploy-v1 standalone-only mode166 v6_one isolator: 0=full, 1=after fill, 2=after HMX finish, 3=after accumulate")
set(HTP_LPBQ_EXACT_K64_MIN_SAFE_PERCENT "50" CACHE STRING "LPBQ deploy-v1 A/B: minimum K64-safe sidecar coverage before enabling exact-K64")
set(HTP_LPBQ_EXACT_K64_ACQUIRE_PER_GROUP "0" CACHE STRING "LPBQ deploy-v1 A/B: hold HMX across N tiles for one exact-K64 group")
set(HTP_LPBQ_EXACT_K64_ISSUE_ACC_OVERLAP "0" CACHE STRING "LPBQ deploy-v1 A/B: issue hi4 before HVX lo4 accumulate in exact-K64 safe groups")
set(HTP_LPBQ_EXACT_K64_FULL_WEIGHT_STREAM "0" CACHE STRING "LPBQ deploy-v1 A/B: expand exact-K64 compact K4 weights into full HMX tiles and issue one long stream")
set(HTP_LPBQ_EXACT_K64_PAIR_PREACCUM "0" CACHE STRING "LPBQ deploy-v1 A/B: pair sidecar-safe exact-K64 after.uh drains before HVX unpack/accumulate")
set(HTP_LPBQ_EXACT_K64_REQUIRE_ALL_SAFE "1" CACHE STRING "LPBQ deploy-v1 correctness guard: enter exact-K64 only when the K64 safe table has no unsafe groups")
set(HTP_LPBQ_EXACT_K64_FORCE_UNSAFE "0" CACHE STRING "LPBQ deploy-v1 diagnostic: force real-layer K64 safe table to unsafe so exact branch exercises only K32/K16 fallback")
set(HTP_LPBQ_HMX_CLEAR_OUT_BEFORE_STORE "1" CACHE STRING "LPBQ deploy-v1 drain A/B: clear HMX after.uh output tile before accumulator store")
set(HTP_LPBQ_HMX_CLEAR_ACTIVE_ROWS_ONLY "1" CACHE STRING "LPBQ deploy-v1 drain A/B: clear only active HMX row-pairs before after.uh store")
set(HTP_LPBQ_HMX_CVT_UH_DRAIN "0" CACHE STRING "LPBQ deploy-v1 drain A/B: use cvt.uh=acc + mxmem=cvt instead of direct after.uh store")
set(HTP_LPBQ_HMX_CVT_UH_SELECTOR2_DRAIN "0" CACHE STRING "LPBQ deploy-v1 drain A/B: use signed-preserving cvt.uh selector 2 instead of direct after.uh store")
set(HTP_LPBQ_FULL_U8_ACQUIRE_PER_M "0" CACHE STRING "LPBQ deploy-v1 narrow HMX lifecycle A/B: hold HMX across one full-U8 M block without enabling the broader R4/nibble path")
set(HTP_LPBQ_HMX_ACQUIRE_PER_M "0" CACHE STRING "LPBQ deploy-v1 bottleneck-first A/B: hold HMX across one R4/nibble M block instead of one K32 tile")
set(HTP_LPBQ_IMMUTABLE_SIDECAR_CACHE "0" CACHE STRING "LPBQ deploy-v1 A/B: skip per-op DSP cache invalidates for read-only sidecar constants")
set(HTP_LPBQ_ROUNDHALF_FOLDED_STORE_FAST "0" CACHE STRING "LPBQ deploy-v1 direct after.uh epilogue A/B: specialize folded roundhalf store when raw_adjust=0")
set(HTP_LPBQ_ROUNDHALF_FOLDED_RAWADJ_BIAS_FAST "0" CACHE STRING "LPBQ deploy-v1 direct after.uh epilogue A/B: fold raw_adjust*out_scale into bias_eff for folded roundhalf store")

# LPBQ deploy-v1 build plumbing: the SDK build_cmake wrapper is inconsistent
# about forwarding arbitrary cache args. Keep cache defaults above as rollback
# defaults, but allow the artifact-local build script to override HTP_* knobs
# through same-name environment variables during a fresh configure.
get_cmake_property(_lpbq_cache_vars CACHE_VARIABLES)
foreach(_lpbq_var IN LISTS _lpbq_cache_vars)
    if(_lpbq_var MATCHES "^HTP_" AND DEFINED ENV{${_lpbq_var}})
        set(${_lpbq_var} "$ENV{${_lpbq_var}}" CACHE STRING "LPBQ env override for ${_lpbq_var}" FORCE)
    endif()
endforeach()

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()
set(ignoreVar ${DEPENDENCIES}${V})

# NOTE(hzx): QAIC generated code are located in ${CMAKE_CURRENT_BINARY_DIR}
set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_BINARY_DIR}
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)
include_directories(${common_incs})

if(${OS_TYPE} MATCHES "HLOS")
    # HLOS dynamic binary
    add_library(htp_ops SHARED 
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/op_export.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/session.c
    )

    # Build interface files
    build_idl(include/htp_ops.idl htp_ops)

    # HLOS executable
    add_executable(htp_ops_test
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/test.c
    )
    
    # Add common HLOS-specific linker options
    set_common_compile_and_link_options(htp_ops_test)

    target_compile_definitions(htp_ops_test PUBLIC VERIFY_PRINT_ERROR)
    if(${CMAKE_SYSTEM_NAME} MATCHES "UbuntuARM")
        target_link_options(htp_ops_test PUBLIC -lpthread -ldl)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        target_link_options(htp_ops_test PUBLIC -llog -ldl)
    endif()

    # Find the DSP from provided domain value, default is CDSP
    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(htp_ops ${dsprpc})

    add_dependencies(htp_ops_test htp_ops)
    target_link_libraries(htp_ops_test htp_ops)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops_test htp_ops)

else()

    # Shared skel library
    set(htp_ops_skel_sources
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_skel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/commu.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/power.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/hmx_mgr.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/worker_pool.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/mmap_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/op_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/op_tests.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/vtcm_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/flash_attn_sp_hdim.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/flash_attn.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/mat_mul.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/precompute_table.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/rms_norm.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/mm_benchmark.c
    )
    if(HTP_LPBQ_HMX_OPCODE_CORPUS)
        list(APPEND htp_ops_skel_sources
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/probes/hmx_opcode_corpus.c
        )
    endif()
    add_library(htp_ops_skel SHARED ${htp_ops_skel_sources})
    build_idl(include/htp_ops.idl htp_ops_skel)

    # Enable HMX inline assembly
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mhmx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mhmx")
    # mat_mul.c owns the pure-FP16 hot loops (FP32/FP16 conversion, VTCM weight
    # publish, and output store). Keep the stronger optimization scoped here so
    # unrelated DSP ops retain the SDK's default ReleaseG flags.
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/mat_mul.c PROPERTIES COMPILE_OPTIONS "-O3")
    target_compile_definitions(htp_ops_skel PRIVATE
        FIGURE8_ENABLE_LUT_EXP=$<IF:$<BOOL:${FIGURE8_ENABLE_LUT_EXP}>,1,0>
        FIGURE8_ENABLE_PROFILE_TIMERS=$<IF:$<BOOL:${FIGURE8_ENABLE_PROFILE_TIMERS}>,1,0>
        HTP_ENABLE_OUT_STATIONARY_PREFILL=$<IF:$<BOOL:${HTP_ENABLE_OUT_STATIONARY_PREFILL}>,1,0>
        HTP_OS_DEBUG_COMPARE=$<IF:$<BOOL:${HTP_OS_DEBUG_COMPARE}>,1,0>
        HTP_MATMUL_PIPELINE_MODE=${HTP_MATMUL_PIPELINE_MODE}
        HTP_MATMUL_WEIGHT_AREA_KB=${HTP_MATMUL_WEIGHT_AREA_KB}
        HTP_MATMUL_ACTIVATION_AREA_KB=${HTP_MATMUL_ACTIVATION_AREA_KB}
        HTP_MATMUL_OUTPUT_AREA_KB=${HTP_MATMUL_OUTPUT_AREA_KB}
        HTP_MATMUL_SCRATCH_AREA_KB=${HTP_MATMUL_SCRATCH_AREA_KB}
        HTP_MATMUL_PIPELINE_OUTPUT_KB=${HTP_MATMUL_PIPELINE_OUTPUT_KB}
        HTP_FP16_WEIGHT_LOAD_MODE=${HTP_FP16_WEIGHT_LOAD_MODE}
        HTP_FP16_DMA_DST_BYPASS=${HTP_FP16_DMA_DST_BYPASS}
        HTP_FP16_DIRECT_FINAL_DMA=${HTP_FP16_DIRECT_FINAL_DMA}
        HTP_FP16_DIRECT_FINAL_PIPELINE=${HTP_FP16_DIRECT_FINAL_PIPELINE}
        HTP_FP16_DIRECT_FINAL_CACHE_INVALIDATE=${HTP_FP16_DIRECT_FINAL_CACHE_INVALIDATE}
        HTP_FP16_DIRECT_FINAL_TOUCH=${HTP_FP16_DIRECT_FINAL_TOUCH}
        HTP_FP16_DECODE_DIRECT_PIPELINE=${HTP_FP16_DECODE_DIRECT_PIPELINE}
        HTP_FP16_DECODE_PIPELINE_MAX_M=${HTP_FP16_DECODE_PIPELINE_MAX_M}
        HTP_FP16_PARALLEL_WEIGHT_PUBLISH=${HTP_FP16_PARALLEL_WEIGHT_PUBLISH}
        HTP_FP16_PUBLISH_USE_MEMCPY=${HTP_FP16_PUBLISH_USE_MEMCPY}
        HTP_FP16_PARALLEL_OUTPUT_STORE=${HTP_FP16_PARALLEL_OUTPUT_STORE}
        HTP_FP16_SMALL_M_VTCM_ENABLE=${HTP_FP16_SMALL_M_VTCM_ENABLE}
        HTP_FP16_SMALL_M_THRESHOLD=${HTP_FP16_SMALL_M_THRESHOLD}
        HTP_FP16_SMALL_M_WEIGHT_AREA_KB=${HTP_FP16_SMALL_M_WEIGHT_AREA_KB}
        HTP_FP16_SMALL_M_ACTIVATION_AREA_KB=${HTP_FP16_SMALL_M_ACTIVATION_AREA_KB}
        HTP_FP16_SMALL_M_OUTPUT_AREA_KB=${HTP_FP16_SMALL_M_OUTPUT_AREA_KB}
        HTP_FP16_SMALL_M_SCRATCH_AREA_KB=${HTP_FP16_SMALL_M_SCRATCH_AREA_KB}
        HTP_FP16_SMALL_M_PIPELINE_OUTPUT_KB=${HTP_FP16_SMALL_M_PIPELINE_OUTPUT_KB}
        HTP_FP16_OS_M_BLOCK_SIZE=${HTP_FP16_OS_M_BLOCK_SIZE}
        HTP_FP16_OS_N_BLOCK_SIZE=${HTP_FP16_OS_N_BLOCK_SIZE}
        HTP_FP16_OS_K_BLOCK_SIZE=${HTP_FP16_OS_K_BLOCK_SIZE}
        HTP_FP16_OS_WEIGHT_GATHER_DMA=${HTP_FP16_OS_WEIGHT_GATHER_DMA}
        HTP_FP16_OS_WEIGHT_GATHER_DMA_SCRATCH=${HTP_FP16_OS_WEIGHT_GATHER_DMA_SCRATCH}
        HTP_FP16_OS_WEIGHT_GATHER_DMA_PREFILL_ONLY=${HTP_FP16_OS_WEIGHT_GATHER_DMA_PREFILL_ONLY}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6=${HTP_LPBQ_R4_FULL_U8_STREAM_V6}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_TILES=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_TILES}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_SMALL_M_MAX=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_SMALL_M_MAX}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_MIN_M=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_MIN_M}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_VALUE=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_VALUE}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_ALT=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_ALT}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_ALT_EXTRA=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ROUND_ALT_EXTRA}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_ROUND_VALUE=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_ROUND_VALUE}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_ROUND_VALUE_ROWS_GT1=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_ROUND_VALUE_ROWS_GT1}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_APPLY_257=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_APPLY_257}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_NEG_EXTRA=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_NEG_EXTRA}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_RSHIFT=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_RSHIFT}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_RSHIFT_ROWS_GT1=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_RSHIFT_ROWS_GT1}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_EXTRA_RSHIFT=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_EXTRA_RSHIFT}
        HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_EXTRA_RSHIFT_ROWS_GT1=${HTP_LPBQ_R4_FULL_U8_SINGLEPASS_RECOVER_CORR_EXTRA_RSHIFT_ROWS_GT1}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_SIGN_COMPENSATE=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_SIGN_COMPENSATE}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_EXACT_RECOVER=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_EXACT_RECOVER}
        HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_EXACT_RECOVER=${HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_EXACT_RECOVER}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_SCALE=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_SCALE}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_RAW_ADJUST=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_RAW_ADJUST}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_ROUND_BIAS_BITS=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_ADAPTIVE_ROUND_BIAS_BITS}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_SCALE_BITS_OVERRIDE=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_SCALE_BITS_OVERRIDE}
        HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_RECOVER_SHIFT_OVERRIDE=${HTP_LPBQ_R4_FULL_U8_STREAM_V6_GROUP_RECOVER_SHIFT_OVERRIDE}
        HTP_LPBQ_R4_FULL_U8_K16_GROUPS=${HTP_LPBQ_R4_FULL_U8_K16_GROUPS}
        HTP_LPBQ_R4_FULL_U8_K32_GROUPS=${HTP_LPBQ_R4_FULL_U8_K32_GROUPS}
        HTP_LPBQ_R4_FULL_U8_K32_SCALE_BITS_OVERRIDE=${HTP_LPBQ_R4_FULL_U8_K32_SCALE_BITS_OVERRIDE}
        HTP_LPBQ_R4_FULL_U8_K32_RECOVER_SHIFT_OVERRIDE=${HTP_LPBQ_R4_FULL_U8_K32_RECOVER_SHIFT_OVERRIDE}
        HTP_LPBQ_FULL_U8_PRECOMPENSATE_HMX_SCALE=${HTP_LPBQ_FULL_U8_PRECOMPENSATE_HMX_SCALE}
        HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_TILES=${HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_TILES}
        HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_DELAY_RECOVER=${HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_DELAY_RECOVER}
        HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_COMPACT_WEIGHT=${HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_COMPACT_WEIGHT}
        HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP=${HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP}
        HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_SMALL_M_ONLY=${HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_SMALL_M_ONLY}
        HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_PREFILL_ONLY=${HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_PREFILL_ONLY}
        HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_ACC_FINAL_ISSUE=${HTP_LPBQ_FULL_U8_STREAM_V6_GROUP_N_TILE_OVERLAP_ACC_FINAL_ISSUE}
        HTP_LPBQ_FULL_U8_V6_COPY_1024_LOOP8=${HTP_LPBQ_FULL_U8_V6_COPY_1024_LOOP8}
        HTP_LPBQ_FULL_U8_V6_COPY_BULK_LOOP8=${HTP_LPBQ_FULL_U8_V6_COPY_BULK_LOOP8}
        HTP_LPBQ_SKIP_UNUSED_FULL_U8_HI_ACT_CLEAR=${HTP_LPBQ_SKIP_UNUSED_FULL_U8_HI_ACT_CLEAR}
        HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH=${HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH}
        HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH_BYTES=${HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH_BYTES}
        HTP_LPBQ_KMAJOR_WEIGHT_GROUP_COPY=${HTP_LPBQ_KMAJOR_WEIGHT_GROUP_COPY}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_LOAD=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_LOAD}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_DST_BYPASS=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_DST_BYPASS}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_SRC_BYPASS=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_SRC_BYPASS}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_PREISSUE_WAIT=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_PREISSUE_WAIT}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_VISIBILITY_SYNC=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_VISIBILITY_SYNC}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_OVERLAP_ACT=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_OVERLAP_ACT}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_DOUBLE_BUFFER=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_DOUBLE_BUFFER}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_N_BLOCK_TILES=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_N_BLOCK_TILES}
        HTP_LPBQ_KMAJOR_WEIGHT_DMA_OVERLAP_ROWBLOCK=${HTP_LPBQ_KMAJOR_WEIGHT_DMA_OVERLAP_ROWBLOCK}
        HTP_LPBQ_NONR4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE=${HTP_LPBQ_NONR4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE}
        HTP_LPBQ_NONR4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE_BLOCKS=${HTP_LPBQ_NONR4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE_BLOCKS}
        HTP_LPBQ_NONR4_GROUPED_V6_NBLOCK_ROWBLOCK_REUSE=${HTP_LPBQ_NONR4_GROUPED_V6_NBLOCK_ROWBLOCK_REUSE}
        HTP_LPBQ_NONR4_GROUPED_V6_NBLOCK_ROWBLOCK_REUSE_TILES=${HTP_LPBQ_NONR4_GROUPED_V6_NBLOCK_ROWBLOCK_REUSE_TILES}
        HTP_LPBQ_FFNDOWN_DECODE_G32_STAGING_AB=${HTP_LPBQ_FFNDOWN_DECODE_G32_STAGING_AB}
        HTP_LPBQ_FFNGATEUP_DECODE_G32_STAGING_AB=${HTP_LPBQ_FFNGATEUP_DECODE_G32_STAGING_AB}
        HTP_LPBQ_WEIGHT_PUBLISH_TRACE_SPLIT=${HTP_LPBQ_WEIGHT_PUBLISH_TRACE_SPLIT}
        HTP_LPBQ_WEIGHT_PATH_TRACE_SPLIT=${HTP_LPBQ_WEIGHT_PATH_TRACE_SPLIT}
        HTP_LPBQ_TRACE_W0_EXPERT_BUCKETS=${HTP_LPBQ_TRACE_W0_EXPERT_BUCKETS}
        HTP_LPBQ_R4_FWHT_V2=${HTP_LPBQ_R4_FWHT_V2}
        HTP_LPBQ_R4_FWHT_DIRECT_V6_STORE=${HTP_LPBQ_R4_FWHT_DIRECT_V6_STORE}
        HTP_LPBQ_R4_FWHT_BLOCK=${HTP_LPBQ_R4_FWHT_BLOCK}
        HTP_LPBQ_R4_FWHT_SCALE_GRANULARITY=${HTP_LPBQ_R4_FWHT_SCALE_GRANULARITY}
        HTP_LPBQ_ACT_ROWS1_DIRECT_V6=${HTP_LPBQ_ACT_ROWS1_DIRECT_V6}
        HTP_LPBQ_ACT_PACKET_REUSE=${HTP_LPBQ_ACT_PACKET_REUSE}
        HTP_LPBQ_ACT_STATIC_SCALE_NONR4=${HTP_LPBQ_ACT_STATIC_SCALE_NONR4}
        HTP_LPBQ_HMX_CONVERT_AFFINE_BENCH=${HTP_LPBQ_HMX_CONVERT_AFFINE_BENCH}
        HTP_LPBQ_HMX_REUSE_SCALE_PAYLOAD_PER_NTILE=${HTP_LPBQ_HMX_REUSE_SCALE_PAYLOAD_PER_NTILE}
        HTP_LPBQ_HMX_HOLD_ACQUIRE_PER_ROWBLOCK=${HTP_LPBQ_HMX_HOLD_ACQUIRE_PER_ROWBLOCK}
        HTP_LPBQ_HMX_BATCH_ISSUE_KGROUPS=${HTP_LPBQ_HMX_BATCH_ISSUE_KGROUPS}
        HTP_LPBQ_HMX_DOUBLE_BUFFER_DRAIN=${HTP_LPBQ_HMX_DOUBLE_BUFFER_DRAIN}
        HTP_LPBQ_HMX_EPILOGUE_LAG_ONE_TILE=${HTP_LPBQ_HMX_EPILOGUE_LAG_ONE_TILE}
        HTP_LPBQ_HMX_OPCODE_CORPUS=${HTP_LPBQ_HMX_OPCODE_CORPUS}
        HTP_LPBQ_HMX_AFTER_ISSUE_HVX_PROBE=${HTP_LPBQ_HMX_AFTER_ISSUE_HVX_PROBE}
        HTP_LPBQ_HMX_AFTER_ISSUE_HVX_PROBE_ITERS=${HTP_LPBQ_HMX_AFTER_ISSUE_HVX_PROBE_ITERS}
        HTP_LPBQ_FULL_V6_WEIGHT_DMA_LOAD=${HTP_LPBQ_FULL_V6_WEIGHT_DMA_LOAD}
        HTP_LPBQ_FULL_V6_WEIGHT_DMA_SCRATCH=${HTP_LPBQ_FULL_V6_WEIGHT_DMA_SCRATCH}
        HTP_LPBQ_FULL_V6_WEIGHT_DMA_DST_BYPASS=${HTP_LPBQ_FULL_V6_WEIGHT_DMA_DST_BYPASS}
        HTP_LPBQ_FULL_V6_WEIGHT_DMA_OVERLAP_ACT=${HTP_LPBQ_FULL_V6_WEIGHT_DMA_OVERLAP_ACT}
        HTP_LPBQ_FULL_V6_WEIGHT_DIRECT_MXMEM_LOAD=${HTP_LPBQ_FULL_V6_WEIGHT_DIRECT_MXMEM_LOAD}
        HTP_LPBQ_FULL_V6_WEIGHT_PUBLISH_PINGPONG=${HTP_LPBQ_FULL_V6_WEIGHT_PUBLISH_PINGPONG}
        HTP_LPBQ_FULL_V6_WEIGHT_PUBLISH_PINGPONG_SMALL_M_MAX=${HTP_LPBQ_FULL_V6_WEIGHT_PUBLISH_PINGPONG_SMALL_M_MAX}
        HTP_LPBQ_FULL_V6_WEIGHT_PARALLEL_PUBLISH=${HTP_LPBQ_FULL_V6_WEIGHT_PARALLEL_PUBLISH}
        HTP_LPBQ_FULL_V6_WEIGHT_ASYNC_PREPUBLISH_ACT=${HTP_LPBQ_FULL_V6_WEIGHT_ASYNC_PREPUBLISH_ACT}
        HTP_LPBQ_FULL_V6_MACRO_KGROUP_PIPELINE=${HTP_LPBQ_FULL_V6_MACRO_KGROUP_PIPELINE}
        HTP_LPBQ_ACTIVATION_HMX_CACHE=${HTP_LPBQ_ACTIVATION_HMX_CACHE}
        HTP_LPBQ_ACTIVATION_HMX_CACHE_MAX_KB=${HTP_LPBQ_ACTIVATION_HMX_CACHE_MAX_KB}
        HTP_LPBQ_ACTIVATION_HVX_QUANT=${HTP_LPBQ_ACTIVATION_HVX_QUANT}
        HTP_LPBQ_ACTIVATION_FP16_STAGED_QUANT=${HTP_LPBQ_ACTIVATION_FP16_STAGED_QUANT}
        HTP_LPBQ_DISABLE_ONLINE_INPUT_SCALE=${HTP_LPBQ_DISABLE_ONLINE_INPUT_SCALE}
        HTP_LPBQ_ACTIVATION_HVX_VECTOR_ROUND_STORE=${HTP_LPBQ_ACTIVATION_HVX_VECTOR_ROUND_STORE}
        HTP_LPBQ_ACTIVATION_HVX_FLOOR_ROUND_STORE=${HTP_LPBQ_ACTIVATION_HVX_FLOOR_ROUND_STORE}
        HTP_LPBQ_R4_PARALLEL_PREFILL=${HTP_LPBQ_R4_PARALLEL_PREFILL}
        HTP_LPBQ_R4_CACHED_X4_KK_OUTER=${HTP_LPBQ_R4_CACHED_X4_KK_OUTER}
        HTP_LPBQ_R4_HVX_DOT=${HTP_LPBQ_R4_HVX_DOT}
        HTP_LPBQ_R4_HVX_DOT16=${HTP_LPBQ_R4_HVX_DOT16}
        HTP_LPBQ_R4_DOT4_ROWPAIR=${HTP_LPBQ_R4_DOT4_ROWPAIR}
        HTP_LPBQ_R4_FP16_HMX_ROTATE=${HTP_LPBQ_R4_FP16_HMX_ROTATE}
        HTP_LPBQ_R4_PREFILL_HMX_DENSE=${HTP_LPBQ_R4_PREFILL_HMX_DENSE}
        HTP_LPBQ_R4_PREFILL_HMX_MIN_M=${HTP_LPBQ_R4_PREFILL_HMX_MIN_M}
        HTP_LPBQ_R4_HMX_DENSE_BLOCK=${HTP_LPBQ_R4_HMX_DENSE_BLOCK}
        HTP_LPBQ_R4_HMX_HIGHBYTE_DIRECT_V6_PROBE=${HTP_LPBQ_R4_HMX_HIGHBYTE_DIRECT_V6_PROBE}
        HTP_LPBQ_R4_HMX_HFSCALE_DIRECT_V6_PROBE=${HTP_LPBQ_R4_HMX_HFSCALE_DIRECT_V6_PROBE}
        HTP_LPBQ_R4_HMX_HFSCALE_VTCM_WEIGHT_CACHE_PROBE=${HTP_LPBQ_R4_HMX_HFSCALE_VTCM_WEIGHT_CACHE_PROBE}
        HTP_LPBQ_R4_HMX_HFSCALE_VTCM_WEIGHT_CACHE_GROUPS=${HTP_LPBQ_R4_HMX_HFSCALE_VTCM_WEIGHT_CACHE_GROUPS}
        HTP_LPBQ_R4_V6_K128_BULK=${HTP_LPBQ_R4_V6_K128_BULK}
        HTP_LPBQ_R4_V6_K128_PARALLEL_ROWS=${HTP_LPBQ_R4_V6_K128_PARALLEL_ROWS}
        HTP_LPBQ_R4_V6_K128_SKIP_INNER_CLEAR=${HTP_LPBQ_R4_V6_K128_SKIP_INNER_CLEAR}
        HTP_LPBQ_R4_V6_K128_BULK_SMALL_M=${HTP_LPBQ_R4_V6_K128_BULK_SMALL_M}
        HTP_LPBQ_R4_SIGN_FHT_FAST=${HTP_LPBQ_R4_SIGN_FHT_FAST}
        HTP_LPBQ_R4_STANDARD_FHT_FORCE=${HTP_LPBQ_R4_STANDARD_FHT_FORCE}
        HTP_LPBQ_R4_STANDARD_FHT_HMX_PROBE=${HTP_LPBQ_R4_STANDARD_FHT_HMX_PROBE}
        HTP_LPBQ_R4_STANDARD_FHT_HMX_INTEGRATED=${HTP_LPBQ_R4_STANDARD_FHT_HMX_INTEGRATED}
        HTP_LPBQ_R4_STANDARD_FHT_HVX_PROBE=${HTP_LPBQ_R4_STANDARD_FHT_HVX_PROBE}
        HTP_LPBQ_R4_STANDARD_FHT_HVX_INTEGRATED=${HTP_LPBQ_R4_STANDARD_FHT_HVX_INTEGRATED}
        HTP_LPBQ_R4_STANDARD_FHT_HVX_NIBBLE_PROBE=${HTP_LPBQ_R4_STANDARD_FHT_HVX_NIBBLE_PROBE}
        HTP_LPBQ_R4_STANDARD_FHT_HVX_NIBBLE_INTEGRATED=${HTP_LPBQ_R4_STANDARD_FHT_HVX_NIBBLE_INTEGRATED}
        HTP_LPBQ_R4_STANDARD_FHT_CORE_PROBE=${HTP_LPBQ_R4_STANDARD_FHT_CORE_PROBE}
        HTP_LPBQ_R4_STANDARD_FHT_ROWPAIR_DIRECT=${HTP_LPBQ_R4_STANDARD_FHT_ROWPAIR_DIRECT}
        HTP_LPBQ_R4_STANDARD_FHT_I16_DIRECT=${HTP_LPBQ_R4_STANDARD_FHT_I16_DIRECT}
        HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_PROBE=${HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_PROBE}
        HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_INTEGRATED=${HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_INTEGRATED}
        HTP_LPBQ_R4_STANDARD_FHT_FP16_DIRECT=${HTP_LPBQ_R4_STANDARD_FHT_FP16_DIRECT}
        HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_NIBBLE_PROBE=${HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_NIBBLE_PROBE}
        HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_NIBBLE_INTEGRATED=${HTP_LPBQ_R4_STANDARD_FHT_HVX_FULL_NIBBLE_INTEGRATED}
        HTP_LPBQ_R4_SIGN_FHT_APPROX=${HTP_LPBQ_R4_SIGN_FHT_APPROX}
        HTP_LPBQ_R4_SIGN_FHT_COLSCALE_APPROX=${HTP_LPBQ_R4_SIGN_FHT_COLSCALE_APPROX}
        HTP_LPBQ_R4_PREFETCH_COLS8=${HTP_LPBQ_R4_PREFETCH_COLS8}
        HTP_LPBQ_R4_PREFETCH_K128_ROWPAR_COLS8=${HTP_LPBQ_R4_PREFETCH_K128_ROWPAR_COLS8}
        HTP_LPBQ_R4_QUANT_STORE8_FAST=${HTP_LPBQ_R4_QUANT_STORE8_FAST}
        HTP_LPBQ_R4_U8_V6_QUANT_STORE8_FAST=${HTP_LPBQ_R4_U8_V6_QUANT_STORE8_FAST}
        HTP_LPBQ_R4_ROWS1_REDUCE8_FAST=${HTP_LPBQ_R4_ROWS1_REDUCE8_FAST}
        HTP_LPBQ_R4_ROWS1_DOT4_SPLIT=${HTP_LPBQ_R4_ROWS1_DOT4_SPLIT}
        HTP_LPBQ_R4_DOT8_R128_UNROLL=${HTP_LPBQ_R4_DOT8_R128_UNROLL}
        HTP_LPBQ_R4_DOT_PACK_PRECOMPUTE_ROW_STORE=${HTP_LPBQ_R4_DOT_PACK_PRECOMPUTE_ROW_STORE}
        HTP_LPBQ_R4_K128_DOT_PACK_PRECOMPUTE_ROW_STORE=${HTP_LPBQ_R4_K128_DOT_PACK_PRECOMPUTE_ROW_STORE}
        HTP_LPBQ_R4_V6_K128_ROWGROUP4_DOT2=${HTP_LPBQ_R4_V6_K128_ROWGROUP4_DOT2}
        HTP_LPBQ_R4_V6_K128_BULK_ROWPAIR=${HTP_LPBQ_R4_V6_K128_BULK_ROWPAIR}
        HTP_LPBQ_R4_V6_K128_COLGROUP_PARALLEL=${HTP_LPBQ_R4_V6_K128_COLGROUP_PARALLEL}
        HTP_LPBQ_R4_V6_K128_WORKERS_MAX=${HTP_LPBQ_R4_V6_K128_WORKERS_MAX}
        HTP_LPBQ_R4_V6_K128_SKIP_ACC_ZERO=${HTP_LPBQ_R4_V6_K128_SKIP_ACC_ZERO}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_R128_UNROLL=${HTP_LPBQ_R4_V6_K128_ROWPAIR_R128_UNROLL}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_DOT8=${HTP_LPBQ_R4_V6_K128_ROWPAIR_DOT8}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_REDUCE8=${HTP_LPBQ_R4_V6_K128_ROWPAIR_REDUCE8}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_STORE2_FAST=${HTP_LPBQ_R4_V6_K128_ROWPAIR_STORE2_FAST}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_FUSED_STORE4=${HTP_LPBQ_R4_V6_K128_ROWPAIR_FUSED_STORE4}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_FUSED_STORE4_R128_UNROLL=${HTP_LPBQ_R4_V6_K128_ROWPAIR_FUSED_STORE4_R128_UNROLL}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_ALIGNED_LOADS=${HTP_LPBQ_R4_V6_K128_ROWPAIR_ALIGNED_LOADS}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_PREBASE_STORE8=${HTP_LPBQ_R4_V6_K128_ROWPAIR_PREBASE_STORE8}
        HTP_LPBQ_R4_V6_K128_ROWPAIR_CONST_LANES=${HTP_LPBQ_R4_V6_K128_ROWPAIR_CONST_LANES}
        HTP_LPBQ_R4_SCALE_CACHE_R128_UNROLL=${HTP_LPBQ_R4_SCALE_CACHE_R128_UNROLL}
        HTP_LPBQ_R4_TRACE_DOT_PACK_SPLIT=${HTP_LPBQ_R4_TRACE_DOT_PACK_SPLIT}
        HTP_LPBQ_R4_NIBBLE_ISSUE_ACC_OVERLAP=${HTP_LPBQ_R4_NIBBLE_ISSUE_ACC_OVERLAP}
        HTP_LPBQ_R4_NIBBLE_ACTIVE_ROW_ACCUM=${HTP_LPBQ_R4_NIBBLE_ACTIVE_ROW_ACCUM}
        HTP_LPBQ_R4_NIBBLE_RAW_ACCUM=${HTP_LPBQ_R4_NIBBLE_RAW_ACCUM}
        HTP_LPBQ_R4_NIBBLE_RAW_ACCUM_OVERLAP=${HTP_LPBQ_R4_NIBBLE_RAW_ACCUM_OVERLAP}
        HTP_LPBQ_R4_ACCUM_DECODE_UNROLL=${HTP_LPBQ_R4_ACCUM_DECODE_UNROLL}
        HTP_LPBQ_R4_STRUCTURED_FWHT=${HTP_LPBQ_R4_STRUCTURED_FWHT}
        HTP_LPBQ_R4_STRUCTURED_FWHT_STAGE_UNROLL=${HTP_LPBQ_R4_STRUCTURED_FWHT_STAGE_UNROLL}
        HTP_LPBQ_R4_STRUCTURED_FWHT_DIRECT_K128=${HTP_LPBQ_R4_STRUCTURED_FWHT_DIRECT_K128}
        HTP_LPBQ_ADAPTIVE_BUCKET_FAST_0_3=${HTP_LPBQ_ADAPTIVE_BUCKET_FAST_0_3}
        HTP_LPBQ_ADAPTIVE_BUCKET_FAST_0_3_ROWS1=${HTP_LPBQ_ADAPTIVE_BUCKET_FAST_0_3_ROWS1}
        HTP_LPBQ_ADAPTIVE_ACCUM_VARSHIFT_ROWS1=${HTP_LPBQ_ADAPTIVE_ACCUM_VARSHIFT_ROWS1}
        HTP_LPBQ_ADAPTIVE_SHIFT0_FAST_PATH=${HTP_LPBQ_ADAPTIVE_SHIFT0_FAST_PATH}
        HTP_LPBQ_ADAPTIVE_UNIFORM_SHIFT0_NOSHIFT=${HTP_LPBQ_ADAPTIVE_UNIFORM_SHIFT0_NOSHIFT}
        HTP_LPBQ_ADAPTIVE_SHIFT0_SINGLEPASS=${HTP_LPBQ_ADAPTIVE_SHIFT0_SINGLEPASS}
        HTP_LPBQ_ADAPTIVE_PROBE_SCALE_CACHE=${HTP_LPBQ_ADAPTIVE_PROBE_SCALE_CACHE}
        HTP_LPBQ_ADAPTIVE_PROBE_ONLY_SHIFT5_FAST=${HTP_LPBQ_ADAPTIVE_PROBE_ONLY_SHIFT5_FAST}
        HTP_LPBQ_ADAPTIVE_PROBE_SCALAR_SMALL_ROWS=${HTP_LPBQ_ADAPTIVE_PROBE_SCALAR_SMALL_ROWS}
        HTP_LPBQ_ADAPTIVE_RETRY_SKIP_INIT=${HTP_LPBQ_ADAPTIVE_RETRY_SKIP_INIT}
        HTP_LPBQ_ADAPTIVE_SCALE_LOW32_ONLY=${HTP_LPBQ_ADAPTIVE_SCALE_LOW32_ONLY}
        HTP_LPBQ_ADAPTIVE_SCALE_SKIP_UNCHANGED=${HTP_LPBQ_ADAPTIVE_SCALE_SKIP_UNCHANGED}
        HTP_LPBQ_ADAPTIVE_DERIVE_I32_MAX=${HTP_LPBQ_ADAPTIVE_DERIVE_I32_MAX}
        HTP_LPBQ_ADAPTIVE_SCALE_WORD_LUT=${HTP_LPBQ_ADAPTIVE_SCALE_WORD_LUT}
        HTP_LPBQ_ADAPTIVE_DERIVE_ROW0_EXACT_ROWS1=${HTP_LPBQ_ADAPTIVE_DERIVE_ROW0_EXACT_ROWS1}
        HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE=${HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE}
        HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE_BLOCKS=${HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_WEIGHT_REUSE_BLOCKS}
        HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_HMX_ACQUIRE_PER_KGROUP=${HTP_LPBQ_R4_GROUPED_V6_ROWBLOCK_HMX_ACQUIRE_PER_KGROUP}
        HTP_LPBQ_R4_ROWBLOCK_PROBE_ONLY_SHIFT5_FAST=${HTP_LPBQ_R4_ROWBLOCK_PROBE_ONLY_SHIFT5_FAST}
        HTP_LPBQ_R4_ROWBLOCK_SKIP_INITIAL_SCALE_LOAD=${HTP_LPBQ_R4_ROWBLOCK_SKIP_INITIAL_SCALE_LOAD}
        # LPBQ deploy-v1 maintenance: HTP_LPBQ_HVX_WEIGHT_COPY_PREFETCH_BYTES is
        # now listed beside the weight-copy knobs above; keep this breadcrumb so
        # future retunes do not accidentally add a second compile definition.
        HTP_LPBQ_HMX_FINE_SUBTRACE=${HTP_LPBQ_HMX_FINE_SUBTRACE}
        HTP_LPBQ_FULL_U8_ROWS1_ACCUM_PAIR_UNROLL=${HTP_LPBQ_FULL_U8_ROWS1_ACCUM_PAIR_UNROLL}
        HTP_LPBQ_SCALED_K128_GROUPS=${HTP_LPBQ_SCALED_K128_GROUPS}
        HTP_LPBQ_SCALED_K_GROUP_TILES=${HTP_LPBQ_SCALED_K_GROUP_TILES}
        HTP_LPBQ_R4_SCALED_K_GROUPS=${HTP_LPBQ_R4_SCALED_K_GROUPS}
        HTP_LPBQ_EXACT_K64_SAFE_GROUPS=${HTP_LPBQ_EXACT_K64_SAFE_GROUPS}
        HTP_LPBQ_EXACT_K_SAFE_GROUP_TILES=${HTP_LPBQ_EXACT_K_SAFE_GROUP_TILES}
        HTP_LPBQ_R4_EXACT_K64_SAFE_GROUPS=${HTP_LPBQ_R4_EXACT_K64_SAFE_GROUPS}
        HTP_LPBQ_EXACT_K64_NON_R4_GROUPS=${HTP_LPBQ_EXACT_K64_NON_R4_GROUPS}
        HTP_LPBQ_EXACT_K64_SAFE_ABS_LIMIT=${HTP_LPBQ_EXACT_K64_SAFE_ABS_LIMIT}
        HTP_LPBQ_EXACT_K64_DEBUG_STOP_STAGE=${HTP_LPBQ_EXACT_K64_DEBUG_STOP_STAGE}
        HTP_LPBQ_MODE166_STOP_AFTER=${HTP_LPBQ_MODE166_STOP_AFTER}
        HTP_LPBQ_MODE166_V6_ONE_STOP_STAGE=${HTP_LPBQ_MODE166_V6_ONE_STOP_STAGE}
        HTP_LPBQ_EXACT_K64_MIN_SAFE_PERCENT=${HTP_LPBQ_EXACT_K64_MIN_SAFE_PERCENT}
        HTP_LPBQ_EXACT_K64_ACQUIRE_PER_GROUP=${HTP_LPBQ_EXACT_K64_ACQUIRE_PER_GROUP}
        HTP_LPBQ_EXACT_K64_ISSUE_ACC_OVERLAP=${HTP_LPBQ_EXACT_K64_ISSUE_ACC_OVERLAP}
        HTP_LPBQ_EXACT_K64_FULL_WEIGHT_STREAM=${HTP_LPBQ_EXACT_K64_FULL_WEIGHT_STREAM}
        HTP_LPBQ_EXACT_K64_PAIR_PREACCUM=${HTP_LPBQ_EXACT_K64_PAIR_PREACCUM}
        HTP_LPBQ_EXACT_K64_REQUIRE_ALL_SAFE=${HTP_LPBQ_EXACT_K64_REQUIRE_ALL_SAFE}
        HTP_LPBQ_EXACT_K64_FORCE_UNSAFE=${HTP_LPBQ_EXACT_K64_FORCE_UNSAFE}
        HTP_LPBQ_HMX_CLEAR_OUT_BEFORE_STORE=${HTP_LPBQ_HMX_CLEAR_OUT_BEFORE_STORE}
        HTP_LPBQ_HMX_CLEAR_ACTIVE_ROWS_ONLY=${HTP_LPBQ_HMX_CLEAR_ACTIVE_ROWS_ONLY}
        HTP_LPBQ_HMX_CVT_UH_DRAIN=${HTP_LPBQ_HMX_CVT_UH_DRAIN}
        HTP_LPBQ_HMX_CVT_UH_SELECTOR2_DRAIN=${HTP_LPBQ_HMX_CVT_UH_SELECTOR2_DRAIN}
        HTP_LPBQ_FULL_U8_ACQUIRE_PER_M=${HTP_LPBQ_FULL_U8_ACQUIRE_PER_M}
        HTP_LPBQ_HMX_ACQUIRE_PER_M=${HTP_LPBQ_HMX_ACQUIRE_PER_M}
        HTP_LPBQ_IMMUTABLE_SIDECAR_CACHE=${HTP_LPBQ_IMMUTABLE_SIDECAR_CACHE}
        HTP_LPBQ_ROUNDHALF_FOLDED_STORE_FAST=${HTP_LPBQ_ROUNDHALF_FOLDED_STORE_FAST}
        HTP_LPBQ_ROUNDHALF_FOLDED_RAWADJ_BIAS_FAST=${HTP_LPBQ_ROUNDHALF_FOLDED_RAWADJ_BIAS_FAST}
    )

    # Link C++ libraries
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.so.1)
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.so.1)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops_skel)

endif()
