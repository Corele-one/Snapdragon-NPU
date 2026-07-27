# Figure 8 Perfetto Trace Notes

The `lut_exp_q*.ntff` and `lut_exp_q*.perfetto.json` files in this directory are Chrome Trace Event JSON files that Perfetto UI can open directly.

These traces are now generated from `FIG8_ATTENTION_EVENT` rows: every slice uses a DSP qtimer start timestamp and end timestamp captured around the corresponding attention-kernel section. This preserves cross-worker overlap and is not reconstructed from cumulative per-component totals.

They are still software-observed traces, not Qualcomm sysmon/ETM hardware traces, because this production device does not expose the NSP/CDSP profiling capability needed for true hardware-unit telemetry.

## Unit Mapping

- `Memory/L2 Load-Pack`: `q_load`, `k_load`, `v_load`. These are software load/pack windows into VTCM and L2-prefetch-assisted memory accesses, not DMA counters.
- `HMX Unit`: `qk_dot`.
- `HVX Unit`: `safe_sm`.
- `Mixed HVX/HMX Section`: `core_acc`, `o_scale`, because the Figure 8 component events wrap HVX state/scatter/setup work together with HMX P*V or scale-dot work.
- `Store/Writeback`: `o_store`.
- `Scalar Unit`: derived uninstrumented control gaps and instant notes only; not measured scalar hardware utilization

## Hardware Trace Blockers Observed

- `qprof --capabilities` exposes apps CPU/GPU/thread metrics only; no `profiler:nsp-dsp-metrics` or `profiler:cdsp-dsp-metrics`.
- `qprof --profile ... profiler:nsp-dsp-metrics` reports: Platform Software support for NPU0 metrics is not available on this product.
- target `qprof --sysmon getinfo/profile/TLP --q6 npu0` reports npu0 metrics/stats support is not available.
- `adb root` reports `adbd cannot run as root in production builds`.
- HexTA ETM setup needs writes to `/sys/bus/coresight/...`; shell user gets permission denied.

## Generated Files

- `lut_exp_q4.ntff`
- `lut_exp_q4.perfetto.json`
- `lut_exp_q8.ntff`
- `lut_exp_q8.perfetto.json`
- `lut_exp_q16.ntff`
- `lut_exp_q16.perfetto.json`
- `lut_exp_q32.ntff`
- `lut_exp_q32.perfetto.json`
- `lut_exp_all_q.ntff`
- `lut_exp_all_q.perfetto.json`
- `event_trace_summary.json`
