# Figure 8 Perfetto Trace Notes

The `scna_lane8_q*.ntff` and `scna_lane8_q*.perfetto.json` files in this directory are Chrome Trace Event JSON files that Perfetto UI can open directly.

These traces are now generated from `FIG8_ATTENTION_EVENT` rows: every slice uses a DSP qtimer start timestamp and end timestamp captured around the corresponding attention-kernel section. This preserves cross-worker overlap and is not reconstructed from cumulative per-component totals.

They are still software-observed traces, not Qualcomm sysmon/ETM hardware traces, because this production device does not expose the NSP/CDSP profiling capability needed for true hardware-unit telemetry.

## Unit Mapping

- `Memory/L2 Load-Pack`: `q_load`, `k_load`, `v_load`. These are software load/pack windows into VTCM and L2-prefetch-assisted memory accesses, not DMA counters.
- `HMX Unit`: `qk_dot`.
- `HVX Unit`: `safe_sm` and nested `scna_exp`; online rescale SCNA events are nested in `core_acc`.
- `Mixed HVX/HMX Section`: `core_acc`, `o_scale`, because the Figure 8 component events wrap HVX state/scatter/setup work together with HMX P*V or scale-dot work.
- `Store/Writeback`: `o_store`.
- `Scalar Unit`: derived uninstrumented control gaps and instant notes only; not measured scalar hardware utilization

## Hardware Trace Blockers Observed

- This run deliberately records event-level DSP qtimer intervals only; PMU, ETM, sysmon, and hardware utilization counters were not collected.
- Accordingly every trace declares `hardware_trace=false` and must not be interpreted as a hardware-counter trace.

## Generated Files

- `scna_lane8_q4.ntff`
- `scna_lane8_q4.perfetto.json`
- `scna_lane8_q8.ntff`
- `scna_lane8_q8.perfetto.json`
- `scna_lane8_q16.ntff`
- `scna_lane8_q16.perfetto.json`
- `scna_lane8_q32.ntff`
- `scna_lane8_q32.perfetto.json`
- `scna_lane8_all_q.ntff`
- `scna_lane8_all_q.perfetto.json`
- `event_trace_summary.json`
