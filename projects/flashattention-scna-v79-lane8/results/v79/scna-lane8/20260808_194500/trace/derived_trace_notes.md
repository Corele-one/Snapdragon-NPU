# SCNA serial vs lane8 Perfetto trace notes

- `hardware_trace=false`: all slices come from DSP qtimer `t0/t1` software events; no PMU, ETM, sysmon, or utilization counters are present.
- Serial and lane8 are independent replays of the same binary, shape, seed, and iteration schedule; timestamp overlap does not mean concurrent execution.
- Per-layout/per-Qo traces and audit outputs are stored in the `serial/` and `lane8/` subdirectories.
- Drag `scna_serial_vs_lane8_all_q.perfetto.json` into https://ui.perfetto.dev to inspect the combined comparison.
