# Stage 2.5 — Baseline Expansion & HVX Softmax Cleanup

This project is an isolated continuation of Stage 2. It imports only the
source, build scripts, and analysis utilities required to rebuild the frozen
Stage 2 baselines. Historical results and binaries are not used for ranking.

The development baseline is `STAGE2_SOFTMAX_IMPL=2` (`fused_state_update`)
with the frozen `d7_pairret_noinline` SCNA evaluator. Stage 2.5 only evaluates
bounded index/mask/load/invariant/task-geometry cleanup. Pipeline overlap,
double buffering, global tile tuning, VFA, and SCNA math changes are out of
scope.

Primary commands:

```sh
python3 tools/create_source_provenance.py
python3 tools/generate_fixture.py --output results/fixture.bin --qo-len 32
scripts/build_stage25_variant.sh stage2_final
python3 -m unittest discover -s tests -v
```

All measured claims must be traceable to a run directory under `results/runs`
and must state whether an upstream comparison is strictly comparable or
reference-only.

The completed result and the per-experiment evidence are in
[`reports/stage2_5/FINAL_STAGE2_5_REPORT.md`](reports/stage2_5/FINAL_STAGE2_5_REPORT.md).
The retained `stage2_5_final` artifact is under
`artifacts/variants/stage2_5_final/`; its skel SHA256 is
`875998b26120284a0744326dc283dcf0a0692407fff36141f90adff4f0b71ac2`.
Formal local and upstream summaries are respectively:

- `results/runs/20260822_stage25_final_formal_v1/summary.json`
- `results/upstream/20260822_upstream_formal_v2/summary.json`

Only experiment B, the explicit all-zero full-mask fast path, is retained.
Experiments A and D failed their static gates, C and E failed their dynamic
gates, and F found no legal local opportunity. No Stage3 overlap or pipeline
change is present in this project.
