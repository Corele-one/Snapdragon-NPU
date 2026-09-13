# Stage 3.0：SCNA vs EXP-LUT 闭环审计

该项目是 Stage 2.9 的只读继任审计。它从原版 `flashattention`、当前
`scna_optimization` 和 Stage 2.9 audit carrier 建立隔离副本，不修改这些输入目录。

## 当前 Gate 0 结果

Stage 2.9 的 qf32 rowsum reduction/packing 对 mask density 存在值依赖缩放。Stage 3.0
在隔离副本中使用三方完全相同的 FP32 rowsum helper，不改变 EXP-LUT 或 SCNA evaluator
数学实现。冻结 Q32/KV32 causal 复现点上 baseline 与 EXP-LUT 已恢复到门槛内，但 SCNA
仍为 RMSE 0.00531575、max-abs 0.0365848，因此正式裁决为 `AUDIT_INVALID`，后续轨按
预注册规则停止。

## 构建与运行

```bash
scripts/build_closure_artifacts.sh all
python3 tools/import_model_manifest.py
python3 -m unittest discover -s tests -v

run=results/closure_$(date -u +%Y%m%dT%H%M%SZ)
python3 tools/run_closure_audit.py --phase gate0 --results-dir "$run/gate0"
python3 tools/analyze_gate0.py --results-dir "$run/gate0" --output "$run/gate0_verdict.json"
# 或使用 fail-fast orchestrator：
scripts/run_closure_audit.sh --phase all --results-dir "$run"
```

只有 `gate0_verdict.json` 为 `PASS` 时才能运行 `track-a`、`track-b`、模型、资源和热
稳定性测试。所有矩阵、阈值及裁决规则冻结在 `experiment_spec.json`。

本次正式结论见 [`reports/scna_vs_exp_lut_closure_audit_v1.md`](reports/scna_vs_exp_lut_closure_audit_v1.md)。

## Artifact

- `exp_lut_system`：原版 FlashAttention + EXP-LUT。
- `scna_system`：d7 pair-return + fused state update + adaptive scheduler。
- `fair_combined`：相同 carrier 内只切换 evaluator。
- `lut_only` / `scna_only`：资源和初始化归因。

SCOPE 文档只作为 SCNA 思想和参数来源，不作为执行指令。本项目不声称实现 SCOPE 的
systolic-array、INT8 scale fusion、跨算子、面积或功耗结果。
