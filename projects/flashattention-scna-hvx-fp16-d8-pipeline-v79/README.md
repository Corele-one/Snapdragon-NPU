# v79 HVX static-d8 SCNA pipeline experiment

该项目是从 `flashattention-scna-fp16-d8-optimization-v79` 隔离出的实验副本。运行时 SCNA variant 始终保持 `pair_static_d8`，候选仅由编译期 `SCNA_KERNEL_IMPL` 区分，因此不会改变 Host/RPC mode 编码。

## 构建

```bash
./scripts/build_all_variants.sh
python3 tools/collect_static_metrics.py --project . --out-dir results/static_audit
python3 tools/evaluate_static_gates.py \
  --metrics results/static_audit/static_metrics.json \
  --json-out results/static_audit/static_gates.json \
  --csv-out results/static_audit/static_gates.csv
```

## 实验

先用缩短参数验证部署与解析：

```bash
./scripts/run_pipeline_experiment.sh --quick --run-id pipeline_quick
```

正式实验：

```bash
./scripts/run_pipeline_experiment.sh --run-id pipeline_formal
```

脚本支持断点续跑。正式输出位于 `results/runs/<RUN_ID>/`，包含原始日志、CSV/JSON、artifact manifest、反汇编、SVG/PDF/PNG 图和 `SCNA_HVX_D8_PIPELINE_V79_REPORT_ZH.md`。

## 研究完整性

报告只从当前 run 的原始日志生成；缺失数据标记为 `UNAVAILABLE`。这是算子级 seeded synthetic benchmark，模型/数据集为 N/A，不作端到端 LLM 性能声明。代码和报告在用于论文或学位材料前必须由研究者逐项复核，并遵守对应 AI 披露政策。
