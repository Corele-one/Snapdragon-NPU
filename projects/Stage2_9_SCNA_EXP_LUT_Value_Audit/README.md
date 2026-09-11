# Stage 2.9：SCNA vs EXP-LUT 综合价值审计

这是 Stage2.75 冻结实现的隔离审计项目。它不修改 Stage2.75，不重新拟合 SCNA，
不引入 DMA-HMX Pipeline 或 INT8 改动。

## 产物

- `fair_combined`：主性能、正确性和模型对比的唯一公平产物。
- `lut_only`：保留 LUT 分配/初始化，用于资源和冷启动归因。
- `scna_only`：移除 LUT 分配/初始化；不支持 LUT 请求且不会回退。

冻结配置为 v79、`pair_static_d8`、`d7_pairret_noinline`、
`fused_state_update` 和 Stage2.75 adaptive scheduler。预注册矩阵及裁决规则见
`experiment_spec.json`。

## Stage 2.9.1：模型链路修复

新增规范为 `model_repair_spec.json` 与 `audit_rerun_spec.json`；原
`experiment_spec.json` 和 `formal_20260910_gate2` 证据保持不变。修复流程先生成
CPU/HMX F16，再从各自 F16 输入用同一个 type 514 quantizer 生成配对模型。逐 tensor
验证器对普通 F16 tensor 做位级比较，对 HMX matmul tensor 做逆置换后的位级比较，
并对量化 tensor 输出反量化误差及各分支来源哈希。

通用 HMX matmul 审计覆盖 F16、IQ4_NL、Q8_0、Qwen 实际投影形状以及顺序、四阶段
pipeline 和 output-stationary 诊断路径。模型内诊断可通过
`LLAMA_NPU_MATMUL_AUDIT_TENSOR=<tensor substring|*>` 启用，只输出 HMX/scalar
对照 JSONL，不进行 CPU fallback。

## 运行

```bash
scripts/build_variants.sh --flavor all
scripts/build_llama_backend.sh
python3 -m unittest discover -s tests -v
python3 tools/build_model_pair.py --spec model_repair_spec.json
generated/model_pair_venv/bin/python tools/verify_model_pair.py --spec model_repair_spec.json
python3 tools/run_matmul_audit.py --spec model_repair_spec.json --results-dir results/matmul-new
python3 tools/run_repair_gates.py --spec experiment_spec.json --repair-spec model_repair_spec.json \
  --model-manifest generated/models/qwen2.5-1.5b-stage291/model_pair_manifest.json \
  --matmul-results results/matmul-new --results-dir results/gates-new --reports-dir reports/gates-new \
  --diagnostic-summary results/model-chain-diagnostic/diagnostic_summary.json
scripts/run_formal_audit.sh \
  --model-manifest generated/models/qwen2.5-1.5b-stage291/model_pair_manifest.json \
  --results-dir results/formal-new-repaired-pair --reports-dir reports/final-new-repaired-pair
```

各 runner 均拒绝混入已有正式证据。正式脚本运行前冻结 spec、模型 manifest、Stage2.75
目录树及 evaluator 源码哈希，并在裁决前复核。若任一模型修复门槛失败，脚本按预注册
规则停止下游矩阵并输出 `AUDIT_INVALID`。

## 当前正式 gate 结果

`results/formal_20260910_gate2` 的唯一裁决为 `AUDIT_INVALID`：CPU WikiText2
一批次 PPL 为 6.7055，真实 HTP baseline PPL 为约 6.69937e12。专用产物、非法模式拒绝、
Attention smoke 和 256 KiB LUT 释放均已验证，但这些证据不能绕过模型硬门槛。

最终报告位于 `reports/final_20260910_gate2/FINAL_STAGE2_9_REPORT.md`。

Stage 2.9.1 的新结果使用版本化的 device-derived 模型 manifest；它保留同源 F16
逐 tensor 位级证明和设备端 type-514 两分支量化 lineage。若模型 gate 被公共 Attention
链路阻断，可用 `tools/summarize_model_chain_failure.py` 固化自定义 FlashAttention 开/关、
模型内首个分歧、短 KV oracle 和 rowsum 数值证据，并通过 `--diagnostic-summary` 纳入新的
门槛报告。该诊断不会修改被冻结的 Attention 或指数 evaluator。
