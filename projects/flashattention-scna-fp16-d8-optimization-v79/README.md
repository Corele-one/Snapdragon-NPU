# SCNA FP16 d8 渐进优化实验（Hexagon v79）

本项目从未优化的阶段一 SCNA 重新建立纵向 baseline，并以独立 DSP library 逐步加入六项优化。横向 baseline 是项目原生 polynomial exp2（Origin HVX）和 EXP-LUT。

七个构建依次为：

1. `stage1_dynamic_row`：runtime width、逐 row、热路径重复参数查找/转换、qf16 链、单 worker；
2. `prepare_once_row`：Attention 提交 worker 前只准备一次参数；
3. `pair_shared_dynamic`：row0/row1 共用一次 SCNA `w/b` load/splat；
4. `pair_static_d8`：固定并展开 d=8；
5. `pair_d8_fma_noinline`：v79 FP16 `vmpyacc + vmax + vadd`，noinline；
6. `pair_d8_fma_inline`：相同 body，always_inline；
7. `optimized`：按预注册统计规则选 call policy，并扩展至多 worker。

“pair shared”只表示 SCNA 参数广播复用。QK、PV、mask、K/V load 和 tiling 在全部变体间固定；本实验不包含 K/V tile 复用、KV DMA/VTCM pipeline，也不归因这类收益。

## 可复现实验

预注册假设、父项、唯一变化、证据白名单和门限在 [`experiment_spec.json`](experiment_spec.json)。

```bash
# SDK 6.6，v79；生成七个带 build ID 的独立 DSP library
./scripts/build_all_variants.sh

# 设备全矩阵、统计分析、报告和所有 gate；支持同 RUN_ID 断点续跑
RUN_ID=scna_d8_v79_20260811 ./scripts/run_optimization.sh

# 单独重生成报告
python3 tools/generate_optimization_report.py \
  --run-dir results/runs/scna_d8_v79_20260811 \
  --spec experiment_spec.json
```

`--workers` 接受 `auto|1..6`。`auto` 取 requested/default HVX contexts、有效 task 数和 `total VTCM / 1 MiB` 的最小值。q=32 时调度器产生 16 个 query-block × KV-head task。

每个 SCNA 请求都会在 DSP 端校验运行时 variant 与编译期 `SCNA_BUILD_VARIANT`。artifact、SHA-256、`-mv79` 编译证据和反汇编分别保存在 `artifacts/variants/` 与 run 的 `static/` 中。

## 输出

每次 run 生成：

- `raw/`：micro、主矩阵、diagnostic、正确性和 worker scaling 日志；
- `manifest.txt` 与 `inline_selection.json`；
- `summary.json`、`summary.csv`、`verification.json`；
- 各构建反汇编与静态 branch/instruction/packet/call/spill/stack/code-size 指标；
- 中文 `REPORT.md`，每个阶段包含数据解释、前置证据、假设、实现改动、结果、静态解释、异常分析和结论。
- Matplotlib 生成的可确定性矢量 SVG：渐进优化柱状图、Qo 横向基线总览与去除 stage1 的细节图、inline/noinline 分组柱状图、worker scaling 折线图、worker timeline、正确性图、逐步贡献（消融风格）图，以及 7 张与阶段数据表一一对应的 Qo 分层图；所有图例使用英文，每张图下均生成解释性的 `Key Finding`，横向细节图同时输出完整绘图数据表。

报告绘图依赖可通过 `python3 -m pip install --user -r requirements-report.txt` 安装。

报告的每一步固定包含 Motivation、Prior Evidence、Hypothesis、Implementation Change、Results、Static Explanation、Verdict and Transition。Motivation 只能引用父阶段 raw log/summary/disassembly；证据缺失时会明确写“预注册工程假设，前置证据不足”。

## 正确性与统计

Attention gate 为 `ret=0`、finite、mask/tail zero、host-FP32 RMSE ≤ 0.002、max-abs ≤ 0.01；不同 worker 数 checksum 必须一致。Micro 同时检查 dense/boundary、pair/single、单调性、非负性、NaN/Inf 和防 benchmark 提升 checksum。

主结论使用五个 session，每个 `5 warmup + 20 measured`；micro 每构建 30 个独立且至少 50 ms 的样本。相邻阶段使用 paired bootstrap 95% CI；CI 跨 1.0× 时为 Inconclusive。inline 未领先至少 1%或 CI 跨 1.0×时选择 noinline。
