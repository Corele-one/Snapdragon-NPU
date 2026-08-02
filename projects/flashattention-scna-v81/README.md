# SCNA FlashAttention on Hexagon v81

该项目是独立的 SCNA/HVX 实验，用 shape-constrained neural approximation 替换 FlashAttention online safe-softmax 中的指数计算，并研究 branchless tree kernel 与 KV DMA/VTCM 流水。

## Benchmark Integrity

**主实验没有混用 v79 与 v81 数据。** 可用于当前 SCNA 性能结论的 baseline、SCNA direct/tree 以及 pipeline off/on，均在同一台 SM8750P 实机的 Hexagon v81 上运行，DSP 构建目标为 `v81`，最终 compile/link flags 已核验包含 `-mv81`。

| 数据范围 | 平台 | 是否用于当前主结论 | 说明 |
|---|---|---:|---|
| `results/v81/scna/stage4-main-20260801/` | SM8750P / v81 | 是 | 修复正确性问题后的 baseline 与 SCNA direct/tree 同轮比较 |
| `results/v81/scna/stage4-correctness-20260801/` | SM8750P / v81 | 是 | FP32 reference、mask、tail、causal、direct/tree gate |
| `results/v81/scna/stage5-pipeline-main-20260801/` | SM8750P / v81 | 是 | 同轮 baseline、pipeline off/on 与 5+20 次测量 |
| `results/v81/scna/stage5-pipeline-correctness-20260801/` | SM8750P / v81 | 是 | pipeline 字节一致性与 FP32 reference |
| `results/v81/scna/sm8750p-*` 早期目录 | SM8750P / v81 | 否 | 阶段一至三历史执行成本；Attention 正确性结论已撤回 |
| `results/v73/`、`results/v79/` | 历史设备/架构 | 否 | 从原项目继承，仅作历史归档，禁止与 v81 SCNA 做 speedup 对比 |

阶段一至三同样是在 v81 上执行，但后来发现当时 baseline 和所有 nonlinear mode 共享 rowsum 写回、v81 reciprocal 选择以及 padded mask stride 问题。因此，早期 microkernel 吞吐仍可用于理解内核演进，早期 Attention latency 不能进入最终 Pareto、speedup 或正确性结论。

本项目后续图表和汇报必须遵守以下规则：

1. SCNA speedup 的分子与分母只能来自同一个 `results/v81/scna/<run>/` 或明确记录的相邻同设备 session。
2. `results/v73/` 和 `results/v79/` 不得进入 v81 主表、主图、置信区间或摘要数字。
3. LUT 只允许作为附录；SCNA 主结论只比较 v81 baseline 与 v81 SCNA。
4. 阶段五中 baseline 未启用 KV pipeline，因此 baseline/SCNA 回答的是“组合方案是否超过现有实现”；pipeline off/on 才是 KV 流水贡献的因果消融。

## 实验范围

SCNA 同时替换 online safe-softmax 的两处指数：

```text
P = f(S - m_cur)
l_new = f(m_prev - m_cur) * l_prev + rowsum(P)
```

- `exp2`：score scale 使用 `log2(e) / sqrt(head_dim)`。
- `exp`：score scale 使用 `1 / sqrt(head_dim)`。
- precision：FP16 与真实 S8 input/weight、S32 中间结果的 INT8 路径。
- width：d8、d16、d32。
- kernel：direct neuron sum 与 branchless breakpoint tree。
- pipeline：KV direct load 与双 VTCM buffer 2D DMA pipeline。
- mask：SCNA 后显式清零 masked/tail lane，不将 `-inf` 输入网络。

主要接口：

```text
--mode baseline|lut-exp|scna-fp16|scna-int8
--scna-function exp2|exp
--scna-kernel direct|tree
--scna-width 8|16|32
--scna-pipeline off|on
--scna-exp-bench
--compare-direct-tree
--compare-pipeline
```

## v81 实验设置

| 项目 | 设置 |
|---|---|
| Device | model `25091RP04C`，SoC `SM8750P`，ADB serial `bde3ddde` |
| DSP | Hexagon v81，128-byte HVX |
| Toolchain | Hexagon SDK 6.6.0.0，Hexagon Tools 19.0.07 |
| Build gate | DSP compile/link command 含 `-mv81` |
| Main shape | `qo_len={4,8,16,32}`，`kv_len=4096`，heads/KV-heads `12/2`，head-dim 128 |
| Repetition | 单 worker，5 warmup + 20 measured |
| Correctness | full/padding/causal，`kv_len=4093/4096`，head-dim 64/128 |

## 关键结果

### Branchless tree

- FP16 d32 pair microkernel：direct/tree 为 `18358/8767 us` 每 100k calls，tree 加速 `2.094x`。
- FP16 d32 指令数由 `586` 降到 `167`，multiply 指令由 `96` 降到 `4`。
- 100 个 DSP output vs FP32 检查、48 个 direct/tree 检查和 100 个 mask/tail probes 均为 0 failure。
- 修复后的阶段四中，最快 SCNA 相对同轮 baseline：q4 `1.085x`、q8 `1.050x`、q16 `0.989x`、q32 `0.831x`。tree 优化降低了 nonlinear 成本，但单靠 SCNA 尚未在长 query 上全面胜出。

### KV DMA pipeline

- 96/96 个 pipeline on/off 配置获得 DSP 加速，bootstrap 95% CI 下界均大于 1。
- FP16 exp tree d8 的 q32 DSP latency 从 `2121.0 us` 降至 `1821.5 us`，pipeline speedup `1.164x [1.128, 1.175]`。
- 该配置 K+V 时间从 `590.5 us` 降至 `303.5 us`；SCNA compute 为 `1076 -> 1079 us`，说明收益来自数据调度。
- 96 个 on/off 输出全部字节一致；FP32 gate 最大 RMSE `8.86e-4`，无 nonfinite。

## 代码与数据

```text
src/htp-ops-lib-main/src/dsp/ops/scna_exp2.c   SCNA direct/tree HVX kernel
src/htp-ops-lib-main/src/dsp/ops/flash_attn.c  Attention integration 与 KV pipeline
src/htp-ops-lib-main/include/dsp/scna_params.h  导出的参数
training/fit_export_scna.py                     exp/exp2 训练、仿真与导出
scripts/run_scna_v81_matrix.sh                  v81 性能矩阵
scripts/run_scna_v81_correctness.sh             FP32/direct-tree 正确性矩阵
scripts/run_scna_v81_pipeline.sh                pipeline off/on 矩阵
scripts/run_scna_v81_pipeline_correctness.sh    pipeline 字节一致性
results/v81/scna/                               v81 raw CSV、log、汇总与 SVG
docs/stage-reports/                             分阶段实验报告
```

阶段报告：

- [阶段一/二：Demo 与热路径重写](docs/stage-reports/SCNA_HVX_阶段一阶段二总结_2026-07-31.md)
- [阶段三：真实 INT8 kernel](docs/stage-reports/SCNA_HVX_阶段三总结_2026-08-01.md)
- [阶段四：branchless tree 与正确性闭环](docs/stage-reports/SCNA_HVX_阶段四_tree重写与正确性闭环_2026-08-01.md)
- [阶段五：KV DMA/VTCM 流水](docs/stage-reports/SCNA_HVX_阶段五_KV_DMA流水_2026-08-01.md)

## 复现

runner 默认使用已部署到 `/data/local/tmp/flashattention_scna_tree` 的 Android host binary 与 v81 DSP skel。运行前必须检查连接设备和构建目标：

```bash
adb devices
rg -- '-mv81' src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v81/build.ninja
```

执行修复后的 correctness、SCNA 主矩阵和 pipeline 消融：

```bash
./scripts/run_scna_v81_correctness.sh
./scripts/run_scna_v81_matrix.sh
./scripts/run_scna_v81_pipeline_correctness.sh
./scripts/run_scna_v81_pipeline.sh
```

LUT 附录默认关闭。仅在明确生成附录时启用：

```bash
RUN_LUT_APPENDIX=1 ./scripts/run_scna_v81_matrix.sh
```

## 解释边界

- 输入是固定 seed 的 synthetic Figure8 case，不等价于模型级 perplexity 或端到端 token latency。
- q4/q8 的小幅 SCNA 优势仍需要跨 session、随机化运行顺序和温度控制复现。
- 当前主结果为单 worker，不能外推到多 worker scalability。
- stage4/5 报告记录了 v81 build gate；早期 SCNA run 尚无独立机器可读 `provenance.txt`。后续采集应像 CPU 实验一样记录设备、二进制 SHA-256、SDK、Tools、`dsp_arch`、温度和 Git 状态。

## 许可边界

llama.cpp 子树保留 MIT `LICENSE`。HTP operator 快照依赖获授权的 Qualcomm Hexagon SDK；不要提交 proprietary SDK headers 或未获授权的二进制。
