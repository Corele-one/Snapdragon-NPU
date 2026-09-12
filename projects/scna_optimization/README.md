# SCNA Optimization（Snapdragon Hexagon v79）

本项目提供一个可独立构建和验证的 FlashAttention SCNA 最终实现，目标设备是 SM8750P 的 Hexagon v79。

SCNA（shape-constrained neural approximation）在这里不是一个完整的 Attention 算法，而是用一个很小的分段线性神经网络近似 Softmax 中的 `exp2`。QK score 已乘 `log2(e)/sqrt(head_dim)`，所以这里计算 `exp2` 与标准 Softmax 中计算自然指数相对应。整个 Attention 仍按下面的数据流执行：

```text
Q × K  --HMX-->  scaled scores
                    |
                    v
              row max / mask
                    |
                    v
        SCNA exp2 + online Softmax state  --HVX-->
                    |
                    v
P × V  --HMX-->  Attention output
```

本文使用的硬件术语：HMX 是 Hexagon 的矩阵计算单元，负责 Q×K 和 P×V；HVX 是向量计算单元，负责 mask、归约、SCNA 和 Softmax 状态；VTCM 是 DSP 片上的紧耦合存储，用来保存每个 worker 的 tile 和 scratch 数据。

最终代码只保留了一条 SCNA 生产路径。它可以概括为：

```text
d7_pairret_noinline + fused_state_update + adaptive scheduler
```

这三个名字分别表示：

- `d7_pairret_noinline`：使用 7 个有效 ReLU neuron 的 FP16/HVX `exp2` evaluator；一次同时计算两行，并直接以两个 HVX vector 返回；热函数禁止内联。
- `fused_state_update`：计算完每一块的 rowmax 和 rowsum 后，直接在寄存器中更新 online Softmax 状态，不再把中间状态写入 VTCM 后重新读取。
- `adaptive scheduler`：根据 query 行数选择每个 task 处理的行数，再按设备的 HVX context、task 数量和 VTCM 容量决定 worker 数。

下面从输入、计算过程和硬件含义解释这三部分。

## 1. FP16 SCNA exp2 evaluator

Softmax 在减去行最大值后，传给指数函数的输入满足 `x <= 0`。本实现把输入限制到 `[-256, 0]`，然后计算：

```text
x_clamped = clamp(x, -256, 0)
y = sum(i=0..6) relu(w_i * x_clamped + b_i)
```

参数仍采用原训练模型的 d8 存储格式，但第 8 个 neuron 在这个输入区间内恒为零，所以运行时只计算前 7 个。这就是名字中 `d7` 的含义。删除恒零 neuron 后，每个 64-lane HVX vector 少做一组 affine、ReLU 和累加，同时不改变有效区间内的函数。

计算全程使用 FP16/HVX：

- 每个 weight 保持为 FP16 scalar，并复制到 `Rhf` scalar-vector operand；不为 weight 建立长生命周期的 broadcast vector。
- bias 只在使用时通过 `vsplat` 生成 vector。
- 两个输入 vector 共用同一组 weight/bias，并把两条依赖链交错排列，从而提高一个 V79 instruction packet 中可并行发射的指令数量。
- 两个输出组合成 `HVX_VectorPair` 直接返回，因此叫 `pairret`；避免通过指针写回再由 caller reload。
- evaluator 标记为 `noinline`。这保持了一个稳定、无栈帧的函数边界，避免内联后 SCNA 临时量与 Softmax caller 的大量 live vector 相互挤压。

当前 v79 artifact 中，热 evaluator 为 **112 instructions、36 packets、0 spill、0 B frame**。源码入口位于：

- [`src/htp-ops-lib-main/include/dsp/scna_exp2_hot.h`](src/htp-ops-lib-main/include/dsp/scna_exp2_hot.h)：clamp、FP16 scalar-weight affine、7-neuron 单行/双行计算。
- [`src/htp-ops-lib-main/src/dsp/ops/scna_exp2.c`](src/htp-ops-lib-main/src/dsp/ops/scna_exp2.c)：参数准备、最终 pair-return evaluator 和运行时配置拒绝。

## 2. 寄存器内 online Softmax 状态融合

FlashAttention 不会一次保存完整的 score matrix，而是按 KV block 更新每一行的 online Softmax 状态。对于上一块状态 `(m_old, l_old)` 和当前块的最大值、概率和，核心更新是：

```text
m_new = max(m_old, block_rowmax)
alpha = exp2(m_old - m_new)
l_new = l_old * alpha + block_rowsum
O     = O * alpha + P × V
```

旧数据流先把 `block_rowmax` 和 `block_rowsum` 写入 compact VTCM buffer，离开 Softmax 循环后再 reload，用它们更新 `m/l/rescale`。

最终实现让这两个值继续保留在 HVX registers 中，并在相同的 row-vector scope 内立即完成 `m_new`、`alpha` 和 `l_new` 计算。这样每个 64-row chunk、每个 KV block 都消除了两类 compact state 的 store/reload 往返，同时没有扩大 SCNA evaluator 的寄存器集合。

该优化只改变状态消费位置，不改变：

- HMX 生成 QK score 时使用的 scale；
- mask、KV tail 和 row reduction 语义；
- SCNA 的 FP16 参数与输入区间；
- HMX 的 `P × V` 计算。

实现位于 [`src/htp-ops-lib-main/src/dsp/ops/flash_attn.c`](src/htp-ops-lib-main/src/dsp/ops/flash_attn.c) 的 FP16 FlashAttention core，构建固定为 `STAGE2_SOFTMAX_IMPL=2`。

## 3. Query/KV-head 自适应调度

一个 scheduler task 由 `(query chunk, KV head)` 唯一确定。task 内负责这一组 query 行对一个 KV head 的完整 K/V traversal；同一个 task 不会被多个 worker 拆分。

如果 query chunk 太小，会产生更多 task 和更多并行机会，但每个 chunk 都要重新遍历 K/V；如果 chunk 太大，K/V 重复读取减少，却可能只剩两个 KV-head task，无法使用足够多的 HVX context。当前策略是在两者之间取平衡：

```text
Q <= 8  -> q_task_rows = 4
Q <= 16 -> q_task_rows = 8
Q > 16  -> q_task_rows = 16

task_count = ceil(Q / q_task_rows) * KV_head_count
workers = min(HVX_context_count, task_count, VTCM_worker_cap)
```

worker 通过 atomic counter 动态领取下一个 task。每个实际 worker 独占 1 MiB VTCM slice，所以 worker 数还必须受 VTCM 总容量限制。

例如本轮 `Q=32、KV heads=2`：`q_task_rows=16`，因此产生 `2 × 2 = 4` 个 task；设备报告 6 个 HVX context、VTCM 最多容纳 8 个 worker，最终实际使用 4 个 worker。这既保留 4-way 并行，又把 K/V traversal 次数从 `q_task_rows=4` 时的 8 个 query chunk 降为 2 个。

实现位于 [`src/htp-ops-lib-main/src/dsp/ops/flash_attn.c`](src/htp-ops-lib-main/src/dsp/ops/flash_attn.c) 的 worker queue 和 `simple_flash_attn_impl`。Host CLI 不提供 worker 数、query chunk 或 kernel variant 实验选项；这些策略在最终实现中自动确定。

## 4. 最终路径如何组合

对每个 worker 领取的 task，执行顺序如下：

1. 把该 KV head 的 K tile 和当前 query chunk 搬入 worker 私有 VTCM。
2. 使用 HMX 计算已经带 `log2(e)/sqrt(head_dim)` scale 的 QK score。
3. HVX 应用 mask、求 rowmax，并把 centered score 送入双行 SCNA evaluator。
4. SCNA 直接返回两个概率 vector；HVX 完成 rowsum。
5. 在 rowmax/rowsum 仍在 registers 时更新 online `m/l/rescale`。
6. 使用 HMX 完成 P×V，并更新输出累加器。
7. worker 通过 atomic queue 领取下一个 `(query chunk, KV head)`，直到 task 耗尽。

因此最终收益来自三个层次：更短的指数 evaluator、较少的 VTCM 中间状态往返，以及更合理的 task 粒度。它不是只替换一个数学函数的单变量优化。

## 5. baseline 和 LUT-exp 的定义

`reference/flashattention/htp-ops-lib-main/` 是 `../flashattention/src/htp-ops-lib-main/` 的逐文件快照，并通过完整 source-tree hash 校验。

- **baseline**：原版 FlashAttention；QK/PV 使用 HMX，Softmax exp2 使用原版 HVX polynomial。
- **LUT-exp**：相同原版源码和 artifact，exp2 改走 VTCM LUT + HVX `vgather`。
- **SCNA**：本项目的 7-neuron pair-return evaluator、寄存器状态融合和自适应 scheduler。

性能表比较的是“原版系统 vs 最终优化系统”，不能解释成只隔离 exp evaluator 的单变量实验。

## 6. SM8750P / v79 真机结果

固定 KV=4096、12 query heads、2 KV heads、head_dim=128、full mask。每种模式和每个 Q 执行 5 个轮换 session；每个 session 包含 5 次 warmup 和 20 次正式测量。每格 `n=100`，合计 **1200/1200** 条有效 Host 样本，全部 `ret=0`。

| Q | baseline µs (95% CI) | LUT-exp µs (95% CI) | SCNA µs (95% CI) | SCNA / baseline paired ratio |
|---:|---:|---:|---:|---:|
| 4 | 561.5 [558.0, 572.0] | 548.0 [548.0, 549.0] | 498.0 [496.5, 499.0] | 0.8768 [0.8709, 0.9322] |
| 8 | 658.0 [657.0, 658.0] | 549.0 [549.0, 577.0] | 526.5 [493.5, 535.0] | 0.7951 [0.7024, 0.8073] |
| 16 | 877.0 [876.0, 881.0] | 659.0 [658.0, 659.0] | 657.0 [656.0, 657.5] | 0.7292 [0.7191, 0.7489] |
| 32 | 1434.0 [1432.5, 1435.0] | 1020.0 [1009.5, 1040.0] | 833.5 [780.5, 861.0] | 0.5780 [0.5490, 0.5869] |

比值按同 session、同 iteration 配对；95% CI 使用固定 seed 的 10,000 次 bootstrap。正式排名关闭 event 输出。独立诊断中的组件时间是跨 task 的 work-sum，不得与 Host wall latency 相加或直接比较。

最终 SCNA 正确性矩阵为 **180/180**：3 masks × 5 个 Q × 2 个 KV 长度 × 2 个 head dimension × 3 个 seed。最大 RMSE `0.000283223`，最大绝对误差 `0.00121131`，nonfinite、masked output 和 tail output 均为 0。Q=32 的 10 个独立进程得到相同 measured checksum `0x19c89fab0065963c`。

完整结果见：

- [`results/v79/RESULTS.md`](results/v79/RESULTS.md)：三方 Host latency。
- [`results/v79/DIAGNOSTICS.md`](results/v79/DIAGNOSTICS.md)：独立组件诊断。
- [`results/v79/correctness/summary.json`](results/v79/correctness/summary.json)：180-case 和 determinism 汇总。
- [`results/v79/evidence/`](results/v79/evidence/)：设备、thermal、artifact、静态汇编和 provenance 证据。

## 7. 构建和复现

需要 Qualcomm Hexagon SDK 6.6.0.0/6.6.0.1、Android toolchain 和可用的 ADB 设备。

```bash
./scripts/build.sh
./scripts/smoke.sh all
./scripts/run_correctness.sh
./scripts/run_benchmark.sh
python3 -m unittest discover -s tests -v
```

`build.sh` 生成两个独立 bundle：

- `artifacts/reference/`：只用于原版 baseline 和 LUT-exp。
- `artifacts/scna/`：只用于最终 SCNA。

`run_correctness.sh` 和 `run_benchmark.sh` 可以从已有完整日志恢复。解析器会拒绝缺失日志、非零返回、模式错配和不完整 session。固定实验参数位于 [`experiment_spec.json`](experiment_spec.json)。

## 8. 设计文档

为便于单独审阅每一步工作，详细实现和效果拆分在 [`docs/`](docs/)：

- [`Stage 1：FP16/HVX SCNA evaluator`](docs/stage1_scna_hvx_evaluator.md)
- [`Stage 2：online Softmax 状态融合`](docs/stage2_softmax_state_fusion.md)
- [`Stage 2.5：为什么没有合入 full-mask 专用路径`](docs/stage2_5_full_mask_decision.md)
- [`Stage 2.75：自适应 worker scheduler`](docs/stage2_75_adaptive_scheduler.md)

这些文档只描述最终实现实际采用的技术和必要的保留边界；历史实验工程已归档，构建和测试不依赖归档目录。
