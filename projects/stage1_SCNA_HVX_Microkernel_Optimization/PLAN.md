## 1. Objective

本阶段唯一目标：

> **在 Hexagon V79 HVX 上继续优化当前 SCNA exponential evaluator，降低 SCNA microkernel 的 packet count、critical-path latency 和实际 Attention 中的 `scna_exp` 开销，同时严格控制 register pressure、spill 和 caller stack。**


```
Production baseline:
    d7_pairret_noinline

Performance reference:
    EXP-LUT

Historical reference:
    static_d8_ref
```

SCNA 是固定算法方案。

**禁止：**

```
切换到其他 exp approximation
重新使用 Horner polynomial
重新探索 lane-8
重新探索 d7_quad_pipeline
全局 prebroadcast
forced inline
修改 HMX QK/PV
修改 FlashAttention tiling
修改整个 Softmax dataflow
修改 DMA pipeline
```

`d7_pairret_noinline` 已经把 q32 降到 5182.5 us，而 `d7_quad_pipeline` 为 5206.5 us；quad 同时造成 1920 B caller frame 和 116 个 stack references，因此不能继续沿“扩大并行 vector chain”方向搜索。

---

# 2. Core Hypothesis

现阶段不要再假设：

> SCNA 算术运算太多。

已有实验更支持：

\[ \boxed{ T_{\rm SCNA} \approx T_{\rm packet} + T_{\rm dependency} + T_{\rm resource\ conflict} + T_{\rm register/live-range} + T_{\rm ABI} } \]

而不是简单：

\[ T_{\rm SCNA}\propto N_{\rm instruction} \]

最直接的证据是：

```
static_d8:
127 inst
41 packets

d7_serial:
121 inst
46 packets
```

删除一个 neuron 后 instruction 数下降，但 packet 数反而增加，最终没有得到有效性能提升。

所以本阶段的核心问题改成：

\[ \boxed{ \text{How few packets can one SCNA evaluation require without increasing its live register set?} } \]

---

# 3. Mandatory Baseline Reproduction

在改代码之前：

1. checkout / freeze 当前 `d7_pairret_noinline`；
2. 保存 artifact SHA256；
3. 重跑 correctness；
4. 重跑 micro benchmark；
5. 重跑 q=1/4/8/16/32 Attention；
6. 保存反汇编。

必须确认当前数据与报告没有明显漂移。

参考实验配置保持：

```
KV = 4096
query heads = 12
KV heads = 2
head_dim = 128
full mask
workers = 1

q = 1 / 4 / 8 / 16 / 32
```

原报告使用 5 sessions、每 session 5 warmup + 20 measure，并通过 bootstrap 计算 95% CI；新 Stage 1 正式实验继续尽量维持同一统计规范。

如果 baseline 无法复现，先解释漂移，不继续优化。

---

# 4. Experiment A — Packet-Level Bottleneck Analysis

## Priority

**P0，必须第一个执行。**

不要立即修改代码。

先分析当前：

```
d7_pairret_noinline
112 instructions
36 packets
```

究竟为什么需要 36 packets。

---

## A1. Generate packet map

对 evaluator 的反汇编逐 packet 标注：

```
Packet ID
Instruction(s)
Input dependency
Output dependency
Operation class
Likely execution resource
Independent work available?
Critical path?
```

建议生成：

|Packet|Operations|Depends on|Type|Critical|
|---|---|---|---|---|
|P0|...|input|arithmetic|yes|
|P1|...|P0|multiply|yes|
|P2|...|independent|compare|no|
|...|||||

至少分类：

```
multiply
add/FMA
compare
permute/shuffle
splat
load
store
conversion
scalar/vector move
branch/control
```

---

## A2. Calculate metrics

必须报告：

```
total instructions
total packets

instructions / packet

multiply-related packets
permute-related packets
splat-related packets

single-instruction packets
multi-instruction packets

estimated critical-path packets
```

重点找：

### 类型 1：dependency bubble

```
P0: produce A

P1:
    operation must wait for A

P2:
    use A
```

### 类型 2：resource conflict

理论上 independent，但因为 HVX execution resources 无法放进一个 packet。

### 类型 3：under-filled packet

存在：

```
{ instruction }
```

但周围其实有 independent work，有重排机会。

---

## A3. Deliverable

生成：

```
reports/stage1/
01_packet_analysis.md
```

在没有得到这份报告之前，不允许开始 Experiment B。

V79 HVX PRM 是这里的主要 ISA 依据：

[Qualcomm Hexagon V79 HVX Programmer Reference Manual](https://docs.qualcomm.com/doc/80-N2040-61/topic/introduction.html?utm_source=chatgpt.com)

需要重点查：

```
packet rules
instruction resources
latency
multiply
permutation
vector/scalar operand forms
register dependency
```

---

# 5. Experiment B — Two-Accumulator SCNA

## Priority

**P0。**

这是 Stage 1 首个真正代码改动。

当前假设 SCNA 最后的 neuron accumulation 存在较长的 accumulator dependency：

```
acc
 ↓
acc + neuron0
 ↓
acc + neuron1
 ↓
acc + neuron2
 ↓
...
```

尝试改成：

\[ acc_0=f_0+f_2+f_4+f_6 \]\[ acc_1=f_1+f_3+f_5 \]

最后：

\[ y=acc_0+acc_1 \]

也就是：

```
neuron0 ─→ acc0
neuron1 ─→ acc1

neuron2 ─→ acc0
neuron3 ─→ acc1

neuron4 ─→ acc0
neuron5 ─→ acc1

neuron6 ─→ acc0

acc0 + acc1
```

---

## Why try this

它和已经失败的 quad 不一样。

Quad 增加的是：

```
multiple complete vector evaluation chains
```

从而显著扩大 live set。

Two-accumulator 只增加：

```
one additional accumulator
```

目标是减少 accumulator dependency depth，而不是增加 lane/vector width。

---

## Required variants

只做：

```
B0 = current pairret
B1 = pairret + 2 accumulators
```

**不要马上做 3/4 accumulator。**

只有 B1 明确有正证据才允许进一步扩展。

---

## Static gate

B1 必须满足：

```
evaluator spill = 0
evaluator stack = 0

caller stack frame <= baseline
caller stack references <= baseline + small justified delta
```

重点比较：

```
packets
instructions
critical-path estimate
register usage
```

如果：

```
instructions ↓
但 packets ↑
```

直接视为负结果。

---

## Correctness

因为 accumulation association 改变：

```
不要求 bitwise checksum 与 baseline 完全一致
```

但必须完整执行现有 correctness matrix：

```
finite
mask
tail
RMSE <= 0.002
max_abs <= 0.01
```

原报告中的动态候选均通过了这套 72-case correctness gate，因此继续沿用即可。

---

# 6. Experiment C — Remaining Splat Audit

## Priority

**P0。**

当前一个非常有价值的已有结果是：

```
scalar-weight:
splat 17 → 8
Rhf mul 0 → 14

micro:
22.690 → 20.627 ns/64
≈ -9.09%
```

说明：

> vector splat elimination 确实是有效方向。

问题是：

\[ \boxed{\text{剩下 8 个 splat 是什么？}} \]

---

## C1. Audit every remaining splat

生成表：

|Splat|Source|Purpose|Frequency|Can eliminate?|
|---|---|---|---|---|
|S0|...|weight|||
|S1|...|bias|||
|...|||||

分类：

```
weight
bias
threshold
zero
one
scale
other constant
```

---

## C2. Search ISA alternatives

对每一类分别检查：

```
vector op + scalar half
vector compare + scalar operand
immediate form
scalar register operand
cheap rematerialization
```

目标不是把所有参数长期 broadcast 到 vector register。

已经证明：

```
global prebroadcast
```

会导致 1920 B caller frame 和额外 stack references，因此明确禁止重复这一策略。

---

## C3. One constant class at a time

必须分别构建：

```
C1 = bias optimization only
C2 = threshold optimization only
C3 = zero/one optimization only
...
```

不要一次全部修改。

每个 variant 都独立经过：

```
assembly gate
micro
correctness
Attention
```

这样才能知道究竟是哪一个改动有效。

---

# 7. Experiment D — Short-Lifetime Constant Scheduling

## Priority

**P0/P1。**

目前实验已经说明：

```
长期保存大量 vector constants
        ↓
register pressure
        ↓
spill / caller frame
```

因此反过来尝试：

\[ \boxed{\text{short live range}} \]

---

## D1. Inspect current constant lifetime

不要：

```
prepare W0...W6
prepare B0...B6

then evaluate all neurons
```

尝试：

```
prepare W0/B0
compute neuron0
W0/B0 dead

prepare W1/B1
compute neuron1
W1/B1 dead
```

或者：

```
prepare neuron0 + neuron1 constants
compute neuron0/neuron1
release them

prepare neuron2 + neuron3
...
```

---

## Goal

允许：

```
少量重复 cheap setup
```

换取：

```
lower peak register pressure
better packet scheduling
zero spills
```

这里不以 instruction 数最少为唯一目标。

---

# 8. Experiment E — Two-Neuron Local Scheduling

## Priority

**P1。**

如果 Experiment A 发现两个相邻 neuron 之间存在可利用的独立操作，则尝试：

```
Neuron 0 stage 1
Neuron 1 stage 1

Neuron 0 stage 2
Neuron 1 stage 2

Neuron 0 accumulate
Neuron 1 accumulate
```

而不是：

```
complete neuron0
complete neuron1
```

---

## Key distinction

这不是：

```
lane-8
quad pipeline
```

而是只在：

\[ \boxed{\text{2 neurons × 1 input block}} \]

之间制造有限 ILP。

原则：

> ILP 增量必须小于 register pressure 增量。

---

## Stop immediately if

出现：

```
caller frame significant increase

stack references significant increase

evaluator spill

packets >= baseline
```

则否决。

不要为了 micro 上 <1% 收益接受巨大的 live range 增长。

---

# 9. Experiment F — Iteration Software Pipelining

## Priority

**P1。**

如果 Experiment E 仍不能填满 packet，可以尝试把：

```
iteration i compute
```

与：

```
iteration i+1 preparation
```

交错。

例如：

```
pair i:
    SCNA arithmetic

pair i+1:
    input/parameter preparation
```

目标：

\[ prep_{i+1} \parallel compute_i \]

而不是 quad 那种：

\[ compute_i \parallel compute_{i+1} \parallel compute_{i+2} \parallel compute_{i+3} \]

---

## Register gate

这是硬门禁：

```
caller frame <= d7_pairret_noinline
evaluator stack = 0
spill = 0
```

只允许非常有限的 live-set 增长。

FlashAttention-V 在其他 vector ISA 上也观察到 loop reordering / unrolling 可以通过暴露独立工作改善 vector utilization；这里只借鉴这种 ILP 思路，不能照搬其 RVV/SVE 实现。[arXiv](https://arxiv.org/abs/2608.18656?utm_source=chatgpt.com)

[FlashAttention for Scalable Vector Architectures](https://arxiv.org/abs/2608.18656?utm_source=chatgpt.com)

---

# 10. Experiment G — ABI and Small-Q Fixed Cost

## Priority

**P1。**

目前 `pairret_noinline` 有非常明显的 Qo 特征：

```
q1   1.0153 × static
q4   1.0147 ×
q8   0.9678 ×
q16  0.9676 ×
q32  0.9640 ×
```

即：

> evaluator 优化随着工作量增大越来越明显，而 q1/q4 被固定成本稀释。

因此单独分析：

```
function call
argument setup
return value movement
loop control
prologue/epilogue
branch
```

---

## G1. Do not re-try forced inline

forced inline 已经导致 caller stack 增长，报告已经否决。

所以不要简单：

```
always_inline
```

---

## G2. Test call amortization

可以 prototype：

```
one call
    ├─ process pair0
    ├─ release temporary registers
    └─ process pair1
```

而不是：

```
hold pair0 + pair1 simultaneously
```

目标仅是 amortize：

```
call
setup
return
```

而不是扩大计算并行度。

---

## G3. Small-Q specialization

如果确实证明 fixed ABI cost 是 q1/q4 的主要问题，可以允许：

```
q <= threshold:
    short-Q SCNA schedule

q > threshold:
    normal pairret SCNA
```

**算法仍然必须是相同 SCNA。**

不要使用 LUT fallback。

---

# 11. Experiment H — Safe-Softmax Range Folding

## Priority

**P1/P2。**

你已经利用：

\[ x\le0 \]

证明第 8 neuron 恒零并删除了它。报告说明这个数学化简是正确的，只是由于 packet scheduling 没有单独转化成性能收益。

继续检查剩余 7 neurons：

\[ h_i(x)=\operatorname{ReLU}(w_ix+b_i) \]

根据**代码中实际可证明的输入区间**分析：

\[ w_ix+b_i \]

是否：

### 恒负

\[ w_ix+b_i\le0 \]

则：

```
neuron removable
```

### 恒正

\[ w_ix+b_i\ge0 \]

则：

\[ \operatorname{ReLU}(w_ix+b_i) = w_ix+b_i \]

可以删除该 neuron 的 compare/ReLU path。

---

## Important

不要自己假定存在额外：

\[ x_{\min} \]

只有代码/数学上确实能够证明输入区间时才使用。

这不是重新训练 SCNA。

这是对现有模型进行：

```
range-aware constant folding
```

---

# 12. Experiment I — Fused Scale Feasibility

## Priority

Stage 1 最后的独立实验。

这里只研究：

\[ \boxed{\text{SCNA evaluator 能否吸收 input scale}} \]

**不要在这里重写整个 Softmax。**

先确定实际 scale：

```
per tensor?
per head?
per row?
per tile?
per channel?
```

然后根据实际 SCNA：

\[ f(x)=\sum_i a_i\operatorname{ReLU}(w_ix+b_i)+c \]

推导：

\[ f(sx) \]

是否能够通过参数变换实现。

例如在满足相应条件时可能有：

\[ w_i' = sw_i \]

使：

\[ f(sx) = \sum_i a_i\operatorname{ReLU}(w_i'x+b_i)+c \]

但必须根据真实 scale granularity 和调用位置证明，禁止仅凭公式直接实现。

---

## Benchmark

最终增加：

|Path|Scale cost|SCNA/LUT cost|Total|
|---|---|---|---|
|scale + EXP-LUT||||
|scale + SCNA||||
|fused-scale SCNA|fused|||

这个实验的意义是：

即使：

\[ T_{\rm SCNA}>T_{\rm LUT} \]

仍有可能：

\[ \boxed{ T_{\rm fused\ SCNA} < T_{\rm scale}+T_{\rm LUT} } \]

但这只是 Stage 1 的 microkernel/fusion 结论。

**整个 Softmax dataflow 的 fusion 留给 Stage 2。**

---

# 13. Experiment Priority Order

Codex 严格按照：

```
A. Packet analysis
        ↓
B. Two accumulators
        ↓
C. Remaining splat audit
        ↓
D. Short-lifetime constants
        ↓
E. Two-neuron scheduling
        ↓
F. Iteration software pipeline
        ↓
G. ABI / small-Q
        ↓
H. Range folding
        ↓
I. Scale fusion
```

不要同时开九个实现。

原则：

```
分析
→ 一个 variant
→ static gate
→ micro
→ correctness
→ full Attention
→ report
→ 再决定下一项
```

---

# 14. Mandatory Static Gate

**所有 candidate 在上设备前先检查 assembly。**

至少记录：

|Metric|Baseline|Candidate|
|---|---|---|
|Instructions|112||
|Packets|36||
|Splats|8||
|Rhf mul|14||
|Eval spill|0||
|Eval stack|0 B||
|Caller stack refs|baseline||
|Caller frame|baseline||

硬性拒绝：

```
eval spill > 0
```

原则上拒绝：

```
large caller frame increase
large stack-reference increase
```

除非存在非常明显的 packet/latency 收益，否则不能进入正式 Attention benchmark。

---

# 15. Evaluation Protocol

每个通过 static gate 的 candidate 都需要：

### Level 1 — Micro

```
ns / 64 useful elements
95% CI
relative to pairret
relative to EXP-LUT if available
```

### Level 2 — Correctness

完整 72-case matrix。

### Level 3 — Full Attention

```
q1
q4
q8
q16
q32
```

### Level 4 — Stage diagnostic

至少继续记录：

```
Host
scna_exp sum
profiled_total sum
```

因为原报告已经证明 evaluator 收益会被其他 Attention 阶段稀释：`pairret` 的 diagnostic `scna_exp` 从 1285 降到了 967，即约 -24.75%，而端到端收益明显更小。

---

# 16. Success Criteria

Stage 1 不再要求：

```
SCNA pure exp must beat LUT
```

分三级判断。

### Goal A — Microkernel

优先目标：

\[ T_{\rm SCNA} \le1.10T_{\rm LUT} \]

如果进入：

\[ \le1.05T_{\rm LUT} \]

视为非常强的结果。

---

### Goal B — Attention

至少：

```
q8
q16
q32
```

相对于当前 `d7_pairret_noinline` 获得稳定 improvement。

注意：

> 新 Stage 1 的 baseline 是 pairret，不再是 static-d8。

---

### Goal C — Fused Scale

如果：

\[ T_{\rm fused-scale+SCNA} \le T_{\rm scale+EXP-LUT} \]

则即使 pure SCNA 仍略慢于 LUT，也可以判定 Stage 1 达成核心目标。

---

# 17. Mandatory Experiment Reporting

这一节是**强制要求**。

\[ \boxed{\text{实验完成但没有报告 = 实验未完成}} \]

Codex 每完成一个 Experiment，不允许只留下代码和数字。

---

## 17.1 Directory

建议：

```
reports/stage1_scna/
├── 00_baseline.md
├── 01_packet_analysis.md
├── 02_two_accumulator.md
├── 03_splat_audit.md
├── 04_constant_lifetime.md
├── 05_local_scheduling.md
├── 06_software_pipeline.md
├── 07_abi_small_q.md
├── 08_range_folding.md
├── 09_scale_fusion.md
├── experiment_log.md
└── FINAL_STAGE1_REPORT.md
```

没有执行的实验不需要创建假报告。

---

# 17.2 Every experiment report must contain

## 1. Question

明确一句话：

> 本实验试图回答什么？

例如：

> SCNA 剩余性能瓶颈是否来自单 accumulator dependency chain？

---

## 2. Hypothesis

例如：

> Two-accumulator evaluation can shorten the dependency chain without reproducing quad pipeline's register-pressure problem.

---

## 3. Code Changes

记录：

```
files changed
functions changed
commit
artifact SHA256
compile flags
```

描述核心改动。

不要贴大量无关源码。

---

## 4. Assembly Evidence

必须包含：

```
instructions
packets
splat
Rhf mul
spill
stack
caller frame
caller stack references
```

如果性能结论依赖 packet scheduling，贴关键 packet 片段。

---

## 5. Correctness

至少：

```
cases
max RMSE
max abs
finite
mask
tail
PASS / FAIL
```

---

## 6. Microbenchmark

至少：

|Version|ns/64|95% CI|Ratio|
|---|---|---|---|
|pairret baseline|||1.000|
|candidate||||

---

## 7. Full Attention

|Version|q1|q4|q8|q16|q32|
|---|---|---|---|---|---|
|pairret||||||
|candidate||||||

必须给：

```
ratio
95% CI
```

而不只是绝对 latency。

---

## 8. Stage Diagnostic

至少：

```
scna_exp
profiled_total
Host
```

用于区分：

```
microkernel 没变快
```

与：

```
microkernel 变快但被其他阶段稀释
```

---

## 9. Interpretation

这是最重要的一节之一。

禁止只写：

> Candidate is 4% faster.

必须写：

> Candidate q32 improves by X%, while evaluator packet count decreases from 36 to Y and caller stack remains unchanged. The result therefore supports/rejects the hypothesis that accumulator dependency was a meaningful bottleneck.

也就是：

\[ \boxed{ \text{数据} + \text{assembly/architecture evidence} + \text{解释} } \]

---

## 10. Decision

每个实验最后必须明确：

```
Decision:
    KEEP / REJECT / INCONCLUSIVE

Reason:

Current best implementation:

Remaining bottleneck:

Recommended next experiment:
```

---

# 17.3 Negative results are mandatory

任何失败 variant 不得删除。

例如：

```
Experiment:
2-neuron interleaving

Expected:
reduce dependency bubbles

Observed:
packets 36 → 35
but caller frame 0 → 1024 B
Attention q32 regressed 1.8%

Interpretation:
additional ILP did not amortize register-pressure cost

Decision:
REJECT
```

你的当前报告已经很好地体现了这种做法：inline、prebroadcast、quad 都保留了失败原因和验证证据，而不是只保留最好结果。SCNA_HVX_D8_PIPELINE_V79_REPORT_ZH.mdMD

继续保持这一标准。

---

# 18. Final Stage 1 Report

阶段结束必须生成：

```
reports/stage1_scna/FINAL_STAGE1_REPORT.md
```

结构固定为：

```
# Stage 1: SCNA HVX Microkernel Optimization

## 1. Executive Summary

## 2. Starting Point
d7_pairret_noinline

## 3. Bottleneck Analysis
packet
dependency
register pressure
ABI

## 4. Experiments
A...
B...
C...

## 5. Correctness

## 6. Microbenchmark Results

## 7. Full Attention Results

## 8. Assembly / Packet Analysis

## 9. Fused-Scale Result

## 10. Ablation

## 11. Negative Results

## 12. Current Best SCNA Kernel

## 13. Remaining Gap to EXP-LUT

## 14. Bottleneck Remaining After Stage 1

## 15. Recommendation for Stage 2
```

---

# 19. Required Final Ablation Table

最终必须至少有：

|Variant|Packets|Spill|Micro|q8|q16|q32|vs LUT|
|---|---|---|---|---|---|---|---|
|static-d8|41|0||||||
|pairret baseline|36|0||||||
|two-acc||||||||
|best splat opt||||||||
|best lifetime opt||||||||
|best scheduling||||||||
|**Stage-1 final**||||||||
|EXP-LUT|—|—|||||1.00×|

同时单独给：

|Scale path|Latency|
|---|---|
|separate scale + EXP-LUT||
|separate scale + SCNA||
|fused scale + SCNA||

---

# 20. Hard Boundary of This Plan

在新项目 `/home/corleone/code/Snapdragon-NPU/projects/stage1_SCNA_HVX_Microkernel_Optimization` 中独立实现；仅复制必要源码、脚本和测试，不复制旧结果、构建产物或虚拟环境，不改动原项目及现有工作树。

以现有 `d7_pairret_noinline` 为纵向基线；Origin HVX、EXP-LUT 为横向基线。三者必须在新实验中同机、同参数重新测量，不复用历史延迟。

最近的一次优化项目位于 /home/corleone/code/Snapdragon-NPU/projects/flashattention-scna-hvx-fp16-d8-pipeline-v79

报告为/home/corleone/code/Snapdragon-NPU/projects/flashattention-scna-hvx-fp16-d8-pipeline-v79/results/runs/20260821_pipeline_formal_v1/SCNA_HVX_D8_PIPELINE_V79_REPORT_ZH.md

完成 Stage 1 后：

**STOP。**

不要自行继续做：

```
register-resident whole Softmax
Softmax pass fusion
row/head interleaving outside SCNA
HMX/HVX overlap
S/P double buffering
K/V DMA overlap
Br/Bc cost model
```

最后只提交：

```
1. Stage-1 final code
2. reproducible benchmark
3. correctness result
4. assembly evidence
5. FINAL_STAGE1_REPORT.md
6. recommendation for Stage 2
```

然后等待下一份独立计划。

---

## References Codex should read

第一优先级仍然是 **V79 HVX Programmer Reference Manual**，重点不是通读，而是围绕当前 36 packets 查 packet/resource/latency/scalar-vector operand。  
[V79 HVX Programmer Reference Manual](https://docs.qualcomm.com/doc/80-N2040-61/topic/introduction.html?utm_source=chatgpt.com)

SCNA 数学结构以 **SCOPE** 为准，不在本阶段重新设计 approximation algorithm。

FlashAttention-V 只用于参考 **loop reorder / unroll / vector register utilization / ILP** 思路，不允许把 Stage 1 扩展成完整 Attention 优化。[arXiv](https://arxiv.org/abs/2608.18656?utm_source=chatgpt.com)  
[FlashAttention-V](https://arxiv.org/abs/2608.18656?utm_source=chatgpt.com)

