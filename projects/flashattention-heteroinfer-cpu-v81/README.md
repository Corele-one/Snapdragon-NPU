# HeteroInfer-Inspired CPU Scheduling for FlashAttention v81

该项目只研究 CPU host stub 等待、请求生命周期和 DSP dispatch profiling。Attention 数学与 DSP baseline kernel 保持不变，不包含 SCNA kernel、SCNA mode、SCNA 参数或 GPU-NPU tensor parallelism。

## Benchmark Integrity

**本项目主矩阵全部是 SM8750P / Hexagon v81 实测。** legacy、spin、predictive 使用相同 baseline `flash_attn.c`、相同 v81 DSP binary、相同输入和同一设备；等待策略之间的输出逐字节一致。

| 数据范围 | 平台 | 是否用于当前主结论 | 说明 |
|---|---|---:|---|
| `results/v81/heteroinfer-cpu/stage1-main-20260801/` | SM8750P / v81 | 是 | 36 配置、720 measured samples，含机器可读 provenance |
| `results/v81/heteroinfer-cpu/stage1-correctness-20260801/` | SM8750P / v81 | 是 | 9 个 FP32 gate 和跨 wait-policy 字节一致性 |
| `results/v73/`、`results/v79/` | 历史设备/架构 | 否 | 从干净 baseline 快照继承，仅作历史归档，禁止进入本实验图表和 speedup |

该项目的结果不能与 SCNA 项目的延迟拼接成同一 speedup。CPU 调度收益只在 baseline kernel 固定的条件下解释；HeteroInfer 论文中的 GPU-NPU 端到端收益也没有被套用到本实验。

## 固定 DSP Kernel

runner 在构建和运行前校验：

```text
src/htp-ops-lib-main/src/dsp/ops/flash_attn.c
SHA-256 = 6a8b025bda0cbaa36c3b48e3d379b7502631cdd584a7571e926e6633483356cb
```

若该 hash 改变，`verify_baseline_kernel` 会直接终止实验。这保证三种 CPU 策略比较的 DSP Attention kernel 完全相同。

## 实现范围

### Wait policy

- `legacy`：原始 `usleep(50)` 轮询。
- `spin`：持续 acquire-load completion flag，仅每 1024 polls 检查超时钟。
- `predictive`：根据同 shape 校准样本的滚动 P10 预测完成时间，先休眠，再以 100 us guard 短时间忙轮询。

### Host control path

- `--wait-policy legacy|spin|predictive`
- `--host-cpu N`
- `--host-sync-calibration N`
- request descriptor 一次预构建。
- rpcmem、channel 和 DSP fd mapping 全程常驻。
- 记录 wall latency、thread CPU、sleep/spin time、poll count 与 prediction error。
- DSP dispatch 分解为 mapping、validate-in、compute、validate-out 和 total。

## v81 实验设置

| 项目 | 设置 |
|---|---|
| Device | model `25091RP04C`，SoC `SM8750P`，Android 16，ADB serial `bde3ddde` |
| DSP | Hexagon v81 |
| Toolchain | Hexagon SDK 6.6.0.0，Hexagon Tools 19.0.07 |
| Build gate | DSP compile/link command 含 `-mv81` |
| Main shape | `qo_len={4,8,16,32}`，`kv_len=4096`，heads/KV-heads `12/2`，head-dim 128 |
| Per configuration | 20 legacy pre-calibration + 5 warmup + 20 measured |
| CPU placement | unpinned、CPU0 low-capacity、CPU6 high-capacity |
| Policy order | rotating order，降低固定执行顺序偏差 |
| Statistics | median、p50、p95，4000 次 bootstrap 95% CI |

设备实际只有两个 CPU capacity/frequency domain：CPU0-5 为 3.5328 GHz/capacity 792，CPU6-7 为 4.32 GHz/capacity 1024。因此报告使用 low-capacity/high-capacity，不虚构 little/middle/prime 分类。

## 关键结果

- 9/9 个 FP32 correctness gate 通过，legacy/spin/predictive 在每个 case 中输出 SHA-256 完全一致。
- legacy host control residual 为 `41.0-110.5 us`；spin 为 `10.0-15.0 us`；predictive 为 `9.5-16.0 us`。
- predictive 消除了 `73.2%-89.1%` 的 legacy control residual。
- 固定 DSP latency 后，predictive normalized speedup 为 `1.0247x-1.0757x`，12/12 配置的 bootstrap 95% CI 下界大于 1。
- predictive thread CPU time 中位数约为 spin 的 `5.07%`，即 CPU 时间降低约 `19.7x`。
- observed speedup 为 `0.979x-1.341x`，但 DSP dispatch 同时偏移 `-20.0%` 到 `+5.6%`。超过 Amdahl 上限的部分属于 DSP/Fabric/DVFS 耦合，不能归因于 CPU 等待策略。
- 720 个 measured samples 中，DSP dispatch `total - sum(components)` 为 `0-1 us`，中位数 `0 us`。

## 代码与数据

```text
src/htp-ops-lib-main/src/host/test.c          wait policy、校准和 host metrics
src/htp-ops-lib-main/src/dsp/op_executor.cc   DSP dispatch 分解
scripts/heteroinfer_cpu_common.sh              baseline hash、v81 build/deploy gate
scripts/run_heteroinfer_cpu_v81.sh             主性能矩阵
scripts/run_heteroinfer_cpu_correctness.sh     FP32 与字节一致性矩阵
scripts/analyze_heteroinfer_cpu.py              统计、置信区间和 SVG
results/v81/heteroinfer-cpu/                   独立 raw data 与汇总
docs/stage-reports/                            分阶段报告
```

详细报告：

- [阶段一：预测等待与控制面剖析](docs/stage-reports/HeteroInfer_CPU_阶段一_预测等待与控制面剖析_2026-08-01.md)

机器可读实验信息：

- [主矩阵 provenance](results/v81/heteroinfer-cpu/stage1-main-20260801/provenance.txt)
- [正确性 provenance](results/v81/heteroinfer-cpu/stage1-correctness-20260801/provenance.txt)
- [主矩阵 summary](results/v81/heteroinfer-cpu/stage1-main-20260801/summary/summary.md)

## 复现

runner 会校验 baseline hash、构建 Android/v81 binaries、检查 `-mv81`、部署到设备并采集 provenance：

```bash
cd /home/corleone/code/Snapdragon-NPU/projects/flashattention-heteroinfer-cpu-v81
adb devices
./scripts/run_heteroinfer_cpu_correctness.sh
./scripts/run_heteroinfer_cpu_v81.sh
```

复用已构建和部署的 binary 时可显式跳过 build，但仍会执行 baseline hash gate：

```bash
SKIP_BUILD=1 ./scripts/run_heteroinfer_cpu_v81.sh
```

默认结果分别写入：

```text
results/v81/heteroinfer-cpu/stage1-correctness-<stamp>/
results/v81/heteroinfer-cpu/stage1-main-<stamp>/
```

## 结果解释规则

1. CPU 调度的主结论使用 normalized speedup，即固定同组 legacy DSP 中位数后只比较 host control residual。
2. observed wall speedup 保留真实系统行为，但不能在 DSP dispatch 发生变化时全部归因于 wait policy。
3. spin 的延迟收益必须连同约 91%-100% 单核占用一起报告。
4. 三种策略必须先通过输出字节一致性和 FP32 reference gate。
5. 任何 v73/v79、SCNA 或 LUT 数据都不得进入本实验主图、主表或摘要结论。

## 研究边界

- 当前是 standalone synthetic Figure8 benchmark，尚未接入完整 llama runtime 的多 op 请求队列。
- 未采集完整模型 token latency、能耗或跨时段独立 session 方差。
- 未实现 HeteroInfer 的 GPU-NPU tensor parallelism。
- CPU 活跃度可能影响共享 fabric 和 DSP DVFS，因此下一阶段需要同步采集 CPU/fabric/DSP clocks。

## 许可边界

llama.cpp 子树保留 MIT `LICENSE`。HTP operator 快照依赖获授权的 Qualcomm Hexagon SDK；不要提交 proprietary SDK headers 或未获授权的二进制。
