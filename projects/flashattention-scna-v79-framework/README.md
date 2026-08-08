# Hexagon v79 上的 SCNA FlashAttention 实验

这是 SCNA 优化实验的独立项目。当前阶段已迁入 FlashAttention 的完整 HTP/FastRPC 基线、Figure 8 profile ABI 和性能分析工具；SCNA 近似本身尚未实现。后续 SCNA 代码应在不改变基线语义的前提下接入同一条 benchmark 链路。

## 本项目所对标的 baseline

本项目的两条 baseline 共用相同的 FlashAttention 分块、online safe-softmax、HMX QK/PV 矩阵计算、VTCM 分配、输入 shape 与 profile timer；唯一受控变量是 safe-softmax 中 `exp2` 的求值方式。

| CLI mode | 对标实现 | safe-softmax 非线性路径 | 用途 |
|---|---|---|---|
| `baseline` | Origin HVX / 不使用 LUT 的 FlashAttention | HVX 向量多项式 `exp2`，`mode_flags=0` | SCNA 的主要原始基线。 |
| `lut-exp` | LUT-EXP FlashAttention | 使用已在 backend 初始化时准备的 VTCM `exp2` 表，并以 HVX `vgather` 查询，`LLM_NPU_MODE_LUT_EXP=1` | 查表非线性路径对照。 |

实现位置为：

- `src/htp-ops-lib-main/src/dsp/ops/flash_attn.c`：两条路径的实际分支；`enable_vgather_exp` 由运行时 flag 决定。
- `src/htp-ops-lib-main/src/dsp/ops/precompute_table.c`：LUT-EXP 的 VTCM `exp2` 表初始化。
- `src/htp-ops-lib-main/src/host/test.c`：`htp_ops_test --figure8-attn --mode baseline|lut-exp` standalone benchmark。
- `src/htp-ops-lib-main/include/op_reg.h`：FlashAttention request、mode flag 与 Figure 8 profile ABI。

`baseline` 并不表示整个 attention 都只使用 HVX：QK/PV 主计算仍走原始 HMX 路径；“Origin HVX”特指不经 LUT 的 HVX safe-softmax/指数求值。为让同一 skeleton 支持 runtime 切换，backend 初始化会准备 LUT 表；但 `baseline` 的计时热路径不会执行 `vgather`，因此对比衡量的是两种求值路径而非建表成本。两条 baseline 的比较必须固定 DSP 架构、设备、shape、warmup、迭代次数和采集会话，不能用跨架构结果计算 speedup。

## v79 与 v81，以及本项目选择 v79 的原因

| 项目 | Hexagon v79 | Hexagon v81 |
|---|---|---|
| ISA/运行时目标 | 使用 `DSP_ARCH=v79`，DSP 编译参数为 `-mv79`，链接 v79 对应 QuRT/runtime。 | 使用 `DSP_ARCH=v81`，编译参数为 `-mv81`，需要 SDK 同时提供 v81 QuRT/runtime。 |
| 代码生成 | 本项目当前唯一的性能目标；迁入基线已以 v79 构建核验。 | 新一代 ISA；参考代码对某些 v81 BF16/FP8 HVX intrinsic 还需要专门的 IEEE FP lowering 设置。 |
| 数据可比性 | 可作为本项目 baseline、SCNA 变体和消融实验的统一分母。 | 不能与 v79 延迟、吞吐或 timer 数据混合后得出 speedup。 |

选择 v79 的原因是本项目的实验契约是 **v79 baseline 对 v79 SCNA**：迁入的 Origin-HVX 与 LUT-EXP 对照均已按 `-mv79` 构建，能让后续变更只归因于 SCNA 实现，而非 ISA、QuRT runtime 或工具链变化。v81 不是 v79 的性能等价替代：即使同一设备能够加载较低 ISA 的 binary，也不能把该结果宣称为原生 v81 性能；如需做 v81 实验，必须独立以 `-mv81` 重建两条 baseline 与全部 SCNA 变体，并单独采集数据。

## 目录结构

```text
flashattention-scna-v79-v2/
├── README.md
├── docs/
│   └── 环境验证执行总结_2026-08-07.md
├── results/
│   └── README.md
├── scripts/
│   ├── build.sh
│   ├── deploy_and_smoke.sh
│   ├── run_figure8_baselines.sh
│   └── use_hexagon_sdk_6_6.sh
├── src/htp-ops-lib-main/
│   ├── include/                 # FastRPC、message 和 profile ABI
│   ├── src/host/                # htp_ops_test 与环境 ping
│   └── src/dsp/                 # HTP baseline、VTCM/HMX/HVX 实现
└── tools/
    ├── parse_figure8_attention_timers.py
    ├── compare_figure8_lut_exp.py
    ├── generate_figure8_perfetto_trace.py
    ├── parse_figure8_long_kv_breakdown.py
    ├── parse_llm_inference_trace.py
    └── serve_trace_cors.py
```

## 前置条件

- Linux/WSL、CMake、Ninja、Python 3；
- 已获授权的 Qualcomm Hexagon SDK（默认环境脚本使用 6.6.0.0）；
- SDK 配置的 Android NDK；
- 设备验证需要 `adb` 及已连接、授权的 Snapdragon 设备。

若 SDK 不在默认位置：

```bash
export SCNA_HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0
```

## 构建 v79

```bash
./scripts/build.sh --dsp-arch v79
```

构建产物为：

```text
src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/
    htp_ops_test
    libhtp_ops.so
    scna_env_smoke
    libscna_env.so
src/htp-ops-lib-main/hexagon_ReleaseG_toolv*_v79/ship/
    libhtp_ops_skel.so
    libscna_env_skel.so
```

可用以下命令核验实际 ISA，而不是仅依赖输出目录名：

```bash
rg -- '-mv79' src/htp-ops-lib-main/hexagon_ReleaseG_toolv*_v79/build.ninja
```

## 部署和单次验证

先确认设备在线：

```bash
adb devices -l
```

部署所有 artifact 并执行无状态 FastRPC ping（不申请 VTCM/HMX）：

```bash
./scripts/deploy_and_smoke.sh --mode ping
```

执行单次 Origin-HVX baseline 或 LUT-EXP：

```bash
./scripts/deploy_and_smoke.sh --mode baseline --qo-len 4 --kv-len 4096
./scripts/deploy_and_smoke.sh --mode lut-exp --qo-len 4 --kv-len 4096
```

成功的 attention 运行会输出 `FIG8_ATTENTION_*` 记录和 `ret=0`。这只表示 FastRPC 与 kernel 调用完成；数值正确性仍应在后续 SCNA 接入时单独建立 CPU reference gate。

## 采集与分析 baseline

使用默认 shape（`qo_len=4,8,16,32`、`kv_len=4096`、heads/KV-heads=`12/2`、head dimension=`128`、5 warmup + 20 次测量）采集两条 baseline：

```bash
./scripts/run_figure8_baselines.sh
```

该脚本会先执行 `ping`，再保存原始日志至 `results/local/v79/`，最后依次调用：

```bash
python3 tools/parse_figure8_attention_timers.py \
  --input-dir results/local/v79/baseline \
  --out-dir results/local/v79/baseline

python3 tools/compare_figure8_lut_exp.py \
  --baseline-summary results/local/v79/baseline/attention_timers_summary.json \
  --lut-exp-summary results/local/v79/lut_exp/attention_timers_summary.json \
  --out-dir results/local/v79/lut_exp
```

其他迁入工具的用途：

- `generate_figure8_perfetto_trace.py`：由原始 Figure 8 log 生成 Perfetto/NTFF 事件；
- `parse_figure8_long_kv_breakdown.py`：解析长 KV 场景并生成 baseline/LUT-EXP 分解对比；
- `parse_llm_inference_trace.py`：解析完整 llama.cpp HTP inference trace；
- `serve_trace_cors.py`：为本地 trace 查看提供 CORS 静态服务。

不要将 Figure 8 软件 timer 直接解释为 HMX/HVX PMU utilization；它们是 profile software event，而不是硬件 active-cycle 计数器。

## 后续 SCNA 接入边界

后续应在 `src/htp-ops-lib-main/src/dsp/ops/` 增加 SCNA evaluator，并让 host mode 或专用 flag 选择它。SCNA、`baseline` 和 `lut-exp` 必须共用同一输入、mask、shape、profile buffer 与 VTCM/HMX 调度路径，才能将差异归因于非线性求值实现。请保留 `scna_env_smoke` 和两条 baseline，作为独立环境诊断与性能分母。

请勿提交 SDK 专有头文件、构建目录或本地原始 benchmark 结果。
