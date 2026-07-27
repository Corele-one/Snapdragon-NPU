# Snapdragon NPU research implementations

本仓库整理了 Snapdragon Hexagon NPU 上三条 LLM 推理实现线：W4/W8 权重仅量化、pure FP16 和 FlashAttention。仓库保留了 llama.cpp/GGML 的 C++ host 框架、HTP FastRPC 接口、Hexagon DSP kernel、构建与分析脚本，以及可公开保留的实验结果。

> 本仓库当前应保持为私有研究快照。HTP operator 源码没有随附明确的上游许可证，并依赖用户自行从获授权的 Qualcomm Hexagon/QAIRT SDK 环境补齐专有头文件。请先阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，不要仅依据仓库根目录的 MIT `LICENSE` 推断所有子目录都可重新分发。

## 项目总览

| 目录 | 功能 | 主要精度/路径 | 详细说明 |
|---|---|---|---|
| `projects/w4-w8-a16/` | 原始 llama.cpp HTP 权重仅量化参考实现 | W4/W8 weight，FP16 HMX activation/compute，FP32 host ABI | [README](projects/w4-w8-a16/README.md) |
| `projects/pure-fp16/` | 完整 pure-FP16 推理与后续 DMA/VTCM/output-stationary 优化 | FP16 weight + FP16 HMX compute，FP32 host ABI | [README](projects/pure-fp16/README.md) |
| `projects/flashattention/` | HTP FlashAttention、runtime mode、Figure 8 trace/对比工具 | FP16 HMX QK/PV + HVX softmax | [README](projects/flashattention/README.md) |
| `reference/w4-w8-a16-mllm/` | 字面意义上的 QNN graph W4A16/W8A16 参考快照 | QNN AOT W4A16/W8A16 | [SNAPSHOT](reference/w4-w8-a16-mllm/SNAPSHOT.md) |

三个 `projects/*/src/` 都是可独立阅读和构建的 source-only 闭包。为保证每条实现线能单独复现，llama.cpp 框架在三条实现中分别保留；这会产生源码重复，但避免不同实验线之间的源码漂移和隐式依赖。

## 术语与数据流

### W4/W8 A16

`projects/w4-w8-a16/` 的准确描述是：

> Weight-only W4/W8 with FP16 HMX activation/compute and FP32 host ABI.

GGML 的 `Q4_0`、`Q8_0` 和 `IQ4_NL` 权重在 DSP 上由 HVX 解量化为 FP16；输入 activation 从 FP32 caller ABI 转成 FP16；FP16 activation 和 FP16 weight tile 再进入 HMX。它不是把整个 host ABI 改成 FP16，也不要与源码中独立的 W8PC/A8PT INT8 activation 路径混淆。

### Pure FP16

pure-FP16 路径不包含 INT4/INT8 weight dequant：

```text
GGML F16 tensor
  -> llama.cpp/GGML HTP backend
  -> FastRPC W16A32 request
  -> FP32 activation 转 FP16 并进入 VTCM
  -> HMX-layout FP16 weight
  -> HMX FP16 MMA
  -> FP16 output 转回 FP32 caller ABI
```

该目录同时保留后续的 DMA/VTCM、parallel publish、parallel output store、activation cache 和 output-stationary/split-K 实验优化。

### FlashAttention

```text
GGML_OP_FLASH_ATTN_EXT
  -> ggml/src/ggml-htp/htp-ops.cc
  -> FastRPC HTP_OPS_FLASH_ATTN_QO_F32_KV_F16
  -> DSP op_executor
  -> HMX QK/PV + HVX softmax
```

`v73` 与 `v79` 是同一套源码以不同 `DSP_ARCH` 构建；`baseline` 与 `lut_exp` 是运行时 mode 切换，并非两套源码。

## 仓库目录

```text
.
├── LICENSE
├── README.md
├── THIRD_PARTY_NOTICES.md
├── projects
│   ├── w4-w8-a16
│   │   └── src
│   │       ├── htp-ops-lib-main
│   │       └── llama.cpp-npu-htp-backend
│   ├── pure-fp16
│   │   ├── deploy-template
│   │   ├── evidence
│   │   ├── scripts
│   │   ├── src
│   │   │   ├── htp-ops-lib-main
│   │   │   └── llama.cpp-npu-htp-backend
│   │   └── tools
│   └── flashattention
│       ├── results
│       ├── src
│       │   ├── htp-ops-lib-main
│       │   └── llama.cpp-npu-htp-backend
│       └── tools
└── reference
    └── w4-w8-a16-mllm
```

各源码闭包中的关键层次：

- `src/htp-ops-lib-main/include/`：host/DSP ABI、IDL、operator registry、HMX/HVX helper 声明。
- `src/htp-ops-lib-main/src/host/`：Android AArch64 FastRPC stub、session 和 standalone test。
- `src/htp-ops-lib-main/src/dsp/`：DSP dispatch、worker、VTCM/HMX 管理和 kernel。
- `src/llama.cpp-npu-htp-backend/ggml/src/ggml-htp/`：GGML HTP backend、buffer、FastRPC 和 CPU fallback。
- `src/llama.cpp-npu-htp-backend/examples/server/`：HTTP server 集成。
- `src/llama.cpp-npu-htp-backend/extras/convert_hf_to_gguf_htp.py`：HMX layout GGUF 转换。

## 通用环境

复现需要用户自行准备：

- Windows PowerShell 7 或 Windows PowerShell 5.1；
- WSL2 Ubuntu 22.04；
- CMake、Ninja、Python 3；
- Android SDK platform tools（`adb`）和 Android NDK r25c；
- 获授权的 Qualcomm Hexagon SDK 6.x/QAIRT 环境；
- Snapdragon 8 Gen 2 或更高、Hexagon v73+ 且支持 FP16 HMX 的 Android 设备；
- 用户自行取得的 Hugging Face 原始模型权重。

SDK 文件、模型、`.so`、可执行文件和构建缓存没有进入仓库。仓库也不会执行 `adb kill-server` 或重启设备。

## 最短复现入口

### W4/W8 A16

```powershell
Set-Location .\projects\w4-w8-a16
```

按照该目录 [README](projects/w4-w8-a16/README.md) 构建 HTP operator、llama.cpp HTP backend，转换 F16-HMX GGUF，并运行 `IQ4_NL+Q8_0` 量化。

### Pure FP16

先转换或准备 HMX-layout FP16 GGUF，然后：

```powershell
Set-Location .\projects\pure-fp16
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_pure_fp16.ps1 `
  -DspArch v73 `
  -Jobs 8 `
  -ModelPath C:\path\to\qwen2.5-1.5b-instruct.f16-hmx.gguf
```

构建脚本还支持 `-WslEnvScript` 和 `-AndroidNdkRootWsl`，因此干净 clone 不依赖原开发机的盘符。完整部署和 smoke/trace/PD sweep 流程见 [pure-FP16 README](projects/pure-fp16/README.md)。

### FlashAttention

```powershell
Set-Location .\projects\flashattention
```

按照该目录 [README](projects/flashattention/README.md) 构建 v73 或 v79，使用同一份 skel 分别运行 `--mode baseline` 与 `--mode lut-exp`，再用 `tools/` 下的脚本生成 CSV、summary JSON 和 Perfetto/NTFF trace。

## 验证口径

本仓库区分以下验证层级：

1. **静态/语法检查**：Python `py_compile`、PowerShell parser、路径闭包和敏感信息扫描。
2. **构建检查**：需要本地 WSL + 获授权 SDK；生成物不会提交。
3. **standalone RPC/kernel smoke**：返回码 0 只表示调用执行完成。
4. **数值正确性**：需要 CPU reference 对照，例如 FlashAttention 使用 `test-backend-ops -o FLASH_ATTN_EXT -b HTP`。
5. **端到端服务验证**：需要实机、模型、部署产物和 API 请求。

FlashAttention 的 Perfetto/NTFF 文件记录 DSP qtimer 软件事件，不等同于 Qualcomm PMU 的 HMX/HVX active-cycle 或利用率计数。

## 未随仓库提供的内容

- GGUF、Hugging Face 权重及任何训练数据；
- Qualcomm SDK、QAIRT、受限头文件或文档；
- Android/Hexagon 构建输出；
- 设备序列号、凭据和本地绝对路径配置；
- FlashAttention 派生的大型 `.ntff`/`.perfetto.json` 重复文件。

请从各项目 README 开始复现，并在公开传播或再分发前核对 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
