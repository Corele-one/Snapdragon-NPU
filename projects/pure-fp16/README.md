# Pure FP16 implementation

该目录保存完整的 pure-FP16 llama.cpp/HTP 实现，以及后续 DMA、VTCM、parallel publish、parallel output store、activation cache 和 output-stationary/split-K 优化。

## 功能

pure-FP16 路径使用 HMX-layout FP16 weight 和 FP16 HMX MMA。GGML/host 侧 ABI 仍是 FP32 activation/output：

```text
GGML F16 tensor
  -> GGML HTP backend
  -> FastRPC W16A32
  -> FP32 activation 转 FP16
  -> FP16 weight/activation 进入 VTCM/HMX
  -> HMX MMA
  -> FP16 accumulator/output 转回 FP32 ABI
```

这里没有 W4/W8 weight dequant。

## 目录

```text
pure-fp16/
├── README.md
├── deploy-template
│   └── run_server.sh
├── evidence
│   ├── profiling_traces
│   └── sequence_benchmarks
├── scripts
│   ├── build_pure_fp16.ps1
│   ├── common.ps1
│   ├── rebuild_pd_sweep_summary.ps1
│   ├── run_pd_sweep_pure_fp16.ps1
│   ├── run_smoke_pure_fp16.ps1
│   ├── run_trace_pure_fp16.ps1
│   ├── switch_llm_inference_route.ps1
│   └── tune_batch_pure_fp16.ps1
├── src
│   ├── htp-ops-lib-main
│   └── llama.cpp-npu-htp-backend
└── tools
    └── parse_llm_inference_trace.py
```

### 核心源码

- `src/htp-ops-lib-main/src/dsp/ops/mat_mul.c`：W16A32 HMX kernel、DMA/VTCM pipeline、output-stationary 路径。
- `src/htp-ops-lib-main/src/dsp/op_executor.cc`：DSP dispatch、cache、profile/trace。
- `src/htp-ops-lib-main/src/dsp/commu.c`：FastRPC DSP 入口。
- `src/htp-ops-lib-main/src/host/op_export.c`：host RPC wrapper。
- `src/htp-ops-lib-main/src/host/test.c`：standalone test。
- `src/htp-ops-lib-main/include/op_reg.h`：ABI 和 `LLM_NPU_MODE_PURE_FP16`。
- `src/llama.cpp-npu-htp-backend/ggml/src/ggml-htp/htp-ops.cc`：F16 tensor 的 W16A32 dispatch。
- `src/llama.cpp-npu-htp-backend/examples/server/server.cpp`：server mode 集成。
- `src/llama.cpp-npu-htp-backend/extras/convert_hf_to_gguf_htp.py`：FP16-HMX GGUF 转换。

### 优化代码

当前源码保留：

- DMA 到 VTCM scratch，再由 HVX publish 到 HMX buffer；
- 大 prefill 的多阶段 overlap；
- parallel weight publish；
- parallel FP16-to-FP32 output store；
- gate/up 相邻调用 activation cache；
- `K > N` 的 split-K/output-stationary 路径；
- short-M VTCM profile；
- output-stationary K-slice DMA gather；
- 默认关闭、供 A/B 使用的 direct-final DMA/decode direct pipeline 开关。

## 环境

- Windows PowerShell；
- WSL2 Ubuntu 22.04；
- CMake、Ninja；
- Android NDK r25c；
- 获授权的 Hexagon SDK 6.x；
- `adb`；
- v73+、支持 FP16 HMX 的 Snapdragon 设备；
- Python 和 llama.cpp GGUF 转换依赖。

构建脚本默认：

```text
WslDistro       = Ubuntu-22.04
WslEnvScript    = /root/llama-npu-env.sh
AndroidNdkRoot  = /root/Qualcomm/Hexagon_SDK/6.3.0.0/tools/android-ndk-r25c
```

这些值都可通过参数覆盖；README 中的 SDK 6.x 是兼容范围，脚本默认路径不是仓库依赖。

## Reproduction

以下命令从 `projects/pure-fp16/` 执行。

### 1. 模型转换

```bash
python src/llama.cpp-npu-htp-backend/extras/convert_hf_to_gguf_htp.py \
  --outfile /path/to/qwen2.5-1.5b-instruct.f16-hmx.gguf \
  --outtype f16 \
  /path/to/Qwen2.5-1.5B-Instruct
```

`f16-hmx.gguf` 只改变 W16A32 kernel 使用的 storage layout；值仍为 FP16，没有 INT4/INT8 per-group quantization。

### 2. 构建和打包

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_pure_fp16.ps1 `
  -DspArch v73 `
  -Jobs 8 `
  -ModelPath C:\path\to\qwen2.5-1.5b-instruct.f16-hmx.gguf
```

非默认 WSL/SDK：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_pure_fp16.ps1 `
  -DspArch v73 `
  -WslDistro Ubuntu-22.04 `
  -WslEnvScript /opt/hexagon/setup-llama-npu.sh `
  -AndroidNdkRootWsl /opt/android-ndk-r25c `
  -ModelPath C:\path\to\model.f16-hmx.gguf
```

也可设置环境变量代替 `-ModelPath`：

```powershell
$env:SNAPDRAGON_NPU_FP16_MODEL = 'C:\path\to\model.f16-hmx.gguf'
```

脚本会：

1. 构建 Android AArch64 HTP stub 和 Hexagon skel；
2. 构建 `llama-server`、`llama-cli` 和共享库；
3. 从 `deploy-template/run_server.sh` 初始化 `deploy/`；
4. 把运行时文件打包进 `deploy/`；
5. 写入本地 manifest。

`deploy/`、模型、日志和二进制均被 Git 忽略。

### 3. Standalone HTP test

构建后把以下文件推送到隔离目录：

```text
deploy/htp_ops_test
deploy/libhtp_ops.so
deploy/libhtp_ops_skel.so
```

设备上执行：

```bash
cd /data/local/tmp/snapdragon-npu-fp16
chmod 755 htp_ops_test
LD_LIBRARY_PATH=. \
DSP_LIBRARY_PATH="./cdsp;./dsp;." \
./htp_ops_test
```

返回 0 只证明 RPC/kernel smoke 完成；数值正确性仍应对照 host/CPU reference。

### 4. 部署 pure-FP16 server

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\switch_llm_inference_route.ps1 `
  -Mode pure_fp16 `
  -DspArch v73 `
  -ModelPath C:\path\to\qwen2.5-1.5b-instruct.f16-hmx.gguf `
  -Serial <device-serial> `
  -Port 8080
```

脚本只操作指定 AppDir 下的 runtime 和 `llama-server` 进程，不重启设备，也不执行 `adb kill-server`。

`deploy-template/run_server.sh` 会启用 API key。不要在局域网环境使用默认 `llama-npu-local-key`；部署前设置：

```bash
export LLAMA_API_KEY='<strong-random-secret>'
```

或在设备端执行：

```bash
./run_server.sh set-key '<strong-random-secret>'
```

### 5. Smoke

server 已部署后：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_smoke_pure_fp16.ps1 `
  -Mode pure_fp16 `
  -Port 8080 `
  -Serial <device-serial>
```

### 6. PD sweep

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_pd_sweep_pure_fp16.ps1 `
  -Port 8080 `
  -Serial <device-serial> `
  -Batch 512 `
  -Ubatch 512 `
  -Threads 4 `
  -Repeats 3 `
  -SpeedOnly
```

### 7. Detailed trace

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_trace_pure_fp16.ps1 `
  -Port 8080 `
  -Serial <device-serial> `
  -CaseName prefill_512_decode_16 `
  -TargetPrefill 512 `
  -Decode 16
```

期望检查：

- 命令退出码为 0；
- smoke 内容检查通过；
- trace 中出现 `matmul_f16` 或 `matmul_w16a32`；
- pure-FP16 trace 中不出现 `weight_hvx_dequant`。

trace 会改变时间开销，不能把 trace wall time 当作正常推理性能。

## Evidence

`evidence/` 只保留少量历史验证/性能材料，用于说明测试形状和解析格式。它不是对新设备、新 SDK 或当前源码重新验证后的承诺。复现实验时应记录设备 SoC、DSP arch、SDK、commit、模型 hash、build flags 和完整命令。

## 许可边界

llama.cpp 子树保留 MIT `LICENSE`。HTP operator 快照没有明确许可证，并缺少不能公开上传的 Qualcomm proprietary 头文件；请从获授权 SDK 环境补齐并阅读根目录 `THIRD_PARTY_NOTICES.md`。
