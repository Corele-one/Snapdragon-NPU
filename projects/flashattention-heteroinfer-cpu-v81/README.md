# FlashAttention on Snapdragon HTP

该目录保存 FlashAttention 的 DSP kernel、llama.cpp/GGML HTP 接入、Figure 8 timer 工具和 v73/v79 的紧凑测量结果。

## 目录

```text
flashattention/
├── README.md
├── results
│   ├── v73
│   │   ├── baseline
│   │   └── lut_exp
│   └── v79
│       ├── baseline
│       └── lut_exp
├── src
│   ├── htp-ops-lib-main
│   └── llama.cpp-npu-htp-backend
└── tools
    ├── compare_figure8_lut_exp.py
    ├── generate_figure8_perfetto_trace.py
    ├── parse_figure8_attention_timers.py
    ├── parse_figure8_long_kv_breakdown.py
    ├── parse_llm_inference_trace.py
    └── serve_trace_cors.py
```

## 核心实现

### DSP/HTP

- `src/htp-ops-lib-main/src/dsp/ops/flash_attn.c`：`head_dim % 64 == 0` 主路径，HMX QK/PV、HVX safe-softmax、LUT/vgather exp、profile timer。
- `src/htp-ops-lib-main/src/dsp/ops/flash_attn_sp_hdim.c`：非 64 倍数 head dimension fallback。
- `src/htp-ops-lib-main/src/dsp/op_executor.cc`：DSP op dispatch、rpcmem mapping、profile buffer 和 trace。
- `src/htp-ops-lib-main/src/host/test.c`：`htp_ops_test --figure8-attn` standalone benchmark。
- `src/htp-ops-lib-main/include/op_reg.h`：operator ID、mode flag、参数和 profile event ABI。

### llama.cpp/GGML C++ framework

- `src/llama.cpp-npu-htp-backend/src/llama.cpp`：构建 `GGML_OP_FLASH_ATTN_EXT` graph。
- `src/llama.cpp-npu-htp-backend/ggml/src/ggml-htp/`：HTP backend 和 FastRPC request。
- `src/llama.cpp-npu-htp-backend/examples/server/server.cpp`：`npu_mode` 到 `LLAMA_NPU_MODE`。
- `src/llama.cpp-npu-htp-backend/tests/test-backend-ops.cpp`：CPU reference/backends 对照测试入口。

## 路径关系

v73/v79 是同一源码的不同构建目标：

```text
DSP_ARCH=v73
DSP_ARCH=v79
```

baseline/LUT-exp 是 runtime mode：

```text
htp_ops_test --figure8-attn --mode baseline
htp_ops_test --figure8-attn --mode lut-exp
```

standalone 链路：

```text
htp_ops_test --figure8-attn
  -> HTP_OPS_FLASH_ATTN_PROFILE_QO_F32_KV_F16
  -> simple_flash_attn_profiled(...)
  -> mode_flags
  -> enable_vgather_exp
```

完整 llama.cpp 链路：

```text
GGML_OP_FLASH_ATTN_EXT
  -> ggml-htp/htp-ops.cc
  -> HTP_OPS_FLASH_ATTN_QO_F32_KV_F16
  -> op_executor.cc
  -> simple_flash_attn_llm_profiled(...)
```

限制：

- `baseline/lut_exp` 对比应使用主路径，即 `head_dim % 64 == 0`；本仓库 Figure 8 使用 `head_dim=128`。
- `flash_attn_sp_hdim.c` fallback 当前固定启用 vgather exp，不接收相同的 runtime mode。
- `FIGURE8_ENABLE_LUT_EXP` 只影响非-profile 默认值；standalone profile 和完整 llama 路径都由 runtime mode 控制。

## 环境

- WSL2 Ubuntu 22.04；
- CMake、Ninja；
- Android NDK r25c；
- 获授权的 Hexagon SDK 6.x；
- `adb`；
- v73/v79 Snapdragon 设备；
- Python 3。

## Reproduction

以下命令从 `projects/flashattention/` 执行。

### 1. 构建 v73

```powershell
$HtpWsl = (wsl.exe wslpath -a (Resolve-Path .\src\htp-ops-lib-main)).Trim()
wsl.exe -d Ubuntu-22.04 -u root -- bash -lc "
set -e
source /root/llama-npu-env.sh
cd '$HtpWsl'
build_cmake android
build_cmake hexagon DSP_ARCH=v73 \
  FIGURE8_ENABLE_PROFILE_TIMERS=ON \
  FIGURE8_ENABLE_LUT_EXP=OFF
"
```

v79 只需把 `DSP_ARCH=v73` 改为 `DSP_ARCH=v79`。

### 2. 部署 standalone benchmark

```powershell
$Remote = '/data/local/tmp/figure8_attn'
adb shell "mkdir -p $Remote/cdsp $Remote/dsp"
adb push .\src\htp-ops-lib-main\android_ReleaseG_aarch64\ship\htp_ops_test "$Remote/"
adb push .\src\htp-ops-lib-main\android_ReleaseG_aarch64\ship\libhtp_ops.so "$Remote/"
adb push .\src\htp-ops-lib-main\hexagon_ReleaseG_toolv88_v73\ship\libhtp_ops_skel.so "$Remote/cdsp/"
adb push .\src\htp-ops-lib-main\hexagon_ReleaseG_toolv88_v73\ship\libhtp_ops_skel.so "$Remote/dsp/"
adb shell "chmod 755 $Remote/htp_ops_test"
```

实际 Hexagon 输出目录中的 tool version 可能不同，请以构建日志为准。

### 3. 用同一份 skel 跑 baseline/LUT-exp

Baseline：

```powershell
adb shell "cd /data/local/tmp/figure8_attn && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --figure8-attn --mode baseline --qo-len 4 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup 5 --iters 20"
```

LUT-exp：

```powershell
adb shell "cd /data/local/tmp/figure8_attn && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test --figure8-attn --mode lut-exp --qo-len 4 --kv-len 4096 --n-heads 12 --n-kv-heads 2 --head-dim 128 --warmup 5 --iters 20"
```

对 `qo_len=4,8,16,32` 重复运行，并分别保存到：

```text
results/v73/baseline/raw_q*.log
results/v73/lut_exp/raw_q*.log
```

### 4. 解析 timer

```powershell
python .\tools\parse_figure8_attention_timers.py `
  --input-dir .\results\v73\baseline `
  --out-dir .\results\v73\baseline
```

### 5. 生成 Perfetto/NTFF

```powershell
python .\tools\generate_figure8_perfetto_trace.py `
  --input-dir .\results\v73\baseline `
  --out-dir .\results\v73\baseline\ntff
```

仓库忽略大型派生 `.ntff` 和 `.perfetto.json`；它们可从 `raw_q*.log` 重新生成。

### 6. 对比 baseline/LUT-exp

```powershell
python .\tools\compare_figure8_lut_exp.py `
  --baseline-summary .\results\v73\baseline\attention_timers_summary.json `
  --lut-exp-summary .\results\v73\lut_exp\attention_timers_summary.json `
  --out-dir .\results\v73\lut_exp
```

同样方式处理 v79。`results/v79/v73_vs_v79.*` 是随快照保留的静态历史对比；当前工具目录没有独立的 v73-v79 comparator。

### 7. 数值正确性

standalone log 的 `ret=0` 只说明 FastRPC/kernel 完成，不证明 output 与 reference 一致。完整数值验证应构建 llama.cpp tests，并运行：

```text
test-backend-ops -o FLASH_ATTN_EXT -b HTP
```

测试时记录 CPU reference tolerance、model/head shape、DSP arch、SDK 和 build flags。

## Results 语义

- `raw_q*.log`：可重新解析的 qtimer 原始输出。
- `attention_timers.csv` / `attention_timers_summary.json`：parser 输出。
- `baseline_vs_lut_exp.*`：同一 arch、同一 shape 的 runtime mode 对比。
- `v73_vs_v79.*`：历史静态架构对比。
- `ntff/README.md`：派生 trace 的查看说明。

Perfetto/NTFF 是 DSP qtimer 软件事件，不是 Qualcomm PMU utilization。不能把 `core_acc`、`o_scale` 或 mixed section 直接解释成 HMX/HVX hardware active-cycle。

## 许可边界

llama.cpp 子树保留 MIT `LICENSE`。HTP operator 快照没有明确许可证，并缺少不能公开上传的 Qualcomm proprietary 头文件；请从获授权 SDK 环境补齐并阅读根目录 `THIRD_PARTY_NOTICES.md`。
