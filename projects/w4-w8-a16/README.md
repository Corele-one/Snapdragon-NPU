# W4/W8 A16 reference implementation

该目录保存最初参考的 llama.cpp + HTP operator 实现。它的准确计算口径是：

> Weight-only W4/W8 with FP16 HMX activation/compute and FP32 host ABI.

## 目录

```text
w4-w8-a16/
├── README.md
└── src
    ├── htp-ops-lib-main
    │   ├── include
    │   └── src
    │       ├── dsp
    │       └── host
    └── llama.cpp-npu-htp-backend
        ├── common
        ├── examples
        ├── extras
        ├── ggml
        ├── include
        ├── src
        └── tests
```

### `src/htp-ops-lib-main`

- `include/op_reg.h`：operator ID、参数和 host/DSP ABI。
- `include/htp_ops.idl`：FastRPC IDL。
- `include/dsp/quants.h`：Q4_0、IQ4_NL、Q8_0 block 定义。
- `include/dsp/hmx_utils.h`：FP16 HMX tile load、MMA 和 accumulator helper。
- `src/host/`：AArch64 stub、session 和 standalone test。
- `src/dsp/commu.c`：FastRPC DSP 入口。
- `src/dsp/op_executor.cc`：DSP operator dispatch。
- `src/dsp/ops/mat_mul.c`：W4/W8 解量化、VTCM、DMA、HVX/HMX pipeline。

### `src/llama.cpp-npu-htp-backend`

- `ggml/src/ggml-htp/htp-ops.cc`：GGML op 判断、rpcmem/FastRPC 和 Q4/Q8/F16 分派。
- `ggml/src/ggml-htp/op_reg.h`：host 侧 ABI mirror。
- `ggml/src/ggml-htp/`：HTP backend、buffer 和 fallback。
- `extras/convert_hf_to_gguf_htp.py`：把 FP16 tensor 转为 HMX storage layout。
- `examples/quantize/`：llama.cpp quantizer。
- `examples/main/`、`examples/server/`：CLI 和 HTTP server。
- `tests/`：llama.cpp/GGML 测试框架。

## 精度分派

核心分派位于 `ggml/src/ggml-htp/htp-ops.cc`：

- `GGML_TYPE_Q4_0` -> `HTP_OPS_MAT_MUL_PERMUTED_W4D16A32`
- `GGML_TYPE_Q8_0` -> `HTP_OPS_MAT_MUL_PERMUTED_W8D16A32`
- `GGML_TYPE_IQ4_NL` -> `HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL`

`mat_mul.c` 使用 HVX 将量化 weight block 解码成 FP16 tile，并把 FP32 activation 转成 FP16，再交给 HMX。默认示例使用 `IQ4_NL+Q8_0` 混合量化，因此同时覆盖 W4 和 W8。

源码中另有 `W8PC_A8PT` 路径，它是真 INT8 activation 路径，不属于本 README 的 A16 口径。

## 环境

- WSL2 Ubuntu 22.04；
- CMake、Ninja、Python；
- Android NDK r25c；
- 获授权的 Hexagon SDK 6.x；
- v73+、支持 FP16 HMX 的 Snapdragon 设备；
- Hugging Face 模型和 `adb`。

以下示例假设当前目录为 `projects/w4-w8-a16/`，并且 WSL 环境脚本已经提供 `build_cmake`、Android NDK 和 Hexagon toolchain。

## Reproduction

### 1. 构建 HTP operator

PowerShell：

```powershell
$ProjectWsl = (wsl.exe wslpath -a (Resolve-Path .\src\htp-ops-lib-main)).Trim()
wsl.exe -d Ubuntu-22.04 -u root -- bash -lc "
set -e
source /root/llama-npu-env.sh
cd '$ProjectWsl'
build_cmake android
build_cmake hexagon DSP_ARCH=v73
"
```

主要输出通常位于：

```text
src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/
src/htp-ops-lib-main/hexagon_ReleaseG_toolv*_v73/ship/
```

### 2. 构建 llama.cpp HTP backend

在 WSL 中：

```bash
cd src/llama.cpp-npu-htp-backend
cmake -S . -B build-android-htp-v73-shared -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DGGML_HTP=ON \
  -DGGML_OPENMP=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android-htp-v73-shared \
  --target llama-cli llama-quantize ggml-htp -j8
```

### 3. 转换 FP16-HMX 模型

```bash
python src/llama.cpp-npu-htp-backend/extras/convert_hf_to_gguf_htp.py \
  --outfile /path/to/qwen2.5-1.5b.f16-hmx.gguf \
  --outtype f16 \
  /path/to/Qwen2.5-1.5B-Instruct
```

### 4. 量化为混合 W4/W8

```bash
REPACK_FOR_HVX=1 \
src/llama.cpp-npu-htp-backend/build-android-htp-v73-shared/bin/llama-quantize \
  /path/to/qwen2.5-1.5b.f16-hmx.gguf \
  /path/to/qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf \
  IQ4_NL+Q8_0
```

模型不应提交到 Git。

### 5. 部署和运行

把 AArch64 binary/shared libraries、`libhtp_ops.so`、Hexagon `libhtp_ops_skel.so` 和 GGUF 放入设备上的同一隔离目录或其 `cdsp/`、`dsp/` 子目录，然后：

```bash
cd /data/local/tmp/snapdragon-npu-w4w8
export LD_LIBRARY_PATH="$PWD:/vendor/lib64:/system/lib64"
export DSP_LIBRARY_PATH="$PWD:$PWD/cdsp:$PWD/dsp"
./llama-cli \
  -t 4 \
  -fa \
  -m qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf \
  -p "Hello, my name is"
```

### 6. 验证

- 先运行 HTP operator 的 standalone test，确认 FastRPC 和 skel 可加载。
- 再运行 llama.cpp 的 backend op/reference test。
- 最后用固定 prompt 做 CPU/HTP 输出和端到端生成对照。

standalone `ret=0` 仅表示 RPC/kernel 执行完成，不单独证明量化模型的数值或生成质量正确。

## 许可边界

llama.cpp 子树保留 MIT `LICENSE`。HTP operator 快照没有明确许可证，且构建需要用户自行补齐未上传的专有 SDK 头文件。详情见仓库根目录 `THIRD_PARTY_NOTICES.md`。
