# llama.cpp with custom Hexagon NPU backend

This is the code repository for the paper [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://arxiv.org/abs/2509.23324), which supports using the Hexagon NPU on Qualcomm Snapdragon SoCs for LLM inference. This project is primarily a research prototype and is not intended for production environments.

This project consists of two components: one based on llama.cpp (this repository) and an independent operator library [HTP-Ops-lib](https://github.com/haozixu/htp-ops-lib). We assume that users perform cross-compilation on a Linux host to generate executable for Android devices.

**Hardware requirements**: Android phones with Qualcomm Snapdragon 8 Gen 2 or higher SoC

**Software requirements**: CMake, Android NDK, Hexagon SDK 6.x (verified version: 6.0.0.2), Python environment required by llama.cpp

## Getting Started

### Program Compilation

#### llama.cpp (this repository)

The compilation process here is basically the same as that of llama.cpp; see the relevant documentation for details. However, this project includes the following key options when configuring cmake:

+ `-DGGML_HTP=ON`: Enables the Hexagon NPU backend, enabled by default. HTP is the abbreviation for Hexagon Tensor Processor.
+ `-DGGML_OPENMP=OFF`: Disables OpenMP support. Currently, some CPU-related implementations in the NPU hybrid backend are incompatible with OpenMP; please ensure OpenMP is disabled.

Optional options:

+ `-DBUILD_SHARED_LIBS=OFF`: Don't generate shared libraries. This can reduce the number of dynamic link libraries that need to be copied to the device but will increase the size of the executables.

The following shows a complete cross-compilation process:

1. Create a build directory

```sh
mkdir -p build; cd build
```

2. Perform CMake configuration, ensuring that the environment variable `ANDROID_NDK` corresponding to the Android NDK is set

```sh
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DGGML_HTP=ON -DGGML_OPENMP=OFF
```

3. Compile `llama-cli` and `llama-quantize` (to be used later)

```sh
make -j llama-cli llama-quantize
```

After compilation, you should be able to find the executable programs `llama-cli` and `llama-quantize` in the `build/bin` directory. If you did not set `BUILD_SHARED_LIBS` to `OFF` during cmake configuration, you also need to pay attention to the following files:

- `build/src/libllama.so` 
- `build/ggml/libggml.so`
- `build/ggml/libggml-base.so`
- `build/ggml/libggml-cpu.so`
- `build/ggml/src/ggml-htp/libggml-htp.so`

#### The operator library

Detailed documentation can be found in [HTP-Ops-lib](https://github.com/haozixu/htp-ops-lib); here is a brief introduction to the build process.

1. First, ensure the Hexagon SDK environment is set up. Run the following command in the root directory of the Hexagon SDK:

```sh
source setup_sdk_env.source
```

2. Clone the project

```sh
git clone https://github.com/haozixu/htp-ops-lib; cd htp-ops-lib
```

3. Execute the following **two** commands in the root directory of the operator library:

```sh
build_cmake android
build_cmake hexagon DSP_ARCH=v73
```

Here, `DSP_ARCH` specifies the target Hexagon NPU architecture version. We recommend using `v73` by default for better compatibility. (The NPU architecture version on Snapdragon 8 Gen 2 is v73; you can modify this option according to your target hardware.)

After compilation, you should see two directories: `android_ReleaseG_aarch64` and `hexagon_ReleaseG_toolv87_v73` (the actual names may vary depending on the compilation mode, specific toolchain, and target architecture version). Note the following two products:

- `android_ReleaseG_aarch64/libhtp_ops.so`
- `hexagon_ReleaseG_toolv87_v73/libhtp_ops_skel.so`

These two shared objects will be used later. In FastRPC terminology, they are the Stub (`libhtp_ops.so`) and Skeleton (`libhtp_ops_skel.so`) respectively. You can use `ldd` to distinguish between the two shared objects: `libhtp_ops.so` targets the AArch64 architecture and runs on the CPU; `libhtp_ops_skel.so` targets the Q6DSP architecture and runs on the Hexagon NPU (cDSP).

### Model Conversion

We use a modified `convert_hf_to_gguf.py` conversion script, which is located at `extras/convert_hf_to_gguf_htp.py`. You need to check `pyproject.toml` or `requirements.txt` to ensure that the dependencies required by llama.cpp's Python scripts are installed. The basic usage of this script is the same as the original version, and you also need to prepare weight files from HuggingFace.

Currently, this script supports some Qwen and Llama models. To support more models, you can refer to the `modify_tensors` method of the `Model` base class in the script (this method rearranges weights according to the layout requirements of the FP16 HMX unit):

```Python
def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
    new_name = self.map_tensor_name(name)
    if any(self.match_model_tensor_name(new_name, key, bid) for key in [
        gguf.MODEL_TENSOR.ATTN_Q,
        gguf.MODEL_TENSOR.ATTN_K,
        gguf.MODEL_TENSOR.ATTN_V,
        gguf.MODEL_TENSOR.ATTN_OUT,
        gguf.MODEL_TENSOR.FFN_UP,
        gguf.MODEL_TENSOR.FFN_DOWN,
        gguf.MODEL_TENSOR.FFN_GATE,
    ]):
        mat_shape = data_torch.shape
        assert len(mat_shape) == 2

        n, k = mat_shape
        assert n % 32 == 0 and k % 32 == 0

        n_chunks, k_chunks = n // 32, k // 32
        x = data_torch.view(n_chunks, 32, k_chunks, 32)
        x = x.permute(0, 2, 1, 3).contiguous() # shape: [n_chunks, k_chunks, 32, 32]

        y = x.view(n_chunks, k_chunks, 32, 16, 2).permute(0, 1, 3, 2, 4).contiguous()

        data_torch = y.view(n, k)

    return [(new_name, data_torch)]
```

Taking Qwen2.5-1.5B as an example, use the following command to generate FP16 GGUF weights required by Hexagon NPU HMX:

```sh
python extras/convert_hf_to_gguf_htp.py --outfile qwen2.5-1.5b.f16-hmx.gguf --outtype f16 $path_to_hf_model
```

### Quantizing the Model

This step requires the compiled `llama-quantize`. This step does not need to be done on the Android device; you can use `llama-quantize` on the host.

The current backend operator library supports weight matrix quantization types including `Q4_0`, `IQ4_NL`, `Q8_0`, and `F16`. In the current implementation, `IQ4_NL` has higher precision than `Q4_0` but with the same computational overhead. We have preset a hybrid quantization scheme of `IQ4_NL` and `Q8_0`. An example of the model quantization command using this recommended scheme is as follows:

```sh
REPACK_FOR_HVX=1 ./build/bin/llama-quantize qwen2.5-1.5b.f16-hmx.gguf qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf IQ4_NL+Q8_0
```

Note that running this command requires setting the environment variable `REPACK_FOR_HVX` to enable weight rearrangement for HVX.

### Running the Program

It is assumed that the Android device is connected via adb (Termux is an alternative but we recommend adb; root permission may be required in termux). The following commands are all run on the device.

First, create a workspace where all required files will be placed:

```sh
mkdir -p /data/local/tmp/llama.cpp; cd /data/local/tmp/llama.cpp
```

To perform LLM inference on the device, transfer the produced files from the above steps to the Android device, including:

- Quantized model GGUF file (e.g., `qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf`)
- Executable programs (e.g., `llama-cli`)
- Necessary dynamic-linked libraries (e.g., `libhtp_ops.so`, `libhtp_ops_skel.so` and possible llama.cpp shared objects)

Set the following two environment variables before running. Keep `/system/lib64` ahead of `/vendor/lib64`: on recent Android releases, `libcdsprpc.so` may load `libbinder_ndk.so`, which requires the framework Binder implementation from `/system/lib64`. The workspace still comes first so the HTP host stub can be found.

- `LD_LIBRARY_PATH=/data/local/tmp/llama.cpp:/system/lib64:/vendor/lib64`
- `DSP_LIBRARY_PATH=/data/local/tmp/llama.cpp`

Run `llama-cli`:

```sh
LD_LIBRARY_PATH=/data/local/tmp/llama.cpp:/system/lib64:/vendor/lib64 DSP_LIBRARY_PATH=/data/local/tmp/llama.cpp ./llama-cli -t 4 -fa -m qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf -p "Hello my name is"
```

The `-fa` option is set here to enable the FlashAttention kernel.

## Known Issues

1. Currently, the size of models that can run on the device is limited; we recommend using models below 4B. This is mainly because the Hexagon cDSP (NPU) is a 32-bit processor with a 32-bit virtual address space. Our current design uses a single NPU session, and the dynamic virtual address mapping/unmapping scheme faces some limitations (see https://github.com/quic/fastrpc/issues/137 for details). QNN uses multiple NPU sessions to avoid this issue, which we have not yet supported.

2. For stability reasons, the upstream llama.cpp version used in this repository is somewhat outdated. Developers can migrate the HTP backend to the new upstream llama.cpp code by themselves. (Note: The CPU operator implementation in the current HTP backend directly reuses the CPU backend implementation; careful refactoring is required to avoid conflicts.)

## Troubleshooting

1. "unable to load libcdsprpc.so"

The HTP backend relies on the interface provided by `libcdsprpc.so` for rpcmem (dmabuf) operations. This dynamic link library is usually located in `/vendor/lib64`. Add both system paths while preserving the order `...:/system/lib64:/vendor/lib64`; reversing them can load an incompatible vendor `libbinder.so` before `libbinder_ndk.so` resolves its framework symbols.

2. Inference gets stuck

The main reason for inference getting stuck is a fatal error in the operator library. Use the following method to inspect the logs generated on the NPU side:

+ Create a `.farf` file on the device

```sh
echo 0x1f > $name_of_your_executable.farf
```

Replace `$name_of_your_executable` with the actual name of the executable file used. Example: If an error occurs when executing `llama-cli`, enable logging with the command `echo 0x1f > llama-cli.farf`.

+ Inspect adb log output from the host

```sh
adb logcat -s adsprpc
```

3. Garbage inference output

One possibility is that the NPU operator library `libhtp_ops.so` is not loaded correctly, and it falls back to CPU computation, resulting in incorrect results. Please check the prompt output in stdout/stderr. Also make sure all required transformations for HMX and HVX are applied to the model weights.

## Citation

If you find our work helpful, please cite us.

```bibtex
@article{hao2025scaling,
  title={Scaling LLM Test-Time Compute with Mobile NPU on Smartphones},
  author={Zixu Hao and Jianyu Wei and Tuowei Wang and Minxing Huang and Huiqiang Jiang and Shiqi Jiang and Ting Cao and Ju Ren},
  journal={arXiv preprint arXiv:2509.23324},
  year={2025}
}
```
