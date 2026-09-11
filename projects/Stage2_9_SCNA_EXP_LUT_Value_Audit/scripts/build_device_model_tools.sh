#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
llama_tree="/home/corleone/code/Snapdragon-NPU/projects/flashattention/src/llama.cpp-npu-htp-backend"
set +u
source "$script_dir/use_hexagon_sdk_6_6.sh"
set -u
ndk="$HEXAGON_SDK_ROOT/tools/android-ndk-r25c"
clangxx="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++"
android_build="$llama_tree/build-android"
output="$project_dir/artifacts/device_model_tools"
mkdir -p "$output"

"$clangxx" -std=c++17 -O2 -fPIE -pie \
  -I"$llama_tree/ggml/include" \
  "$project_dir/tools/gguf_hmx_permute.cpp" \
  -L"$android_build/ggml/src" -lggml-base \
  -Wl,-rpath,'$ORIGIN' -o "$output/gguf-hmx-permute"
"$clangxx" -std=c++17 -O2 -fPIE -pie \
  -I"$llama_tree/ggml/include" \
  "$project_dir/tools/gguf_quant_lineage.cpp" \
  -L"$android_build/ggml/src" -lggml-base \
  -Wl,-rpath,'$ORIGIN' -o "$output/gguf-quant-lineage"

cp -a "$android_build/bin/llama-quantize" "$output/"
cp -a "$android_build/src/libllama.so" \
  "$android_build/ggml/src/libggml.so" \
  "$android_build/ggml/src/libggml-base.so" \
  "$android_build/ggml/src/libggml-cpu.so" \
  "$android_build/ggml/src/ggml-htp/libggml-htp.so" "$output/"
cp -a "$ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$output/"
(cd "$output" && sha256sum gguf-hmx-permute gguf-quant-lineage llama-quantize lib*.so > sha256.txt)
printf '%s\n' "$output"
