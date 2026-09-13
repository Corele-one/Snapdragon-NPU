#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
work_tree="$project_dir/work/llama.cpp-stage30"
"$script_dir/prepare_llama_backend.sh" >/dev/null
set +u
source "$script_dir/use_hexagon_sdk_6_6.sh"
set -u
ndk="$HEXAGON_SDK_ROOT/tools/android-ndk-r25c"
build_dir="$work_tree/build-android-stage30"
cmake -S "$work_tree" -B "$build_dir" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HTP=ON -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF
cmake --build "$build_dir" --target llama-bench llama-perplexity -j "$(nproc)"
mkdir -p "$project_dir/artifacts/model_host"
cp -a "$build_dir/bin/llama-bench" "$build_dir/bin/llama-perplexity" "$project_dir/artifacts/model_host/"
cp -a "$build_dir/src/libllama.so" "$build_dir/ggml/src/libggml.so" "$build_dir/ggml/src/libggml-base.so" \
  "$build_dir/ggml/src/libggml-cpu.so" "$build_dir/ggml/src/ggml-htp/libggml-htp.so" \
  "$project_dir/artifacts/model_host/"
(cd "$project_dir/artifacts/model_host" && sha256sum llama-bench llama-perplexity lib*.so >sha256.txt)
python3 "$project_dir/tools/create_manifest.py" --project "$project_dir" >/dev/null
echo "$project_dir/artifacts/model_host"
