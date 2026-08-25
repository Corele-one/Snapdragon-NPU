#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
upstream_dir="$root_dir/external/llama.cpp"
sdk_root=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0
cmake_bin="$sdk_root/tools/cmake-3.28.3-linux-x86_64/bin/cmake"
build_dir="$upstream_dir/build-arm64-android-snapdragon-release"

test "$(git -C "$upstream_dir" rev-parse HEAD)" = "0a50d9909a3478e82679f505bf8595d1eee4b0a8"
actual=$(git -C "$upstream_dir" status --porcelain | sed 's/^...//' | LC_ALL=C sort)
allowed=$'CMakeUserPresets.json\nggml/src/ggml-hexagon/htp-opnode.h\ntests/CMakeLists.txt\ntests/test-stage25-fa.cpp'
test "$actual" = "$allowed" || { printf 'unexpected upstream diff:\n%s\n' "$actual" >&2; exit 1; }

export ANDROID_NDK_ROOT="$sdk_root/tools/android-ndk-r25c"
export HEXAGON_SDK_ROOT="$sdk_root"
export HEXAGON_TOOLS_ROOT="$sdk_root/tools/HEXAGON_Tools/19.0.07"
cd "$upstream_dir"
"$cmake_bin" --preset arm64-android-snapdragon-release \
    -DGGML_OPENCL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
"$cmake_bin" --build "$build_dir" --target test-stage25-fa htp-v79 -j "$(nproc)"

artifact_dir="$root_dir/artifacts/upstream_0a50d990"
mkdir -p "$artifact_dir/lib"
cp "$build_dir/bin/test-stage25-fa" "$artifact_dir/"
cp "$build_dir/bin/"libggml{,-base,-cpu,-hexagon}.so "$artifact_dir/lib/"
cp "$build_dir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$artifact_dir/lib/"
git diff --binary >"$artifact_dir/upstream_instrumentation.patch"
git status --porcelain >"$artifact_dir/upstream_status.txt"
git rev-parse HEAD >"$artifact_dir/upstream_commit.txt"
sha256sum "$artifact_dir/test-stage25-fa" "$artifact_dir/lib/"*.so >"$artifact_dir/sha256.txt"

