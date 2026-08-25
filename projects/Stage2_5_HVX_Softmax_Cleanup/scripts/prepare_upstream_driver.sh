#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
upstream_dir="$root_dir/external/llama.cpp"

test "$(git -C "$upstream_dir" rev-parse HEAD)" = "0a50d9909a3478e82679f505bf8595d1eee4b0a8"
if test -n "$(git -C "$upstream_dir" status --porcelain)"; then
    printf '%s\n' "refusing to prepare a dirty upstream checkout" >&2
    exit 1
fi
cp "$root_dir/patches/upstream/test-stage25-fa.cpp" "$upstream_dir/tests/test-stage25-fa.cpp"
git -C "$upstream_dir" apply "$root_dir/patches/upstream/tests-cmake.patch"
git -C "$upstream_dir" apply "$root_dir/patches/upstream/profile-kparams.patch"
mapfile -t changed < <(git -C "$upstream_dir" status --porcelain | sed 's/^...//')
allowed=$'ggml/src/ggml-hexagon/htp-opnode.h\ntests/CMakeLists.txt\ntests/test-stage25-fa.cpp'
actual=$(printf '%s\n' "${changed[@]}" | LC_ALL=C sort)
if test "$actual" != "$allowed"; then
    printf 'unexpected upstream diff:\n%s\n' "$actual" >&2
    exit 1
fi
git -C "$upstream_dir" diff -- . ':!tests/test-stage25-fa.cpp' > "$root_dir/patches/upstream/applied_tracked.patch"
sha256sum "$upstream_dir/tests/test-stage25-fa.cpp" "$root_dir/patches/upstream/"*.patch
