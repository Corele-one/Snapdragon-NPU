#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
checkout="$project_dir/external/llama.cpp"
repository="https://github.com/ggml-org/llama.cpp.git"
commit="0a50d9909a3478e82679f505bf8595d1eee4b0a8"

if [[ ! -d "$checkout/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$repository" "$checkout"
else
  [[ -z "$(git -C "$checkout" status --porcelain)" ]] || {
    echo "Existing upstream checkout is dirty; refusing to overwrite it" >&2
    exit 1
  }
fi

git -C "$checkout" fetch --depth=1 origin "$commit"
git -C "$checkout" checkout --detach "$commit"

actual="$(git -C "$checkout" rev-parse HEAD)"
[[ "$actual" == "$commit" ]] || { echo "Unexpected upstream commit: $actual" >&2; exit 1; }
[[ -z "$(git -C "$checkout" status --porcelain)" ]] || { echo "Upstream checkout is not clean" >&2; exit 1; }
echo "upstream=$repository commit=$actual clean=1"
