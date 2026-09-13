#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
archive="$project_dir/../Archived/flashattention-scna-v81/src/llama.cpp-npu-htp-backend"
work_tree="$project_dir/work/llama.cpp-stage30"
[[ -d "$archive" ]] || { echo "Missing archived llama backend: $archive" >&2; exit 1; }
mkdir -p "$work_tree"
rsync -a --delete --exclude='.git/' --exclude='build-*/' "$archive/" "$work_tree/"
python3 "$project_dir/tools/patch_llama_backend.py" \
  --tree "$work_tree" \
  --audit-op-reg "$project_dir/src/htp-ops-lib-main/include/op_reg.h"
echo "$work_tree"
