#!/usr/bin/env bash
set -euo pipefail
remote_dir="${STAGE30_REMOTE_DIR:-/data/local/tmp/stage3_0_closure_audit}"
if [[ "${1:-}" == "--remote-dir" ]]; then
  [[ $# -ge 3 ]] || { echo "Usage: $0 [--remote-dir PATH] htp_ops_test arguments..." >&2; exit 2; }
  remote_dir="$2"
  shift 2
fi
[[ $# -gt 0 ]] || { echo "No htp_ops_test arguments supplied" >&2; exit 2; }
quoted=()
for arg in "$@"; do printf -v q '%q' "$arg"; quoted+=("$q"); done
adb shell "cd '$remote_dir' && LD_LIBRARY_PATH=. DSP_LIBRARY_PATH='./cdsp;./dsp;.' ./htp_ops_test ${quoted[*]}"
