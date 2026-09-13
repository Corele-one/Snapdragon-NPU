#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
phase="all"
results_dir=""

usage() {
  echo "Usage: $0 --results-dir PATH [--phase gate0|track-a|track-b|model|resource|thermal|all]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) phase="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$results_dir" ]] || { usage; exit 2; }

gate_dir="$results_dir/gate0"
verdict="$results_dir/gate0_verdict.json"

run_gate0() {
  set +e
  python3 "$project_dir/tools/run_closure_audit.py" --phase gate0 --results-dir "$gate_dir"
  local run_ret=$?
  python3 "$project_dir/tools/analyze_gate0.py" --results-dir "$gate_dir" --output "$verdict"
  local analyze_ret=$?
  set -e
  if [[ $run_ret -ne 0 || $analyze_ret -ne 0 ]]; then
    echo "AUDIT_INVALID: Gate 0 failed; all downstream phases are stopped." >&2
    return 3
  fi
}

case "$phase" in
  gate0) run_gate0 ;;
  track-a|track-b)
    python3 "$project_dir/tools/run_closure_audit.py" --phase "$phase" \
      --results-dir "$results_dir/$phase" --gate0-verdict "$verdict"
    ;;
  model|resource|thermal)
    python3 "$project_dir/tools/run_closure_audit.py" --phase "$phase" \
      --results-dir "$results_dir/$phase" --gate0-verdict "$verdict"
    ;;
  all)
    run_gate0
    python3 "$project_dir/tools/run_closure_audit.py" --phase track-a \
      --results-dir "$results_dir/track-a" --gate0-verdict "$verdict"
    python3 "$project_dir/tools/run_closure_audit.py" --phase track-b \
      --results-dir "$results_dir/track-b" --gate0-verdict "$verdict"
    python3 "$project_dir/tools/run_closure_audit.py" --phase model \
      --results-dir "$results_dir/model" --gate0-verdict "$verdict"
    python3 "$project_dir/tools/run_closure_audit.py" --phase resource \
      --results-dir "$results_dir/resource" --gate0-verdict "$verdict"
    python3 "$project_dir/tools/run_closure_audit.py" --phase thermal \
      --results-dir "$results_dir/thermal" --gate0-verdict "$verdict"
    ;;
  *) usage; exit 2 ;;
esac
