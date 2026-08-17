#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
run_id="${SCNA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export SCNA_RUN_ID="$run_id"
python_bin="${SCNA_PYTHON:-python3}"
[[ -x "$project_dir/.venv/bin/python" ]] && python_bin="$project_dir/.venv/bin/python"
"$script_dir/build_all.sh"
"$script_dir/probe_simulator.sh"
"$script_dir/run_micro.sh" --all
"$script_dir/run_attention.sh" --smoke
"$script_dir/run_attention.sh" --sweep
"$script_dir/run_detailed_sample.sh"
"$python_bin" "$project_dir/tools/analyze_results.py" --run-dir "$project_dir/results/runs/$run_id"
"$python_bin" "$project_dir/tools/collect_static_metrics.py" --project-dir "$project_dir" --run-dir "$project_dir/results/runs/$run_id"
"$python_bin" "$project_dir/tools/write_manifest.py" --project-dir "$project_dir" --run-dir "$project_dir/results/runs/$run_id"
"$python_bin" "$project_dir/tools/plot_results.py" --summary "$project_dir/results/runs/$run_id/summary.json" --out-dir "$project_dir/results/runs/$run_id/figures"
"$python_bin" "$project_dir/tools/render_dataflow.py" --out-dir "$project_dir/results/runs/$run_id/figures"
"$python_bin" "$project_dir/tools/generate_report.py" --run-dir "$project_dir/results/runs/$run_id"
echo "Completed: $project_dir/results/runs/$run_id"
