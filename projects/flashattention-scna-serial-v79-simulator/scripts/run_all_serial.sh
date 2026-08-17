#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
run_id="${SCNA_RUN_ID:-scna_all_serial_v79_$(date +%Y%m%d_%H%M%S)}"
export SCNA_RUN_ID="$run_id"
python_bin="${SCNA_PYTHON:-python3}"
[[ -x "$project_dir/.venv/bin/python" ]] && python_bin="$project_dir/.venv/bin/python"

"$script_dir/build_all.sh"
"$script_dir/probe_simulator.sh"
"$script_dir/run_micro.sh" --all
experiment_rc=0
"$script_dir/run_attention.sh" --all-serial || experiment_rc=$?
analysis_rc=0
"$python_bin" "$project_dir/tools/analyze_all_serial.py" --run-dir "$project_dir/results/runs/$run_id" || analysis_rc=$?
"$python_bin" "$project_dir/tools/collect_static_metrics.py" --project-dir "$project_dir" --run-dir "$project_dir/results/runs/$run_id"
"$script_dir/run_runtime_filtered_samples.sh"
"$python_bin" "$project_dir/tools/collect_runtime_filtered_metrics.py" --project-dir "$project_dir" --run-dir "$project_dir/results/runs/$run_id"
"$python_bin" "$project_dir/tools/write_manifest.py" --project-dir "$project_dir" --run-dir "$project_dir/results/runs/$run_id"
if [[ $analysis_rc -eq 0 ]]; then
  "$python_bin" "$project_dir/tools/plot_all_serial.py" --run-dir "$project_dir/results/runs/$run_id"
fi
"$python_bin" "$project_dir/tools/render_all_serial_dataflow.py" --out-dir "$project_dir/results/runs/$run_id/figures"
"$python_bin" "$project_dir/tools/generate_all_serial_report.py" --run-dir "$project_dir/results/runs/$run_id"
echo "Completed: $project_dir/results/runs/$run_id"
[[ $experiment_rc -eq 0 && $analysis_rc -eq 0 ]]
