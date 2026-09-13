#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
spec="$project_dir/experiment_spec.json"
rerun_spec="$project_dir/audit_rerun_spec.json"
model_manifest=""
results_dir=""
reports_dir=""

usage() { echo "Usage: $0 --model-manifest PATH --results-dir PATH --reports-dir PATH [--spec PATH]" >&2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec) spec="$2"; shift 2 ;;
    --model-manifest) model_manifest="$2"; shift 2 ;;
    --results-dir) results_dir="$2"; shift 2 ;;
    --reports-dir) reports_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$model_manifest" && -n "$results_dir" && -n "$reports_dir" ]] || { usage; exit 2; }
[[ -f "$model_manifest" ]] || { echo "Missing model manifest: $model_manifest" >&2; exit 2; }
[[ ! -e "$reports_dir" ]] || { echo "Refusing to overwrite report directory: $reports_dir" >&2; exit 2; }
if [[ -e "$results_dir" ]] && find "$results_dir" -type f -print -quit | grep -q .; then
  echo "Refusing to mix a formal rerun with existing evidence: $results_dir" >&2
  exit 2
fi
python3 - "$model_manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m.get("tensor_validation", {}).get("pass") is True, "tensor validation not passed"
assert m.get("repair_gates", {}).get("pass") is True, "repair gates not passed"
PY

mkdir -p "$results_dir"
on_formal_error() {
  status=$?
  trap - ERR
  if [[ ! -e "$reports_dir" ]]; then
    python3 "$project_dir/tools/analyze_audit.py" --spec "$spec" --results-dir "$results_dir" \
      --reports-dir "$reports_dir" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap on_formal_error ERR
sha256sum "$spec" "$rerun_spec" "$project_dir/model_repair_spec.json" "$model_manifest" \
  >"$results_dir/frozen_inputs.sha256"
python3 "$project_dir/tools/hash_tree.py" --root "$project_dir/../Stage2_75_Worker_Scheduler" \
  --output "$results_dir/stage2_75_before.json"
sha256sum "$project_dir/src/htp-ops-lib-main/src/dsp/ops/scna_exp2.c" \
  "$project_dir/src/htp-ops-lib-main/src/dsp/ops/precompute_table.c" \
  "$project_dir/src/htp-ops-lib-main/src/dsp/ops/flash_attn.c" \
  >"$results_dir/evaluator_sources_before.sha256"

"$script_dir/build_variants.sh" --flavor all
python3 - "$model_manifest" "$project_dir/artifacts/fair_combined/libhtp_ops_skel.so" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.load(open(sys.argv[1]))
gate = manifest.get("repair_gates", {})
if gate.get("pass") is not True:
    raise SystemExit("repair gate status changed before formal build")
evidence = pathlib.Path(gate["evidence"]["path"])
matrix_gate = next(json.loads(line) for line in evidence.read_text().splitlines()
                   if json.loads(line).get("gate") == "matmul_matrix")
matrix_raw = pathlib.Path(matrix_gate["matrix_evidence"]["path"])
record = next(json.loads(line) for line in matrix_raw.read_text().splitlines()
              if json.loads(line).get("record_type") == "matmul_value_audit")
actual = hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
if actual != record["dsp_sha256"]:
    raise SystemExit(f"formal fair_combined DSP differs from repair-gate binary: {actual}")
PY
"$script_dir/build_llama_backend.sh"
python3 -m unittest discover -s "$project_dir/tests" -v
python3 "$project_dir/tools/create_manifest.py" --project "$project_dir"
python3 "$project_dir/tools/run_device_audit.py" --spec "$spec" --phase smoke --results-dir "$results_dir"

set +e
python3 "$project_dir/tools/run_model_audit.py" --spec "$spec" --model-manifest "$model_manifest" \
  --results-dir "$results_dir"
model_ret=$?
set -e
if [[ "$model_ret" -ne 0 ]]; then
  sha256sum --check "$results_dir/frozen_inputs.sha256"
  python3 "$project_dir/tools/analyze_audit.py" --spec "$spec" --results-dir "$results_dir" \
    --reports-dir "$reports_dir" || true
  exit "$model_ret"
fi

python3 "$project_dir/tools/run_device_audit.py" --spec "$spec" --phase all --results-dir "$results_dir"
python3 "$project_dir/tools/run_thermal_audit.py" --spec "$spec" --results-dir "$results_dir"
sha256sum --check "$results_dir/frozen_inputs.sha256"
python3 "$project_dir/tools/hash_tree.py" --root "$project_dir/../Stage2_75_Worker_Scheduler" \
  --check "$results_dir/stage2_75_before.json"
(cd / && sha256sum --check "$results_dir/evaluator_sources_before.sha256")
python3 "$project_dir/tools/create_final_manifest.py" --spec "$spec" --model-manifest "$model_manifest" \
  --results-dir "$results_dir" --output "$results_dir/final_manifest.json"
python3 "$project_dir/tools/analyze_audit.py" --spec "$spec" --results-dir "$results_dir" \
  --reports-dir "$reports_dir"
