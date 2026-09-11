#!/usr/bin/env python3
"""Execute the ordered Stage 2.9.1 model gates and stop at first failure."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shlex
import sys
from pathlib import Path

from model_lineage import file_record, sha256, stable_case_id


PROJECT = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT / "tools/run_model_audit.py"
    spec = importlib.util.spec_from_file_location("stage291_model_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def append(path: Path, record: dict[str, object]) -> None:
    identity = {key: record.get(key) for key in ("record_type", "gate", "mode") if key in record}
    record.setdefault("case_id", stable_case_id(identity))
    record.setdefault("attempt", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def render_report(reports: Path, gate: str, records: list[dict[str, object]],
                  diagnostic: dict[str, object] | None = None) -> None:
    reports.mkdir(parents=True, exist_ok=False)
    details = "\n".join(f"- `{r['gate']}`: pass={r['pass']}, PPL={r.get('ppl')}, ratio={r.get('ratio')}"
                        for r in records)
    verdict = {"verdict": "AUDIT_INVALID", "failed_gate": gate, "gates": records}
    if diagnostic is not None:
        verdict["diagnostic"] = diagnostic
    (reports / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    with (reports / "gate_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("gate", "pass", "ppl", "cpu_ppl", "ratio", "limit"))
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in writer.fieldnames})
    with (reports / "confidence_intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("status", "reason"))
        writer.writeheader()
        writer.writerow({"status": "not_run", "reason": f"baseline gate failed: {gate}"})
    diagnostic_text = ""
    if diagnostic is not None:
        isolation = diagnostic["model_path_isolation"]
        rowsum = diagnostic["rowsum"]
        diagnostic_text = (
            "\n## 故障隔离\n\n"
            f"- 禁用自定义 FlashAttention 的 HMX F16 PPL：`{isolation['no_custom_flash_ppl']}`。\n"
            f"- 启用冻结 FlashAttention 的 HMX F16 PPL：`{isolation['custom_flash_ppl']}`。\n"
            f"- 模型内首个分歧：`{isolation['first_divergent_tensor']}`；其前置 HMX matmul "
            f"全部通过：`{isolation['preceding_model_matmuls_all_pass']}`。\n"
            f"- Q=32/KV=32/causal 同类失败模式：`{', '.join(diagnostic['short_kv_failed_modes'])}`。\n"
            f"- 融合 rowsum={rowsum['stored_fp16']}，标量概率和={rowsum['scalar_expected_fp32']}，"
            f"mask/tail 均为零：`{rowsum['masked_and_tail_zero']}`。\n\n"
            "证据指向被冻结的融合 Attention rowsum/state-update 公共路径，而非 SCNA 或 EXP-LUT "
            "evaluator。该路径不在本轮允许修改范围内，所以不能宣布算法胜负。\n"
        )
    (reports / "FINAL_STAGE2_9_REPORT.md").write_text(
        "# Stage 2.9.1 最终审计\n\n## 裁决\n\n`AUDIT_INVALID`\n\n"
        f"模型修复链路在分级门槛 `{gate}` 停止；按预注册规则没有运行后续 SCNA/EXP-LUT 矩阵，"
        "也没有据此启动 DMA-HMX 或 INT8。\n\n## 门槛证据\n\n" + details + "\n" +
        diagnostic_text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT / "experiment_spec.json")
    parser.add_argument("--repair-spec", type=Path, default=PROJECT / "model_repair_spec.json")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--matmul-results", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-summary", type=Path)
    args = parser.parse_args()
    results = args.results_dir.resolve()
    reports = args.reports_dir.resolve()
    if reports.exists():
        raise RuntimeError(f"refusing to overwrite gate reports: {reports}")
    if results.exists() and any(results.iterdir()):
        raise RuntimeError(f"refusing to overwrite gate results: {results}")
    results.mkdir(parents=True, exist_ok=False) if not results.exists() else None
    raw = results / "raw/repair_gates.jsonl"
    logs = results / "logs/repair_gates"
    logs.mkdir(parents=True, exist_ok=True)
    repair = json.loads(args.repair_spec.resolve().read_text())
    audit_spec = json.loads(args.spec.resolve().read_text())
    manifest_path = args.model_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    diagnostic = None
    if args.diagnostic_summary:
        diagnostic_path = args.diagnostic_summary.resolve()
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic.get("verdict") != "AUDIT_INVALID":
            raise RuntimeError("diagnostic summary must carry AUDIT_INVALID")
        diagnostic["evidence"] = file_record(diagnostic_path)
    if manifest.get("tensor_validation", {}).get("pass") is not True:
        raise RuntimeError("tensor-level model pair validation did not pass")
    matrix_lines = [json.loads(line) for line in (args.matmul_results / "raw/matmul.jsonl").read_text().splitlines()]
    matrix = [line for line in matrix_lines if line.get("record_type") == "matmul_matrix_summary"]
    gate_records: list[dict[str, object]] = []
    case_lines = [line for line in matrix_lines if line.get("record_type") == "matmul_value_audit"]
    repair_spec_hash = sha256(args.repair_spec.resolve())
    final_by_rep: dict[tuple[str, int], dict[str, object]] = {}
    for line in case_lines:
        key = (str(line.get("case_id")), int(line.get("repetition", 0)))
        if int(line.get("attempt", 0)) > int(final_by_rep.get(key, {}).get("attempt", 0)):
            final_by_rep[key] = line
    case_ids = {key[0] for key in final_by_rep}
    stable_provenance = all(len({str(line.get(key)) for line in case_lines}) == 1
                            for key in ("runner_sha256", "host_sha256", "host_build_id",
                                        "dsp_sha256", "dsp_build_id", "matmul_source_sha256",
                                        "executor_source_sha256"))
    current_provenance = {
        "runner_sha256": sha256(PROJECT / "tools/run_matmul_audit.py"),
        "host_sha256": sha256(PROJECT / "artifacts/host/htp_ops_test"),
        "dsp_sha256": sha256(PROJECT / "artifacts/fair_combined/libhtp_ops_skel.so"),
        "matmul_source_sha256": sha256(PROJECT / "src/htp-ops-lib-main/src/dsp/ops/mat_mul.c"),
        "executor_source_sha256": sha256(PROJECT / "src/htp-ops-lib-main/src/dsp/op_executor.cc"),
    }
    provenance_matches_current = all({str(line.get(key)) for line in case_lines} == {value}
                                     for key, value in current_provenance.items())
    matrix_pass = (len(matrix) == 1 and matrix[0].get("pass") is True and
                   matrix[0].get("spec_sha256") == repair_spec_hash and
                   all(line.get("spec_sha256") == repair_spec_hash for line in case_lines) and
                   stable_provenance and provenance_matches_current and
                   len(case_ids) == int(matrix[0].get("cases", -1)) and
                   len(final_by_rep) == 2 * len(case_ids) and
                   all(line.get("pass") is True for line in final_by_rep.values()))
    first = {"record_type": "repair_gate", "gate": "matmul_matrix", "pass": matrix_pass,
             "matrix_evidence": file_record(args.matmul_results / "raw/matmul.jsonl"),
             "repair_spec_sha256": repair_spec_hash, "unique_cases": len(case_ids),
             "stable_provenance": stable_provenance,
             "provenance_matches_current": provenance_matches_current}
    append(raw, first); gate_records.append(first)
    if not matrix_pass:
        render_report(reports, "matmul_matrix", gate_records, diagnostic)
        return 3

    runner = load_runner()
    runner.SPEC = audit_spec
    runner.MODEL_MANIFEST = manifest
    runner.MODEL = dict(audit_spec["model"])
    runner.MODEL["cpu_model"] = manifest["device"]["models"]["cpu_quant"]
    runner.MODEL["hmx_model"] = manifest["device"]["models"]["hmx_quant"]
    runner.deploy(logs)
    dataset = runner.MODEL["dataset"]
    common = ["-f", dataset, "-fa", "-c", "512", "-b", "512", "-ub", "64", "-t", "4", "--chunks", "1"]

    def cpu_ppl(model_path: str, name: str) -> float:
        text = runner.adb_shell(runner.cpu_command(["-m", model_path, *common]), logs / f"{name}.log", timeout=3600)
        return runner.extract_ppl(text)[0]

    def htp_ppl(model_path: str, mode: str, name: str, skip_custom: bool = False) -> float:
        command = runner.htp_command("llama-perplexity", mode, ["-m", model_path, *common])
        if skip_custom:
            command = command.replace("./llama-perplexity", "SKIP_HTP_OPS=1 ./llama-perplexity", 1)
        text = runner.adb_shell(command, logs / f"{name}.log", timeout=3600)
        if skip_custom:
            # The explicit confirmation is emitted by htp-ops.cc, which this
            # gate deliberately disables.  Require independent allocation and
            # graph-placement evidence from the generic HTP/RPCMEM backend.
            markers = ("Initializing HTP backend", "RPCMEM model buffer size",
                       "RPCMEM compute buffer size", "graph splits = 3")
            if any(marker not in text for marker in markers):
                raise RuntimeError(f"generic HTP/RPCMEM evidence missing at gate {name}")
            if "STAGE29_BACKEND_CONFIRM" in text:
                raise RuntimeError(f"custom HTP ops were not disabled at gate {name}")
        elif f"STAGE29_BACKEND_CONFIRM backend=my-htp mode={mode}" not in text:
            raise RuntimeError(f"HTP confirmation missing at gate {name}")
        if __import__("re").search(r"fallback to CPU|all OPs will fallback", text, __import__("re").I):
            raise RuntimeError(f"CPU fallback detected at gate {name}")
        return runner.extract_ppl(text)[0]

    gates = repair["gates"]
    cpu_f16 = cpu_ppl(manifest["device"]["models"]["cpu_f16"], "cpu_f16")
    ratio = abs(cpu_f16 / gates["trusted_cpu_f16_ppl"] - 1.0)
    record = {"record_type": "repair_gate", "gate": "cpu_f16", "ppl": cpu_f16, "ratio": ratio,
              "limit": gates["cpu_f16_vs_trusted_ppl_relative_max"], "pass": math.isfinite(cpu_f16) and ratio <= gates["cpu_f16_vs_trusted_ppl_relative_max"]}
    append(raw, record); gate_records.append(record)
    if not record["pass"]: render_report(reports, "cpu_f16", gate_records, diagnostic); return 3

    standard_htp = htp_ppl(manifest["device"]["models"]["cpu_f16"], "baseline", "standard_f16_htp", True)
    ratio = abs(standard_htp / cpu_f16 - 1.0)
    record = {"record_type": "repair_gate", "gate": "standard_f16_htp", "ppl": standard_htp,
              "ratio": ratio, "limit": gates["standard_f16_htp_vs_cpu_relative_max"],
              "pass": math.isfinite(standard_htp) and ratio <= gates["standard_f16_htp_vs_cpu_relative_max"]}
    append(raw, record); gate_records.append(record)
    if not record["pass"]: render_report(reports, "standard_f16_htp", gate_records, diagnostic); return 3

    hmx_f16 = htp_ppl(manifest["device"]["models"]["hmx_f16"], "baseline", "hmx_f16")
    ratio = abs(hmx_f16 / cpu_f16 - 1.0)
    record = {"record_type": "repair_gate", "gate": "hmx_f16", "ppl": hmx_f16, "ratio": ratio,
              "limit": gates["hmx_f16_vs_cpu_relative_max"],
              "pass": math.isfinite(hmx_f16) and ratio <= gates["hmx_f16_vs_cpu_relative_max"]}
    append(raw, record); gate_records.append(record)
    if not record["pass"]: render_report(reports, "hmx_f16", gate_records, diagnostic); return 3

    cpu_quant = cpu_ppl(manifest["device"]["models"]["cpu_quant"], "cpu_quant")
    hmx_quant = htp_ppl(manifest["device"]["models"]["hmx_quant"], "baseline", "hmx_quant")
    ratio = abs(hmx_quant / cpu_quant - 1.0)
    record = {"record_type": "repair_gate", "gate": "hmx_quant_baseline", "cpu_ppl": cpu_quant,
              "ppl": hmx_quant, "ratio": ratio, "limit": gates["hmx_quant_vs_cpu_quant_relative_max"],
              "backend_confirmed": True, "no_cpu_fallback": True,
              "pass": math.isfinite(hmx_quant) and ratio <= gates["hmx_quant_vs_cpu_quant_relative_max"]}
    append(raw, record); gate_records.append(record)
    manifest["repair_gates"] = {"pass": bool(record["pass"]), "evidence": file_record(raw), "records": gate_records}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not record["pass"]: render_report(reports, "hmx_quant_baseline", gate_records, diagnostic); return 3
    print("all Stage2.9.1 repair gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
