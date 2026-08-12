#!/usr/bin/env python3
"""Fail closed on experiment completeness, correctness, build identity and reproducibility."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ORDER = ["stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
         "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized"]
KV = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def fields(line):
    return dict(KV.findall(line))


def lines(root, pattern):
    for file in sorted(root.glob("raw/**/*.log")):
        for line in file.read_text(errors="replace").splitlines():
            if pattern in line: yield file, line


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); ap.add_argument("--project",required=True); args=ap.parse_args()
    root, project = Path(args.run_dir), Path(args.project)
    failures=[]; checks={}

    expected = {"micro_samples": 7*30, "attention_logs": 225, "accuracy_cases": 7*3*2*2*2*3, "diagnostic_logs": 8, "scaling_logs": 3*7}
    actual = {
        "micro_samples": len(list(root.glob("raw/micro/*_sample*.log"))),
        "attention_logs": len(list(root.glob("raw/attention/*.log"))),
        "accuracy_cases": len(list(root.glob("raw/accuracy/*.log"))),
        "diagnostic_logs": len(list(root.glob("raw/diagnostic/*.log"))),
        "scaling_logs": len(list(root.glob("raw/scaling/*.log"))),
    }
    checks["file_counts"]={"expected":expected,"actual":actual}
    for name, count in expected.items():
        if actual[name] != count: failures.append(f"{name}: expected {count}, got {actual[name]}")

    micro=[]
    for file,line in lines(root,"SCNA_EXP_BENCH"):
        if "_sample" not in file.name: continue
        d=fields(line); micro.append(d)
        variant=next((v for v in ORDER if v in file.name),None)
        if variant is None or int(d.get("variant_id",-1)) != ORDER.index(variant) or int(d.get("build_variant",-2)) != ORDER.index(variant):
            failures.append(f"build id mismatch: {file}")
        for key in ("monotonic_violations","negative_count","nan_count","random_nonfinite_count","paired_single_mismatches"):
            if int(d.get(key,-1)) != 0: failures.append(f"micro {key}: {file}")
        if int(d.get("pair_elapsed_us", 0)) < 50000:
            failures.append(f"micro sample shorter than 50 ms: {file}")
    checks["micro_records"]=len(micro)

    accuracy=[]
    for file,line in lines(root,"FIG8_ATTENTION_COMPARE "):
        d=fields(line); accuracy.append(d)
        if int(d.get("pass",0)) != 1 or float(d.get("rmse","inf")) > .002 or float(d.get("max_abs_error","inf")) > .01:
            failures.append(f"accuracy gate: {file}")
        if int(d.get("candidate_nonfinite",1)) or int(d.get("reference_nonfinite",1)):
            failures.append(f"nonfinite: {file}")
    checks["accuracy_records"]=len(accuracy)
    numeric_count=0
    for file,line in lines(root,"FIG8_NUMERIC "):
        if "raw/accuracy" not in str(file): continue
        numeric_count += 1; d=fields(line)
        if int(d.get("masked_p_nonzero",-1)) != 0 or int(d.get("tail_p_nonzero",-1)) != 0:
            failures.append(f"mask/tail zero gate: {file}")
    if numeric_count == 0: failures.append("no accuracy numeric-debug records")
    checks["accuracy_numeric_records"]=numeric_count

    static=json.loads((root/"static/static_metrics.json").read_text())
    for variant in ORDER:
        row=static.get(variant,{})
        if row.get("missing") or not row.get("mv79_verified"): failures.append(f"missing -mv79 evidence: {variant}")
    static_symbols=static.get("pair_static_d8",{}).get("symbols",{})
    static_pair=[body for name,body in static_symbols.items() if "pair_static_d8_qf16" in name]
    if not static_pair or any(body.get("branches",0) > 1 for body in static_pair):
        failures.append("static d8 runtime-loop gate failed")
    checks["static_d8_symbols"] = static_pair

    checksum_sets={}
    for file in sorted(root.glob("raw/scaling/optimized_w*.log")):
        values=[]
        for line in file.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_CHECKSUM" in line and "phase=measure" in line: values.append(fields(line).get("checksum"))
        checksum_sets[file.name]=sorted(set(values))
    all_checksums={value for values in checksum_sets.values() for value in values}
    if len(all_checksums) != 1: failures.append("optimized worker checksum mismatch")
    checks["worker_checksums"]=checksum_sets

    old_name="flashattention-scna-fp16-d8-" + "ablation-v79"
    refs=[]
    for path in project.rglob("*"):
        if path.is_file() and not any(part in {"results","artifacts","android_ReleaseG_aarch64","hexagon_ReleaseG_toolv19_v79","__pycache__"} for part in path.parts):
            try:
                if old_name in path.read_text(errors="ignore"): refs.append(str(path.relative_to(project)))
            except OSError: pass
    if refs: failures.append(f"old path dependencies: {refs}")
    checks["old_path_dependencies"]=refs

    report_before=hashlib.sha256((root/"REPORT.md").read_bytes()).hexdigest()
    subprocess.run(["python3",str(project/"tools/generate_optimization_report.py"),"--run-dir",str(root),"--spec",str(project/"experiment_spec.json")],check=True)
    report_after=hashlib.sha256((root/"REPORT.md").read_bytes()).hexdigest()
    checks["deterministic_report"] = report_before == report_after
    if report_before != report_after: failures.append("report regeneration is not deterministic")

    result={"pass":not failures,"failures":failures,"checks":checks}
    (root/"verification.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if failures:
        for failure in failures: print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("all experiment gates passed")


if __name__=="__main__": main()
