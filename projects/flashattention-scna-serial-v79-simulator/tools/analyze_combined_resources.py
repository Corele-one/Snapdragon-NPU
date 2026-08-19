#!/usr/bin/env python3
"""Merge the two immutable formal runs with an instrumented resource audit."""
from __future__ import annotations

import argparse, csv, hashlib, json, re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
VARIANTS = ["stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
            "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized"]
QOS = [1, 4, 8, 16, 32]
STAGES = ["q_load", "k_load", "v_load", "qk_dot", "safe_sm", "core_acc", "o_scale", "o_store"]
BUFFER_FIELDS = ["q_bytes", "o0_bytes", "o1_bytes", "k_bytes", "v_bytes", "s_bytes", "p_bytes",
                 "d_bytes", "col_vectors_bytes", "row_vectors_bytes", "hmx_scales_bytes"]

def scalar(value):
    value = value.rstrip(":")
    try: return int(value, 0)
    except ValueError:
        try: return float(value)
        except ValueError: return value

def marker_rows(path, marker):
    rows = []
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.startswith(marker + " "): continue
        row = {k: scalar(v) for k, v in PAIR.findall(line)}
        row.update(source=str(path), line=n); rows.append(row)
    return rows

def sha256(path):
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def median_metric(item, name): return float(item["metrics"][name]["median"])

def primary_matrix(summary):
    matrix = defaultdict(dict)
    for item in summary["attention"].values():
        if item["worker_policy"] != "w1" or item["kv"] != 64 or item["heads"] != 12 or item["kv_heads"] != 2 or item["head_dim"] != 128: continue
        matrix[item["scheme"]][str(item["qo"])] = item
    return dict(matrix)

def source_matrix(summary):
    matrix = defaultdict(dict)
    for item in summary["attention"].values():
        if item["kv"] == 64 and item["heads"] == 12 and item["kv_heads"] == 2 and item["head_dim"] == 128:
            matrix[item["mode"]][str(item["qo"])] = item
    return dict(matrix)

def parse_resources(run_dir):
    cases, all_events, all_layouts = {}, [], []
    for log in sorted((run_dir / "raw/resource").glob("*.log")):
        headers = marker_rows(log, "ATTENTION_RESOURCE_HEADER")
        verifies = marker_rows(log, "ATTENTION_VERIFY")
        timers = marker_rows(log, "ATTENTION_TIMER")
        events = marker_rows(log, "ATTENTION_RESOURCE_EVENT")
        layouts = marker_rows(log, "ATTENTION_VTCM_LAYOUT")
        scna = marker_rows(log, "SCNA_RESOURCE_SUMMARY")
        processes = marker_rows(log, "SIM_PROCESS_RESULT")
        if not headers: continue
        header = headers[-1]; scheme = header["variant"] if header["mode"] == "serial" else header["mode"]
        duplicate_scheme = scheme in cases
        stage = defaultdict(lambda: {"duration_us": 0, "logical_bytes": 0, "logical_ops": 0, "events": 0})
        units = defaultdict(int)
        for event in events:
            dst = stage[str(event["stage"])]; dst["duration_us"] += int(event["duration_us"])
            dst[str(event["value_kind"])] += int(event["value"]); dst["events"] += 1
            units[str(event["unit"])] += 1
            all_events.append({"scheme": scheme, **event})
        unique_layouts = []
        seen = set()
        for layout in layouts:
            key = tuple(layout.get(field) for field in BUFFER_FIELDS + ["used_bytes", "reserved_bytes", "vtcm_total_bytes"])
            if key not in seen: seen.add(key); unique_layouts.append(layout)
            all_layouts.append({"scheme": scheme, **layout})
        layout_ok = bool(unique_layouts) and all(sum(int(x[f]) for f in BUFFER_FIELDS) == int(x["used_bytes"])
            and int(x["used_bytes"]) <= int(x["reserved_bytes"]) <= int(x["vtcm_total_bytes"]) for x in unique_layouts)
        expected_units = {"q_load": "MEMORY", "k_load": "MEMORY", "v_load": "MEMORY", "qk_dot": "HMX",
                          "safe_sm": "HVX", "o_scale": "HMX", "o_store": "STORE"}
        unit_ok = all(any(e["stage"] == s and e["unit"] == u for e in events) for s, u in expected_units.items()) and \
                  any(e["stage"] == "core_acc" and e["unit"] == "HVX" for e in events) and \
                  any(e["stage"] == "core_acc" and e["unit"] == "HMX" for e in events)
        scna_row = scna[-1] if scna else {}
        scna_ok = (int(scna_row.get("figure_scna_events", -1)) == int(scna_row.get("logical_calls", -2))) if scheme in VARIANTS else \
                  int(scna_row.get("figure_scna_events", -1)) == int(scna_row.get("logical_calls", -2)) == 0
        gates = {
            "single_header": len(headers) == 1, "header_pass": header.get("status") == "PASS",
            "event_overflow_zero": int(header.get("figure_overflow", -1)) == int(header.get("llm_overflow", -1)) == 0,
            "stages_complete": set(stage) == set(STAGES), "units_match_source": unit_ok,
            "vtcm_layout_valid": layout_ok, "scna_event_count_self_consistent": scna_ok,
            "verify_pass": len(verifies) == 1 and verifies[0].get("status") == "PASS",
            "process_exit_zero": len(processes) == 1 and int(processes[0].get("exit_code", -1)) == 0,
            "single_measurement": len(timers) == 1,
            "unique_scheme": not duplicate_scheme,
        }
        cases[scheme] = {"header": header, "verify": verifies[-1] if verifies else {}, "timer_instrumented": timers[-1] if timers else {},
                         "stages": dict(stage), "units": dict(units), "layouts": unique_layouts,
                         "scna": scna_row, "gates": gates, "pass": all(gates.values()), "source_log": str(log.relative_to(run_dir))}
    return cases, all_events, all_layouts

def resource_workload_match(resources):
    expected = {"qo":32,"kv":64,"heads":12,"kv_heads":2,"head_dim":128,"requested_workers":1,"active_workers":1}
    return bool(resources) and all(all(int(case["header"].get(k,-1)) == v for k,v in expected.items()) for case in resources.values())

def parse_ints(text): return [int(x) for x in re.findall(r"(?<![A-Za-z])([1-9][0-9]*)", text)]

def validate_native(run_dir):
    range_file = run_dir / "evidence/native_pc_range.txt"
    meta = {k: scalar(v) for k, v in PAIR.findall(range_file.read_text(errors="replace"))} if range_file.exists() else {}
    start, end = int(meta.get("pc_start", 0)), int(meta.get("pc_end", -1))
    trace_dir = run_dir / "metrics/native/pcfilter"
    trace_files = {name: trace_dir / f"{name}.txt" for name in ("memtrace", "coproctrace", "pctrace")}
    details = {}; total_matches = 0
    for name, path in trace_files.items():
        # Trace files can be tens of GiB.  PC-filter validation only needs a
        # verified in-range sample, so scan a bounded prefix and retain the
        # complete raw file as evidence without loading it into memory.
        text = ""
        if path.exists():
            with path.open("rb") as stream: text = stream.read(4 * 1024 * 1024).decode(errors="replace")
        addresses = [int(x, 16) for x in re.findall(r"(?:0x)?([0-9a-fA-F]{8,})", text)]
        matches = sum(start <= x <= end for x in addresses); total_matches += matches
        details[name] = {"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size if path.exists() else 0,
                         "pc_matches_in_4mib_prefix": matches}
    filtered_exit = int(meta.get("exit_code", -1))
    filtered_valid = filtered_exit == 0 and all(v["bytes"] > 0 for v in details.values()) and total_matches > 0
    timing_log = run_dir / "raw/native/optimized_unfiltered_timing.log"
    proc = marker_rows(timing_log, "SIM_PROCESS_RESULT") if timing_log.exists() else []
    pmu = run_dir / "metrics/native/unfiltered/pmu.txt"; packet = run_dir / "metrics/native/unfiltered/packet_analyze.txt"
    pmu_text = pmu.read_text(errors="replace") if pmu.exists() else ""
    relevant = {}
    for line in pmu_text.splitlines():
        if re.search(r"HVX|COPROC|LOAD|STORE|L2|VTCM|HMX", line, re.I):
            values = parse_ints(line); relevant[line.split()[0] if line.split() else "unknown"] = max(values) if values else 0
    unfiltered_valid = bool(proc) and int(proc[-1].get("exit_code", -1)) == 0 and pmu.exists() and pmu.stat().st_size > 0 and \
                       packet.exists() and packet.stat().st_size > 0 and any(v > 0 for v in relevant.values())
    return {"kernel_attributable": filtered_valid, "status": "VALIDATED" if filtered_valid else "UNAVAILABLE",
            "pc_range": {"start": start, "end": end}, "filtered": {"exit_code": filtered_exit, "files": details,
            "pc_matches": total_matches, "validated": filtered_valid},
            "unfiltered": {"validated_full_process_only": unfiltered_valid, "pmu_bytes": pmu.stat().st_size if pmu.exists() else 0,
            "packet_bytes": packet.stat().st_size if packet.exists() else 0, "relevant_nonzero": relevant,
            "scope": "full process including QuRT/loader; never promoted to kernel evidence"},
            "authority": "simulator_trace" if filtered_valid else "software_events_and_static_disassembly"}

def write_csvs(run_dir, combined, events):
    with (run_dir / "combined_attention.csv").open("w", newline="") as f:
        fields = ["scheme", "qo", "metric", "median", "min", "max", "count", "values"]
        w = csv.DictWriter(f, fields); w.writeheader()
        for scheme, qs in combined["primary_attention"].items():
            for qo, item in qs.items():
                for metric, value in item["metrics"].items():
                    if isinstance(value, dict): w.writerow({"scheme": scheme, "qo": qo, "metric": metric, **value})
    with (run_dir / "resource_events.csv").open("w", newline="") as f:
        fields = ["scheme", "stage", "unit", "worker", "block_r", "block_c", "chunk_r", "chunk_c", "value_kind", "value", "duration_us", "source", "line"]
        w = csv.DictWriter(f, fields, extrasaction="ignore"); w.writeheader(); w.writerows(events)
    with (run_dir / "reproducibility.csv").open("w", newline="") as f:
        fields = ["mode", "qo", "run1_us", "run2_us", "difference_us", "relative_percent"]
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(combined["reproducibility"])
    layout = next(iter(combined["resource_audit"].values()))["layouts"][0]
    with (run_dir / "vtcm_layout.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["buffer", "bytes", "percent_of_used"]); w.writeheader()
        for field in BUFFER_FIELDS: w.writerow({"buffer": field.removesuffix("_bytes"), "bytes": layout[field], "percent_of_used": 100*layout[field]/layout["used_bytes"]})

def build(source_run, primary_run, run_dir):
    source = json.loads((source_run / "summary.json").read_text()); primary = json.loads((primary_run / "summary_all_serial.json").read_text())
    static = json.loads((primary_run / "static/static_metrics.json").read_text())
    pmat, smat = primary_matrix(primary), source_matrix(source)
    repro = []
    mapping = {"origin":"origin", "exp-lut":"exp-lut", "stage1":"stage1_dynamic_row", "optimized":"optimized"}
    for old, new in mapping.items():
        for qo in QOS:
            a = median_metric(smat[old][str(qo)], "kernel_us"); b = median_metric(pmat[new][str(qo)], "kernel_us")
            repro.append({"mode": old, "qo": qo, "run1_us": a, "run2_us": b, "difference_us": b-a, "relative_percent": 100*(b-a)/a})
    ablation=[]
    for parent, child in zip(VARIANTS, VARIANTS[1:]):
        for qo in QOS:
            a=median_metric(pmat[parent][str(qo)],"kernel_us"); b=median_metric(pmat[child][str(qo)],"kernel_us")
            ablation.append({"parent":parent,"child":child,"qo":qo,"parent_us":a,"child_us":b,"speedup":a/b,"change_percent":100*(a-b)/a})
    resources, events, layouts = parse_resources(run_dir); native = validate_native(run_dir)
    environment = (run_dir / "evidence/environment.txt").read_text(errors="replace") if (run_dir / "evidence/environment.txt").exists() else ""
    expected = {"origin", "exp-lut", *VARIANTS}
    gates = {"source_run_pass": source.get("pass") is True, "primary_run_pass": primary.get("pass") is True,
             "resource_matrix_complete": set(resources) == expected, "resource_cases_pass": bool(resources) and all(x["pass"] for x in resources.values()),
             "resource_workloads_match": resource_workload_match(resources),
             "primary_measurements_are_five": all(item["metrics"]["kernel_us"]["count"] == 5 for qs in pmat.values() for item in qs.values()),
             "reproduction_matrix_complete": len(repro) == 20, "native_zero_not_interpreted_as_no_access": native["status"] in {"VALIDATED","UNAVAILABLE"}}
    combined = {"schema_version":1,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"run_id":run_dir.name,
      "scope":"Hexagon Simulator diagnostic; not Snapdragon device performance",
      "setup":{"operator":"FlashAttention + serial d8 SCNA exp2","dataset":"deterministic synthetic Q/K/V/mask generated by harness; no external dataset",
        "primary":{"qo":QOS,"kv":64,"heads":12,"kv_heads":2,"head_dim":128,"mask":"full","workers":1,"warmup":1,"measurements":5},
        "micro":{"warmup":5,"measurements":1000,"processes":2},"baselines":["origin","exp-lut","stage1_dynamic_row"],
        "simulator":{"model":"v79na_1","vtcm_bytes":8388608,"hvx128_contexts":6,"hmx":"legacy adapter","sdk":"6.6.0.0","tools_clang":"19.0.07"},
        "formal_run_host":"未记录","resource_audit_host":environment},
      "source_runs":{"independent":{"path":str(source_run),"summary_sha256":sha256(source_run/"summary.json")},
                     "matched_primary":{"path":str(primary_run),"summary_sha256":sha256(primary_run/"summary_all_serial.json")}},
      "micro":source["micro"],"primary_attention":pmat,"reproducibility":repro,"ablation":ablation,
      "static_metrics":static,"resource_audit":resources,"native_trace":native,"gates":gates,"pass":all(gates.values())}
    write_csvs(run_dir, combined, events)
    (run_dir/"combined_summary.json").write_text(json.dumps(combined,indent=2,sort_keys=True)+"\n")
    evidence = {"source_files":[{"path":str(p),"sha256":sha256(p)} for p in [source_run/"summary.json",primary_run/"summary_all_serial.json",primary_run/"static/static_metrics.json"]],
                "native":native,"resource_logs":[str(p.relative_to(run_dir)) for p in sorted((run_dir/"raw/resource").glob("*.log"))]}
    (run_dir/"evidence_manifest.json").write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
    (run_dir/"verification_combined.json").write_text(json.dumps({"pass":combined["pass"],"gates":gates},indent=2,sort_keys=True)+"\n")
    native_line = ("NATIVE_TRACE_VALIDATION status={status} kernel_attributable={kernel} filtered_exit={exit} "
                   "pc_matches={matches} memtrace_bytes={mem} coproctrace_bytes={copro} pctrace_bytes={pc} "
                   "unfiltered_full_process_valid={full}\n").format(
        status=native["status"], kernel=int(native["kernel_attributable"]), exit=native["filtered"]["exit_code"],
        matches=native["filtered"]["pc_matches"], mem=native["filtered"]["files"]["memtrace"]["bytes"],
        copro=native["filtered"]["files"]["coproctrace"]["bytes"], pc=native["filtered"]["files"]["pctrace"]["bytes"],
        full=int(native["unfiltered"]["validated_full_process_only"]))
    (run_dir/"raw/native/native_validation.log").write_text(native_line)
    return combined

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-run",type=Path,required=True); p.add_argument("--primary-run",type=Path,required=True); p.add_argument("--run-dir",type=Path,required=True); a=p.parse_args()
    result=build(a.source_run.resolve(),a.primary_run.resolve(),a.run_dir.resolve()); print(json.dumps({"pass":result["pass"],"gates":result["gates"]},sort_keys=True)); return 0 if result["pass"] else 1
if __name__ == "__main__": raise SystemExit(main())
