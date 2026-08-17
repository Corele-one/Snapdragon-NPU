#!/usr/bin/env python3
"""Parse the matched all-serial simulator campaign and enforce its gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import numeric_summary, parse_logs  # noqa: E402

VARIANTS = [
    "stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
    "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized",
]
BUILD_IDS = {name: index for index, name in enumerate(VARIANTS)}
QOS = (1, 4, 8, 16, 32)
TIMER_FIELDS = (
    "kernel_us", "profiled_total_us", "q_load_us", "k_load_us", "v_load_us", "qk_dot_us",
    "safe_sm_us", "core_acc_us", "o_scale_us", "o_store_us", "scna_exp_us", "param_prepare_us",
)


def identity(row):
    mode = str(row["mode"])
    variant = str(row.get("variant", "none"))
    scheme = variant if mode == "serial" else mode
    requested = int(row.get("requested_workers", row.get("workers", 1)))
    policy = "auto" if requested == 0 else f"w{requested}"
    return (scheme, policy, int(row["qo"]), int(row["kv"]), int(row["heads"]),
            int(row["kv_heads"]), int(row["head_dim"]))


def item_id(key):
    scheme, policy, qo, kv, heads, kvh, dim = key
    return f"{scheme}_{policy}_q{qo}_kv{kv}_h{heads}_kh{kvh}_d{dim}"


def dedupe(rows, marker):
    seen, output, duplicates = {}, [], []
    fields = ("mode", "variant", "requested_workers", "qo", "kv", "heads", "kv_heads", "head_dim", "tail_check")
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in seen:
            duplicates.append({"marker": marker, "identity": list(key), "first": seen[key]["source"], "second": row["source"]})
        else:
            seen[key] = row
            output.append(row)
    return output, duplicates


def expected_active_workers(qo):
    # Harness: min(HVX contexts=6, query-block tasks=ceil(Qo/4)*KVH=2, VTCM task cap=8).
    return min(6, ((qo + 3) // 4) * 2, 8)


def summarize_group(key, samples):
    scheme, policy, qo, kv, heads, kvh, dim = key
    metrics = {field: numeric_summary(row[field] for row in samples) for field in TIMER_FIELDS}
    kernel, profiled = metrics["kernel_us"]["median"], metrics["profiled_total_us"]["median"]
    safe, scna = metrics["safe_sm_us"]["median"], metrics["scna_exp_us"]["median"]
    metrics.update({
        "scna_share_attention_percent": 100.0 * scna / kernel if kernel else 0.0,
        "scna_share_profiled_percent": 100.0 * scna / profiled if profiled else 0.0,
        "scna_share_safe_sm_percent": 100.0 * scna / safe if safe else 0.0,
    })
    return {
        "scheme": scheme, "worker_policy": policy, "qo": qo, "kv": kv, "heads": heads,
        "kv_heads": kvh, "head_dim": dim, "samples": samples, "metrics": metrics,
        "active_workers": sorted({int(row.get("active_workers", row.get("workers", -1))) for row in samples}),
    }


def evaluate_gates(rows, attention, verifications, duplicates):
    perf_expected = {(scheme, "w1", qo, 64, 12, 2, 128)
                     for scheme in ["origin", "exp-lut", *VARIANTS] for qo in QOS}
    tail_expected = {(variant, "w1", 3, 65, 2, 1, 64) for variant in VARIANTS}
    auto_expected = {("optimized", "auto", qo, 64, 12, 2, 128) for qo in QOS}
    observed = {tuple((item[field] for field in
                ("scheme", "worker_policy", "qo", "kv", "heads", "kv_heads", "head_dim")))
                for item in attention.values()}
    perf_counts = {key: attention[item_id(key)]["metrics"]["kernel_us"]["count"] for key in perf_expected if item_id(key) in attention}
    timer_rows = [row for row in rows["ATTENTION_TIMER"] if row.get("source", "").startswith("raw/attention/")]
    serial_rows = [row for row in timer_rows if row.get("mode") == "serial"]
    micro_rows = [row for row in rows["SCNA_SIM_RESULT"] if row.get("source", "").startswith("raw/micro/")]
    micro_variants = {str(row.get("variant")) for row in micro_rows if row.get("status") == "PASS"}
    serial_ids_ok = all(BUILD_IDS.get(str(row.get("variant"))) == int(row.get("build_id", -1)) for row in serial_rows)
    single_workers_ok = all(int(row.get("active_workers", -1)) == 1 for row in timer_rows
                            if int(row.get("requested_workers", 1)) == 1)
    auto_workers_ok = all(int(row.get("active_workers", -1)) == expected_active_workers(int(row["qo"]))
                          for row in serial_rows if int(row.get("requested_workers", 1)) == 0)
    return {
        "capability_pass": any(row.get("status") == "PASS" for row in rows["SIM_CAPABILITY"]),
        "micro_all_variants_present": micro_variants == set(VARIANTS),
        "micro_all_pass": bool(micro_rows) and all(row.get("status") == "PASS" for row in micro_rows),
        "attention_no_duplicate_verification": not duplicates,
        "attention_performance_matrix_complete": perf_expected <= observed and all(value == 5 for value in perf_counts.values()),
        "attention_tail_matrix_complete": tail_expected <= observed,
        "attention_auto_matrix_complete": auto_expected <= observed,
        "attention_variant_build_ids_match": bool(serial_rows) and serial_ids_ok,
        "attention_single_workers_match": single_workers_ok,
        "attention_auto_workers_match": auto_workers_ok,
        "attention_all_verifications_pass": bool(verifications) and all(row.get("status") == "PASS" for row in verifications),
        "attention_numeric_thresholds_pass": bool(verifications) and all(
            float(row.get("rmse", 999)) <= 0.002 and float(row.get("max_abs", 999)) <= 0.01
            and int(row.get("candidate_nonfinite", 999)) == 0
            and int(row.get("reference_nonfinite", 999)) == 0 for row in verifications),
        "attention_tail_mask_pass": bool(timer_rows) and all(
            int(row.get("tail_nonzero", 999)) == 0 and int(row.get("masked_nonzero", 999)) == 0 for row in timer_rows),
        "processes_all_zero": bool(rows["SIM_PROCESS_RESULT"]) and all(
            int(row.get("exit_code", -1)) == 0 for row in rows["SIM_PROCESS_RESULT"]
            if not row.get("source", "").startswith("raw/detailed/")),
    }


def build_summary(run_dir):
    rows = parse_logs(run_dir / "raw")
    grouped = defaultdict(list)
    for row in rows["ATTENTION_TIMER"]:
        if row.get("source", "").startswith("raw/attention/"):
            grouped[identity(row)].append(row)
    attention = {item_id(key): summarize_group(key, samples) for key, samples in sorted(grouped.items())}
    verifications, duplicates = dedupe(
        [row for row in rows["ATTENTION_VERIFY"] if row.get("source", "").startswith("raw/attention/")],
        "ATTENTION_VERIFY",
    )
    gates = evaluate_gates(rows, attention, verifications, duplicates)
    return {
        "schema_version": 2, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name, "scope": "Hexagon Simulator diagnostic; not Snapdragon device performance",
        "variants": VARIANTS, "attention": attention, "verifications": verifications,
        "duplicates": duplicates, "capability": rows["SIM_CAPABILITY"],
        "micro": rows["SCNA_SIM_RESULT"], "processes": rows["SIM_PROCESS_RESULT"],
        "simulator_totals": rows["SIMULATOR_TOTAL"], "gates": gates, "pass": all(gates.values()),
    }


def write_csv(summary, path):
    fields = ["scheme", "worker_policy", "qo", "kv", "heads", "kv_heads", "head_dim", "metric", "median", "min", "max", "count", "values"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for item in summary["attention"].values():
            for name, metric in item["metrics"].items():
                if not isinstance(metric, dict): continue
                writer.writerow({**{field: item[field] for field in fields[:7]}, "metric": name,
                                 "median": metric["median"], "min": metric["min"], "max": metric["max"],
                                 "count": metric["count"], "values": json.dumps(metric["values"])})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); summary = build_summary(run_dir)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (run_dir / "summary.json").write_text(payload); (run_dir / "summary_all_serial.json").write_text(payload)
    verification = {"pass": summary["pass"], "gates": summary["gates"], "duplicates": summary["duplicates"]}
    (run_dir / "verification_all_serial.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    write_csv(summary, run_dir / "attention_all_serial.csv")
    print(json.dumps(verification, sort_keys=True)); return 0 if summary["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
