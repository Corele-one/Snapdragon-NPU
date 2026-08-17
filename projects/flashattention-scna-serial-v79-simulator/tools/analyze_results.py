#!/usr/bin/env python3
"""Parse simulator markers into a reproducible JSON/CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

MARKERS = {
    "SIM_CAPABILITY",
    "SCNA_SIM_RESULT",
    "ATTENTION_TIMER",
    "ATTENTION_VERIFY",
    "ATTENTION_SMOKE_RESULT",
    "SIM_PROCESS_RESULT",
}
PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
TOTAL_RE = re.compile(r"Total:\s+Insns=(\d+)\s+Pcycles=(\d+)")


def scalar(value: str):
    value = value.rstrip(":")
    if value in {"PASS", "FAIL", "SKIP"}:
        return value
    try:
        return int(value, 0)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_logs(root: Path):
    rows = defaultdict(list)
    for path in sorted(root.rglob("*.log")):
        relative = str(path.relative_to(root.parent))
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            total = TOTAL_RE.search(line)
            if total:
                rows["SIMULATOR_TOTAL"].append({
                    "instructions": int(total.group(1)),
                    "pcycles": int(total.group(2)),
                    "scope": "full_process_including_qurt_loader",
                    "source": relative,
                    "line": line_number,
                })
            marker = line.split(maxsplit=1)[0] if line else ""
            if marker not in MARKERS:
                continue
            row = {key: scalar(value) for key, value in PAIR_RE.findall(line)}
            row["source"] = relative
            row["line"] = line_number
            rows[marker].append(row)
    return rows


def numeric_summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def build_summary(run_dir: Path):
    rows = parse_logs(run_dir / "raw")
    micro_by_variant = defaultdict(list)
    for row in rows["SCNA_SIM_RESULT"]:
        if row.get("status") == "PASS" and row.get("source", "").startswith("raw/micro/"):
            micro_by_variant[str(row["variant"])].append(row)
    micro = {}
    for variant, samples in sorted(micro_by_variant.items()):
        micro[variant] = {
            "samples": samples,
            "elapsed_us": numeric_summary(row["elapsed_us"] for row in samples),
            "pair_elapsed_us": numeric_summary(row["pair_elapsed_us"] for row in samples),
            "prepare_elapsed_us": numeric_summary(row["prepare_elapsed_us"] for row in samples),
        }

    grouped = defaultdict(list)
    timer_fields = [
        "kernel_us", "profiled_total_us", "q_load_us", "k_load_us", "v_load_us", "qk_dot_us",
        "safe_sm_us", "core_acc_us", "o_scale_us", "o_store_us", "scna_exp_us", "param_prepare_us",
    ]
    for row in rows["ATTENTION_TIMER"]:
        if not row.get("source", "").startswith("raw/attention/"):
            continue
        key = (str(row["mode"]), int(row["qo"]), int(row["kv"]), int(row["heads"]),
               int(row["kv_heads"]), int(row["head_dim"]))
        grouped[key].append(row)
    attention = {}
    for key, samples in sorted(grouped.items()):
        mode, qo, kv, heads, kv_heads, dim = key
        ident = f"{mode}_q{qo}_kv{kv}_h{heads}_kh{kv_heads}_d{dim}"
        metrics = {field: numeric_summary(row[field] for row in samples) for field in timer_fields}
        total = metrics["profiled_total_us"]["median"]
        kernel = metrics["kernel_us"]["median"]
        safe = metrics["safe_sm_us"]["median"]
        scna = metrics["scna_exp_us"]["median"]
        metrics["scna_share_attention_percent"] = 100.0 * scna / kernel if kernel else 0.0
        metrics["scna_share_profiled_percent"] = 100.0 * scna / total if total else 0.0
        metrics["scna_share_safe_sm_percent"] = 100.0 * scna / safe if safe else 0.0
        attention[ident] = {
            "mode": mode, "qo": qo, "kv": kv, "heads": heads, "kv_heads": kv_heads, "head_dim": dim,
            "samples": samples, "metrics": metrics,
        }

    verification_map = {}
    for row in rows["ATTENTION_VERIFY"]:
        if not row.get("source", "").startswith("raw/attention/"):
            continue
        key = tuple(row.get(field) for field in ("mode", "qo", "kv", "heads", "kv_heads", "head_dim", "tail_check"))
        verification_map[key] = row
    verifications = list(verification_map.values())
    processes = [row for row in rows["SIM_PROCESS_RESULT"] if not row.get("source", "").startswith("raw/detailed/")]
    expected_variants = {
        "stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
        "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized",
    }
    observed_variants = set(micro)
    micro_rows = [row for row in rows["SCNA_SIM_RESULT"] if row.get("source", "").startswith("raw/micro/")]
    numeric_limits = {
        "rmse": 0.003, "max_abs": 0.02, "dense_rmse": 0.011, "dense_max_abs": 0.16,
        "random_rmse": 0.011, "random_max_abs": 0.16,
    }
    expected_sweep = {
        (mode, qo, 64, 12, 2, 128)
        for mode in ("origin", "exp-lut", "stage1", "optimized")
        for qo in (1, 4, 8, 16, 32)
    }
    observed_attention = {
        (item["mode"], item["qo"], item["kv"], item["heads"], item["kv_heads"], item["head_dim"])
        for item in attention.values()
    }
    expected_smoke = {
        ("optimized", 1, 64, 2, 1, 64),
        ("optimized", 3, 65, 2, 1, 64),
    }
    attention_timer_rows = [row for row in rows["ATTENTION_TIMER"] if row.get("source", "").startswith("raw/attention/")]
    gates = {
        "capability_pass": any(row.get("status") == "PASS" for row in rows["SIM_CAPABILITY"]),
        "micro_all_variants_present": observed_variants == expected_variants,
        "micro_all_pass": bool(micro_by_variant) and all(
            row.get("status") == "PASS" for row in rows["SCNA_SIM_RESULT"]
            if row.get("source", "").startswith("raw/micro/")
        ),
        "micro_numeric_pass": bool(micro_rows) and all(
            all(float(row[field]) <= limit for field, limit in numeric_limits.items())
            and int(row.get("canonical_oracle_mismatches", 999)) <= 1
            for row in micro_rows
        ),
        "micro_invariants_pass": bool(micro_rows) and all(
            int(row.get(field, 999)) == 0
            for row in micro_rows
            for field in ("random_nonfinite_count", "monotonic_violations", "negative_count", "nan_count", "paired_single_mismatches")
        ),
        "attention_all_pass": bool(verifications) and all(row.get("status") == "PASS" for row in verifications),
        "attention_sweep_complete": expected_sweep <= observed_attention,
        "attention_smoke_tail_complete": expected_smoke <= observed_attention,
        "attention_tail_mask_pass": bool(attention_timer_rows) and all(
            int(row.get("tail_nonzero", 999)) == 0 and int(row.get("masked_nonzero", 999)) == 0
            for row in attention_timer_rows
        ),
        "processes_all_zero": bool(processes) and all(row.get("exit_code") == 0 for row in processes),
    }
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "scope": "Hexagon simulator diagnostic; not device performance",
        "capability": rows["SIM_CAPABILITY"],
        "micro": micro,
        "attention": attention,
        "verifications": verifications,
        "smoke": rows["ATTENTION_SMOKE_RESULT"],
        "processes": processes,
        "simulator_totals": rows["SIMULATOR_TOTAL"],
        "gates": gates,
        "pass": all(gates.values()),
    }
    return summary


def write_csv(summary, path: Path):
    fields = ["mode", "qo", "kv", "heads", "kv_heads", "head_dim", "metric", "median", "min", "max", "count"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in summary["attention"].values():
            for name, metric in item["metrics"].items():
                if not isinstance(metric, dict):
                    continue
                writer.writerow({
                    "mode": item["mode"], "qo": item["qo"], "kv": item["kv"], "heads": item["heads"],
                    "kv_heads": item["kv_heads"], "head_dim": item["head_dim"], "metric": name,
                    "median": metric["median"], "min": metric["min"], "max": metric["max"], "count": metric["count"],
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.run_dir.resolve())
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.run_dir / "verification.json").write_text(json.dumps({"pass": summary["pass"], "gates": summary["gates"]}, indent=2) + "\n")
    write_csv(summary, args.run_dir / "attention_summary.csv")
    print(json.dumps({"pass": summary["pass"], "gates": summary["gates"]}, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
