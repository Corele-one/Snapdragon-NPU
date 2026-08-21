#!/usr/bin/env python3
"""Apply preregistered stack/spill gates to v79 SCNA candidates."""
import argparse
import csv
import json
from pathlib import Path

STRICT = {"d7_pairret_inline", "d7_prebroadcast"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text())
    baseline = metrics.get("static_d8_ref", {})
    rows = []
    for name, data in metrics.items():
        evaluator_spill = int(data.get("spill_memory", 0))
        evaluator_stack = int(data.get("stack_frame_bytes", 0))
        caller = data.get("caller", {})
        base_caller = baseline.get("caller", {})
        caller_spill_delta = int(caller.get("spill_memory", 0)) - int(base_caller.get("spill_memory", 0))
        caller_stack_delta = int(caller.get("stack_frame_bytes", 0)) - int(base_caller.get("stack_frame_bytes", 0))
        static_pass = not data.get("missing", False)
        reasons = []
        if name in STRICT:
            if evaluator_spill > int(baseline.get("spill_memory", 0)):
                static_pass = False; reasons.append("evaluator_spill_increase")
            if evaluator_stack > int(baseline.get("stack_frame_bytes", 0)):
                static_pass = False; reasons.append("evaluator_stack_increase")
            if caller_spill_delta > 0:
                static_pass = False; reasons.append("caller_spill_increase")
            if caller_stack_delta > 0:
                static_pass = False; reasons.append("caller_stack_increase")
        rows.append({
            "kernel_impl": name, "static_pass": int(static_pass),
            "reasons": ";".join(reasons), "instructions": data.get("instructions"),
            "packets": data.get("packets"), "splat": data.get("splat"),
            "scalar_weight_multiply": data.get("scalar_weight_multiply"),
            "evaluator_spill": evaluator_spill, "evaluator_stack": evaluator_stack,
            "caller_spill_delta": caller_spill_delta, "caller_stack_delta": caller_stack_delta,
        })
    Path(args.json_out).write_text(json.dumps({"schema_version": 3, "rows": rows}, indent=2) + "\n")
    with Path(args.csv_out).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
