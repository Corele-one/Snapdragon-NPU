#!/usr/bin/env python3
"""Quantify small-Q fixed-cost dilution from a formal baseline run."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from analyze_baseline import bootstrap_median, fields, paired_ratio


QOS = (1, 4, 8, 16, 32)
LABELS = ("static_d8_ref", "d7_pairret_noinline")
STAGES = ("profiled_total", "safe_sm", "scna_exp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = {label: {q: defaultdict(lambda: {"host": None, **{stage: 0.0 for stage in STAGES}})
                    for q in QOS} for label in LABELS}
    for path in sorted((args.run_dir / "raw/attention").glob("*.log")):
        match = re.match(r"(.+)_q(\d+)_s(\d+)\.log", path.name)
        if not match or match.group(1) not in LABELS:
            continue
        label, q, session = match.group(1), int(match.group(2)), int(match.group(3))
        for line in path.read_text(errors="replace").splitlines():
            data = fields(line)
            if data.get("phase") != "measure":
                continue
            iteration = int(data["iteration"])
            key = (session, iteration)
            if "FIG8_ATTENTION_HOST_TIMING" in line and int(data.get("ret", "1")) == 0:
                rows[label][q][key]["host"] = float(data["host_elapsed_us"])
            elif "FIG8_ATTENTION_TIMERS" in line:
                for stage in STAGES:
                    rows[label][q][key][stage] += float(data.get(stage, 0))

    summary = {label: {} for label in LABELS}
    raw = {label: {} for label in LABELS}
    for label in LABELS:
        for q in QOS:
            complete = {key: row for key, row in rows[label][q].items() if row["host"] is not None}
            raw[label][q] = complete
            summary[label][q] = {
                metric: bootstrap_median([row[metric] for row in complete.values()], seed=0xA810 + q)
                for metric in ("host", *STAGES)
            }
            fractions = [row["scna_exp"] / row["host"] for row in complete.values() if row["host"]]
            summary[label][q]["scna_over_host"] = bootstrap_median(fractions, seed=0xA820 + q)

    ratios = {q: paired_ratio(
        {key: row["host"] for key, row in raw["d7_pairret_noinline"][q].items()},
        {key: row["host"] for key, row in raw["static_d8_ref"][q].items()},
        seed=0xA830 + q,
    ) for q in QOS}

    result = {
        "schema_version": 1,
        "source_run": str(args.run_dir),
        "summary": summary,
        "pairret_over_static_host": ratios,
        "assembly": {
            "evaluator_instructions": 112,
            "evaluator_packets": 36,
            "evaluator_frame_bytes": 0,
            "call_packet_already_filled_instructions": 3,
            "return_packet_instructions": 3,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
