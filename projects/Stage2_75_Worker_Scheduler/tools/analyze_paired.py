#!/usr/bin/env python3
"""Paired same-session ratios for Stage 2.75 scheduler measurements."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_scheduler import METRICS, median_ci, parse_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--comparison", action="append", required=True,
                        help="candidate:baseline label pair")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    samples: dict[tuple[object, ...], dict[str, dict[tuple[int, int], dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    metas: dict[tuple[object, ...], dict[str, object]] = {}
    for path in sorted((args.run_dir / "raw/attention").glob("*.log")):
        meta, rows = parse_log(path)
        condition = (meta["q"], meta["kv"], meta["q_task_rows"], meta["requested_workers"], meta["mapping"])
        metas[condition] = {key: meta[key] for key in
                            ("q", "kv", "q_task_rows", "requested_workers", "mapping")}
        label = str(meta["label"])
        session = int(meta["session"])
        for row in rows:
            samples[condition][label][(session, int(row["iteration"]))] = row

    results = []
    metric_names = ["host_us", *METRICS]
    for comparison_index, spec in enumerate(args.comparison):
        candidate, baseline = spec.split(":", 1)
        for condition_index, condition in enumerate(sorted(samples, key=str)):
            by_label = samples[condition]
            if candidate not in by_label or baseline not in by_label:
                continue
            matched = sorted(set(by_label[candidate]) & set(by_label[baseline]))
            if not matched:
                continue
            metrics = {}
            for metric_index, metric in enumerate(metric_names):
                ratios = [by_label[candidate][point][metric] / by_label[baseline][point][metric]
                          for point in matched if by_label[baseline][point][metric] != 0]
                metrics[metric] = median_ci(ratios, args.draws,
                                            27575 + comparison_index * 10007 + condition_index * 101 + metric_index)
            results.append({"candidate": candidate, "baseline": baseline, "meta": metas[condition],
                            "matched_points": len(matched), "ratio_candidate_over_baseline": metrics})

    output = {"schema_version": 1, "bootstrap_draws": args.draws,
              "pairing": "same session and measurement iteration", "comparisons": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
