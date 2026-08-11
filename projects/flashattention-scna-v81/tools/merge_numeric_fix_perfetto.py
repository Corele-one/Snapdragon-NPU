#!/usr/bin/env python3
"""Merge pre/post numeric-fix trace replays into aligned, non-concurrent tracks."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-sha256", required=True)
    parser.add_argument("--post-sha256", required=True)
    args = parser.parse_args()

    sources = [
        ("pre", "Origin-HVX", "pre/baseline/perfetto/baseline_q32.perfetto.json", args.pre_sha256),
        ("post", "Origin-HVX", "post/baseline/perfetto/baseline_q32.perfetto.json", args.post_sha256),
        ("pre", "SCNA direct d8", "pre/scna-d8/perfetto/scna_fp16_q32.perfetto.json", args.pre_sha256),
        ("post", "SCNA direct d8", "post/scna-d8/perfetto/scna_fp16_q32.perfetto.json", args.post_sha256),
    ]
    merged = []
    audits = []
    for index, (revision, mode, relative, binary_hash) in enumerate(sources, start=1):
        path = args.trace_root / relative
        trace = json.loads(path.read_text(encoding="utf-8"))
        process_pid = 1000 + index
        durations = []
        process_metadata = 0
        thread_metadata = 0
        for event in trace["traceEvents"]:
            copied = dict(event)
            copied["pid"] = process_pid
            copied["args"] = dict(copied.get("args", {}))
            copied["args"].update({
                "revision": revision,
                "evaluator": mode,
                "binary_sha256": binary_hash,
                "qo_len": 32,
                "trace_execution": "independent replay; aligned for visual comparison, not concurrent execution",
            })
            if copied.get("ph") == "M" and copied.get("name") == "process_name":
                copied["args"]["name"] = f"{revision.upper()} | {mode} | Qo32"
                process_metadata += 1
            elif copied.get("ph") == "M" and copied.get("name") == "thread_name":
                thread_metadata += 1
            if copied.get("ph") == "X":
                duration = copied.get("dur", 0)
                if duration < 0:
                    raise SystemExit(f"negative duration in {path}: {duration}")
                durations.append(duration)
            merged.append(copied)
        audits.append({
            "revision": revision,
            "mode": mode,
            "source": str(path),
            "binary_sha256": binary_hash,
            "process_metadata": process_metadata,
            "thread_metadata": thread_metadata,
            "duration_events": len(durations),
            "negative_durations": sum(value < 0 for value in durations),
            "hardware_trace": trace.get("metadata", {}).get("hardware_trace"),
        })

    payload = {
        "displayTimeUnit": "us",
        "traceEvents": merged,
        "metadata": {
            "format": "Chrome Trace Event JSON accepted by Perfetto UI",
            "hardware_trace": False,
            "timeline_source": "event-level software instrumentation using DSP qtimer t0/t1",
            "comparison_note": "Four independent trace replays are aligned at t=0 for comparison and were not executed concurrently.",
            "shape": "Qo=32, KV=4096, heads/KV-heads=12/2, head-dim=128",
            "warmup": 1,
            "measured_iterations": 3,
            "sources": audits,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    audit_path = args.output.with_name("numeric_fix_pre_post_q32_trace_audit.json")
    audit_path.write_text(json.dumps({"trace_file": str(args.output), "sources": audits}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
