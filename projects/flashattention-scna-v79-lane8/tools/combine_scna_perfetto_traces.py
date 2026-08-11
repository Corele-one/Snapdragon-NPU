#!/usr/bin/env python3
"""Combine independent serial/lane8 trace replays into separate processes."""

import argparse
import json
from pathlib import Path


def remap(events, pid_base, process_label):
    pids = sorted({event.get("pid", 0) for event in events})
    mapping = {old: pid_base + index for index, old in enumerate(pids)}
    result = []
    for event in events:
        copied = dict(event)
        copied["pid"] = mapping[event.get("pid", 0)]
        if copied.get("ph") == "M" and copied.get("name") == "process_name":
            copied["args"] = {"name": f"{process_label} / {copied.get('args', {}).get('name', '')}"}
        result.append(copied)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--lane8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane8-label", default="SCNA lane8")
    args = parser.parse_args()
    serial = json.loads(args.serial.read_text(encoding="utf-8"))
    lane8 = json.loads(args.lane8.read_text(encoding="utf-8"))
    events = remap(serial["traceEvents"], 1000, "SCNA serial (independent replay)")
    events += remap(lane8["traceEvents"], 2000, f"{args.lane8_label} (independent replay)")
    events.append({
        "name": "comparison_note", "cat": "metadata", "ph": "i", "s": "g", "pid": 0, "tid": 0, "ts": 0,
        "args": {
            "layouts_execute_concurrently": False,
            "note": f"Serial and {args.lane8_label} are independent replays from the same binary/shape/seed; overlapping timestamps do not imply concurrent execution.",
            "hardware_trace": False,
        },
    })
    output = {
        "displayTimeUnit": "us",
        "traceEvents": events,
        "metadata": {
            "format": "Chrome Trace Event JSON accepted by Perfetto UI",
            "hardware_trace": False,
            "independent_replays": True,
            "serial_source": str(args.serial),
            "lane8_source": str(args.lane8),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    serial_summary_path = args.serial.parent / "event_trace_summary.json"
    lane8_summary_path = args.lane8.parent / "event_trace_summary.json"
    combined_summary = {
        "hardware_trace": False,
        "independent_replays": True,
        "serial": json.loads(serial_summary_path.read_text(encoding="utf-8")),
        "lane8": json.loads(lane8_summary_path.read_text(encoding="utf-8")),
    }
    summary_path = args.output.parent / "event_trace_summary.json"
    summary_path.write_text(json.dumps(combined_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    notes_path = args.output.parent / "derived_trace_notes.md"
    notes_path.write_text(
        f"# SCNA serial vs {args.lane8_label} Perfetto trace notes\n\n"
        "- `hardware_trace=false`: all slices come from DSP qtimer `t0/t1` software events; no PMU, ETM, sysmon, or utilization counters are present.\n"
        f"- Serial and {args.lane8_label} are independent replays of the same binary, shape, seed, and iteration schedule; timestamp overlap does not mean concurrent execution.\n"
        "- Per-layout/per-Qo traces and audit outputs are stored in their corresponding subdirectories.\n"
        f"- Drag `{args.output.name}` into https://ui.perfetto.dev to inspect the combined comparison.\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
