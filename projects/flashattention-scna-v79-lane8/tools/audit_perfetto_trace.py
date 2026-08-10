#!/usr/bin/env python3
"""Audit Figure8 Chrome Trace JSON before opening it in Perfetto."""

import argparse
import json
from pathlib import Path


REQUIRED_BASE = {"q_load", "k_load", "qk_dot", "safe_sm", "v_load", "core_acc", "o_scale", "o_store"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--layout", choices=("serial", "lane8"), required=True)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--binary-sha256", required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    errors = []
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        errors.append("traceEvents is absent or is not a list")
        events = []
    metadata_events = [event for event in events if event.get("ph") == "M"]
    if not any(event.get("name") == "process_name" for event in metadata_events):
        errors.append("process_name metadata is absent")
    if not any(event.get("name") == "thread_name" for event in metadata_events):
        errors.append("thread_name metadata is absent")
    slices = [event for event in events if event.get("ph") == "X"]
    for event in slices:
        if event.get("dur", -1) < 0:
            errors.append(f"negative duration: {event.get('name')}")
    components = {event.get("args", {}).get("component") for event in slices}
    missing = REQUIRED_BASE - components
    if missing:
        errors.append(f"missing required components: {sorted(missing)}")
    if "scna_exp" not in components:
        errors.append("SCNA trace has no scna_exp slice")

    notes = [event.get("args", {}) for event in events if event.get("name") == "trace_note"]
    if not notes:
        errors.append("trace_note metadata is absent")
    for note in notes:
        if note.get("hardware_trace") is not False:
            errors.append("hardware_trace must be false")
        if note.get("layout") != args.layout:
            errors.append(f"layout mismatch: {note.get('layout')} != {args.layout}")
        if note.get("scna_width") != args.width:
            errors.append(f"width mismatch: {note.get('scna_width')} != {args.width}")
        if note.get("binary_sha256") != args.binary_sha256:
            errors.append("binary SHA-256 mismatch")

    parent_slices = [
        event for event in slices
        if event.get("args", {}).get("component") in {"safe_sm", "core_acc"}
    ]
    for child in [event for event in slices if event.get("args", {}).get("component") == "scna_exp"]:
        child_args = child.get("args", {})
        child_start = child.get("ts", 0)
        child_end = child_start + child.get("dur", 0)
        contained = False
        for parent in parent_slices:
            parent_args = parent.get("args", {})
            same_call = all(
                child_args.get(key) == parent_args.get(key)
                for key in ("iteration", "worker", "kv_head", "block_c")
            )
            parent_start = parent.get("ts", 0)
            parent_end = parent_start + parent.get("dur", 0)
            if same_call and parent_start <= child_start and child_end <= parent_end:
                contained = True
                break
        if not contained:
            errors.append(f"scna_exp is outside safe_sm/core_acc: {child.get('name')}")

    overflow = sum(item.get("event_overflow_total", 0) for item in summary.get("by_qo_len", {}).values())
    if overflow != 0:
        errors.append(f"event buffer overflow total is {overflow}")
    prov = summary.get("provenance", {})
    if prov.get("binary_sha256") != args.binary_sha256:
        errors.append("summary binary SHA-256 mismatch")
    if summary.get("layouts") != [args.layout]:
        errors.append(f"summary layouts mismatch: {summary.get('layouts')}")
    if summary.get("scna_widths") != [args.width]:
        errors.append(f"summary widths mismatch: {summary.get('scna_widths')}")

    result = {
        "trace": str(args.trace),
        "hardware_trace": False,
        "slice_count": len(slices),
        "scna_exp_slices": sum(
            1 for event in slices if event.get("args", {}).get("component") == "scna_exp"
        ),
        "event_overflow_total": overflow,
        "errors": errors,
        "pass": not errors,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
