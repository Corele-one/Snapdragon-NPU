#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


COMPONENTS = [
    "q_load",
    "k_load",
    "v_load",
    "qk_dot",
    "safe_sm",
    "core_acc",
    "o_scale",
    "o_store",
    "scna_exp",
]

UNIT_BY_COMPONENT = {
    "q_load": "Memory/L2 Load-Pack",
    "k_load": "Memory/L2 Load-Pack",
    "v_load": "Memory/L2 Load-Pack",
    "qk_dot": "HMX Unit",
    "safe_sm": "HVX Unit",
    "core_acc": "Mixed HVX/HMX Section",
    "o_scale": "Mixed HVX/HMX Section",
    "o_store": "Store/Writeback",
    "scna_exp": "HVX Unit",
}

COLOR_BY_COMPONENT = {
    "q_load": "good",
    "k_load": "rail_response",
    "v_load": "rail_animation",
    "qk_dot": "thread_state_running",
    "safe_sm": "rail_load",
    "core_acc": "thread_state_iowait",
    "o_scale": "thread_state_uninterruptible",
    "o_store": "cq_build_running",
    "scna_exp": "rail_load",
}

TEXT_FIELDS = {
    "mode",
    "phase",
    "component",
    "layout",
    "mask_mode",
    "binary_sha256",
    "isa",
    "seed",
    "sdk",
    "tools",
}

TIMER_RE = re.compile(r"FIG8_ATTENTION_TIMERS\s+(.*)")
HOST_RE = re.compile(r"FIG8_ATTENTION_HOST_TIMING\s+(.*)")
EVENT_COUNT_RE = re.compile(r"FIG8_ATTENTION_EVENT_COUNT\s+(.*)")
EVENT_RE = re.compile(r"FIG8_ATTENTION_EVENT\s+(.*)")
PROVENANCE_RE = re.compile(r"SCNA_PROVENANCE\s+(.*)")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def mode_to_slug(mode):
    return mode.replace("-", "_")


def read_text_auto(path):
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:256]:
        return data.decode("utf-16-le", errors="replace")
    return data.decode("utf-8", errors="replace")


def parse_key_values(text):
    out = {}
    for key, value in KEY_VALUE_RE.findall(text):
        if key in TEXT_FIELDS:
            out[key] = value
        else:
            out[key] = int(value)
    return out


def parse_logs(input_dir):
    timer_rows = []
    host_rows = []
    event_count_rows = []
    event_rows = []
    provenance_rows = []
    for path in sorted(input_dir.glob("raw_q*.log")):
        for line in read_text_auto(path).splitlines():
            provenance_match = PROVENANCE_RE.search(line)
            if provenance_match:
                row = parse_key_values(provenance_match.group(1))
                row["raw_file"] = path.name
                provenance_rows.append(row)
                continue
            event_match = EVENT_RE.search(line)
            if event_match:
                row = parse_key_values(event_match.group(1))
                row["raw_file"] = path.name
                if row.get("phase") == "measure":
                    event_rows.append(row)
                continue
            event_count_match = EVENT_COUNT_RE.search(line)
            if event_count_match:
                row = parse_key_values(event_count_match.group(1))
                row["raw_file"] = path.name
                if row.get("phase") == "measure":
                    event_count_rows.append(row)
                continue
            timer_match = TIMER_RE.search(line)
            if timer_match:
                row = parse_key_values(timer_match.group(1))
                row["raw_file"] = path.name
                if row.get("phase") == "measure":
                    timer_rows.append(row)
                continue
            host_match = HOST_RE.search(line)
            if host_match:
                row = parse_key_values(host_match.group(1))
                row["raw_file"] = path.name
                if row.get("phase") == "measure":
                    host_rows.append(row)
    return timer_rows, host_rows, event_count_rows, event_rows, provenance_rows


def make_tid(unit, worker, kv_head):
    unit_base = {
        "Kernel Wall": 1000,
        "Memory/L2 Load-Pack": 2000,
        "HVX Unit": 3000,
        "HMX Unit": 4000,
        "Mixed HVX/HMX Section": 5000,
        "Store/Writeback": 6000,
        "Scalar Unit": 7000,
    }[unit]
    return unit_base + worker * 100 + kv_head


def add_metadata(events, pid, name, tids):
    events.append(
        {
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "tid": 0,
            "args": {"name": name},
        }
    )
    for tid, track_name in sorted(tids.items()):
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"name": track_name},
            }
        )


def event_duration_us(row):
    direct = row.get("dur_us", 0)
    endpoints = row.get("t1_us", 0) - row.get("t0_us", 0)
    return max(direct, endpoints, 0)


def lane_key(row):
    return (row["iteration"], row["worker"], row["kv_head"])


def lane_args(row, host_by_iter):
    return {
        "mode": row["mode"],
        "layout": row.get("layout", "n/a"),
        "scna_layout": row.get("scna_layout", -1),
        "scna_width": row.get("scna_width", 0),
        "qo_len": row["qo_len"],
        "kv_len": row["kv_len"],
        "n_heads": row["n_heads"],
        "n_kv_heads": row["n_kv_heads"],
        "head_dim": row["head_dim"],
        "iteration": row["iteration"],
        "worker": row["worker"],
        "kv_head": row["kv_head"],
        "host_elapsed_us": host_by_iter.get(row["iteration"], 0),
    }


def add_control_gaps(events, pid, lane_rows, base_ts, host_by_iter):
    prev_t1 = min(row["t0_us"] for row in lane_rows)
    first = lane_rows[0]
    args = {
        **lane_args(first, host_by_iter),
        "unit": "Scalar Unit",
        "duration_source": "gap between adjacent DSP qtimer events on the same software lane",
        "note": "This is an uninstrumented control/scheduling gap, not hardware scalar utilization.",
    }
    scalar_tid = make_tid("Scalar Unit", first["worker"], first["kv_head"])
    events.append(
        {
            "name": "scalar_track_note",
            "cat": "Scalar Unit",
            "ph": "i",
            "s": "t",
            "pid": pid,
            "tid": scalar_tid,
            "ts": prev_t1 - base_ts,
            "args": {
                **args,
                "duration_source": "instant marker only",
                "note": "No hardware scalar counter is available; duration slices on this track are derived gaps only.",
            },
        }
    )
    for row in lane_rows:
        if row["t0_us"] > prev_t1:
            events.append(
                {
                    "name": "control_gap_uninstrumented",
                    "cat": "Scalar Unit",
                    "ph": "X",
                    "pid": pid,
                    "tid": scalar_tid,
                    "ts": prev_t1 - base_ts,
                    "dur": row["t0_us"] - prev_t1,
                    "args": args,
                }
            )
        prev_t1 = max(prev_t1, row["t1_us"])


def build_trace_for_qo(qo_len, event_rows, host_rows, pid, provenance):
    rows = sorted(
        [row for row in event_rows if row["qo_len"] == qo_len],
        key=lambda row: (row["iteration"], row["t0_us"], row["t1_us"], row["worker"], row["kv_head"]),
    )
    if not rows:
        raise ValueError(f"no FIG8_ATTENTION_EVENT rows for qo_len={qo_len}")

    base_ts = min(row["t0_us"] for row in rows)
    host_by_iter = {
        row["iteration"]: row["host_elapsed_us"]
        for row in host_rows
        if row["qo_len"] == qo_len
    }

    tids = {}
    for row in rows:
        worker = row["worker"]
        kv_head = row["kv_head"]
        for unit in [
            "Kernel Wall",
            "Memory/L2 Load-Pack",
            "HVX Unit",
            "HMX Unit",
            "Mixed HVX/HMX Section",
            "Store/Writeback",
            "Scalar Unit",
        ]:
            tids.setdefault(make_tid(unit, worker, kv_head), f"{unit} / worker {worker} / kv_head {kv_head}")

    first = rows[0]
    meta_args = {
        "source": "FIG8_ATTENTION_EVENT DSP qtimer start/end records",
        "timeline_source": "event-level software instrumentation, not cumulative reconstruction",
        "hardware_trace": False,
        "mode": first["mode"],
        "layout": first.get("layout", "n/a"),
        "scna_layout": first.get("scna_layout", -1),
        "scna_width": first.get("scna_width", 0),
        "binary_sha256": provenance.get("binary_sha256", "unknown"),
        "isa": provenance.get("isa", "v79"),
        "qo_len": qo_len,
        "kv_len": first["kv_len"],
        "n_heads": first["n_heads"],
        "n_kv_heads": first["n_kv_heads"],
        "head_dim": first["head_dim"],
        "unit_mapping": (
            "Memory/L2 Load-Pack=q_load/k_load/v_load; HMX=qk_dot; HVX=safe_sm; "
            "HVX=scna_exp nested under safe_sm or online core_acc; Mixed HVX/HMX=core_acc/o_scale because these Figure 8 component timers wrap both vector setup/state work and HMX dot work; "
            "Store/Writeback=o_store; "
            "Scalar=derived uninstrumented gaps only"
        ),
    }

    events = []
    mode_slug = mode_to_slug(first["mode"])
    add_metadata(events, pid, f"Figure8 {mode_slug} q{qo_len} event-level DSP qtimer", tids)
    events.append(
        {
            "name": "trace_note",
            "cat": "metadata",
            "ph": "i",
            "s": "p",
            "pid": pid,
            "tid": 0,
            "ts": 0,
            "args": meta_args,
        }
    )

    by_lane = defaultdict(list)
    for row in rows:
        by_lane[lane_key(row)].append(row)

    for lane, lane_rows in sorted(by_lane.items()):
        lane_rows = sorted(lane_rows, key=lambda row: (row["t0_us"], row["t1_us"]))
        iteration, worker, kv_head = lane
        start_us = min(row["t0_us"] for row in lane_rows)
        end_us = max(row["t1_us"] for row in lane_rows)
        lane_base_args = lane_args(lane_rows[0], host_by_iter)
        events.append(
            {
                "name": f"flash_attn_prefill q{qo_len} iter{iteration} kv{kv_head}",
                "cat": "Figure8 attention kernel",
                "ph": "X",
                "pid": pid,
                "tid": make_tid("Kernel Wall", worker, kv_head),
                "ts": start_us - base_ts,
                "dur": max(end_us - start_us, 0),
                "args": {
                    **lane_base_args,
                    "duration_source": "min/max of event-level DSP qtimer records on this lane",
                },
            }
        )
        add_control_gaps(events, pid, lane_rows, base_ts, host_by_iter)

    for row in rows:
        component = row["component"]
        unit = UNIT_BY_COMPONENT.get(component, "Scalar Unit")
        name_suffix = f"r{row['block_r']}"
        if row["block_c"] >= 0:
            name_suffix += f"_c{row['block_c']}"
        events.append(
            {
                "name": f"{component} {name_suffix}",
                "cat": unit,
                "ph": "X",
                "pid": pid,
                "tid": make_tid(unit, row["worker"], row["kv_head"]),
                "ts": row["t0_us"] - base_ts,
                "dur": event_duration_us(row),
                "cname": COLOR_BY_COMPONENT.get(component, "generic_work"),
                "args": {
                    **lane_args(row, host_by_iter),
                    "component": component,
                    "component_id": row["component_id"],
                    "unit": unit,
                    "block_r": row["block_r"],
                    "block_c": row["block_c"],
                    "absolute_t0_us": row["t0_us"],
                    "absolute_t1_us": row["t1_us"],
                    "duration_source": "DSP qtimer event start/end delta",
                },
            }
        )

    return {
        "displayTimeUnit": "us",
        "traceEvents": events,
        "metadata": {
            "format": "Chrome Trace Event JSON accepted by Perfetto UI",
            "source": "Figure 8 attention DSP qtimer event records",
            "hardware_trace": False,
            "timeline_source": "event-level software instrumentation",
            "limitations": [
                "This is not Qualcomm sysmon/ETM hardware-unit telemetry.",
                "Unit labels are mapped from measured code sections; Q/K/V load-pack and output store are not hardware counters.",
                "Scalar Unit slices are derived uninstrumented gaps, not hardware scalar utilization.",
                "The core_acc and o_scale sections are shown as mixed because this Figure 8 event stream does not split their HVX setup/state subwork from HMX dot work.",
            ],
        },
    }


def summarize_events(event_rows, event_count_rows):
    summary = {
        "timeline_source": "FIG8_ATTENTION_EVENT DSP qtimer start/end records",
        "hardware_trace": False,
        "modes": sorted({row["mode"] for row in event_rows}),
        "layouts": sorted({row.get("layout", "n/a") for row in event_rows}),
        "scna_widths": sorted({row.get("scna_width", 0) for row in event_rows}),
        "lut_exp_values": sorted({row.get("lut_exp", -1) for row in event_rows}),
        "by_qo_len": {},
    }
    for qo_len in sorted({row["qo_len"] for row in event_rows}):
        rows = [row for row in event_rows if row["qo_len"] == qo_len]
        count_rows = [row for row in event_count_rows if row["qo_len"] == qo_len]
        component_counts = Counter(row["component"] for row in rows)
        component_durations = defaultdict(int)
        for row in rows:
            component_durations[row["component"]] += event_duration_us(row)
        summary["by_qo_len"][str(qo_len)] = {
            "event_rows": len(rows),
            "measured_iterations": len({row["iteration"] for row in rows}),
            "event_count_rows": len(count_rows),
            "event_overflow_total": sum(row.get("overflow", 0) for row in count_rows),
            "event_count_min": min((row.get("events", 0) for row in count_rows), default=0),
            "event_count_max": max((row.get("events", 0) for row in count_rows), default=0),
            "component_event_counts": {component: component_counts.get(component, 0) for component in COMPONENTS},
            "component_event_duration_us": {
                component: component_durations.get(component, 0) for component in COMPONENTS
            },
        }
    return summary


def preserve_existing(out_dir, names):
    archive_dir = out_dir / "qprof_thread_only_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = out_dir / name
        dst = archive_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def write_notes(path, outputs, blocker_lines, prefix):
    lines = [
        "# Figure 8 Perfetto Trace Notes",
        "",
        f"The `{prefix}_q*.ntff` and `{prefix}_q*.perfetto.json` files in this directory are Chrome Trace Event JSON files that Perfetto UI can open directly.",
        "",
        "These traces are now generated from `FIG8_ATTENTION_EVENT` rows: every slice uses a DSP qtimer start timestamp and end timestamp captured around the corresponding attention-kernel section. This preserves cross-worker overlap and is not reconstructed from cumulative per-component totals.",
        "",
        "They are still software-observed traces, not Qualcomm sysmon/ETM hardware traces, because this production device does not expose the NSP/CDSP profiling capability needed for true hardware-unit telemetry.",
        "",
        "## Unit Mapping",
        "",
        "- `Memory/L2 Load-Pack`: `q_load`, `k_load`, `v_load`. These are software load/pack windows into VTCM and L2-prefetch-assisted memory accesses, not DMA counters.",
        "- `HMX Unit`: `qk_dot`.",
        "- `HVX Unit`: `safe_sm` and nested `scna_exp`; online rescale SCNA events are nested in `core_acc`.",
        "- `Mixed HVX/HMX Section`: `core_acc`, `o_scale`, because the Figure 8 component events wrap HVX state/scatter/setup work together with HMX P*V or scale-dot work.",
        "- `Store/Writeback`: `o_store`.",
        "- `Scalar Unit`: derived uninstrumented control gaps and instant notes only; not measured scalar hardware utilization",
        "",
        "## Hardware Trace Blockers Observed",
        "",
    ]
    lines.extend(f"- {line}" for line in blocker_lines)
    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- `{out}`" for out in outputs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--binary-sha256", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timer_rows, host_rows, event_count_rows, event_rows, provenance_rows = parse_logs(args.input_dir)
    provenance = dict(provenance_rows[-1]) if provenance_rows else {}
    if args.binary_sha256:
        provenance["binary_sha256"] = args.binary_sha256
    provenance.setdefault("isa", "v79")
    qo_lens = sorted({row["qo_len"] for row in event_rows})
    if not qo_lens:
        raise SystemExit("no measured FIG8_ATTENTION_EVENT rows found")
    modes = sorted({row["mode"] for row in event_rows})
    layouts = sorted({row.get("layout", "n/a") for row in event_rows})
    if len(modes) == 1 and modes[0] == "scna-fp16" and len(layouts) == 1:
        prefix = f"scna_{mode_to_slug(layouts[0])}"
    else:
        prefix = mode_to_slug(modes[0]) if len(modes) == 1 else "mixed"

    output_names = []
    for qo_len in qo_lens:
        output_names.extend([f"{prefix}_q{qo_len}.ntff", f"{prefix}_q{qo_len}.perfetto.json"])
    output_names.extend([f"{prefix}_all_q.ntff", f"{prefix}_all_q.perfetto.json", "event_trace_summary.json"])
    preserve_existing(args.out_dir, output_names)

    generated = []
    combined_events = []
    combined_metadata = []
    for index, qo_len in enumerate(qo_lens):
        trace = build_trace_for_qo(qo_len, event_rows, host_rows, 100 + index, provenance)
        for suffix in [".ntff", ".perfetto.json"]:
            name = f"{prefix}_q{qo_len}{suffix}"
            path = args.out_dir / name
            path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
            generated.append(name)
        combined_events.extend(trace["traceEvents"])
        combined_metadata.append(trace["metadata"])

    combined = {
        "displayTimeUnit": "us",
        "traceEvents": combined_events,
        "metadata": {
            "format": "Chrome Trace Event JSON accepted by Perfetto UI",
            "source": "Figure 8 attention DSP qtimer event records",
            "hardware_trace": False,
            "timeline_source": "event-level software instrumentation",
            "per_q_metadata": combined_metadata,
        },
    }
    for suffix in [".ntff", ".perfetto.json"]:
        name = f"{prefix}_all_q{suffix}"
        path = args.out_dir / name
        path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        generated.append(name)

    summary = summarize_events(event_rows, event_count_rows)
    summary["provenance"] = {key: value for key, value in provenance.items() if key != "raw_file"}
    summary_path = args.out_dir / "event_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated.append(summary_path.name)

    blockers = [
        "This run deliberately records event-level DSP qtimer intervals only; PMU, ETM, sysmon, and hardware utilization counters were not collected.",
        "Accordingly every trace declares `hardware_trace=false` and must not be interpreted as a hardware-counter trace.",
    ]
    write_notes(args.out_dir / "derived_trace_notes.md", generated, blockers, prefix)

    print(f"parsed_timer_rows={len(timer_rows)}")
    print(f"parsed_event_count_rows={len(event_count_rows)}")
    print(f"parsed_event_rows={len(event_rows)}")
    print(f"generated_traces={len(generated)}")
    for name in generated:
        path = args.out_dir / name
        print(f"{name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
