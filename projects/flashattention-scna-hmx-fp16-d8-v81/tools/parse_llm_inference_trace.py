#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


HOST_RE = re.compile(r"LLMTRACE_HOST_EVENT\s+(.*)")
DSP_RE = re.compile(r"LLMTRACE_DSP_EVENT\s+(.*)")
DSP_STAGE_RE = re.compile(r"LLMTRACE_DSP_STAGE_EVENT\s+(.*)")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")

TEXT_FIELDS = {
    "mode",
    "phase",
    "op",
    "category",
    "tensor",
    "weight",
    "timing_source",
    "event_type",
    "prev_category",
    "next_category",
    "stage",
    "unit",
    "stage_timing_source",
    "parent_trace_id",
}

DSP_PHASE_ORDER = ["validate_in", "compute", "validate_out"]
DISPLAY_COMPONENTS = [
    ("qkvo_matrix_compute_us", "QKVO"),
    ("ffn_compute_us", "FFN"),
    ("attention_compute_us", "Attention"),
    ("host_npu_comm_us", "Host/NPU comm"),
    ("rmsnorm_residual_host_us", "Host RMSNorm/residual"),
    ("rope_kv_cache_host_us", "RoPE/KV/mask"),
    ("swiglu_host_us", "SwiGLU"),
    ("logits_sample_host_us", "Logits/sample"),
    ("host_scheduler_misc_us", "Host misc"),
    ("other_npu_compute_us", "Other NPU"),
]
MAIN_COMPONENTS = [name for name, _ in DISPLAY_COMPONENTS]

HOST_GAP_COMPONENTS = {
    "rmsnorm_attn_host": "rmsnorm_residual_host_us",
    "residual_rmsnorm_ffn_host": "rmsnorm_residual_host_us",
    "residual_block_tail_host": "rmsnorm_residual_host_us",
    "rope_kv_cache_mask_host": "rope_kv_cache_host_us",
    "swiglu_host": "swiglu_host_us",
    "logits_sample_host": "logits_sample_host_us",
    "tokenizer_prompt_setup_host": "host_scheduler_misc_us",
    "qkv_projection_dispatch_host": "host_scheduler_misc_us",
    "attention_output_pack_host": "host_scheduler_misc_us",
    "ffn_gate_up_dispatch_host": "host_scheduler_misc_us",
    "host_scheduler_misc": "host_scheduler_misc_us",
}


def read_text_auto(path):
    data = path.read_bytes()
    # Android keeps server.log open while writing. If the file is truncated
    # under the live process, later writes can create a sparse NUL prefix.
    # Strip that prefix before deciding whether the payload is UTF-16.
    data = data.lstrip(b"\x00")
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:256]:
        return data.decode("utf-16-le", errors="replace")
    return data.decode("utf-8", errors="replace")


def parse_value(key, value):
    if key in TEXT_FIELDS:
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_key_values(text):
    return {key: parse_value(key, value) for key, value in KEY_VALUE_RE.findall(text)}


def parse_logs(paths):
    host_rows = []
    dsp_rows = []
    dsp_stage_rows = []
    for path in paths:
        for line_no, line in enumerate(read_text_auto(path).splitlines(), start=1):
            host_match = HOST_RE.search(line)
            if host_match:
                row = parse_key_values(host_match.group(1))
                row["source_file"] = str(path)
                row["line_no"] = line_no
                host_rows.append(row)
                continue
            dsp_match = DSP_RE.search(line)
            if dsp_match:
                row = parse_key_values(dsp_match.group(1))
                row["source_file"] = str(path)
                row["line_no"] = line_no
                dsp_rows.append(row)
                continue
            stage_match = DSP_STAGE_RE.search(line)
            if stage_match:
                row = parse_key_values(stage_match.group(1))
                row["source_file"] = str(path)
                row["line_no"] = line_no
                dsp_stage_rows.append(row)
    return host_rows, dsp_rows, dsp_stage_rows


def component_for_category(category):
    if category in {"q_matrix", "k_matrix", "v_matrix", "o_matrix"}:
        return "qkvo_matrix_compute_us"
    if category in {"ffn_gate", "ffn_up", "ffn_down"}:
        return "ffn_compute_us"
    if category == "attention":
        return "attention_compute_us"
    if category in HOST_GAP_COMPONENTS:
        return HOST_GAP_COMPONENTS[category]
    return "other_npu_compute_us"


def trace_sort_key(row):
    try:
        trace_id = int(row.get("trace_id", 0))
    except (TypeError, ValueError):
        trace_id = 0
    return (
        str(row.get("source_file", "")),
        int(row.get("t0_us", 0) or 0),
        int(row.get("line_no", 0) or 0),
        trace_id,
        str(row.get("trace_id", "")),
    )


def infer_host_gap_category(prev, next_row, gap_us):
    prev_cat = prev.get("category", "")
    next_cat = next_row.get("category", "")
    next_phase = next_row.get("phase", "")

    if prev_cat == "ffn_down" and next_cat == "q_matrix" and next_phase == "decode" and gap_us >= 1000:
        return "logits_sample_host"
    if next_cat == "q_matrix":
        return "residual_block_tail_host" if prev_cat == "ffn_down" else "rmsnorm_attn_host"
    if prev_cat in {"q_matrix", "k_matrix"} and next_cat in {"k_matrix", "v_matrix"}:
        return "qkv_projection_dispatch_host"
    if prev_cat == "v_matrix" and next_cat == "attention":
        return "rope_kv_cache_mask_host"
    if prev_cat == "attention" and next_cat == "o_matrix":
        return "attention_output_pack_host"
    if prev_cat == "o_matrix" and next_cat == "ffn_gate":
        return "residual_rmsnorm_ffn_host"
    if prev_cat == "ffn_gate" and next_cat == "ffn_up":
        return "ffn_gate_up_dispatch_host"
    if prev_cat == "ffn_up" and next_cat == "ffn_down":
        return "swiglu_host"
    if prev_cat == "ffn_down":
        return "residual_block_tail_host"
    return "host_scheduler_misc"


def add_host_gap_rows(rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row.get("source_file", "")].append(row)

    gap_rows = []
    for _, items in by_source.items():
        items = sorted(items, key=trace_sort_key)
        for prev, next_row in zip(items, items[1:]):
            if prev.get("mode") != next_row.get("mode"):
                continue
            gap_us = int(next_row.get("t0_us", 0) or 0) - int(prev.get("t1_us", 0) or 0)
            if gap_us <= 0:
                continue
            category = infer_host_gap_category(prev, next_row, gap_us)
            phase = next_row.get("phase") if prev.get("phase") != next_row.get("phase") else prev.get("phase")
            gap_rows.append(
                {
                    "trace_id": f"gap_{prev.get('trace_id')}_{next_row.get('trace_id')}",
                    "mode": next_row.get("mode", "unknown"),
                    "phase": phase or "unknown",
                    "op": "host_gap",
                    "op_index": "",
                    "category": category,
                    "component": component_for_category(category),
                    "tensor": "",
                    "weight": "",
                    "m": 0,
                    "k": 0,
                    "n": 0,
                    "qo_len": 0,
                    "kv_len": 0,
                    "n_heads": 0,
                    "n_kv_heads": 0,
                    "head_dim": 0,
                    "t0_us": int(prev.get("t1_us", 0) or 0),
                    "t1_us": int(next_row.get("t0_us", 0) or 0),
                    "dur_us": gap_us,
                    "prepare_us": 0,
                    "build_us": 0,
                    "wait_us": 0,
                    "unmap_us": 0,
                    "dsp_validate_in_us": 0,
                    "dsp_compute_us": 0,
                    "dsp_validate_out_us": 0,
                    "dsp_total_us": 0,
                    "host_npu_comm_us": 0,
                    "input_bytes": 0,
                    "output_bytes": 0,
                    "matched_dsp_events": 0,
                    "timing_source": "host_gap_inferred_from_host_timestamps",
                    "event_type": "host_gap",
                    "prev_category": prev.get("category", ""),
                    "next_category": next_row.get("category", ""),
                    "source_file": next_row.get("source_file", ""),
                    "line_no": next_row.get("line_no", 0),
                }
            )
    return sorted(rows + gap_rows, key=trace_sort_key)


def combine_rows(host_rows, dsp_rows):
    dsp_by_id = defaultdict(list)
    for row in dsp_rows:
        dsp_by_id[row["trace_id"]].append(row)

    combined = []
    for host in host_rows:
        trace_id = host["trace_id"]
        phases = dsp_by_id.get(trace_id, [])
        phase_dur = defaultdict(int)
        phase_input = defaultdict(int)
        phase_output = defaultdict(int)
        for phase in phases:
            phase_dur[phase["phase"]] += max(int(phase.get("dur_us", 0)), 0)
            phase_input[phase["phase"]] += int(phase.get("input_bytes", 0))
            phase_output[phase["phase"]] += int(phase.get("output_bytes", 0))

        host_total_us = max(int(host.get("dur_us", 0)), 0)
        host_prepare_us = max(int(host.get("prepare_us", 0)), 0)
        host_build_us = max(int(host.get("build_us", 0)), 0)
        host_wait_us = max(int(host.get("wait_us", 0)), 0)
        host_unmap_us = max(int(host.get("unmap_us", 0)), 0)

        if phases:
            dsp_total_us = phase_dur.get("total")
            if not dsp_total_us:
                dsp_total_us = sum(phase_dur[p] for p in DSP_PHASE_ORDER)
            dsp_compute_us = phase_dur.get("compute", 0)
            host_comm_us = max(host_total_us - dsp_total_us, 0)
            timing_source = "matched_dsp_qtimer"
        else:
            # Some production Android builds do not route DSP FARF messages back
            # to server.log/logcat. In that case, keep the event-level timeline
            # useful by treating host wait as NPU-side elapsed time and the
            # host-visible setup/teardown slices as communication/scheduling.
            dsp_total_us = host_wait_us
            dsp_compute_us = host_wait_us
            host_comm_us = host_prepare_us + host_build_us + host_unmap_us
            timing_source = "host_wait_fallback"

        row = {
            **host,
            "dsp_validate_in_us": phase_dur.get("validate_in", 0),
            "dsp_compute_us": dsp_compute_us,
            "dsp_validate_out_us": phase_dur.get("validate_out", 0),
            "dsp_total_us": dsp_total_us,
            "host_npu_comm_us": host_comm_us,
            "component": component_for_category(host.get("category", "other")),
            "input_bytes": max(phase_input.values(), default=0),
            "output_bytes": max(phase_output.values(), default=0),
            "matched_dsp_events": len(phases),
            "timing_source": timing_source,
            "event_type": "npu_op",
            "prev_category": "",
            "next_category": "",
        }
        combined.append(row)
    return add_host_gap_rows(combined)


def summarize(rows):
    summary = {
        "timeline_source": "event-level host timestamps; host gaps are inferred from adjacent NPU op boundaries; matched DSP qtimer durations are used when available, otherwise host wait fallback",
        "hardware_trace": False,
        "communication_model": "with DSP events: host observed op time - matched DSP total time; without DSP events: host prepare + request build + unmap",
        "host_gap_model": "gaps between adjacent LLMTRACE_HOST_EVENT rows are measured host wall-clock time and classified heuristically by neighboring op categories",
        "component_denominator": MAIN_COMPONENTS,
        "by_mode_phase": {},
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("mode", "unknown"), row.get("phase", "unknown"))].append(row)

    for (mode, phase), items in sorted(grouped.items()):
        totals = defaultdict(int)
        npu_items = [row for row in items if row.get("event_type") != "host_gap"]
        gap_items = [row for row in items if row.get("event_type") == "host_gap"]
        totals["op_count"] = len(npu_items)
        totals["host_gap_count"] = len(gap_items)
        totals["matched_op_count"] = sum(1 for row in npu_items if row.get("matched_dsp_events", 0) > 0)
        totals["host_wait_fallback_op_count"] = sum(1 for row in npu_items if row.get("timing_source") == "host_wait_fallback")
        totals["host_total_us"] = sum(int(row.get("dur_us", 0)) for row in npu_items)
        totals["host_prepare_us"] = sum(int(row.get("prepare_us", 0)) for row in npu_items)
        totals["host_build_us"] = sum(int(row.get("build_us", 0)) for row in npu_items)
        totals["host_wait_us"] = sum(int(row.get("wait_us", 0)) for row in npu_items)
        totals["host_unmap_us"] = sum(int(row.get("unmap_us", 0)) for row in npu_items)
        totals["host_npu_comm_us"] = sum(int(row.get("host_npu_comm_us", 0)) for row in npu_items)
        totals["host_cpu_inferred_gap_us"] = sum(int(row.get("dur_us", 0)) for row in gap_items)
        if items:
            totals["phase_wall_span_us"] = max(int(row.get("t1_us", row.get("t0_us", 0))) for row in items) - min(
                int(row.get("t0_us", 0)) for row in items
            )
        for row in items:
            if row.get("event_type") == "host_gap":
                totals[row["component"]] += int(row.get("dur_us", 0))
            else:
                totals[row["component"]] += int(row.get("dsp_compute_us", 0))

        denom = sum(totals[name] for name in MAIN_COMPONENTS)
        percentages = {
            name.replace("_us", "_pct"): (totals[name] / denom * 100.0 if denom else 0.0)
            for name in MAIN_COMPONENTS
        }
        summary["by_mode_phase"][f"{mode}/{phase}"] = {
            **{key: totals[key] for key in sorted(totals)},
            "profiled_total_us": denom,
            **percentages,
        }
    return summary


def write_csv(path, rows):
    fieldnames = [
        "trace_id",
        "mode",
        "phase",
        "op",
        "op_index",
        "category",
        "component",
        "tensor",
        "weight",
        "m",
        "k",
        "n",
        "qo_len",
        "kv_len",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "t0_us",
        "t1_us",
        "dur_us",
        "prepare_us",
        "build_us",
        "wait_us",
        "unmap_us",
        "dsp_validate_in_us",
        "dsp_compute_us",
        "dsp_validate_out_us",
        "dsp_total_us",
        "host_npu_comm_us",
        "input_bytes",
        "output_bytes",
        "matched_dsp_events",
        "timing_source",
        "event_type",
        "prev_category",
        "next_category",
        "source_file",
        "line_no",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stage_unit_id(unit):
    return {
        "dma": 1,
        "hvx": 2,
        "hmx": 3,
        "store": 4,
        "mem": 5,
        "memory": 5,
        "scalar": 6,
    }.get(str(unit or "").lower(), 0)


def normalized_stage_unit(raw):
    unit = str(raw.get("unit", "other") or "other").lower()
    if (raw.get("op") or "") != "flash_attn":
        return unit

    stage = str(raw.get("stage", "") or "")
    try:
        worker = int(raw.get("worker", -1))
    except (TypeError, ValueError):
        worker = -1

    if stage in {"flash_q_load", "flash_k_load", "flash_v_load"}:
        return "mem"
    if stage in {"flash_o_store", "output_store"}:
        return "store"

    # Older profiling-only logs temporarily routed GGML_OP_FLASH_ATTN_EXT to
    # naive_flash_attn_profiled and emitted every compute stage as unit=hvx
    # with worker=-1. Keep those old logs honest instead of drawing them as HMX.
    if worker < 0 and stage in {"flash_qk_dot", "flash_safe_sm", "flash_core_acc", "flash_o_scale"}:
        return "scalar"

    if stage in {"flash_qk_dot", "flash_o_scale"}:
        return "hmx"
    if stage == "flash_safe_sm":
        return "hvx"
    return unit


def build_stage_rows(rows, dsp_stage_rows):
    host_by_trace = {
        str(row.get("trace_id")): row
        for row in rows
        if row.get("event_type") == "npu_op" and row.get("trace_id") is not None
    }
    raw_by_trace = defaultdict(list)
    for row in dsp_stage_rows:
        raw_by_trace[str(row.get("trace_id", ""))].append(row)
    raw_min_by_trace = {
        trace_id: min(int(row.get("raw_t0_us", row.get("t0_us", 0)) or 0) for row in items)
        for trace_id, items in raw_by_trace.items()
        if items
    }

    stage_rows = []
    for raw in dsp_stage_rows:
        trace_id = str(raw.get("trace_id", ""))
        host = host_by_trace.get(trace_id, {})
        raw_t0 = int(raw.get("raw_t0_us", raw.get("t0_us", 0)) or 0)
        raw_t1 = int(raw.get("raw_t1_us", raw.get("t1_us", 0)) or 0)
        raw_min = raw_min_by_trace.get(trace_id, raw_t0)
        host_wait_start = (
            int(host.get("t0_us", 0) or 0)
            + int(host.get("prepare_us", 0) or 0)
            + int(host.get("build_us", 0) or 0)
        )
        if host:
            aligned_t0 = host_wait_start + (raw_t0 - raw_min)
            aligned_t1 = host_wait_start + (raw_t1 - raw_min)
        else:
            aligned_t0 = raw_t0
            aligned_t1 = raw_t1
        dur = max(int(raw.get("dur_us", aligned_t1 - aligned_t0) or 0), 0)
        if aligned_t1 < aligned_t0:
            aligned_t1 = aligned_t0 + dur

        category = raw.get("category") or host.get("category", "")
        stage = raw.get("stage", "unknown")
        unit = normalized_stage_unit(raw)
        stage_rows.append(
            {
                "parent_trace_id": trace_id,
                "mode": raw.get("mode") or host.get("mode", "unknown"),
                "phase": raw.get("phase") or host.get("phase", "unknown"),
                "op": raw.get("op") or host.get("op", ""),
                "op_index": raw.get("op_index", host.get("op_index", "")),
                "category": category,
                "component": component_for_category(category),
                "stage": stage,
                "stage_id": raw.get("stage_id", ""),
                "unit": unit,
                "unit_id": stage_unit_id(unit),
                "worker": raw.get("worker", -1),
                "t0_us": aligned_t0,
                "t1_us": aligned_t1,
                "dur_us": max(aligned_t1 - aligned_t0, dur),
                "raw_t0_us": raw_t0,
                "raw_t1_us": raw_t1,
                "m": raw.get("m", host.get("m", 0)),
                "k": raw.get("k", host.get("k", 0)),
                "n": raw.get("n", host.get("n", 0)),
                "qo_len": raw.get("qo_len", host.get("qo_len", 0)),
                "kv_len": raw.get("kv_len", host.get("kv_len", 0)),
                "n_heads": raw.get("n_heads", host.get("n_heads", 0)),
                "n_kv_heads": raw.get("n_kv_heads", host.get("n_kv_heads", 0)),
                "head_dim": raw.get("head_dim", host.get("head_dim", 0)),
                "mr": raw.get("mr", ""),
                "nc": raw.get("nc", ""),
                "kk": raw.get("kk", ""),
                "chunk_m": raw.get("chunk_m", ""),
                "chunk_n": raw.get("chunk_n", ""),
                "chunk_k": raw.get("chunk_k", ""),
                "bytes": raw.get("bytes", 0),
                "stage_timing_source": "dsp_qtimer_profile_buffer",
                "source_file": raw.get("source_file", ""),
                "line_no": raw.get("line_no", 0),
            }
            )
    return sorted(stage_rows, key=lambda r: (str(r.get("source_file", "")), int(r.get("t0_us", 0)), int(r.get("line_no", 0))))


def write_stage_csv(path, rows):
    fieldnames = [
        "parent_trace_id",
        "mode",
        "phase",
        "op",
        "op_index",
        "category",
        "component",
        "stage",
        "stage_id",
        "unit",
        "unit_id",
        "worker",
        "t0_us",
        "t1_us",
        "dur_us",
        "raw_t0_us",
        "raw_t1_us",
        "m",
        "k",
        "n",
        "qo_len",
        "kv_len",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "mr",
        "nc",
        "kk",
        "chunk_m",
        "chunk_n",
        "chunk_k",
        "bytes",
        "stage_timing_source",
        "source_file",
        "line_no",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_metadata(events, pid, name, tids):
    events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": name}})
    for tid, label in tids.items():
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": label}})


def trace_event(name, cat, pid, tid, ts, dur, args=None):
    return {
        "name": name,
        "cat": cat,
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": int(ts),
        "dur": max(int(dur), 1),
        "args": args or {},
    }


def stage_tid(unit):
    unit = str(unit or "").lower()
    return {
        "dma": 230,
        "mem": 235,
        "memory": 235,
        "hvx": 240,
        "hmx": 250,
        "store": 260,
        "scalar": 270,
    }.get(unit, 270)


def build_trace(rows, stage_rows, mode_filter=None):
    selected = [row for row in rows if mode_filter is None or row.get("mode") == mode_filter]
    if not selected:
        return {"displayTimeUnit": "us", "traceEvents": [], "metadata": {}}
    selected_stages = [row for row in stage_rows if mode_filter is None or row.get("mode") == mode_filter]
    base_ts = min(int(row["t0_us"]) for row in selected)
    events = []
    tids = {
        100: "Host RPC total",
        110: "Host prepare/build/wait/unmap",
        120: "Host CPU inferred gaps",
        200: "DSP validate_in",
        210: "DSP compute",
        220: "DSP validate_out",
    }
    if selected_stages:
        lane_labels = {
            "dma": (230, "DSP qtimer DMA (real DMA only)"),
            "mem": (235, "DSP qtimer memory/L2 load-pack"),
            "memory": (235, "DSP qtimer memory/L2 load-pack"),
            "hvx": (240, "DSP qtimer HVX"),
            "hmx": (250, "DSP qtimer HMX"),
            "store": (260, "DSP qtimer store"),
            "scalar": (270, "DSP qtimer scalar/other"),
        }
        for unit in sorted({str(row.get("unit", "")).lower() for row in selected_stages}):
            lane = lane_labels.get(unit)
            if lane:
                tids[lane[0]] = lane[1]
    add_metadata(events, 1, f"LLM inference {mode_filter or 'all_modes'}", tids)

    for row in sorted(selected, key=trace_sort_key):
        start = int(row["t0_us"]) - base_ts
        args = {
            key: row.get(key)
            for key in [
                "trace_id",
                "mode",
                "phase",
                "op",
                "category",
                "component",
                "tensor",
                "weight",
                "m",
                "k",
                "n",
                "timing_source",
                "prev_category",
                "next_category",
            ]
        }
        if row.get("event_type") == "host_gap":
            events.append(trace_event(row["category"], "host_cpu_gap", 1, 120, start, row["dur_us"], args))
            continue
        events.append(trace_event(row["category"], "host_total", 1, 100, start, row["dur_us"], args))

        cursor = start
        for name, dur_key in [
            ("host_prepare_rpcmem", "prepare_us"),
            ("host_build_request", "build_us"),
            ("host_wait_for_dsp", "wait_us"),
        ]:
            dur = int(row.get(dur_key, 0))
            if dur > 0:
                events.append(trace_event(name, "host_npu_comm", 1, 110, cursor, dur, args))
            cursor += dur
        unmap_us = int(row.get("unmap_us", 0))
        if unmap_us > 0:
            events.append(trace_event("host_unmap_rpcmem", "host_npu_comm", 1, 110, start + int(row["dur_us"]) - unmap_us, unmap_us, args))

        dsp_cursor = start + int(row.get("prepare_us", 0)) + int(row.get("build_us", 0))
        for dsp_name, dur_key, tid in [
            ("dsp_validate_in", "dsp_validate_in_us", 200),
            ("dsp_compute", "dsp_compute_us", 210),
            ("dsp_validate_out", "dsp_validate_out_us", 220),
        ]:
            dur = int(row.get(dur_key, 0))
            if dur > 0:
                events.append(trace_event(dsp_name, row["component"], 1, tid, dsp_cursor, dur, args))
            dsp_cursor += dur

    for row in sorted(selected_stages, key=lambda r: (int(r["t0_us"]), str(r.get("parent_trace_id", "")), r.get("stage", ""))):
        start = int(row["t0_us"]) - base_ts
        args = {
            key: row.get(key)
            for key in [
                "parent_trace_id",
                "mode",
                "phase",
                "op",
                "category",
                "component",
                "m",
                "k",
                "n",
                "qo_len",
                "kv_len",
                "stage",
                "stage_id",
                "unit",
                "unit_id",
                "worker",
                "raw_t0_us",
                "raw_t1_us",
                "bytes",
                "chunk_m",
                "chunk_n",
                "chunk_k",
                "stage_timing_source",
            ]
        }
        events.append(trace_event(row["stage"], "dsp_qtimer_stage", 1, stage_tid(row.get("unit")), start, row["dur_us"], args))

    return {
        "displayTimeUnit": "us",
        "traceEvents": events,
        "metadata": {
            "format": "Chrome Trace Event JSON accepted by Perfetto UI",
            "source": "LLMTRACE_HOST_EVENT, LLMTRACE_DSP_EVENT, and LLMTRACE_DSP_STAGE_EVENT software instrumentation",
            "hardware_trace": False,
            "timeline_source": "host wall clock for RPC boundaries and inferred host gaps; DSP stage lanes are aligned qtimer events copied from the profiling-only DSP profile buffer",
            "stage_timing_source": "LLMTRACE_DSP_STAGE_EVENT profile-buffer rows only; no estimated stage rows are generated",
        },
    }


def write_markdown(path, summary, stage_row_count=0):
    lines = [
        "# LLM Inference Profiling Breakdown",
        "",
        "Percentages use NPU compute/communication slices plus inferred host CPU gaps between adjacent NPU ops.",
        "",
        "| mode/phase | QKVO | FFN | Attention | Host/NPU comm | Host RMSNorm/residual | RoPE/KV/mask | SwiGLU | Logits/sample | Host misc | Other NPU | total us | wall span us | NPU ops | host gaps | matched ops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, data in summary["by_mode_phase"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    f"{data['qkvo_matrix_compute_pct']:.2f}%",
                    f"{data['ffn_compute_pct']:.2f}%",
                    f"{data['attention_compute_pct']:.2f}%",
                    f"{data['host_npu_comm_pct']:.2f}%",
                    f"{data['rmsnorm_residual_host_pct']:.2f}%",
                    f"{data['rope_kv_cache_host_pct']:.2f}%",
                    f"{data['swiglu_host_pct']:.2f}%",
                    f"{data['logits_sample_host_pct']:.2f}%",
                    f"{data['host_scheduler_misc_pct']:.2f}%",
                    f"{data['other_npu_compute_pct']:.2f}%",
                    str(data["profiled_total_us"]),
                    str(data.get("phase_wall_span_us", 0)),
                    str(data["op_count"]),
                    str(data.get("host_gap_count", 0)),
                    str(data["matched_op_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `QKVO` includes `attn_q`, `attn_k`, `attn_v`, and `attn_output` matmul DSP compute.",
            "- `FFN` includes `ffn_gate`, `ffn_up`, and `ffn_down` matmul DSP compute.",
            "- `Attention` is the fused flash-attention op DSP compute.",
            "- `Host/NPU comm` is estimated as host-observed op duration minus matched DSP total duration, plus request/mapping overhead visible to the host.",
            "- `Host RMSNorm/residual`, `RoPE/KV/mask`, `SwiGLU`, `Logits/sample`, and `Host misc` are inferred from measured host wall-clock gaps between adjacent NPU op boundaries.",
            "- `llm_trace_stage_events.csv` and the Perfetto DSP qtimer lanes are generated only from `LLMTRACE_DSP_STAGE_EVENT` rows copied out of the DSP profiling buffer.",
            "- FlashAttention stage units are normalized to the actual kernel path: `mem` for Q/K/V software load-pack, `hmx` for QK dot / P*V / O scale windows, `hvx` for safe-softmax and state-vector work, and `store` for output writeback. `dma` is used only by kernels that really issue DMA requests.",
            f"- Current parsed DSP qtimer stage rows: `{stage_row_count}`. If this is `0`, the log was captured without `LLAMA_NPU_DETAILED_TRACE=1` or predates the DSP profile-buffer instrumentation.",
            "- If DSP FARF events are not available, `compute` uses host wait time as NPU-side elapsed time and `Host/NPU comm` uses host prepare/build/unmap overhead.",
            "- The trace is event-level software instrumentation, not Qualcomm hardware-counter telemetry. `dma` rows record DSP-side issue/wait windows around DMA requests, not a hardware DMA counter stream.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    host_rows, dsp_rows, dsp_stage_rows = parse_logs(args.log)
    rows = combine_rows(host_rows, dsp_rows)
    stage_rows = build_stage_rows(rows, dsp_stage_rows)
    summary = summarize(rows)

    write_csv(args.out_dir / "llm_trace_events.csv", rows)
    write_stage_csv(args.out_dir / "llm_trace_stage_events.csv", stage_rows)
    (args.out_dir / "llm_trace_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_dir / "llm_trace_breakdown.md", summary, len(stage_rows))

    modes = sorted({row.get("mode", "unknown") for row in rows if row.get("event_type") != "host_gap"})
    traces = [("llm_inference_all_modes", build_trace(rows, stage_rows))]
    traces.extend((f"llm_inference_{mode}", build_trace(rows, stage_rows, mode)) for mode in modes)
    for name, trace in traces:
        for suffix in [".perfetto.json", ".ntff"]:
            (trace_dir / f"{name}{suffix}").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    print(f"host_rows={len(host_rows)}")
    print(f"dsp_rows={len(dsp_rows)}")
    print(f"dsp_stage_rows={len(dsp_stage_rows)}")
    print(f"combined_rows={len(rows)}")
    print(f"host_gap_rows={sum(1 for row in rows if row.get('event_type') == 'host_gap')}")
    print(f"stage_rows={len(stage_rows)}")
    print(f"modes={','.join(modes)}")


if __name__ == "__main__":
    main()
