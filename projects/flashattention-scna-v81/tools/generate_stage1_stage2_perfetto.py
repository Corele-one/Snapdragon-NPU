#!/usr/bin/env python3
"""Generate a Perfetto trace comparing the stage-one and stage-two FP16 d8 paths.

The source logs contain DSP qtimer event intervals.  Because stages were separate
device sessions, this script selects the measured iteration with total DSP time
closest to each stage/Qo median and re-bases that iteration to t=0.  The resulting
side-by-side view is a representative comparison, not a claim that the two runs
were concurrent.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


QO_LENS = (4, 8, 16, 32)
STAGES = (
    (
        "stage1_naive",
        "Stage 1 — naive FP16 direct d8",
        "sm8750p-20260731-2300",
        "scna-fp16_d8_q{qo_len}.log",
        "rail_response",
        True,
    ),
    (
        "stage2_rewrite",
        "Stage 2 — rewritten FP16 direct d8",
        "sm8750p-optimized-final-20260731-2355",
        "scna-fp16_d8_q{qo_len}.log",
        "good",
        True,
    ),
    (
        "stage2_baseline",
        "Stage 2 batch — native HVX exp2 baseline",
        "sm8750p-optimized-final-20260731-2355",
        "baseline_d16_q{qo_len}.log",
        "thread_state_running",
        False,
    ),
)

EVENT_RE = re.compile(r"FIG8_ATTENTION_EVENT\s+(.*)")
TIMER_RE = re.compile(r"FIG8_ATTENTION_TIMERS\s+(.*)")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")

COMPONENT_COLOR = {
    "q_load": "good",
    "k_load": "rail_response",
    "v_load": "rail_animation",
    "qk_dot": "thread_state_running",
    "safe_sm": "rail_load",
    "core_acc": "thread_state_iowait",
    "o_scale": "thread_state_uninterruptible",
    "o_store": "cq_build_running",
}


def parse_kv(text):
    values = {}
    for key, value in KEY_VALUE_RE.findall(text):
        if key in {"mode", "phase", "component"}:
            values[key] = value
        else:
            values[key] = int(value)
    return values


def parse_log(path):
    timers = []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVENT_RE.search(line)
        if match:
            row = parse_kv(match.group(1))
            if row.get("phase") == "measure":
                events.append(row)
            continue
        match = TIMER_RE.search(line)
        if match:
            row = parse_kv(match.group(1))
            if row.get("phase") == "measure":
                timers.append(row)
    if not timers or not events:
        raise ValueError(f"{path}: no measured timer/event rows")
    return timers, events


def select_representative(timers, events):
    by_iteration = defaultdict(list)
    for row in timers:
        by_iteration[row["iteration"]].append(row)
    totals = {
        iteration: sum(row["profiled_total"] for row in rows)
        for iteration, rows in by_iteration.items()
    }
    median_total = statistics.median(totals.values())
    chosen_iteration = min(totals, key=lambda iteration: (abs(totals[iteration] - median_total), iteration))
    chosen_timers = by_iteration[chosen_iteration]
    chosen_events = [row for row in events if row["iteration"] == chosen_iteration]
    if not chosen_events:
        raise ValueError(f"iteration {chosen_iteration}: timer rows have no qtimer events")
    return {
        "iteration": chosen_iteration,
        "timers": chosen_timers,
        "events": chosen_events,
        "total_median_us": median_total,
        "total_selected_us": totals[chosen_iteration],
        "scna_median_us": statistics.median(
            sum(row["scna_exp"] for row in rows) for rows in by_iteration.values()
        ),
        "scna_selected_us": sum(row["scna_exp"] for row in chosen_timers),
    }


def metadata(events, pid, tid, name, has_scna_track):
    events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": name}})
    events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": name}})
    if has_scna_track:
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid + 1,
                "args": {"name": name.replace("Timeline", "SCNA evaluator (aggregate timer)")},
            }
        )


def add_stage(events, stage, qo_len, sample, pid, timeline_tid, scna_tid, color, has_scna_track):
    timeline = sorted(sample["events"], key=lambda row: (row["t0_us"], row["t1_us"], row["kv_head"]))
    base_ts = min(row["t0_us"] for row in timeline)
    end_ts = max(row["t1_us"] for row in timeline)
    common_args = {
        "stage": stage,
        "qo_len": qo_len,
        "selected_iteration": sample["iteration"],
        "selection": "measured iteration whose summed profiled_total is closest to the median",
        "dsp_total_median_us": sample["total_median_us"],
        "dsp_total_selected_us": sample["total_selected_us"],
        "scna_median_us": sample["scna_median_us"],
        "scna_selected_us": sample["scna_selected_us"],
        "time_origin": "each stage/Qo session independently rebased to zero for side-by-side visual comparison",
    }
    events.append(
        {
            "name": f"DSP attention kernel — Qo{qo_len}",
            "cat": "Kernel wall",
            "ph": "X",
            "pid": pid,
            "tid": timeline_tid,
            "ts": 0,
            "dur": max(end_ts - base_ts, 1),
            "cname": color,
            "args": {**common_args, "duration_source": "min/max DSP qtimer event timestamps"},
        }
    )
    for row in timeline:
        events.append(
            {
                "name": row["component"],
                "cat": "DSP qtimer component",
                "ph": "X",
                "pid": pid,
                "tid": timeline_tid,
                "ts": row["t0_us"] - base_ts,
                "dur": max(row["t1_us"] - row["t0_us"], 1),
                "cname": COMPONENT_COLOR.get(row["component"], "generic_work"),
                "args": {
                    **common_args,
                    "kv_head": row["kv_head"],
                    "worker": row["worker"],
                    "block_r": row["block_r"],
                    "block_c": row["block_c"],
                    "duration_source": "DSP qtimer start/end interval",
                },
            }
        )

    if has_scna_track:
        safe_start_by_head = {}
        for row in timeline:
            if row["component"] == "safe_sm":
                safe_start_by_head.setdefault(row["kv_head"], row["t0_us"] - base_ts)
        for row in sorted(sample["timers"], key=lambda row: row["kv_head"]):
            events.append(
                {
                    "name": f"SCNA exp evaluator — kv_head {row['kv_head']}",
                    "cat": "SCNA aggregate timer",
                    "ph": "X",
                    "pid": pid,
                    "tid": scna_tid,
                    "ts": safe_start_by_head.get(row["kv_head"], 0),
                    "dur": max(row["scna_exp"], 1),
                    "cname": color,
                    "args": {
                        **common_args,
                        "kv_head": row["kv_head"],
                        "duration_source": "scna_exp aggregate timer; slice is anchored at the first safe_sm event, so its placement is illustrative rather than event-level",
                    },
                }
            )


def build_trace(results):
    events = []
    pid = 42
    for qo_index, qo_len in enumerate(QO_LENS):
        for stage_index, (slug, title, _, _, color, has_scna_track) in enumerate(STAGES):
            timeline_tid = 100 + qo_index * 10 + stage_index * 2
            metadata(events, pid, timeline_tid, f"Timeline | Qo{qo_len} | {title}", has_scna_track)
            add_stage(
                events,
                title,
                qo_len,
                results[(slug, qo_len)],
                pid,
                timeline_tid,
                timeline_tid + 1,
                color,
                has_scna_track,
            )
    return {
        "displayTimeUnit": "us",
        "traceEvents": events,
        "metadata": {
            "format": "Chrome Trace Event JSON accepted by Perfetto UI",
            "title": "SCNA FP16 d8 — Stage 1 vs Stage 2",
            "source": "Measured FIG8_ATTENTION_EVENT DSP qtimer intervals and FIG8_ATTENTION_TIMERS scna_exp aggregates",
            "scope": "FP16 direct d8 historical implementation only; stage-one/two full Attention results predate later correctness fixes",
            "comparison_method": "one median-representative measured iteration per stage and Qo, independently rebased to zero",
            "limitations": [
                "The two stage sessions were not concurrent; aligned timestamps are for visual comparison only.",
                "SCNA evaluator slices use measured scna_exp duration but are anchored to the first safe_sm event because no sub-SCNA qtimer intervals were recorded.",
                "These are software-observed DSP qtimer records, not hardware-unit telemetry.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for slug, _, directory, log_template, _, _ in STAGES:
        for qo_len in QO_LENS:
            path = args.results_root / directory / log_template.format(qo_len=qo_len)
            timers, events = parse_log(path)
            results[(slug, qo_len)] = select_representative(timers, events)

    trace = build_trace(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    print(f"trace={args.out}")
    print(f"events={len(trace['traceEvents'])}")
    for qo_len in QO_LENS:
        before = results[("stage1_naive", qo_len)]
        after = results[("stage2_rewrite", qo_len)]
        print(
            f"Qo{qo_len}: SCNA median {before['scna_median_us']:.1f} -> {after['scna_median_us']:.1f} us "
            f"({before['scna_median_us'] / after['scna_median_us']:.2f}x)"
        )


if __name__ == "__main__":
    main()
