#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
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
]

MODE_DIRS = {
    "baseline": "baseline",
    "lut-exp": "lut_exp",
}

TIMER_RE = re.compile(r"FIG8_ATTENTION_TIMERS\s+(.*)")
HOST_RE = re.compile(r"FIG8_ATTENTION_HOST_TIMING\s+(.*)")
EVENT_COUNT_RE = re.compile(r"FIG8_ATTENTION_EVENT_COUNT\s+(.*)")
CONFIG_RE = re.compile(r"FIG8_ATTENTION_CONFIG\s+(.*)")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


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
        if key in {"mode", "phase"}:
            out[key] = value
        else:
            out[key] = int(value)
    return out


def median(values):
    return statistics.median(values) if values else 0.0


def mean(values):
    return statistics.fmean(values) if values else 0.0


def pct_change(before, after):
    return ((after - before) / before * 100.0) if before else 0.0


def parse_logs(root_dir):
    timer_rows = []
    host_rows = []
    event_count_rows = []
    config_rows = []

    for mode, dirname in MODE_DIRS.items():
        mode_dir = root_dir / dirname
        if not mode_dir.exists():
            continue
        for path in sorted(mode_dir.glob("raw_*.log")):
            text = read_text_auto(path)
            for line in text.splitlines():
                config_match = CONFIG_RE.search(line)
                if config_match:
                    row = parse_key_values(config_match.group(1))
                    row["source"] = str(path.relative_to(root_dir))
                    config_rows.append(row)
                    continue
                timer_match = TIMER_RE.search(line)
                if timer_match:
                    row = parse_key_values(timer_match.group(1))
                    row["source"] = str(path.relative_to(root_dir))
                    if row.get("phase") == "measure":
                        timer_rows.append(row)
                    continue
                host_match = HOST_RE.search(line)
                if host_match:
                    row = parse_key_values(host_match.group(1))
                    row["source"] = str(path.relative_to(root_dir))
                    if row.get("phase") == "measure":
                        host_rows.append(row)
                    continue
                event_count_match = EVENT_COUNT_RE.search(line)
                if event_count_match:
                    row = parse_key_values(event_count_match.group(1))
                    row["source"] = str(path.relative_to(root_dir))
                    if row.get("phase") == "measure":
                        event_count_rows.append(row)

    return timer_rows, host_rows, event_count_rows, config_rows


def aggregate_iteration_rows(timer_rows, host_rows, event_count_rows):
    host_by_iter = {}
    for row in host_rows:
        key = (row["mode"], row["qo_len"], row["kv_len"], row["iteration"])
        host_by_iter[key] = row["host_elapsed_us"]

    events_by_iter = defaultdict(list)
    for row in event_count_rows:
        key = (row["mode"], row["qo_len"], row.get("kv_len", -1), row["iteration"])
        events_by_iter[key].append(row)

    grouped = defaultdict(list)
    for row in timer_rows:
        key = (
            row["mode"],
            row["qo_len"],
            row["kv_len"],
            row["n_heads"],
            row["n_kv_heads"],
            row["head_dim"],
            row["iteration"],
        )
        grouped[key].append(row)

    out = []
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][6])):
        mode, qo_len, kv_len, n_heads, n_kv_heads, head_dim, iteration = key
        iter_key = (mode, qo_len, kv_len, iteration)
        event_counts = events_by_iter.get(iter_key, [])
        agg = {
            "mode": mode,
            "qo_len": qo_len,
            "kv_len": kv_len,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "iteration": iteration,
            "record_count": len(rows),
            "lut_exp_values": ";".join(str(v) for v in sorted({row["lut_exp"] for row in rows})),
            "host_elapsed_us": host_by_iter.get(iter_key, 0),
            "event_count": sum(row.get("events", 0) for row in event_counts),
            "event_overflow": sum(row.get("overflow", 0) for row in event_counts),
            "profiled_total": sum(row["profiled_total"] for row in rows),
        }
        for component in COMPONENTS:
            agg[component] = sum(row[component] for row in rows)
        pct_den = sum(agg[component] for component in COMPONENTS)
        for component in COMPONENTS:
            agg[f"{component}_pct"] = (agg[component] / pct_den * 100.0) if pct_den else 0.0
        out.append(agg)
    return out


def summarize(iter_rows, config_rows):
    by_shape = defaultdict(list)
    for row in iter_rows:
        by_shape[(row["mode"], row["qo_len"], row["kv_len"])].append(row)

    summary = {
        "experiment": "isolated Figure 8 flash-attention long-KV cache breakdown",
        "aggregation": "per iteration sums across kv_head records; comparison rows use mean across measured iterations",
        "percentage_denominator": "sum(profiled_components)",
        "components": COMPONENTS,
        "config_rows": config_rows,
        "by_shape": {},
    }

    for (mode, qo_len, kv_len), rows in sorted(by_shape.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        rows = sorted(rows, key=lambda row: row["iteration"])
        med = {component: median([row[component] for row in rows]) for component in COMPONENTS}
        avg = {component: mean([row[component] for row in rows]) for component in COMPONENTS}
        med_total = sum(med.values())
        avg_total = sum(avg.values())
        med_pct = {component: (med[component] / med_total * 100.0) if med_total else 0.0 for component in COMPONENTS}
        avg_pct = {component: (avg[component] / avg_total * 100.0) if avg_total else 0.0 for component in COMPONENTS}
        host_values = [row["host_elapsed_us"] for row in rows if row["host_elapsed_us"]]
        shape_key = f"{mode}_q{qo_len}_kv{kv_len}"
        summary["by_shape"][shape_key] = {
            "mode": mode,
            "qo_len": qo_len,
            "kv_len": kv_len,
            "valid_iterations": len(rows),
            "record_count_values": sorted(set(row["record_count"] for row in rows)),
            "lut_exp_values": sorted(set(row["lut_exp_values"] for row in rows)),
            "event_overflow_total": sum(row["event_overflow"] for row in rows),
            "event_count_median": median([row["event_count"] for row in rows]),
            "median_us": {**med, "profiled_total": med_total},
            "mean_us": {**avg, "profiled_total": avg_total},
            "median_pct": med_pct,
            "mean_pct": avg_pct,
            "host_elapsed_us": {
                "median": median(host_values),
                "mean": mean(host_values),
                "count": len(host_values),
            },
        }
    return summary


def build_comparison_rows(summary):
    by_pair = {}
    for data in summary["by_shape"].values():
        by_pair[(data["qo_len"], data["kv_len"], data["mode"])] = data

    rows = []
    shapes = sorted({(q, kv) for q, kv, mode in by_pair})
    for qo_len, kv_len in shapes:
        base = by_pair.get((qo_len, kv_len, "baseline"))
        lut = by_pair.get((qo_len, kv_len, "lut-exp"))
        if not base or not lut:
            continue
        row = {
            "qo_len": qo_len,
            "kv_len": kv_len,
            "baseline_iterations": base["valid_iterations"],
            "lut_exp_iterations": lut["valid_iterations"],
            "baseline_lut_exp_values": ";".join(base["lut_exp_values"]),
            "lut_exp_lut_exp_values": ";".join(lut["lut_exp_values"]),
            "aggregation": "mean",
            "baseline_profiled_total_us": base["mean_us"]["profiled_total"],
            "lut_exp_profiled_total_us": lut["mean_us"]["profiled_total"],
            "profiled_total_delta_us": lut["mean_us"]["profiled_total"] - base["mean_us"]["profiled_total"],
            "profiled_total_delta_pct": pct_change(base["mean_us"]["profiled_total"], lut["mean_us"]["profiled_total"]),
            "baseline_host_mean_us": base["host_elapsed_us"]["mean"],
            "lut_exp_host_mean_us": lut["host_elapsed_us"]["mean"],
            "host_delta_us": lut["host_elapsed_us"]["mean"] - base["host_elapsed_us"]["mean"],
            "host_delta_pct": pct_change(base["host_elapsed_us"]["mean"], lut["host_elapsed_us"]["mean"]),
            "baseline_event_overflow_total": base["event_overflow_total"],
            "lut_exp_event_overflow_total": lut["event_overflow_total"],
        }
        for component in COMPONENTS:
            base_us = base["mean_us"][component]
            lut_us = lut["mean_us"][component]
            row[f"baseline_{component}_us"] = base_us
            row[f"lut_exp_{component}_us"] = lut_us
            row[f"{component}_delta_us"] = lut_us - base_us
            row[f"{component}_delta_pct"] = pct_change(base_us, lut_us)
            row[f"baseline_{component}_pct"] = base["mean_pct"][component]
            row[f"lut_exp_{component}_pct"] = lut["mean_pct"][component]
            row[f"{component}_pct_point_delta"] = lut["mean_pct"][component] - base["mean_pct"][component]
        rows.append(row)
    return rows


def write_iteration_csv(path, rows):
    fieldnames = [
        "mode",
        "qo_len",
        "kv_len",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "iteration",
        "record_count",
        "lut_exp_values",
        "host_elapsed_us",
        "event_count",
        "event_overflow",
        "profiled_total",
        *COMPONENTS,
        *[f"{component}_pct" for component in COMPONENTS],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(path, rows):
    fieldnames = [
        "qo_len",
        "kv_len",
        "baseline_iterations",
        "lut_exp_iterations",
        "aggregation",
        "baseline_lut_exp_values",
        "lut_exp_lut_exp_values",
        "baseline_profiled_total_us",
        "lut_exp_profiled_total_us",
        "profiled_total_delta_us",
        "profiled_total_delta_pct",
        "baseline_host_mean_us",
        "lut_exp_host_mean_us",
        "host_delta_us",
        "host_delta_pct",
        "baseline_event_overflow_total",
        "lut_exp_event_overflow_total",
    ]
    for component in COMPONENTS:
        fieldnames.extend(
            [
                f"baseline_{component}_us",
                f"lut_exp_{component}_us",
                f"{component}_delta_us",
                f"{component}_delta_pct",
                f"baseline_{component}_pct",
                f"lut_exp_{component}_pct",
                f"{component}_pct_point_delta",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_optional(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def end_to_end_reference_lines(markdown_path):
    sequence_root = markdown_path.parent.parent
    seq_rows = read_csv_optional(sequence_root / "sequence_lut_exp_10run_throughput_delta.csv")
    try:
        llm_root = sequence_root.parents[1]
    except IndexError:
        llm_root = sequence_root
    prompt_rows = read_csv_optional(
        llm_root
        / "batch_tile_sweeps"
        / "prompt_length_capacity_mode_sweep_20260523"
        / "prompt_length_lut_exp_10run_delta.csv"
    )

    lines = []
    if seq_rows:
        lines.extend(
            [
                "## End-to-End Reference",
                "",
                "These rows are full llama-server prefill throughput, not isolated attention. They explain why isolated attention kernel wall time and full prefill token/s must be judged with separate tables.",
                "",
                "| case | baseline mean prefill tok/s | LUT-exp mean prefill tok/s | LUT vs baseline | faster |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in seq_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["case"],
                        f"{float(row['baseline_mean_prefill_tok_s']):.3f}",
                        f"{float(row['lut_exp_mean_prefill_tok_s']):.3f}",
                        f"{float(row['lut_vs_baseline_prefill_delta_pct']):+.3f}%",
                        row["prefill_faster_mode"],
                    ]
                )
                + " |"
            )
        lines.append("")

    if prompt_rows:
        lines.extend(
            [
                "Prompt-length cross-check:",
                "",
                "| target prompt | baseline mean tok/s | LUT-exp mean tok/s | LUT vs baseline | faster |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in prompt_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["target_prompt_tokens"],
                        f"{float(row['baseline_mean_tok_s']):.3f}",
                        f"{float(row['lut_exp_mean_tok_s']):.3f}",
                        f"{float(row['lut_vs_baseline_delta_pct']):+.3f}%",
                        row["faster_mode"],
                    ]
                )
                + " |"
            )
        lines.append("")

    return lines


def write_comparison_markdown(path, rows):
    lines = [
        "# Long-KV Figure 8 Attention Breakdown",
        "",
        "## Scope First",
        "",
        "This file is an isolated DSP FlashAttention kernel benchmark, not full llama-server prefill throughput. It fixes `qo_len=64` and sweeps only `kv_len`. Use `kernel wall us` / `wall delta` to judge whether the standalone kernel is faster; `component total us` is an internal qtimer breakdown and can disagree with wall time for short cases because it excludes outer call overhead and sums per-worker component regions.",
        "",
        *end_to_end_reference_lines(path),
        "## Isolated Attention Kernel Timers",
        "",
        "Fixed shape: `qo_len=64`, `n_heads=12`, `n_kv_heads=2`, `head_dim=128` unless noted in raw logs.",
        "Comparison uses mean over the measured iterations. Negative delta means LUT-exp is faster. The primary speed column is `wall delta`, not `component delta`.",
        "",
        "| qo_len | kv_len | baseline kernel wall us | LUT-exp kernel wall us | wall delta | baseline component total us | LUT-exp component total us | component delta | baseline safe_sm us | LUT-exp safe_sm us | safe_sm delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["qo_len"]),
                    str(row["kv_len"]),
                    f"{row['baseline_host_mean_us']:.1f}",
                    f"{row['lut_exp_host_mean_us']:.1f}",
                    f"{row['host_delta_pct']:+.2f}%",
                    f"{row['baseline_profiled_total_us']:.1f}",
                    f"{row['lut_exp_profiled_total_us']:.1f}",
                    f"{row['profiled_total_delta_pct']:+.2f}%",
                    f"{row['baseline_safe_sm_us']:.1f}",
                    f"{row['lut_exp_safe_sm_us']:.1f}",
                    f"{row['safe_sm_delta_pct']:+.2f}%",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Component Time Delta",
            "",
            "| kv_len | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["kv_len"]),
                    *[f"{row[f'{component}_delta_pct']:.2f}%" for component in COMPONENTS],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Validation Notes",
            "",
            "- `baseline_lut_exp_values` should be `0`; `lut_exp_lut_exp_values` should be `1`.",
            "- `baseline_event_overflow_total` and `lut_exp_event_overflow_total` should both be `0` for every row.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timer_rows, host_rows, event_count_rows, config_rows = parse_logs(args.root_dir)
    iter_rows = aggregate_iteration_rows(timer_rows, host_rows, event_count_rows)
    summary = summarize(iter_rows, config_rows)
    comparison_rows = build_comparison_rows(summary)

    write_iteration_csv(args.out_dir / "long_kv_attention_timers.csv", iter_rows)
    (args.out_dir / "long_kv_attention_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_comparison_csv(args.out_dir / "long_kv_baseline_vs_lut_exp.csv", comparison_rows)
    (args.out_dir / "long_kv_baseline_vs_lut_exp_summary.json").write_text(
        json.dumps(
            {
                "comparison": "baseline vs lut-exp",
                "aggregation": summary["aggregation"],
                "components": COMPONENTS,
                "rows": comparison_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_comparison_markdown(args.out_dir / "long_kv_baseline_vs_lut_exp.md", comparison_rows)

    print(f"parsed_timer_records={len(timer_rows)}")
    print(f"parsed_iteration_rows={len(iter_rows)}")
    print(f"compared_shapes={len(comparison_rows)}")


if __name__ == "__main__":
    main()
