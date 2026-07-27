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

TIMER_RE = re.compile(r"FIG8_ATTENTION_TIMERS\s+(.*)")
HOST_RE = re.compile(r"FIG8_ATTENTION_HOST_TIMING\s+(.*)")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def mode_to_slug(mode):
    return mode.replace("-", "_")


def parse_key_values(text):
    out = {}
    for key, value in KEY_VALUE_RE.findall(text):
        if key in {"mode", "phase"}:
            out[key] = value
        else:
            out[key] = int(value)
    return out


def read_text_auto(path):
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:256]:
        return data.decode("utf-16-le", errors="replace")
    return data.decode("utf-8", errors="replace")


def median(values):
    return statistics.median(values) if values else 0


def mean(values):
    return statistics.fmean(values) if values else 0.0


def parse_logs(input_dir):
    timer_rows = []
    host_rows = []
    raw_chunks = []

    for path in sorted(input_dir.glob("raw_q*.log")):
        text = read_text_auto(path)
        raw_chunks.append(f"===== {path.name} =====\n{text.rstrip()}\n")
        for line in text.splitlines():
            timer_match = TIMER_RE.search(line)
            if timer_match:
                row = parse_key_values(timer_match.group(1))
                if row.get("phase") == "measure":
                    timer_rows.append(row)
                continue
            host_match = HOST_RE.search(line)
            if host_match:
                row = parse_key_values(host_match.group(1))
                if row.get("phase") == "measure":
                    host_rows.append(row)

    return timer_rows, host_rows, "\n".join(raw_chunks)


def aggregate_iteration_rows(timer_rows):
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
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][6])):
        mode, qo_len, kv_len, n_heads, n_kv_heads, head_dim, iteration = key
        agg = {
            "mode": mode,
            "qo_len": qo_len,
            "kv_len": kv_len,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "iteration": iteration,
            "record_count": len(rows),
            "profiled_total": sum(row["profiled_total"] for row in rows),
        }
        for component in COMPONENTS:
            agg[component] = sum(row[component] for row in rows)
        pct_den = sum(agg[component] for component in COMPONENTS)
        for component in COMPONENTS:
            agg[f"{component}_pct"] = (agg[component] / pct_den * 100.0) if pct_den else 0.0
        out.append(agg)
    return out


def summarize(iter_rows, host_rows):
    host_by_q_iter = {
        (row["qo_len"], row["iteration"]): row["host_elapsed_us"]
        for row in host_rows
        if row.get("phase") == "measure"
    }

    by_q = defaultdict(list)
    for row in iter_rows:
        by_q[row["qo_len"]].append(row)

    summary = {
        "mode": sorted({row["mode"] for row in iter_rows}),
        "percentage_denominator": "sum(profiled_components)",
        "aggregation": "per iteration sums across kv_head records, then median/mean across measured iterations",
        "components": COMPONENTS,
        "by_qo_len": {},
    }

    for qo_len, rows in sorted(by_q.items()):
        rows = sorted(rows, key=lambda row: row["iteration"])
        med = {component: median([row[component] for row in rows]) for component in COMPONENTS}
        avg = {component: mean([row[component] for row in rows]) for component in COMPONENTS}
        med_total = sum(med.values())
        avg_total = sum(avg.values())
        med_pct = {component: (med[component] / med_total * 100.0) if med_total else 0.0 for component in COMPONENTS}
        avg_pct = {component: (avg[component] / avg_total * 100.0) if avg_total else 0.0 for component in COMPONENTS}
        host_values = [host_by_q_iter[(qo_len, row["iteration"])] for row in rows if (qo_len, row["iteration"]) in host_by_q_iter]

        summary["by_qo_len"][str(qo_len)] = {
            "valid_iterations": len(rows),
            "record_count_values": sorted(set(row["record_count"] for row in rows)),
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


def write_csv(path, rows):
    fieldnames = [
        "mode",
        "qo_len",
        "kv_len",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "iteration",
        "record_count",
        "profiled_total",
        *COMPONENTS,
        *[f"{component}_pct" for component in COMPONENTS],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, summary, mode_label):
    lines = [
        f"# Figure 8 {mode_label} Attention Percentages",
        "",
        "Percentages use median component time divided by the sum of median profiled components.",
        "",
        "| qo_len | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for qo_len, data in summary["by_qo_len"].items():
        pct = data["median_pct"]
        lines.append(
            "| "
            + " | ".join(
                [
                    qo_len,
                    *[f"{pct[component]:.2f}%" for component in COMPONENTS],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Median Component Time (us)",
            "",
            "| qo_len | profiled_total | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for qo_len, data in summary["by_qo_len"].items():
        med = data["median_us"]
        lines.append(
            "| "
            + " | ".join(
                [
                    qo_len,
                    f"{med['profiled_total']:.1f}",
                    *[f"{med[component]:.1f}" for component in COMPONENTS],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timer_rows, host_rows, raw_text = parse_logs(args.input_dir)
    iter_rows = aggregate_iteration_rows(timer_rows)
    summary = summarize(iter_rows, host_rows)

    (args.out_dir / "attention_timers_raw.log").write_text(raw_text, encoding="utf-8")
    write_csv(args.out_dir / "attention_timers.csv", iter_rows)
    (args.out_dir / "attention_timers_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    modes = summary.get("mode", [])
    mode_slug = mode_to_slug(modes[0]) if len(modes) == 1 else "mixed"
    mode_label = modes[0] if len(modes) == 1 else "mixed-mode"
    write_markdown(args.out_dir / f"figure8_{mode_slug}_percentages.md", summary, mode_label)

    print(f"parsed_timer_records={len(timer_rows)}")
    print(f"parsed_iteration_rows={len(iter_rows)}")


if __name__ == "__main__":
    main()
