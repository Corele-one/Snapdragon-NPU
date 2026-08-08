#!/usr/bin/env python3
import argparse
import csv
import json
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


def pct_change(before, after):
    return ((after - before) / before * 100.0) if before else 0.0


def load_summary(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(baseline, lut_exp):
    rows = []
    common_q = sorted(
        set(baseline["by_qo_len"]).intersection(lut_exp["by_qo_len"]),
        key=lambda value: int(value),
    )
    for qo_len in common_q:
        base_q = baseline["by_qo_len"][qo_len]
        lut_q = lut_exp["by_qo_len"][qo_len]
        row = {
            "qo_len": int(qo_len),
            "baseline_iterations": base_q["valid_iterations"],
            "lut_exp_iterations": lut_q["valid_iterations"],
            "baseline_profiled_total_us": base_q["median_us"]["profiled_total"],
            "lut_exp_profiled_total_us": lut_q["median_us"]["profiled_total"],
            "profiled_total_delta_us": lut_q["median_us"]["profiled_total"] - base_q["median_us"]["profiled_total"],
            "profiled_total_delta_pct": pct_change(
                base_q["median_us"]["profiled_total"],
                lut_q["median_us"]["profiled_total"],
            ),
            "baseline_host_median_us": base_q["host_elapsed_us"]["median"],
            "lut_exp_host_median_us": lut_q["host_elapsed_us"]["median"],
            "host_delta_us": lut_q["host_elapsed_us"]["median"] - base_q["host_elapsed_us"]["median"],
            "host_delta_pct": pct_change(
                base_q["host_elapsed_us"]["median"],
                lut_q["host_elapsed_us"]["median"],
            ),
        }
        for component in COMPONENTS:
            base_us = base_q["median_us"][component]
            lut_us = lut_q["median_us"][component]
            row[f"baseline_{component}_us"] = base_us
            row[f"lut_exp_{component}_us"] = lut_us
            row[f"{component}_delta_us"] = lut_us - base_us
            row[f"{component}_delta_pct"] = pct_change(base_us, lut_us)
            row[f"baseline_{component}_pct"] = base_q["median_pct"][component]
            row[f"lut_exp_{component}_pct"] = lut_q["median_pct"][component]
            row[f"{component}_pct_point_delta"] = lut_q["median_pct"][component] - base_q["median_pct"][component]
        rows.append(row)
    return rows


def write_csv(path, rows):
    fieldnames = [
        "qo_len",
        "baseline_iterations",
        "lut_exp_iterations",
        "baseline_profiled_total_us",
        "lut_exp_profiled_total_us",
        "profiled_total_delta_us",
        "profiled_total_delta_pct",
        "baseline_host_median_us",
        "lut_exp_host_median_us",
        "host_delta_us",
        "host_delta_pct",
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


def write_markdown(path, rows):
    lines = [
        "# Figure 8 Baseline vs LUT-exp",
        "",
        "Comparison uses median per-iteration profiled component time. Negative delta means LUT-exp is faster.",
        "",
        "| qo_len | baseline total us | LUT-exp total us | total delta | baseline safe_sm us | LUT-exp safe_sm us | safe_sm delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["qo_len"]),
                    f"{row['baseline_profiled_total_us']:.1f}",
                    f"{row['lut_exp_profiled_total_us']:.1f}",
                    f"{row['profiled_total_delta_pct']:.2f}%",
                    f"{row['baseline_safe_sm_us']:.1f}",
                    f"{row['lut_exp_safe_sm_us']:.1f}",
                    f"{row['safe_sm_delta_pct']:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Percentage Point Change",
            "",
            "| qo_len | q_load | k_load | v_load | qk_dot | safe_sm | core_acc | o_scale | o_store |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["qo_len"]),
                    *[f"{row[f'{component}_pct_point_delta']:.2f}" for component in COMPONENTS],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--lut-exp-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_summary(args.baseline_summary)
    lut_exp = load_summary(args.lut_exp_summary)
    rows = build_rows(baseline, lut_exp)

    write_csv(args.out_dir / "baseline_vs_lut_exp.csv", rows)
    (args.out_dir / "baseline_vs_lut_exp_summary.json").write_text(
        json.dumps(
            {
                "comparison": "baseline vs lut-exp",
                "aggregation": "median per-iteration profiled component time",
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "baseline_vs_lut_exp.md", rows)
    print(f"compared_qo_lens={len(rows)}")


if __name__ == "__main__":
    main()
