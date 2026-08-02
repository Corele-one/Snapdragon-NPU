#!/usr/bin/env python3
"""Validate and summarize SCNA v81 correctness logs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


PAIR_RE = re.compile(r"(\w+)=([^\s]+)")


def fields(line: str) -> dict[str, str]:
    return dict(PAIR_RE.findall(line))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, str] = {}
    reference_rows: list[dict[str, object]] = []
    direct_tree_rows: list[dict[str, object]] = []
    mask_probes: list[dict[str, object]] = []
    failures: list[str] = []
    seen_probe: set[tuple[str, str, str, str, str, str, str]] = set()

    for line in args.input.read_text(encoding="utf-8", errors="replace").splitlines():
        if "FIG8_ATTENTION_CONFIG" in line:
            config = fields(line)
            continue
        if "FIG8_ATTENTION_COMPARE" in line and "reference_mode=host-fp32" in line:
            data = fields(line)
            row = {**config, **data}
            reference_rows.append(row)
            if int(data.get("candidate_nonfinite", "1")) or int(data.get("reference_nonfinite", "1")):
                failures.append(f"nonfinite output: {config}")
            if data.get("candidate_mode") == "baseline":
                if float(data["rmse"]) > 0.002 or float(data["max_abs_error"]) > 0.01:
                    failures.append(f"baseline FP32 gate: rmse={data['rmse']} max={data['max_abs_error']} {config}")
            continue
        if "FIG8_ATTENTION_DIRECT_TREE_COMPARE" in line:
            data = fields(line)
            direct_tree_rows.append({**config, **data})
            if data.get("gate") != "pass" or data.get("ret") != "0":
                failures.append(f"direct/tree gate: {data}")
            continue
        if "FIG8_NUMERIC" in line and data_key(config) not in seen_probe:
            data = fields(line)
            key = data_key(config)
            seen_probe.add(key)
            row = {**config, **data}
            mask_probes.append(row)
            if (config.get("kv_len") == "4093" or config.get("mask_mode") in ("padding", "causal")) and \
                    int(data.get("p0_last_bits", "-1"), 16) != 0:
                failures.append(f"masked/tail P lane is nonzero: {row}")

    def write_csv(name: str, rows: list[dict[str, object]]) -> None:
        path = args.out_dir / name
        names = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv("fp32_reference.csv", reference_rows)
    write_csv("direct_tree.csv", direct_tree_rows)
    write_csv("mask_tail_probes.csv", mask_probes)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in reference_rows:
        grouped[(str(row.get("candidate_mode", "unknown")), str(row.get("scna_function", "none")),
                 str(row.get("scna_kernel", "none")))].append(row)
    report = [
        "# V81 SCNA Correctness Matrix", "",
        f"- FP32 comparisons: {len(reference_rows)}",
        f"- Direct/tree comparisons: {len(direct_tree_rows)}",
        f"- Mask/tail probes: {len(mask_probes)}",
        f"- Gate failures: {len(failures)}", "",
        "| Mode | Function | Kernel | Cases | Max RMSE | Max absolute error | Nonfinite |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for key, rows in sorted(grouped.items()):
        report.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(rows)} | "
            f"{max(float(row['rmse']) for row in rows):.6g} | "
            f"{max(float(row['max_abs_error']) for row in rows):.6g} | "
            f"{sum(int(row.get('candidate_nonfinite', 0)) for row in rows)} |")
    if failures:
        report += ["", "## Failures", ""] + [f"- {failure}" for failure in failures]
    (args.out_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} correctness gate(s) failed")


def data_key(config: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return tuple(config.get(name, "") for name in
                 ("mode", "scna_function", "scna_kernel", "scna_width", "mask_mode", "kv_len", "head_dim"))


if __name__ == "__main__":
    main()
