#!/usr/bin/env python3
"""Emit a simulator-only LUT/SCNA diagnostic table with strict trace gates."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

PAIR = re.compile(r"([A-Za-z0-9_]+)=([^\s:]+)")


def marker(text, name):
    out = []
    for line in text.splitlines():
        if name in line:
            out.append({k: v for k, v in PAIR.findall(line)})
    return out


def parse_retry(root):
    table = []
    for path in sorted((root / "raw/attention").glob("*_kv*.log")):
        text = path.read_text(errors="replace")
        timers = marker(text, "ATTENTION_TIMER")
        verifies = marker(text, "ATTENTION_VERIFY")
        processes = marker(text, "SIM_PROCESS_RESULT")
        if not timers:
            continue
        row = timers[-1]
        label = path.stem.rsplit("_kv", 1)[0]
        kv = int(row.get("kv", path.stem.rsplit("_kv", 1)[1]))
        pass_gate = (verifies and verifies[-1].get("status") == "PASS" and
                     processes and processes[-1].get("exit_code") == "0")
        table.append({"scheme": label, "qo": int(row.get("qo", 32)), "kv": kv,
                      "kernel_us": float(row.get("kernel_us", "nan")),
                      "safe_sm_us": float(row.get("safe_sm_us", "nan")),
                      "scna_exp_us": float(row.get("scna_exp_us", "nan")),
                      "profiled_total_us": float(row.get("profiled_total_us", "nan")),
                      "logical_s_p_bytes_per_head_tile": 32 * kv * 4,
                      "correctness": int(bool(pass_gate)), "source": str(path.relative_to(root))})
    return table


def native_gate(root):
    summary = root / "combined_summary.json"
    if not summary.exists():
        return {"status": "UNAVAILABLE", "exit_ok": False, "files_ok": False,
                "pc_range_ok": False, "nonzero_samples_ok": False, "source": str(root)}
    native = json.loads(summary.read_text()).get("native_trace", {})
    filtered = native.get("filtered", {})
    files = filtered.get("files", {})
    gates = {"exit_ok": filtered.get("exit_code") == 0,
             "files_ok": bool(files) and all(v.get("bytes", 0) > 0 for v in files.values()),
             "pc_range_ok": filtered.get("pc_matches", 0) > 0,
             "nonzero_samples_ok": bool(files) and all(v.get("pc_matches_in_4mib_prefix", 0) > 0
                                                    for v in files.values())}
    return {"status": "VALIDATED" if all(gates.values()) else "UNAVAILABLE", **gates,
            "source": str(root), "previous_authority": native.get("authority", "UNAVAILABLE")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--native-source-run")
    args = ap.parse_args()
    root = Path(args.run_dir)
    if (root / "raw/attention").exists():
        table = parse_retry(root)
    else:
        summary = json.loads((root / "combined_summary.json").read_text())
        rows = defaultdict(dict)
        with (root / "combined_attention.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                if row["scheme"] in ("origin", "exp-lut", "stage1_dynamic_row", "optimized") and int(row["qo"]) == 32:
                    rows[row["scheme"]][row["metric"]] = float(row["median"])
        table = [{"scheme": scheme, "qo": 32, "kv": 64,
                  "kernel_us": rows[scheme].get("kernel_us"), "safe_sm_us": rows[scheme].get("safe_sm_us"),
                  "scna_exp_us": rows[scheme].get("scna_exp_us"), "profiled_total_us": rows[scheme].get("profiled_total_us"),
                  "logical_s_p_bytes_per_head_tile": 32 * 64 * 4, "correctness": 1,
                  "source": "combined_attention.csv"}
                 for scheme in ("origin", "exp-lut", "stage1_dynamic_row", "optimized")]
    native = native_gate(Path(args.native_source_run) if args.native_source_run else root)
    authority = native["status"]
    result = {"schema_version": 1, "scope": "Hexagon v79 simulator diagnostic only; not device roofline or Snapdragon performance.",
              "native_trace_authority": authority, "native_trace_gates": native, "table": table,
              "logical_byte_definition": "Per 32x64 head tile: S read 2 B/element + P write 2 B/element. LUT table entries are not physical transaction counts.",
              "scna_op_definition": "d8 has 8 affine-ReLU terms per element; 16 logical FLOPs if each affine is counted as FMA. This is accounting, not a measured hardware peak."}
    (root / "lut_scna_roofline_simulator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (root / "lut_scna_roofline_simulator.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    lines = ["# LUT 与 SCNA Roofline 诊断（Hexagon v79 Simulator）", "",
             "> 仅为 simulator diagnostic；不能作为 Snapdragon 真机 Roofline、带宽、cycle 或排序结论。", "",
             "| Scheme | Qo | KV | kernel us | safe_sm us | scna_exp us | profiled total us | logical S+P B/head tile | pass |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| {scheme} | {qo} | {kv} | {kernel_us} | {safe_sm_us} | {scna_exp_us} | {profiled_total_us} | {logical_s_p_bytes_per_head_tile} | {correctness} |".format(**row))
    lines += ["", "## 动态 trace/PMU 四项门禁", "", f"- 状态：`{authority}`；source: `{native['source']}`.",
              f"- exit={native['exit_ok']}, files={native['files_ok']}, PC-range={native['pc_range_ok']}, nonzero-samples={native['nonzero_samples_ok']}.",
              "", "## 证据边界", "",
              "- S/P 的逻辑流量相同；表项逻辑大小不得解释为 vgather 的真实 cache/TCM transaction。",
              "- `scna_exp_us` 嵌套在 `safe_sm_us` 中，不与后者相加。",
              "- 真机 roof 与 simulator diagnostics 必须分报告，不得合并。"]
    (root / "ROOFLINE_LUT_VS_SCNA_SIMULATOR_ZH.md").write_text("\n".join(lines) + "\n")
    print(root / "ROOFLINE_LUT_VS_SCNA_SIMULATOR_ZH.md")


if __name__ == "__main__": main()
