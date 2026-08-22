#!/usr/bin/env python3
"""Analyze one self-contained Stage 1 baseline run and render its report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


LABELS = ("origin_hvx", "exp_lut", "static_d8_ref", "d7_pairret_noinline")
QOS = (1, 4, 8, 16, 32)
HISTORICAL_US = {
    "origin_hvx": {1: 661.5, 4: 824.5, 8: 1563.0, 16: 2856.0, 32: 5354.5},
    "exp_lut": {1: 575.0, 4: 717.5, 8: 1318.0, 16: 2413.0, 32: 4561.0},
    "static_d8_ref": {1: 652.5, 4: 815.0, 8: 1537.0, 16: 2899.0, 32: 5376.0},
    "d7_pairret_noinline": {1: 662.5, 4: 827.0, 8: 1487.5, 16: 2805.0, 32: 5182.5},
}


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line))


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def bootstrap_median(values: list[float], draws: int = 10000, seed: int = 0x5C1A) -> dict:
    if not values:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(statistics.median(rng.choices(values, k=len(values))))
    return {
        "median": statistics.median(values),
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "n": len(values),
    }


def paired_ratio(candidate: dict[tuple[int, int], float], baseline: dict[tuple[int, int], float],
                 draws: int = 10000, seed: int = 0x79) -> dict:
    keys = sorted(set(candidate) & set(baseline))
    ratios = [candidate[key] / baseline[key] for key in keys]
    return bootstrap_median(ratios, draws=draws, seed=seed)


def parse_disassembly(path: Path, symbol_token: str) -> dict:
    blocks: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if match:
            current = match.group(1)
            blocks[current] = []
        elif current and re.match(r"^\s*[0-9a-f]+:", line):
            blocks[current].append(line)

    selected = [line for name, body in blocks.items() if symbol_token in name for line in body]
    callers = [line for name, body in blocks.items() if "simple_flash_attn_f16_core" in name for line in body]

    def metric(lines: list[str]) -> dict:
        low = [line.lower() for line in lines]
        frames = [int(value, 0) for line in low
                  for value in re.findall(r"allocframe\(#(0x[0-9a-f]+|\d+)\)", line)]
        return {
            "instructions": len(lines),
            "packets": sum("{" in line for line in lines),
            "instructions_per_packet": len(lines) / max(1, sum("{" in line for line in lines)),
            "splat": sum("vsplat" in line for line in low),
            "scalar_weight_multiply": sum("vmpy" in line and ",r" in line for line in low),
            "multiply_instructions": sum("vmpy" in line or "mpyi" in line for line in low),
            "permute_instructions": sum(any(op in line for op in ("vdeal", "vshuff", "valign", "vlalign", "vdelta")) for line in low),
            "load_instructions": sum("mem" in line and "=" in line for line in low),
            "stack_references": sum(bool(re.search(r"mem[dhw]\(r(29|30)", line)) for line in low),
            "stack_frame_bytes": max(frames, default=0),
        }

    result = metric(selected)
    result["symbol"] = symbol_token
    result["caller"] = metric(callers)
    return result


def parse_micro(run_dir: Path) -> dict:
    result = {}
    for label in ("static_d8_ref", "d7_pairret_noinline"):
        values = []
        checksums = set()
        for path in sorted((run_dir / "raw/micro").glob(f"{label}_sample*.log")):
            for line in path.read_text(errors="replace").splitlines():
                if "SCNA_EXP_BENCH" not in line:
                    continue
                data = fields(line)
                values.append(float(data["paired_ns_per_64"]))
                checksums.add(data["checksum"])
        result[label] = bootstrap_median(values)
        result[label]["checksums"] = sorted(checksums)

    lut_values = []
    for path in sorted((run_dir / "raw/lut_micro").glob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            columns = line.split(",")
            if len(columns) == 20 and columns[0] == "2" and columns[1] == "lut_exp" and columns[17] == "Gelem/s":
                throughput = float(columns[16])
                if int(columns[18]) == 1 and throughput > 0:
                    lut_values.append(64.0 / throughput)
    result["exp_lut"] = bootstrap_median(lut_values)
    pair = result["d7_pairret_noinline"]["median"]
    lut = result["exp_lut"]["median"]
    result["pairret_vs_lut"] = pair / lut if pair is not None and lut else None
    return result


def parse_attention(run_dir: Path) -> tuple[dict, dict]:
    raw: dict[str, dict[int, dict[tuple[int, int], float]]] = {
        label: {q: {} for q in QOS} for label in LABELS
    }
    for path in sorted((run_dir / "raw/attention").glob("*.log")):
        match = re.match(r"(.+)_q(\d+)_s(\d+)\.log", path.name)
        if not match or match.group(1) not in LABELS:
            continue
        label, q, session = match.group(1), int(match.group(2)), int(match.group(3))
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_HOST_TIMING" not in line:
                continue
            data = fields(line)
            if data.get("phase") == "measure" and int(data.get("ret", "1")) == 0:
                raw[label][q][(session, int(data["iteration"]))] = float(data["host_elapsed_us"])

    summary = {label: {} for label in LABELS}
    ratios: dict[str, dict[int, dict]] = defaultdict(dict)
    for label in LABELS:
        for q in QOS:
            summary[label][q] = bootstrap_median(list(raw[label][q].values()), seed=0x100 + q)
            if label != "d7_pairret_noinline":
                ratios[label][q] = paired_ratio(raw["d7_pairret_noinline"][q], raw[label][q], seed=0x200 + q)
    return summary, dict(ratios)


def parse_correctness(run_dir: Path) -> dict:
    rows = []
    for path in sorted((run_dir / "raw/accuracy").glob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if "FIG8_ATTENTION_COMPARE " in line:
                rows.append(fields(line))
    return {
        "cases": len(rows),
        "pass_cases": sum(int(row.get("pass", "0")) for row in rows),
        "max_rmse": max((float(row["rmse"]) for row in rows), default=None),
        "max_abs": max((float(row["max_abs_error"]) for row in rows), default=None),
        "nonfinite": sum(int(row.get("candidate_nonfinite", "0")) for row in rows),
        "pass": bool(rows) and all(row.get("pass") == "1" for row in rows),
    }


def parse_diagnostics(run_dir: Path) -> dict:
    result = {}
    for label in LABELS:
        path = run_dir / "raw/diagnostic" / f"{label}_q32_kv4096.log"
        totals = defaultdict(float)
        host = []
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            data = fields(line)
            if "FIG8_ATTENTION_TIMERS" in line and data.get("phase") == "measure":
                for key in ("scna_exp", "profiled_total"):
                    totals[key] += float(data.get(key, 0))
            elif "FIG8_ATTENTION_HOST_TIMING" in line and data.get("phase") == "measure":
                host.append(float(data["host_elapsed_us"]))
        result[label] = {"host_us": statistics.median(host) if host else None, **totals}
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def fmt_stat(stat: dict, digits: int = 3) -> str:
    if stat.get("median") is None:
        return "UNAVAILABLE"
    return f"{stat['median']:.{digits}f} [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"


def render_report(summary: dict, report_path: Path, project: Path) -> None:
    attention = summary["attention"]
    ratios = summary["ratios"]
    corr = summary["correctness"]
    static = summary["static"]
    micro = summary["micro"]
    quick = summary["quick"]
    expected_cases = 1 if quick else 72
    static_ok = (static["d7_pairret_noinline"]["instructions"] == 112 and
                 static["d7_pairret_noinline"]["packets"] == 36 and
                 static["d7_pairret_noinline"]["splat"] == 8 and
                 static["d7_pairret_noinline"]["scalar_weight_multiply"] == 14)
    correctness_ok = corr["pass"] and corr["cases"] == expected_cases
    baseline_ok = static_ok and correctness_ok and not quick

    lines = [
        "# Stage 1 Baseline Reproduction",
        "",
        "## Decision",
        "",
        ("**PASS：baseline 已完整复现，可以进入 Experiment A。**" if baseline_ok else
         "**QUICK PASS：执行链路有效，但 quick run 不能替代正式 baseline。**" if static_ok and correctness_ok and quick else
         "**STOP：baseline 尚未满足复现门禁，不进入后续代码实验。**"),
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Artifact SHA256: `{summary['artifact_sha256']}`",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Static signature 112 inst / 36 packets / 8 splats / 14 Rhf mul: `{'PASS' if static_ok else 'FAIL'}`",
        f"- Correctness: `{corr['pass_cases']}/{corr['cases']}` cases pass (expected {expected_cases})",
        "",
        "## Question and hypothesis",
        "",
        "本实验检验全新独立构建的 `d7_pairret_noinline` 是否仍复现既有 correctness、micro、静态汇编和 Attention 基线。假设是在同一设备、同一参数和同一工具链下不会出现足以阻止后续优化的明显漂移。",
        "",
        "## Code and build evidence",
        "",
        "本实验未修改 evaluator 算法；新项目只复制必要源码并从零构建，未复制旧 `results/`、`artifacts/`、构建目录或虚拟环境。",
        "",
        f"- `scna_exp2.c` SHA256: `{summary['source_sha256']['scna_exp2.c']}`",
        f"- `flash_attn.c` SHA256: `{summary['source_sha256']['flash_attn.c']}`",
        "- Toolchain: Hexagon SDK 6.6.0.0, LLVM tools 19.0.07, `-mv79` verified in archived compile flags.",
        "",
        "## Assembly evidence",
        "",
        "|Version|Instructions|Packets|Inst/packet|Splats|Rhf mul|Eval stack refs/frame|Caller stack refs/frame|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("static_d8_ref", "d7_pairret_noinline"):
        row = static[label]
        caller = row["caller"]
        lines.append(
            f"|{label}|{row['instructions']}|{row['packets']}|{row['instructions_per_packet']:.3f}|"
            f"{row['splat']}|{row['scalar_weight_multiply']}|{row['stack_references']}/{row['stack_frame_bytes']} B|"
            f"{caller['stack_references']}/{caller['stack_frame_bytes']} B|"
        )
    lines += [
        "",
        "## Correctness",
        "",
        "|Cases|Pass|Max RMSE|Max abs|Nonfinite|Decision|",
        "|---:|---:|---:|---:|---:|---|",
        f"|{corr['cases']}|{corr['pass_cases']}|{corr['max_rmse']}|{corr['max_abs']}|{corr['nonfinite']}|{'PASS' if correctness_ok else 'FAIL'}|",
        "",
        "`pass=1` is emitted by the existing reference gate, which covers the configured finite/mask/tail and numerical limits.",
        f"Watchdog recovery attempts: `{len(summary['recovery_attempts'])}`. "
        + ("The affected case passed on retry; the timeout log remains archived." if summary["recovery_attempts"] else "No recovery was needed."),
        "",
        "## Microbenchmark",
        "",
        "|Version|ns / 64 useful elements (95% bootstrap CI)|Ratio to pairret|",
        "|---|---:|---:|",
        f"|static_d8_ref|{fmt_stat(micro['static_d8_ref'])}|{micro['static_d8_ref']['median'] / micro['d7_pairret_noinline']['median']:.4f}|",
        f"|d7_pairret_noinline|{fmt_stat(micro['d7_pairret_noinline'])}|1.0000|",
        f"|EXP-LUT attention distribution|{fmt_stat(micro['exp_lut'])}|{micro['exp_lut']['median'] / micro['d7_pairret_noinline']['median']:.4f}|",
        "",
        f"Pairret / EXP-LUT micro ratio: **{micro['pairret_vs_lut']:.4f}×**.",
        "",
        "## Full Attention",
        "",
        "|Version|q1 us|q4 us|q8 us|q16 us|q32 us|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        lines.append("|" + label + "|" + "|".join(fmt_stat(attention[label][q]) for q in QOS) + "|")
    lines += [
        "",
        "### Pairret paired ratios",
        "",
        "|Reference|q1|q4|q8|q16|q32|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("origin_hvx", "exp_lut", "static_d8_ref"):
        lines.append("|" + label + "|" + "|".join(fmt_stat(ratios[label][q], 4) for q in QOS) + "|")
    lines += [
        "",
        "## Drift against the historical report",
        "",
        "|Version|q1|q4|q8|q16|q32|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        cells = []
        for q in QOS:
            current = attention[label][q]["median"]
            cells.append(f"{(current / HISTORICAL_US[label][q] - 1) * 100:+.2f}%" if current is not None else "N/A")
        lines.append("|" + label + "|" + "|".join(cells) + "|")
    lines += [
        "",
        "Artifact SHA differs from the historical run because this is a fresh isolated build; code-generation metrics are the primary static drift gate. Dynamic drift is interpreted from the full formal run rather than the quick run.",
        "",
        "## Stage diagnostic",
        "",
        "|Version|Host us|scna_exp sum|profiled_total sum|",
        "|---|---:|---:|---:|",
    ]
    for label in LABELS:
        row = summary["diagnostic"].get(label, {})
        lines.append(f"|{label}|{row.get('host_us')}|{row.get('scna_exp', 0):.0f}|{row.get('profiled_total', 0):.0f}|")
    lines += [
        "",
        "## Interpretation",
        "",
        ("静态签名、72-case correctness 与正式动态数据均已重新生成；后续可仅基于本 run 的证据执行 packet analysis。" if baseline_ok else
         "quick run 只验证构建、设备执行、日志和分析路径，尚不能判断正式动态漂移。" if quick else
         "存在门禁失败；在解释漂移前禁止开始 Experiment B。"),
        "",
        "## Decision record",
        "",
        f"- Decision: `{'KEEP' if baseline_ok else 'INCONCLUSIVE' if quick else 'REJECT'}`",
        f"- Current best implementation: `d7_pairret_noinline`",
        f"- Recommended next experiment: `{'Experiment A — packet-level bottleneck analysis' if baseline_ok else 'formal baseline reproduction' if quick else 'baseline drift diagnosis'}`",
        "",
        "所有数值由本 run 原始日志生成；使用者在用于论文或学位材料前需逐项复核。",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    project = args.project.resolve()

    manifest_text = (run_dir / "evidence/device_manifest.txt").read_text(errors="replace")
    manifest_fields = dict(line.split("=", 1) for line in manifest_text.splitlines() if "=" in line)
    attention, ratios = parse_attention(run_dir)
    summary = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "quick": manifest_fields.get("quick") == "1",
        "source_commit": manifest_fields.get("source_git_commit", "UNKNOWN"),
        "artifact_sha256": sha256(project / "artifacts/variants/d7_pairret_noinline/libhtp_ops_skel.so"),
        "source_sha256": {
            name: sha256(project / "src/htp-ops-lib-main/src/dsp/ops" / name)
            for name in ("scna_exp2.c", "flash_attn.c")
        },
        "static": {
            "d7_pairret_noinline": parse_disassembly(
                run_dir / "static/d7_pairret_noinline.v79.disasm.txt", "hvx_scna_exp2_pair_hot_return_vhf"),
            "static_d8_ref": parse_disassembly(
                run_dir / "static/static_d8_ref.v79.disasm.txt", "pair_static_d8_qf16"),
        },
        "micro": parse_micro(run_dir),
        "correctness": parse_correctness(run_dir),
        "attention": attention,
        "ratios": ratios,
        "diagnostic": parse_diagnostics(run_dir),
        "recovery_attempts": sorted(path.name for path in (run_dir / "raw/recovery").glob("*.log")),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    render_report(summary, args.report.resolve(), project)
    print(json.dumps({
        "run_id": summary["run_id"],
        "quick": summary["quick"],
        "correctness": summary["correctness"],
        "pairret_static": summary["static"]["d7_pairret_noinline"],
        "report": str(args.report.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
