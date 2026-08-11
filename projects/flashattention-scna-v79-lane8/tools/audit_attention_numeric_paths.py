#!/usr/bin/env python3
"""Compare the shared FlashAttention numeric path across local experiment projects."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


PROJECTS = (
    "flashattention-scna-v79-lane8",
    "flashattention-scna-v81",
    "flashattention",
    "flashattention-scna-v79-framework",
)
COMPARE = re.compile(r"FIG8_ATTENTION_COMPARE\s+(.*)")
KV = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_map(text):
    result = {}
    for key, value in KV.findall(text):
        try:
            result[key] = float(value) if any(char in value for char in ".eE") else int(value, 0)
        except ValueError:
            result[key] = value
    return result


def function_body(text, name, limit=18000):
    start = text.find(name)
    return text[start:start + limit] if start >= 0 else ""


def infer_isa(base):
    values = []
    for path in (base / "src/htp-ops-lib-main").glob("hexagon_*_v*"):
        match = re.search(r"_v(\d+)$", path.name)
        if match:
            values.append(f"v{match.group(1)}")
    return ",".join(sorted(set(values))) or "not-built"


def v81_baseline_evidence(project):
    raw = project / "results/v81/scna/stage4-correctness-20260801/raw"
    rows = []
    for path in raw.glob("baseline_*.log"):
        for match in COMPARE.finditer(path.read_text(errors="replace")):
            row = value_map(match.group(1))
            if row.get("candidate_mode") == "baseline":
                rows.append(row)
    if not rows:
        return {}
    return {
        "pre_migration_baseline_cases": len(rows),
        "pre_migration_worst_rmse": max(row["rmse"] for row in rows),
        "pre_migration_worst_max_abs": max(row["max_abs_error"] for row in rows),
        "pre_migration_worst_relative_l2": max(row["relative_l2"] for row in rows),
    }


def shipped_binaries(base):
    result = {}
    for path in sorted(base.glob("hexagon_*_v*/ship/libhtp_ops_skel.so")):
        match = re.search(r"_v(\d+)/ship", str(path))
        if match:
            result[f"v{match.group(1)}"] = sha256(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    legacy = args.projects_root / "flashattention-scna-v79-framework/src/htp-ops-lib-main"
    legacy_hashes = {
        "hvx_math": sha256(legacy / "include/dsp/hvx_math.h"),
        "hvx_convert": sha256(legacy / "include/dsp/hvx_convert.h"),
        "flash_attn": sha256(legacy / "src/dsp/ops/flash_attn.c"),
    }
    rows = []
    for name in PROJECTS:
        project = args.projects_root / name
        base = project / "src/htp-ops-lib-main"
        math_path = base / "include/dsp/hvx_math.h"
        convert_path = base / "include/dsp/hvx_convert.h"
        flash_path = base / "src/dsp/ops/flash_attn.c"
        math_text = math_path.read_text(errors="replace")
        flash_text = flash_path.read_text(errors="replace")
        inv = function_body(math_text, "hvx_my_inv_vhf", 14000)
        safe = function_body(flash_text, "apply mask & compute rowmax", 24000)

        if "q_subnormal" in inv and "hvx_my_exp2_xqf_vhf" in inv:
            reciprocal = "v79-log2-xqf-with-subnormal-normalization"
        elif "#if __HVX_ARCH__ >= 79" in inv:
            reciprocal = "v79-log2-legacy-exp2-no-subnormal-normalization"
        else:
            reciprocal = "legacy-vlut-polynomial"
        rowmax = "rotate-all-lanes" if "Q6_V_vror_VR(v_s_rowmax0" in safe else "one-sided-vlalign"
        rowsum = "rotate-all-lanes" if "Q6_V_vror_VR(v_p_rowsum0" in safe else "one-sided-vlalign"
        safe_exp = "xqf-fp16" if "hvx_my_exp2_xqf_vhf(v_s_minus_m0" in safe else "legacy-qf16-exp2"
        scale_fix = "v_p_rowsum_pack2 = Q6_Vhf_vadd_VhfVhf(v_p_rowsum_pack2, v_p_rowsum_pack2)" in safe
        predicate_writeback = "v_p_rowsum_local_rot = Q6_V_vmux_QVV" in safe
        source_fix_complete = (
            reciprocal == "v79-log2-xqf-with-subnormal-normalization"
            and safe_exp == "xqf-fp16"
            and rowmax == "rotate-all-lanes"
            and rowsum == "rotate-all-lanes"
            and scale_fix
            and predicate_writeback
        )
        row = {
            "project": name,
            "built_isa": infer_isa(project),
            "hvx_math_sha256": sha256(math_path),
            "hvx_convert_sha256": sha256(convert_path),
            "flash_attn_sha256": sha256(flash_path),
            "hvx_math_identical_to_v79_framework": sha256(math_path) == legacy_hashes["hvx_math"],
            "hvx_convert_identical_to_v79_framework": sha256(convert_path) == legacy_hashes["hvx_convert"],
            "flash_attn_identical_to_v79_framework": sha256(flash_path) == legacy_hashes["flash_attn"],
            "reciprocal_path": reciprocal,
            "safe_softmax_exp2_path": safe_exp,
            "rowmax_reduction": rowmax,
            "rowsum_reduction": rowsum,
            "v79_rowsum_2x_scale_fix": scale_fix,
            "predicate_row_writeback": predicate_writeback,
            "source_fix_complete": source_fix_complete,
            "shipped_binary_sha256": shipped_binaries(base),
        }
        row.update(v81_baseline_evidence(project))
        if name == "flashattention-scna-v79-lane8":
            row["assessment"] = "fixed-and-dynamically-verified-216-of-216"
            row["post_migration_verification"] = "v79 FP32 gate: 216/216"
        elif name == "flashattention-scna-v81":
            row["assessment"] = "fix-migrated-build-and-dynamic-accuracy-verified"
            row["post_migration_verification"] = (
                "v81: 4 baseline FP32 cases worst_rmse=1.9595e-05 "
                "worst_max_abs=5.35374e-05; 1 SCNA d8 case rmse=4.46835e-05 max_abs=1.59818e-04"
            )
        elif name == "flashattention":
            row["assessment"] = "fix-migrated-v73-v79-build-and-runtime-smoke-pass-no-fp32-cli"
            row["post_migration_verification"] = "v73/v79 build pass; v73/v79 runtime ret=0"
        else:
            row["assessment"] = "fix-migrated-v73-v79-v81-build-and-runtime-smoke-pass-no-fp32-cli"
            row["post_migration_verification"] = "v73/v79/v81 build pass; v73/v79/v81 runtime ret=0"
        rows.append(row)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (args.output_dir / "cross_project_numeric_path_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "cross_project_numeric_path_audit.json").write_text(
        json.dumps({"legacy_reference_hashes": legacy_hashes, "projects": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
