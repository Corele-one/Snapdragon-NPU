#!/usr/bin/env python3
"""Audit the two Stage 1.5 candidate spaces without rebuilding a DSP artifact.

The audit deliberately consumes the frozen Stage 1 result and the archived
quad-pipeline experiment.  It answers whether either proposal denotes a new
candidate under the deployed HVX lane mapping; it does not estimate a new
latency or manufacture a device result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASELINE = "d7_pairret_noinline"
QUAD = "d7_quad_pipeline"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence is missing: {path}")
    return path


def require_contains(path: Path, text: str) -> None:
    if text not in path.read_text(encoding="utf-8", errors="replace"):
        raise ValueError(f"evidence marker {text!r} is absent from {path}")


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def render_markdown(audit: dict) -> str:
    baseline = audit["baseline"]
    layout = audit["candidate_1_layout"]
    batch = audit["candidate_2_batching"]
    return f"""# Stage 1.5: Layout and Batching Audit

## Decision

**No new device candidate was built.** Candidate 1 fails the lane-mapping
feasibility gate, and Candidate 2 is an archived `d7_quad_pipeline` design
under a different name. The production kernel remains `{BASELINE}`.

## Frozen evidence

| Item | Value |
|---|---:|
| Baseline artifact SHA-256 | `{baseline['artifact_sha256']}` |
| `scna_exp2.c` SHA-256 | `{baseline['source_sha256']['scna_exp2.c']}` |
| `flash_attn.c` SHA-256 | `{baseline['source_sha256']['flash_attn.c']}` |
| Evaluator static signature | {baseline['instructions']} instructions / {baseline['packets']} packets / {baseline['splats']} splats / {baseline['rhf_mul']} Rhf multiply |
| Evaluator spill / stack frame | {baseline['evaluator_spill']} / {baseline['evaluator_stack_frame_bytes']} B |

The baseline static fields are read from the final formal-run summary.  The
artifact and both source files are hashed by this audit; the referenced
disassembly and formal summary must also be present.

## Candidate 1 — weight layout / parameter packing

The deployed evaluator maps one score to one FP16 HVX lane: 64 independent
scores are returned by one vector evaluation. The seven active neurons are
therefore a time dimension. Each neuron uses a scalar-register (`Rhf`) weight
that is already broadcast to every score lane; only its bias needs a vector
splat.

| Mapping | Per vector evaluation |
|---|---:|
| Baseline | 64 distinct scores × 7 sequential neurons → 64 outputs |
| Proposed neuron-in-lane packing | 8 score groups × 8 neuron lanes → 8 partial outputs before reduction |

Packing neurons into lanes gives only {layout['distinct_scores_per_packed_vector']}
distinct scores per vector. To preserve 64 outputs, it requires at least
{layout['packed_vectors_per_baseline_block']} packed evaluations, replication
of every score across neuron lanes, segmented horizontal reduction, and output
restoration. These operations are mandatory data-layout work, not removable
weight loads. The effective-output gate therefore fails before packet analysis:
the proposal cannot produce 64 independent results in one vector evaluation.

Even before those mandatory transforms, the affine vector-operation lower bound
is {layout['packed_affine_vector_steps_lower_bound']} packed affine steps for a
64-score block, versus {layout['baseline_affine_vector_steps']} baseline affine
steps. The packed form is therefore already worse in the compute-only lower
bound, before its reduction and permutation cost is counted.

**Decision: `NOT BUILT — SIMD dimension mismatch`.** A plain memory layout
change cannot eliminate the seven bias splats, and an implementation would add
copy/permute/reduction work while reducing useful score-lane occupancy.

## Candidate 2 — evaluation granularity

The current call consumes two 64-lane row vectors for one column block:
`row0[c:c+64]` and `row1[c:c+64]`. Thus its block=1 batch already contains two
independent vector chains. Batching two adjacent column blocks means four live
input vectors (`x00`, `x01`, `x10`, `x11`), then four output vectors, followed
by two `SCNA_CONSUME_P_BLOCK` calls. This is exactly the archived quad shape.

| Form | Vectors live for SCNA | Status |
|---|---:|---|
| Current pairret, one column block | 2 | production baseline |
| Proposed two column blocks | 4 | archived `{QUAD}` |

Archived quad evidence: {batch['quad_instructions']} instructions /
{batch['quad_packets']} packets, evaluator spill {batch['quad_evaluator_spill']},
caller stack delta +{batch['quad_caller_stack_delta']} B, caller stack-reference
delta +{batch['quad_caller_spill_delta']}, and q32 median
{batch['pairret_q32_us']:.1f} µs (`{BASELINE}`) versus
{batch['quad_q32_us']:.1f} µs (`{QUAD}`) in the same historical formal run.
Those q32 values are archival evidence only; they are not compared with Stage 1
final-run absolute timings.

**Decision: `NOT BUILT — already-covered quad candidate`.** It violates the
Stage 1.5 caller-frame gate and has no historical q32 win.

## Gate and next action

A future candidate may enter static → micro → 72-case correctness →
q1/4/8/16/32 Attention only if it both preserves 64 useful outputs per vector
without lane replication/reduction and is not equivalent to four simultaneously
live score vectors. No current Stage 1.5 proposal meets those prerequisites.

## Evidence paths

""" + "\n".join(f"- `{item['path']}` — SHA-256 `{item['sha256']}`" for item in audit["evidence"]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pipeline-project", type=Path, default=None,
                        help="archived quad-pipeline project (defaults to sibling project)")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    workspace = project.parent
    pipeline = (args.pipeline_project or workspace / "flashattention-scna-hvx-fp16-d8-pipeline-v79").resolve()
    json_out = args.json_out or project / "reports/stage1_scna/STAGE15_LAYOUT_BATCHING_AUDIT.json"
    report_out = args.report_out or project / "reports/stage1_scna/STAGE15_LAYOUT_BATCHING_AUDIT.md"

    baseline_artifact = require(project / f"artifacts/variants/{BASELINE}/libhtp_ops_skel.so")
    scna_source = require(project / "src/htp-ops-lib-main/src/dsp/ops/scna_exp2.c")
    attention_source = require(project / "src/htp-ops-lib-main/src/dsp/ops/flash_attn.c")
    final_summary_path = require(project / "results/runs/20260822_stage1_final_validation_formal_v1/summary.json")
    baseline_disasm = require(project / "results/runs/20260822_stage1_final_validation_formal_v1/static/d7_pairret_noinline.v79.disasm.txt")
    final_report = require(project / "reports/stage1_scna/FINAL_STAGE1_REPORT.md")
    plan = require(project / "PLAN.md")
    quad_artifact = require(pipeline / f"artifacts/variants/{QUAD}/libhtp_ops_skel.so")
    quad_disasm = require(pipeline / f"results/static_audit/{QUAD}.v79.disasm.txt")
    quad_summary_path = require(pipeline / "results/runs/20260821_pipeline_formal_v1/summary.json")
    quad_gates_path = require(pipeline / "results/static_audit/static_gates.json")
    quad_source = require(pipeline / "src/htp-ops-lib-main/src/dsp/ops/flash_attn.c")

    for path, marker in ((scna_source, "Q6_Vqf16_vmpy_VhfRhf"),
                         (scna_source, "hvx_scna_exp2_pair_hot_return_vhf"),
                         (attention_source, "for (int c = scna_c_begin; c < n_cols; c += 64)"),
                         (attention_source, "hvx_scna_exp2_pair_hot_return_vhf"),
                         (quad_source, "for (; scna_c_begin + 64 < n_cols; scna_c_begin += 128)"),
                         (quad_source, "scna_d7_scalar_quad_inline"),
                         (plan, "d7_quad_pipeline"),
                         (final_report, "112 instructions、36 packets")):
        require_contains(path, marker)

    final_summary = json.loads(final_summary_path.read_text())
    static = final_summary["static"][BASELINE]
    expected_static = {"instructions": 112, "packets": 36, "splat": 8,
                       "scalar_weight_multiply": 14, "stack_frame_bytes": 0}
    for key, value in expected_static.items():
        if static.get(key) != value:
            raise ValueError(f"frozen baseline mismatch: {key}={static.get(key)!r}, expected {value!r}")

    quad_summary = json.loads(quad_summary_path.read_text())
    quad_gate = next(row for row in json.loads(quad_gates_path.read_text())["rows"]
                     if row["kernel_impl"] == QUAD)
    if (quad_gate["instructions"], quad_gate["packets"], quad_gate["caller_stack_delta"],
            quad_gate["caller_spill_delta"]) != (123, 43, 1920, 116):
        raise ValueError("archived quad static evidence no longer matches the registered result")

    evidence_paths = [baseline_artifact, scna_source, attention_source, final_summary_path,
                      baseline_disasm, final_report, plan, quad_artifact, quad_disasm,
                      quad_summary_path, quad_gates_path, quad_source]
    audit = {
        "schema_version": 1,
        "decision": "NOT_BUILT_BOTH_CANDIDATES",
        "production_kernel": BASELINE,
        "baseline": {
            "artifact_sha256": sha256(baseline_artifact),
            "source_sha256": {"scna_exp2.c": sha256(scna_source), "flash_attn.c": sha256(attention_source)},
            "instructions": static["instructions"], "packets": static["packets"],
            "splats": static["splat"], "rhf_mul": static["scalar_weight_multiply"],
            "evaluator_spill": 0, "evaluator_stack_frame_bytes": static["stack_frame_bytes"],
        },
        "candidate_1_layout": {
            "decision": "NOT_BUILT_SIMD_DIMENSION_MISMATCH",
            "baseline_distinct_scores_per_vector": 64,
            "active_neurons": 7,
            "packed_neuron_lanes": 8,
            "distinct_scores_per_packed_vector": 8,
            "packed_vectors_per_baseline_block": 8,
            "baseline_affine_vector_steps": 7,
            "packed_affine_vector_steps_lower_bound": 8,
            "mandatory_operations": ["score replication", "segmented horizontal reduction", "output restoration"],
        },
        "candidate_2_batching": {
            "decision": "NOT_BUILT_ALREADY_COVERED_QUAD",
            "baseline_vectors_per_column_block": 2,
            "two_column_blocks_vectors": 4,
            "quad_instructions": quad_gate["instructions"], "quad_packets": quad_gate["packets"],
            "quad_evaluator_spill": quad_gate["evaluator_spill"],
            "quad_caller_stack_delta": quad_gate["caller_stack_delta"],
            "quad_caller_spill_delta": quad_gate["caller_spill_delta"],
            "pairret_q32_us": quad_summary["latency"][BASELINE]["32"]["median"],
            "quad_q32_us": quad_summary["latency"][QUAD]["32"]["median"],
        },
        "evidence": [{"path": relative(path, workspace), "sha256": sha256(path)} for path in evidence_paths],
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    report_out.write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({"decision": audit["decision"], "json": str(json_out), "report": str(report_out)}))


if __name__ == "__main__":
    main()
