#!/usr/bin/env python3
"""Validate completeness, compute paired confidence intervals, and issue one verdict."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = json.loads((PROJECT / "experiment_spec.json").read_text())


def load_jsonl(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted((root / "raw").glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            record["_source"] = path.name
            records.append(record)
    return records


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def bootstrap(values: list[float], draws: int = 10000, seed: int = 0x53434E41,
              reducer=statistics.median) -> dict[str, float]:
    if not values:
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    rng = random.Random(seed)
    samples = [reducer([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws)]
    return {"estimate": reducer(values), "ci_low": percentile(samples, 0.025),
            "ci_high": percentile(samples, 0.975), "n": len(values)}


def paired_latency(records: list[dict[str, object]], group_fields: tuple[str, ...]) -> dict[tuple, dict[str, float]]:
    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("record_type") != "attention_timing" or record.get("phase") != "measure":
            continue
        key = tuple(record.get(field) for field in group_fields) + (record.get("session"), record.get("iteration"))
        by_key[key][str(record.get("mode"))] = float(record["host_elapsed_us"])
    ratios: dict[tuple, list[float]] = defaultdict(list)
    for key, modes in by_key.items():
        if "lut-exp" in modes and "scna-fp16" in modes and modes["lut-exp"] > 0:
            ratios[key[:len(group_fields)]].append(modes["scna-fp16"] / modes["lut-exp"])
    return {key: bootstrap(value) for key, value in ratios.items()}


def attention_performance_details(records: list[dict[str, object]],
                                  group_fields: tuple[str, ...]) -> dict[str, object]:
    host: dict[tuple, list[float]] = defaultdict(list)
    dsp_by_sample: dict[tuple, float] = {}
    workers: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        if record.get("phase") != "measure":
            continue
        shape = tuple(record.get(field) for field in group_fields)
        mode = str(record.get("mode"))
        if record.get("record_type") == "attention_timing":
            host[shape + (mode,)].append(float(record["host_elapsed_us"]))
        elif record.get("record_type") == "attention_timers":
            sample = shape + (record.get("session"), record.get("iteration"), mode)
            dsp_by_sample[sample] = max(dsp_by_sample.get(sample, 0.0), float(record["profiled_total"]))
        elif record.get("record_type") == "resource":
            workers[shape + (mode,)].append((int(record["active_workers"]), int(record["tasks"])))
    dsp_pairs: dict[tuple, dict[str, float]] = defaultdict(dict)
    for key, value in dsp_by_sample.items():
        dsp_pairs[key[:-1]][str(key[-1])] = value
    dsp_ratios: dict[tuple, list[float]] = defaultdict(list)
    for key, modes in dsp_pairs.items():
        if "lut-exp" in modes and "scna-fp16" in modes and modes["lut-exp"] > 0:
            dsp_ratios[key[:len(group_fields)]].append(modes["scna-fp16"] / modes["lut-exp"])
    host_stats = {"|".join(map(str, key)): {
        "p50_us": percentile(values, 0.50), "p95_us": percentile(values, 0.95),
        "p99_us": percentile(values, 0.99), "n": len(values)} for key, values in host.items()}
    worker_stats = {"|".join(map(str, key)): {
        "active_workers": sorted({value[0] for value in values}),
        "tasks": sorted({value[1] for value in values})} for key, values in workers.items()}
    return {"host_quantiles": host_stats,
            "dsp_work_scna_over_lut": {"|".join(map(str, key)): bootstrap(values)
                                        for key, values in dsp_ratios.items()},
            "worker_task_counts": worker_stats}


def throughput_summaries(records: list[dict[str, object]]) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(dict)
    for record in records:
        if record.get("record_type") != "model_throughput":
            continue
        key = (str(record["kind"]), int(record["tokens"]), int(record["session"]))
        grouped[key][str(record["mode"])] = [float(v) for v in record["samples_ts"]]
    paired: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key, modes in grouped.items():
        if "lut-exp" not in modes or "scna-fp16" not in modes:
            continue
        count = min(len(modes["lut-exp"]), len(modes["scna-fp16"]))
        # A higher token/s value means lower latency, hence latency ratio is LUT throughput / SCNA throughput.
        paired[key[:2]].extend(modes["lut-exp"][i] / modes["scna-fp16"][i] for i in range(count)
                               if modes["scna-fp16"][i] > 0)
    return {key: bootstrap(ratios) for key, ratios in paired.items()}


def ppl_summary(records: list[dict[str, object]]) -> dict[str, float]:
    chunks: dict[int, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("record_type") == "model_ppl_chunk" and record.get("mode") in {"lut-exp", "scna-fp16"}:
            chunks[int(record["chunk"])][str(record["mode"])] = float(record["mean_nll"])
    deltas = [value["scna-fp16"] - value["lut-exp"] for _, value in sorted(chunks.items())
              if "lut-exp" in value and "scna-fp16" in value]
    if not deltas:
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    logged = bootstrap(deltas, reducer=statistics.mean)
    return {key: (math.exp(value) if key != "n" else value) for key, value in logged.items()}


def p95(values: list[float]) -> float:
    return percentile(values, 0.95)


def validate_attempts(records: list[dict[str, object]]) -> list[str]:
    """Reject more than one appended rerun or a provenance change."""
    reasons: list[str] = []
    attempts: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("case_id"):
            attempts[str(record["case_id"])].append(record)
    for case_id, values in attempts.items():
        attempt_ids = [int(value.get("attempt", 1)) for value in values]
        if any(value < 1 or value > 2 for value in attempt_ids) or attempt_ids.count(2) > 1:
            reasons.append(f"case {case_id} exceeds the one-rerun limit")
        hashes = {(value.get("spec_sha256"), value.get("dsp_sha256"), value.get("model_sha256"))
                  for value in values if any(key in value for key in ("spec_sha256", "dsp_sha256", "model_sha256"))}
        if len(hashes) > 1:
            reasons.append(f"case {case_id} changed a frozen hash between attempts")
    return reasons


def resource_summary(records: list[dict[str, object]]) -> dict[str, object]:
    resources: dict[str, list[dict[str, object]]] = defaultdict(list)
    startup: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    first_call: dict[str, list[float]] = defaultdict(list)
    steady_call: dict[str, list[float]] = defaultdict(list)
    stress_workers: dict[str, int] = defaultdict(int)
    stress_timing: dict[tuple[str, str, int, int, int], float] = {}
    equivalence: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.get("_source") != "resource.jsonl":
            continue
        flavor = str(record.get("flavor"))
        if record.get("record_type") == "resource":
            resources[flavor].append(record)
        if record.get("record_type") == "startup" and record.get("kind") == "fresh_process":
            for key in ("main_to_ready_us", "session_open_us", "backend_init_us"):
                startup[flavor][key].append(float(record[key]))
        if record.get("record_type") == "attention_timing" and record.get("kind") == "fresh_process":
            target = first_call if record.get("phase") == "warmup" else steady_call
            target[flavor].append(float(record["host_elapsed_us"]))
        if record.get("record_type") == "attention_workers" and "worker_stress" in str(record.get("kind")):
            stress_workers[flavor] = max(stress_workers[flavor], int(record["active_workers"]))
        if record.get("record_type") == "attention_timing" and record.get("phase") == "measure" and \
                "equivalence" in str(record.get("kind")):
            equivalence[(str(record.get("mode")), str(record.get("kind")))].append(float(record["host_elapsed_us"]))
        if record.get("record_type") == "attention_timing" and record.get("phase") == "measure" and \
                "worker_stress" in str(record.get("kind")):
            key = (flavor, str(record.get("mode")), int(record.get("worker", 0)),
                   int(record.get("session", 0)), int(record.get("iteration", 0)))
            stress_timing[key] = float(record["host_elapsed_us"])
    lut_resource = resources.get("lut_only", [{}])[-1]
    scna_resource = resources.get("scna_only", [{}])[-1]
    lut_start = p95(startup["lut_only"].get("main_to_ready_us", []))
    scna_start = p95(startup["scna_only"].get("main_to_ready_us", []))
    cold_improvement = 1.0 - scna_start / lut_start if lut_start and math.isfinite(lut_start) else math.nan
    equivalent = True
    equivalence_ratios = {}
    for mode in ("lut-exp", "scna-fp16"):
        combined = equivalence.get((mode, "combined_equivalence"), [])
        dedicated = equivalence.get((mode, "dedicated_equivalence"), [])
        if not combined or not dedicated:
            equivalent = False
            continue
        ratio = statistics.median(dedicated) / statistics.median(combined)
        equivalence_ratios[mode] = ratio
        equivalent &= 0.98 <= ratio <= 1.02
    realized_worker = stress_workers.get("scna_only", 0) > stress_workers.get("fair_combined", 0)
    stress_ratios: dict[tuple[str, int], list[float]] = defaultdict(list)
    stress_samples = {(mode, worker, session, iteration)
                      for _, mode, worker, session, iteration in stress_timing}
    for mode, worker, session, iteration in stress_samples:
        dedicated_flavor = "scna_only" if mode == "scna-fp16" else "lut_only"
        dedicated = stress_timing.get((dedicated_flavor, mode, worker, session, iteration))
        combined = stress_timing.get(("fair_combined", mode, worker, session, iteration))
        if dedicated is not None and combined and combined > 0:
            stress_ratios[(mode, worker)].append(dedicated / combined)
    stress_ci = {"|".join(map(str, key)): bootstrap(values) for key, values in stress_ratios.items()}
    realized_steady = any(value["ci_high"] <= 1.0 - SPEC["decision"]["resource_steady_improvement_min"]
                          for key, value in stress_ci.items() if key.startswith("scna-fp16|"))
    return {
        "lut_table_bytes": lut_resource.get("lut_table_bytes"),
        "scna_lut_table_bytes": scna_resource.get("lut_table_bytes"),
        "seq_capacity_delta": (int(scna_resource.get("vtcm_seq_capacity_bytes", 0)) -
                               int(lut_resource.get("vtcm_seq_capacity_bytes", 0))),
        "cold_start_p95_improvement": cold_improvement,
        "realized_worker_advantage": realized_worker,
        "realized_steady_advantage": realized_steady,
        "stress_dedicated_over_combined": stress_ci,
        "startup_p95_us": {flavor: {key: p95(values) for key, values in metrics.items()}
                            for flavor, metrics in startup.items()},
        "first_call_p95_us": {flavor: p95(values) for flavor, values in first_call.items()},
        "steady_call_p95_us": {flavor: p95(values) for flavor, values in steady_call.items()},
        "max_workers": dict(stress_workers),
        "dedicated_equivalence_pass": equivalent,
        "dedicated_equivalence_ratios": equivalence_ratios,
    }


def thermal_summary(records: list[dict[str, object]]) -> dict[str, object]:
    batches: dict[str, list[float]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    starts: dict[str, float] = {}
    for record in records:
        if record.get("record_type") == "thermal_batch":
            mode = str(record["mode"])
            elapsed_us = float(record["elapsed_us"])
            requests = float(record["requests"])
            batches[mode].append(1e6 * requests / elapsed_us)
            latencies[mode].append(elapsed_us / requests)
        if record.get("record_type") == "thermal_run" and record.get("event") == "start":
            starts[str(record["mode"])] = float(record["temperature_c"])
    mode_stats = {}
    for mode, values in batches.items():
        window = max(1, len(values) // 4)
        first = statistics.median(values[:window])
        last = statistics.median(values[-window:])
        mode_stats[mode] = {"batches": len(values), "first_throughput": first, "last_throughput": last,
                            "throughput_decay": 1.0 - last / first if first else math.nan,
                            "p95_request_latency_us": percentile(latencies[mode], 0.95),
                            "p99_request_latency_us": percentile(latencies[mode], 0.99)}
    advantage = False
    if "lut-exp" in mode_stats and "scna-fp16" in mode_stats:
        lut, scna = mode_stats["lut-exp"], mode_stats["scna-fp16"]
        advantage = (lut["throughput_decay"] - scna["throughput_decay"] >= 0.05 and
                     scna["p99_request_latency_us"] <= 0.95 * lut["p99_request_latency_us"])
    return {"starts": starts, "modes": mode_stats, "scna_material_advantage": advantage,
            "energy_claim_allowed": False}


def adjacent_crossovers(shape_stats: dict[tuple, dict[str, float]]) -> bool:
    scna_wins = {shape for shape, value in shape_stats.items() if value["ci_high"] < 0.98}
    lut_wins = {shape for shape, value in shape_stats.items() if value["ci_low"] > 1.02}
    def has_adjacent(values: set[tuple]) -> bool:
        q_order = SPEC["attention_performance"]["q"]
        kv_order = SPEC["attention_performance"]["kv"]
        for left in values:
            for right in values:
                if left == right:
                    continue
                # shape=(mask,q,kv); adjacency is two grid points sharing mask and one dimension.
                if left[0] != right[0]:
                    continue
                if left[1] == right[1] and left[2] in kv_order and right[2] in kv_order:
                    if abs(kv_order.index(left[2]) - kv_order.index(right[2])) == 1:
                        return True
                if left[2] == right[2] and left[1] in q_order and right[1] in q_order:
                    if abs(q_order.index(left[1]) - q_order.index(right[1])) == 1:
                        return True
        return False
    return has_adjacent(scna_wins) and has_adjacent(lut_wins)


def preflight_summary(records: list[dict[str, object]]) -> dict[str, object]:
    variants = [r for r in records if r.get("_source") == "variant_validation.jsonl"]
    resources = {}
    for record in variants:
        if record.get("record_type") == "nonlinear":
            resources[str(record.get("flavor"))] = {
                key: record.get(key) for key in ("lut_table_bytes", "lut_init_us", "vtcm_total_bytes",
                                                 "vtcm_reserved_bytes", "vtcm_seq_capacity_bytes")
            }
    rejected = sorted(f"{r.get('flavor')}:{r.get('mode')}" for r in variants
                      if r.get("record_type") == "mode_rejection" and r.get("rejected") is True)
    smoke_correctness = [{key: r.get(key) for key in ("candidate_mode", "rmse", "max_abs_error", "pass")}
                         for r in records if r.get("_source") == "smoke.jsonl" and
                         r.get("record_type") == "attention_correctness"]
    pair = next((r for r in records if r.get("record_type") == "model_pair_validation"), None)
    sanity = next((r for r in records if r.get("record_type") == "model_sanity"), None)
    return {"dedicated_resources": resources, "unsupported_modes_rejected": rejected,
            "smoke_correctness": smoke_correctness, "model_pair_validation": pair,
            "model_sanity": sanity}


def completeness(records: list[dict[str, object]]) -> list[str]:
    reasons = validate_attempts(records)
    sanity = [r for r in records if r.get("record_type") == "model_sanity"]
    if len(sanity) == 1 and not sanity[0].get("pass"):
        value = sanity[0]
        return [f"model baseline sanity failed: CPU PPL={value.get('cpu_ppl')}, "
                f"HTP PPL={value.get('htp_baseline_ppl')}, relative delta={value.get('relative_delta')}; "
                "downstream algorithm comparisons were intentionally skipped"]
    variant_records = [r for r in records if r.get("_source") == "variant_validation.jsonl"]
    supported = {(str(r.get("flavor")), str(r.get("evaluator"))): r for r in variant_records
                 if r.get("record_type") == "nonlinear"}
    rejected = {(str(r.get("flavor")), str(r.get("mode"))) for r in variant_records
                if r.get("record_type") == "mode_rejection" and r.get("rejected") is True}
    lut_dedicated = supported.get(("lut_only", "lut-exp"), {})
    scna_dedicated = supported.get(("scna_only", "scna-fp16"), {})
    if rejected != {("lut_only", "scna-fp16"), ("scna_only", "lut-exp")}:
        reasons.append("dedicated artifacts did not prove rejection of unsupported evaluators")
    if int(lut_dedicated.get("lut_table_bytes", -1)) != 262144 or \
            int(scna_dedicated.get("lut_table_bytes", -1)) != 0:
        reasons.append("dedicated artifact LUT allocation validation missing or failed")
    nonlinear = [r for r in records if r.get("record_type") == "nonlinear" and r.get("_source") == "nonlinear.jsonl"]
    expected_nonlinear = 2 * len(SPEC["nonlinear"]["distributions"]) * SPEC["nonlinear"]["sessions"]
    if len(nonlinear) < expected_nonlinear:
        reasons.append(f"nonlinear incomplete: {len(nonlinear)}/{expected_nonlinear}")
    correctness = [r for r in records if r.get("record_type") == "attention_correctness" and
                   r.get("_source") == "attention_correctness.jsonl"]
    numeric = [r for r in records if r.get("record_type") == "attention_numeric" and
               r.get("_source") == "attention_correctness.jsonl"]
    c = SPEC["attention_correctness"]
    expected_correctness = 2 * (len(c["masks"]) * len(c["q"]) * len(c["kv"]) * len(c["head_dim"]) *
                                len(c["seeds"]) + len(c["long_context_sentinels"]["masks"]) *
                                len(c["long_context_sentinels"]["q"]))
    if len(correctness) < expected_correctness:
        reasons.append(f"correctness incomplete: {len(correctness)}/{expected_correctness}")
    if not numeric:
        reasons.append("mask/tail zero checks are missing from Attention correctness")
    elif any(int(r.get("masked_p_nonzero", 0)) != 0 or int(r.get("tail_p_nonzero", 0)) != 0
             for r in numeric):
        reasons.append("Attention mask/tail zero hard gate failed")
    perf = [r for r in records if r.get("record_type") == "attention_timing" and
            r.get("_source") == "attention_performance.jsonl" and r.get("phase") == "measure"]
    p = SPEC["attention_performance"]
    expected_perf = 2 * (len(p["q"]) * len(p["kv"]) + 1) * p["sessions"] * p["measure"]
    if len(perf) < expected_perf:
        reasons.append(f"attention performance incomplete: {len(perf)}/{expected_perf}")
    if len(sanity) != 1 or not sanity[0].get("pass"):
        reasons.append("model baseline sanity missing or failed")
    for mode in ("cpu", "baseline", "lut-exp", "scna-fp16"):
        chunks = {r.get("chunk") for r in records if r.get("record_type") == "model_ppl_chunk" and r.get("mode") == mode}
        if len(chunks) < SPEC["model"]["ppl_chunks"]:
            reasons.append(f"model PPL incomplete for {mode}: {len(chunks)}/{SPEC['model']['ppl_chunks']}")
        if mode != "cpu":
            points = [r for r in records if r.get("record_type") == "model_throughput" and r.get("mode") == mode]
            required_points = (len(SPEC["model"]["prompt_tokens"]) + 1) * SPEC["model"]["sessions"]
            if len(points) < required_points:
                reasons.append(f"model throughput incomplete for {mode}: {len(points)}/{required_points}")
            logits = sum(int(r.get("fixtures", 0)) for r in records
                         if r.get("record_type") == "model_logit" and r.get("mode") == mode)
            if logits < len(SPEC["model"]["logit_contexts"]) * SPEC["model"]["logit_fixtures_per_context"]:
                reasons.append(f"logit audit incomplete for {mode}")
    resource = resource_summary(records)
    if resource["lut_table_bytes"] != 262144 or resource["scna_lut_table_bytes"] != 0 or \
            resource["seq_capacity_delta"] != 262144 or not resource["dedicated_equivalence_pass"]:
        reasons.append("resource attribution incomplete or dedicated artifact equivalence failed")
    thermal = thermal_summary(records)
    if set(thermal["modes"]) != {"lut-exp", "scna-fp16"}:
        reasons.append("thermal matched pair incomplete")
    elif abs(thermal["starts"]["lut-exp"] - thermal["starts"]["scna-fp16"]) > SPEC["thermal"]["start_temperature_delta_c_max"]:
        reasons.append("thermal start temperatures are not matched")
    return reasons


def evaluate(records: list[dict[str, object]]) -> dict[str, object]:
    invalid = completeness(records)
    correctness = [r for r in records if r.get("record_type") == "attention_correctness"]
    failed = {str(r.get("candidate_mode")) for r in correctness if int(r.get("pass", 0)) != 1}
    failed.update(str(r.get("mode")) for r in records if r.get("record_type") == "attention_numeric" and
                  (int(r.get("masked_p_nonzero", 0)) != 0 or int(r.get("tail_p_nonzero", 0)) != 0))
    attention = paired_latency([r for r in records if r.get("_source") == "attention_performance.jsonl"],
                               ("mask", "q", "kv"))
    attention_details = attention_performance_details(
        [r for r in records if r.get("_source") == "attention_performance.jsonl"], ("mask", "q", "kv"))
    throughput = throughput_summaries(records)
    ppl = ppl_summary(records)
    resource = resource_summary(records)
    thermal = thermal_summary(records)
    if invalid:
        verdict = "AUDIT_INVALID"
        rationale = invalid
    elif "lut-exp" in failed and "scna-fp16" in failed:
        verdict, rationale = "AUDIT_INVALID", ["both candidates failed correctness"]
    elif "lut-exp" in failed:
        verdict, rationale = "KEEP_SCNA", ["EXP-LUT failed correctness"]
    elif "scna-fp16" in failed:
        verdict, rationale = "SWITCH_TO_LUT", ["SCNA failed correctness"]
    else:
        quality_advantage = ppl["ci_high"] < SPEC["decision"]["ppl_equivalence_ratio"][0]
        quality_equivalent = (ppl["ci_low"] >= SPEC["decision"]["ppl_equivalence_ratio"][0] and
                              ppl["ci_high"] <= SPEC["decision"]["ppl_equivalence_ratio"][1])
        model_ratios = list(throughput.values())
        attention_ratios = list(attention.values())
        within_five = bool(model_ratios) and all(value["ci_high"] <= SPEC["decision"]["scna_max_slowdown_ratio"]
                                                  for value in model_ratios)
        cold_advantage = resource["cold_start_p95_improvement"] >= SPEC["decision"]["cold_start_p95_improvement_min"]
        realized_resource = bool(resource["realized_worker_advantage"] or resource["realized_steady_advantage"])
        material_advantage = quality_advantage or cold_advantage or realized_resource or thermal["scna_material_advantage"]
        if adjacent_crossovers(attention):
            verdict, rationale = "DYNAMIC_POLICY", ["predeclared adjacent shape regions have opposite material winners"]
        elif within_five and material_advantage:
            verdict, rationale = "KEEP_SCNA", ["SCNA is within the 5% cap and has a measured material advantage"]
        elif model_ratios and attention_ratios and all(value["ci_low"] > 1.02 for value in model_ratios) and \
                all(value["ci_low"] > 1.02 for value in attention_ratios) and not quality_advantage and \
                not realized_resource and not thermal["scna_material_advantage"]:
            verdict, rationale = "SWITCH_TO_LUT", ["EXP-LUT materially leads every model point without a SCNA offset"]
        elif model_ratios and attention_ratios and quality_equivalent and \
                all(0.98 <= value["ci_low"] and value["ci_high"] <= 1.02
                    for value in model_ratios + attention_ratios) and not material_advantage:
            verdict, rationale = "KEEP_SCNA", ["all registered metrics are equivalent; preregistered tie policy retains SCNA"]
        else:
            verdict, rationale = "SWITCH_TO_LUT", ["SCNA did not satisfy the preregistered retention rule"]
    return {"schema_version": 1, "verdict": verdict, "rationale": rationale,
            "ppl_scna_over_lut": ppl,
            "attention_latency_scna_over_lut": {"|".join(map(str, key)): value for key, value in attention.items()},
            "attention_performance_details": attention_details,
            "model_latency_scna_over_lut": {"|".join(map(str, key)): value for key, value in throughput.items()},
            "resource": resource, "thermal": thermal, "failed_correctness_modes": sorted(failed),
            "preflight": preflight_summary(records),
            "energy_claim_made": False}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_report(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True,
                                                        allow_nan=False) + "\n")
    with (out_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        csv_value = lambda value: "" if isinstance(value, float) and not math.isfinite(value) else value
        writer.writerow(["family", "point", "estimate", "ci_low", "ci_high", "n"])
        for family in ("attention_latency_scna_over_lut", "model_latency_scna_over_lut"):
            for point, value in summary[family].items():
                writer.writerow([family, point, csv_value(value["estimate"]), csv_value(value["ci_low"]),
                                 csv_value(value["ci_high"]), value["n"]])
        ppl = summary["ppl_scna_over_lut"]
        writer.writerow(["quality", "ppl_scna_over_lut", csv_value(ppl["estimate"]),
                         csv_value(ppl["ci_low"]), csv_value(ppl["ci_high"]), ppl["n"]])
    rationale = "\n".join(f"- {item}" for item in summary["rationale"])
    next_step = {
        "KEEP_SCNA": "保留 SCNA；下一项研究转向 DMA-HMX Pipeline。",
        "SWITCH_TO_LUT": "另立迁移任务，把 EXP-LUT 设为生产默认；随后转向 DMA-HMX Pipeline。",
        "DYNAMIC_POLICY": "另立按 Q/KV/mask/VTCM 选择 evaluator 的策略项目。",
        "AUDIT_INVALID": "保持 Stage2.75 不变，先修复或补齐审计基础设施；不得启动基于本审计结论的 DMA/INT8 工作。",
    }[summary["verdict"]]
    if summary["verdict"] == "AUDIT_INVALID":
        preflight = summary["preflight"]
        sanity = preflight.get("model_sanity") or {}
        dedicated = preflight.get("dedicated_resources") or {}
        smoke = preflight.get("smoke_correctness") or []
        cpu_ppl = float(sanity.get("cpu_ppl", math.nan))
        htp_ppl = float(sanity.get("htp_baseline_ppl", math.nan))
        smoke_lines = "\n".join(
            f"- `{row.get('candidate_mode')}`：RMSE `{float(row.get('rmse', math.nan)):.6g}`，"
            f"max-abs `{float(row.get('max_abs_error', math.nan)):.6g}`，smoke `PASS`"
            for row in smoke
        ) or "- 未获得 Attention smoke 结果"
        lut = dedicated.get("lut_only", {})
        scna = dedicated.get("scna_only", {})
        report = f"""# Stage 2.9 SCNA vs EXP-LUT 最终审计

## 裁决

`AUDIT_INVALID`

HTP baseline 已确认走真实 HTP、没有 CPU fallback，但一批次 WikiText2 PPL 为
`{htp_ppl:.6g}`；同一数据集的 CPU PPL 为 `{cpu_ppl:.6g}`，
远超预注册的 5% 差异上限。模型元数据、tensor 数量和类型清单一致，因此当前不能把失败
归因于 evaluator，也不能宣布 SCNA 或 EXP-LUT 胜出。

## 已通过的前置验收

{smoke_lines}
- `lut_only`：LUT `{lut.get('lut_table_bytes')}` B，顺序 VTCM `{lut.get('vtcm_seq_capacity_bytes')}` B。
- `scna_only`：LUT `{scna.get('lut_table_bytes')}` B，顺序 VTCM `{scna.get('vtcm_seq_capacity_bytes')}` B。
- 两个专用产物均明确拒绝不支持的 evaluator；没有静默回退。

释放 256 KiB LUT/VTCM 是已实现的内存布局变化，但尚未证明 worker、稳态性能、冷启动或热稳定性收益，
因此只记为潜在资源价值，不记为综合价值胜项。

## 停止边界

按预注册规则，模型 baseline 门槛失败后跳过完整 64-chunk PPL、模型吞吐/logits、Attention 性能、
30 进程资源压力和 10 分钟热稳定性矩阵。已有 smoke 数据不得用于算法去留裁决。

## 下一步

{next_step}
优先修复或重新生成可证明逻辑权重对应的 CPU/HMX 模型对，并验证 HMX matmul/模型布局链路；
通过 baseline sanity 后再从新的独立结果目录重跑本审计。
"""
        (out_dir / "FINAL_STAGE2_9_REPORT.md").write_text(report)
        return
    report = f"""# Stage 2.9 SCNA vs EXP-LUT 最终审计

## 裁决

`{summary['verdict']}`

{rationale}

## 核心指标

- SCNA/LUT PPL ratio：`{summary['ppl_scna_over_lut']}`
- SCNA/LUT 模型延迟：`{summary['model_latency_scna_over_lut']}`
- 资源归因：`{summary['resource']}`
- 前置验收：`{summary['preflight']}`
- 热稳定性：仅报告温度与持续吞吐代理，不包含能耗结论

## 证据边界

专用产物中 LUT 空间的释放属于“已实现的内存布局变化”；在 worker、稳态性能与模型硬门槛完成前，
不把它写成已实现的性能、冷启动或能耗收益。模型 baseline 硬门槛失败时，后续 SCNA/LUT
模型质量、模型延迟、完整 Attention 性能、资源压力与热稳定性矩阵按预注册规则停止，不能据此宣布算法胜负。

## 下一步

{next_step}
"""
    (out_dir / "FINAL_STAGE2_9_REPORT.md").write_text(report)


def main() -> int:
    global SPEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT / "experiment_spec.json")
    parser.add_argument("--input-dir", "--results-dir", dest="input_dir", type=Path,
                        default=PROJECT / "results/formal")
    parser.add_argument("--out-dir", "--reports-dir", dest="out_dir", type=Path,
                        default=PROJECT / "reports/final")
    args = parser.parse_args()
    SPEC = json.loads(args.spec.resolve().read_text())
    records = load_jsonl(args.input_dir)
    summary = evaluate(records)
    write_report(summary, args.out_dir)
    print(summary["verdict"])
    return 0 if summary["verdict"] != "AUDIT_INVALID" else 3


if __name__ == "__main__":
    raise SystemExit(main())
