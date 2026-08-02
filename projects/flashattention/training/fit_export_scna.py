#!/usr/bin/env python3
"""Fit and export shape-constrained SCNA exp/exp2 models for Hexagon HVX.

The fitted model is sum_i relu(w_i * x + b_i), with non-negative w/b.
For a convex target this is exactly a hinge representation of its secant PWL
fit.  The exporter also converts the hinge sum into a balanced breakpoint tree
whose leaves contain cumulative slope/intercept pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DOMAIN = (-256.0, 0.0)
ACTIVE_DOMAIN = (-16.0, 0.0)
INPUT_MULTIPLIER = 8.0
WIDTHS = (8, 16, 32)
FUNCTIONS = ("exp2", "exp")


def fp16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def c_round(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def target_function(name: str) -> Callable[[float], float]:
    if name == "exp2":
        return lambda x: math.pow(2.0, x)
    if name == "exp":
        return math.exp
    raise ValueError(name)


def target_derivative_at_zero(name: str) -> float:
    return math.log(2.0) if name == "exp2" else 1.0


@dataclass
class TreeParams:
    depth: int
    thresholds: list[float]
    slopes: list[float]
    biases: list[float]


@dataclass
class IntTreeParams:
    depth: int
    thresholds: list[int]
    slopes: list[int]
    biases: list[int]


@dataclass
class Model:
    function: str
    width: int
    wk: list[float]
    bk: list[float]
    wk_fp16: list[float]
    bk_fp16: list[float]
    qw: list[int]
    qb: list[int]
    weight_scale: float
    output_scale: float
    fp_tree: TreeParams
    int_tree: IntTreeParams
    validation: dict[str, object]


def fit_hinges(function: str, width: int) -> tuple[list[float], list[float]]:
    fn = target_function(function)
    left, right = ACTIVE_DOMAIN
    knots = [left + (right - left) * i / (width - 1) for i in range(width)]
    slopes = [(fn(knots[i + 1]) - fn(knots[i])) / (knots[i + 1] - knots[i])
              for i in range(width - 1)]
    weights = [slopes[0]]
    weights.extend(slopes[i] - slopes[i - 1] for i in range(1, len(slopes)))
    weights.append(target_derivative_at_zero(function) - slopes[-1])
    biases = [-weight * knot for weight, knot in zip(weights, knots)]
    if any(weight <= 0.0 for weight in weights) or any(bias < 0.0 for bias in biases):
        raise RuntimeError(f"shape constraint failed for {function} d{width}")
    return weights, biases


def quantize_int8(weights: list[float], biases: list[float]) -> tuple[list[int], list[int], float, float]:
    weight_scale = max(abs(value) for value in weights) / 127.0
    output_scale = weight_scale / INPUT_MULTIPLIER
    qw = [max(-127, min(127, c_round(value / weight_scale))) for value in weights]
    qb = [0 if weight == 0 else max(-32768, min(32767, c_round(value / output_scale)))
          for value, weight in zip(biases, qw)]
    return qw, qb, weight_scale, output_scale


def balanced_nodes(transitions: list[float | int], leaves: int, padded_transition: float | int) -> list[float | int]:
    nodes: list[float | int] = [padded_transition] * (leaves - 1)

    def fill(node: int, lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        nodes[node] = transitions[mid - 1] if mid <= len(transitions) else padded_transition
        fill(node * 2 + 1, lo, mid)
        fill(node * 2 + 2, mid, hi)

    fill(0, 0, leaves)
    return nodes


def build_fp_tree(weights: list[float], biases: list[float]) -> TreeParams:
    hinges = sorted((-bias / weight, weight, bias) for weight, bias in zip(weights, biases) if weight > 0.0)
    leaves = next_power_of_two(len(hinges) + 1)
    transitions = [item[0] for item in hinges]
    slopes = [0.0]
    intercepts = [0.0]
    slope = 0.0
    intercept = 0.0
    for _, weight, bias in hinges:
        slope = fp16(fp16(slope) + fp16(weight))
        intercept = fp16(fp16(intercept) + fp16(bias))
        slopes.append(slope)
        intercepts.append(intercept)
    while len(slopes) < leaves:
        slopes.append(slopes[-1])
        intercepts.append(intercepts[-1])
    return TreeParams(
        depth=int(math.log2(leaves)),
        thresholds=[fp16(float(item)) for item in balanced_nodes(transitions, leaves, 0.0)],
        slopes=[fp16(value) for value in slopes],
        biases=[fp16(value) for value in intercepts],
    )


def build_int_tree(qw: list[int], qb: list[int], width: int) -> IntTreeParams:
    hinges: list[tuple[int, int, int]] = []
    for weight, bias in zip(qw, qb):
        if weight <= 0:
            continue
        qx_min = math.floor(-bias / weight) + 1
        threshold = max(-129, min(0, qx_min - 1))
        hinges.append((threshold, weight, bias))
    hinges.sort()
    leaves = next_power_of_two(width + 1)
    transitions = [item[0] for item in hinges]
    slopes = [0]
    intercepts = [0]
    slope = 0
    intercept = 0
    for _, weight, bias in hinges:
        slope += weight
        intercept += bias
        slopes.append(slope)
        intercepts.append(intercept)
    while len(slopes) < leaves:
        slopes.append(slopes[-1])
        intercepts.append(intercepts[-1])
    return IntTreeParams(
        depth=int(math.log2(leaves)),
        thresholds=[int(item) for item in balanced_nodes(transitions, leaves, 0)],
        slopes=slopes[:leaves],
        biases=intercepts[:leaves],
    )


def fp_direct(x: float, weights: list[float], biases: list[float]) -> float:
    xh = fp16(max(DOMAIN[0], min(DOMAIN[1], x)))
    total = 0.0
    for weight, bias in zip(weights, biases):
        affine = fp16(xh * fp16(weight) + fp16(bias))
        total = fp16(total + max(affine, 0.0))
    return total


def fp_tree_eval(x: float, tree: TreeParams) -> float:
    xh = fp16(max(DOMAIN[0], min(DOMAIN[1], x)))
    node = 0
    leaf = 0
    for _ in range(tree.depth):
        right = xh > tree.thresholds[node]
        leaf = leaf * 2 + int(right)
        node = node * 2 + 1 + int(right)
    return fp16(xh * tree.slopes[leaf] + tree.biases[leaf])


def int_direct(x: float, qw: list[int], qb: list[int], output_scale: float) -> float:
    qx = int(max(-128, min(0, math.trunc(max(ACTIVE_DOMAIN[0], min(0.0, x)) * INPUT_MULTIPLIER))))
    return sum(max(qx * weight + bias, 0) for weight, bias in zip(qw, qb)) * output_scale


def int_tree_eval(x: float, tree: IntTreeParams, output_scale: float) -> float:
    qx = int(max(-128, min(0, math.trunc(max(ACTIVE_DOMAIN[0], min(0.0, x)) * INPUT_MULTIPLIER))))
    node = 0
    leaf = 0
    for _ in range(tree.depth):
        right = qx > tree.thresholds[node]
        leaf = leaf * 2 + int(right)
        node = node * 2 + 1 + int(right)
    value = qx * tree.slopes[leaf] + tree.biases[leaf]
    return max(value, 0) * output_scale


def validate(function: str, weights: list[float], biases: list[float], qw: list[int], qb: list[int],
             output_scale: float, fp_tree: TreeParams, int_tree: IntTreeParams, grid_points: int) -> dict[str, object]:
    fn = target_function(function)
    variants = {
        "fp16_direct": lambda x: fp_direct(x, weights, biases),
        "fp16_tree": lambda x: fp_tree_eval(x, fp_tree),
        "int8_direct": lambda x: int_direct(x, qw, qb, output_scale),
        "int8_tree": lambda x: int_tree_eval(x, int_tree, output_scale),
    }
    stats = {name: {"sq": 0.0, "max": 0.0, "previous": None, "monotonic_violations": 0,
                    "negative_count": 0, "nonfinite_count": 0} for name in variants}
    direct_tree_max = {"fp16": 0.0, "int8": 0.0}
    for index in range(grid_points):
        x = DOMAIN[0] + (DOMAIN[1] - DOMAIN[0]) * index / (grid_points - 1)
        expected = fn(x)
        values = {name: evaluator(x) for name, evaluator in variants.items()}
        direct_tree_max["fp16"] = max(direct_tree_max["fp16"], abs(values["fp16_direct"] - values["fp16_tree"]))
        direct_tree_max["int8"] = max(direct_tree_max["int8"], abs(values["int8_direct"] - values["int8_tree"]))
        for name, value in values.items():
            entry = stats[name]
            error = value - expected
            entry["sq"] += error * error
            entry["max"] = max(entry["max"], abs(error))
            if entry["previous"] is not None and value < entry["previous"]:
                entry["monotonic_violations"] += 1
            if value < 0.0:
                entry["negative_count"] += 1
            if not math.isfinite(value):
                entry["nonfinite_count"] += 1
            entry["previous"] = value
    result: dict[str, object] = {"grid_points": grid_points, "domain": list(DOMAIN)}
    for name, entry in stats.items():
        result[name] = {
            "rmse": math.sqrt(float(entry["sq"]) / grid_points),
            "max_abs": entry["max"],
            "monotonic_violations": entry["monotonic_violations"],
            "negative_count": entry["negative_count"],
            "nonfinite_count": entry["nonfinite_count"],
        }
    result["direct_tree_max_abs"] = direct_tree_max
    result["boundary"] = {
        name: {"at_min": evaluator(DOMAIN[0]), "at_zero": evaluator(0.0)}
        for name, evaluator in variants.items()
    }
    return result


def fit_model(function: str, width: int, grid_points: int) -> Model:
    wk, bk = fit_hinges(function, width)
    wk_fp16 = [fp16(value) for value in wk]
    bk_fp16 = [fp16(value) for value in bk]
    qw, qb, weight_scale, output_scale = quantize_int8(wk, bk)
    fp_tree = build_fp_tree(wk_fp16, bk_fp16)
    int_tree = build_int_tree(qw, qb, width)
    validation = validate(function, wk_fp16, bk_fp16, qw, qb, output_scale, fp_tree, int_tree, grid_points)
    return Model(function, width, wk, bk, wk_fp16, bk_fp16, qw, qb, weight_scale, output_scale,
                 fp_tree, int_tree, validation)


def c_float(values: list[float]) -> str:
    rendered = []
    for value in values:
        if value == 0.0:
            value = 0.0
        token = f"{value:.9g}"
        if "." not in token and "e" not in token:
            token += ".0"
        rendered.append(token + "f")
    return ", ".join(rendered)


def c_int(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def wrapped_array(declaration: str, values: list[float] | list[int], formatter: Callable[[list], str]) -> str:
    chunks = [values[index:index + 8] for index in range(0, len(values), 8)]
    body = "\n".join(f"  {formatter(chunk)}," for chunk in chunks)
    return f"static const {declaration} = {{\n{body}\n}};\n"


def render_header(models: list[Model]) -> str:
    lines = [
        "// Generated by training/fit_export_scna.py. Do not edit by hand.",
        "#pragma once", "", "#include <stdint.h>", "",
        "#define SCNA_MIN_INPUT (-256.0f)",
        "#define SCNA_MAX_INPUT 0.0f",
        "#define SCNA_INT8_MIN_INPUT (-16.0f)",
        "#define SCNA_INT8_INPUT_MULTIPLIER 8.0f",
        "#define SCNA_MAX_WIDTH 32",
        "#define SCNA_MAX_TREE_NODES 63",
        "#define SCNA_MAX_TREE_LEAVES 64", "",
        "typedef struct {",
        "  int function;", "  int width;", "  int tree_depth;", "  int tree_leaves;",
        "  const __fp16 *wk_fp16;", "  const __fp16 *bk_fp16;",
        "  const int8_t *wk_int8;", "  const int16_t *bk_int16;",
        "  const __fp16 *tree_threshold_fp16;", "  const __fp16 *tree_slope_fp16;",
        "  const __fp16 *tree_bias_fp16;", "  const int16_t *tree_threshold_int16;",
        "  const int16_t *tree_slope_int16;", "  const int16_t *tree_bias_int16;",
        "  float weight_scale;", "  float output_scale;", "} scna_params_t;", "",
    ]
    for model in models:
        prefix = f"scna_{model.function}_d{model.width}"
        leaves = 1 << model.fp_tree.depth
        lines.extend([
            wrapped_array(f"__fp16 {prefix}_wk[{model.width}]", model.wk_fp16, c_float),
            wrapped_array(f"__fp16 {prefix}_bk[{model.width}]", model.bk_fp16, c_float),
            wrapped_array(f"int8_t {prefix}_qw[{model.width}]", model.qw, c_int),
            wrapped_array(f"int16_t {prefix}_qb[{model.width}]", model.qb, c_int),
            wrapped_array(f"__fp16 {prefix}_tree_t[{leaves - 1}]", model.fp_tree.thresholds, c_float),
            wrapped_array(f"__fp16 {prefix}_tree_m[{leaves}]", model.fp_tree.slopes, c_float),
            wrapped_array(f"__fp16 {prefix}_tree_c[{leaves}]", model.fp_tree.biases, c_float),
            wrapped_array(f"int16_t {prefix}_tree_qt[{leaves - 1}]", model.int_tree.thresholds, c_int),
            wrapped_array(f"int16_t {prefix}_tree_qm[{leaves}]", model.int_tree.slopes, c_int),
            wrapped_array(f"int16_t {prefix}_tree_qc[{leaves}]", model.int_tree.biases, c_int),
        ])
    lines.extend(["static inline const scna_params_t *scna_get_params(int function, int width) {",
                  "  static const scna_params_t params[] = {"])
    for function_index, function in enumerate(FUNCTIONS):
        for model in [item for item in models if item.function == function]:
            prefix = f"scna_{model.function}_d{model.width}"
            leaves = 1 << model.fp_tree.depth
            lines.append(
                f"    {{ {function_index}, {model.width}, {model.fp_tree.depth}, {leaves}, "
                f"{prefix}_wk, {prefix}_bk, {prefix}_qw, {prefix}_qb, {prefix}_tree_t, {prefix}_tree_m, "
                f"{prefix}_tree_c, {prefix}_tree_qt, {prefix}_tree_qm, {prefix}_tree_qc, "
                f"{model.weight_scale:.9g}f, {model.output_scale:.9g}f }},")
    lines.extend([
        "  };",
        "  const int function_index = function == 1 ? 1 : 0;",
        "  const int width_index = width == 8 ? 0 : width == 16 ? 1 : 2;",
        "  return &params[function_index * 3 + width_index];",
        "}", "",
    ])
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("src/htp-ops-lib-main/include/dsp/scna_params.h"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("training/artifacts"))
    parser.add_argument("--grid-points", type=int, default=16385)
    args = parser.parse_args()
    if args.grid_points < 2:
        parser.error("--grid-points must be at least 2")

    models = [fit_model(function, width, args.grid_points) for function in FUNCTIONS for width in WIDTHS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_header(models), encoding="utf-8")

    for model in models:
        checkpoint = {
            "format": "scope-effective-parameters-v1",
            "fit": {"method": "convex_secant_hinge", "active_domain": list(ACTIVE_DOMAIN),
                    "deployment_domain": list(DOMAIN)},
            "model": {"function": model.function, "num_units": model.width, "reparam": "positive"},
            "parameters": {"effective": {"w": model.wk, "k": [1.0] * model.width, "b": model.bk}},
        }
        checkpoint_path = args.artifact_dir / "checkpoints" / model.function / f"d{model.width}" / "best.json"
        write_json(checkpoint_path, checkpoint)
        checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        metadata = {
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checksum,
            "function": model.function, "width": model.width,
            "domain": list(DOMAIN), "active_domain": list(ACTIVE_DOMAIN),
            "shape_constraints": {"weights_nonnegative": True, "biases_nonnegative": True},
            "fp16": {"weights": model.wk_fp16, "biases": model.bk_fp16},
            "int8": {"input_dtype": "s8", "weight_dtype": "s8", "bias_dtype": "s16",
                     "tree_accumulator_dtype": "s32", "input_multiplier": INPUT_MULTIPLIER,
                     "weight_scale": model.weight_scale, "output_scale": model.output_scale,
                     "weights": model.qw, "biases": model.qb},
            "tree": {"depth": model.fp_tree.depth, "leaves": 1 << model.fp_tree.depth,
                     "representation": "balanced_branchless_breakpoint"},
            "validation": model.validation,
        }
        write_json(args.artifact_dir / "metadata" / model.function / f"d{model.width}.json", metadata)

    print(json.dumps({"models": len(models), "output": str(args.output),
                      "artifact_dir": str(args.artifact_dir), "grid_points": args.grid_points}, sort_keys=True))


if __name__ == "__main__":
    main()
