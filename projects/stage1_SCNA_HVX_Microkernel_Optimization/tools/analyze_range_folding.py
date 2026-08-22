#!/usr/bin/env python3
"""Prove SCNA affine sign bounds on the deployed safe-softmax domain."""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


def fp16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def array_values(text: str, name: str) -> list[float]:
    match = re.search(rf"{name}\[SCNA_D8_WIDTH\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match:
        raise ValueError(f"array {name} not found")
    return [fp16(float(value)) for value in re.findall(r"\(__fp16\)\s*([-+0-9.eE]+)f", match.group(1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = args.header.read_text()
    weights = array_values(text, "scna_exp2_d8_wk")
    biases = array_values(text, "scna_exp2_d8_bk")
    x_min, x_max = -256.0, 0.0
    rows = []
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        values = (weight * x_min + bias, weight * x_max + bias)
        lower, upper = min(values), max(values)
        rows.append({
            "neuron": index,
            "weight_fp16": weight,
            "bias_fp16": bias,
            "affine_min": lower,
            "affine_max": upper,
            "breakpoint": (-bias / weight) if weight else None,
            "always_nonpositive": upper <= 0.0,
            "always_nonnegative": lower >= 0.0,
            "classification": (
                "identically_zero_after_relu" if upper <= 0.0 else
                "relu_removable" if lower >= 0.0 else
                "sign_changes_on_domain"
            ),
        })
    result = {
        "schema_version": 1,
        "domain": [x_min, x_max],
        "coefficient_source": str(args.header),
        "rows": rows,
        "removable_neurons": [row["neuron"] for row in rows if row["always_nonpositive"]],
        "relu_removable_neurons": [row["neuron"] for row in rows if row["always_nonnegative"]],
        "additional_fold_beyond_neuron7": any(
            row["neuron"] < 7 and (row["always_nonpositive"] or row["always_nonnegative"])
            for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
