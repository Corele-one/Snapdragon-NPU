#!/usr/bin/env python3
"""Exhaustively audit scalar FP16 models of the SCNA d8 rewrites.

This is a host-side gate over all 65,536 binary16 bit patterns.  It does not
claim to emulate Hexagon qf16 instruction rounding; the DSP microbenchmark is
the authority for emitted-code accuracy.
"""

import argparse
import json
import math
import struct
from pathlib import Path

W = [2.586841583251953e-05, 0.00010031461715698242, 0.0004892349243164062,
     0.002384185791015625, 0.011627197265625, 0.05670166015625,
     0.2763671875, 0.345458984375]
B = [0.0004138946533203125, 0.0013751983642578125, 0.005588531494140625,
     0.0218048095703125, 0.0797119140625, 0.25927734375, 0.6318359375, 0.0]
BREAKPOINT = [-16.0, -13.7109375, -11.421875, -9.1484375,
              -6.85546875, -4.57421875, -2.287109375]
PREFIX_W = [2.586841583251953e-05, 0.00012612342834472656,
            0.0006155967712402344, 0.0030002593994140625,
            0.01462554931640625, 0.07135009765625, 0.34765625]
PREFIX_B = [0.0004138946533203125, 0.001789093017578125,
            0.00737762451171875, 0.0291748046875, 0.10888671875,
            0.3681640625, 1.0]


def f16(value):
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def from_bits(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def clamp(x):
    if x < -256.0:
        return -256.0
    if x > 0.0:
        return 0.0
    return x


def terms(x):
    x = clamp(x)
    return [max(0.0, f16(f16(B[i]) + x * f16(W[i]))) for i in range(8)]


def serial(x):
    total = 0.0
    for value in terms(x):
        total = f16(total + value)
    return total


def tree(x):
    value = terms(x)[:7]
    partial = [f16(value[0] + value[4]), f16(value[1] + value[5]),
               f16(value[2] + value[6]), value[3]]
    return f16(f16(partial[0] + partial[1]) + f16(partial[2] + partial[3]))


def piecewise(x):
    x = clamp(x)
    slope = intercept = 0.0
    for threshold, prefix_w, prefix_b in zip(BREAKPOINT, PREFIX_W, PREFIX_B):
        if x > threshold:
            slope, intercept = prefix_w, prefix_b
    return max(0.0, f16(intercept + x * slope))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    args = parser.parse_args()

    finite = []
    nonfinite_inputs = 0
    dead_nonzero = 0
    tree_max = piecewise_max = 0.0
    tree_sq = piecewise_sq = 0.0
    tree_changed = piecewise_changed = 0
    for bits in range(1 << 16):
        x = from_bits(bits)
        if not math.isfinite(x):
            nonfinite_inputs += 1
            continue
        expected = serial(x)
        tree_value = tree(x)
        piecewise_value = piecewise(x)
        dead_nonzero += terms(x)[7] != 0.0
        tree_error = abs(tree_value - expected)
        piecewise_error = abs(piecewise_value - expected)
        tree_max = max(tree_max, tree_error)
        piecewise_max = max(piecewise_max, piecewise_error)
        tree_sq += tree_error * tree_error
        piecewise_sq += piecewise_error * piecewise_error
        tree_changed += tree_error != 0.0
        piecewise_changed += piecewise_error != 0.0
        finite.append((x, tree_value, piecewise_value))

    unique = {}
    for x, tv, pv in finite:
        unique[x] = (tv, pv)
    monotonic_tree = monotonic_piecewise = 0
    previous_tree = previous_piecewise = -math.inf
    for x in sorted(unique):
        tv, pv = unique[x]
        monotonic_tree += tv < previous_tree
        monotonic_piecewise += pv < previous_piecewise
        previous_tree, previous_piecewise = tv, pv

    count = len(finite)
    result = {
        "schema_version": 1,
        "all_fp16_patterns": 1 << 16,
        "finite_inputs": count,
        "nonfinite_inputs_excluded_from_accuracy": nonfinite_inputs,
        "dead_neuron_nonzero_count": dead_nonzero,
        "qf16_tree": {
            "changed_outputs": tree_changed,
            "rmse_vs_serial_fp16": math.sqrt(tree_sq / count),
            "max_abs_vs_serial_fp16": tree_max,
            "monotonic_violations": monotonic_tree,
        },
        "piecewise_d8": {
            "changed_outputs": piecewise_changed,
            "rmse_vs_serial_fp16": math.sqrt(piecewise_sq / count),
            "max_abs_vs_serial_fp16": piecewise_max,
            "monotonic_violations": monotonic_piecewise,
        },
    }
    result["pass"] = (dead_nonzero == 0 and monotonic_tree == 0 and monotonic_piecewise == 0 and
                      tree_max <= 0.01 and piecewise_max <= 0.01)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
