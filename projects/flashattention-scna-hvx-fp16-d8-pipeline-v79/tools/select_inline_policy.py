#!/usr/bin/env python3
"""Apply the preregistered inline/noinline rule to single-worker DSP totals."""
import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

KV = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
TIMER = re.compile(r"FIG8_ATTENTION_TIMERS\s+(.*)")


def fields(text):
    out = {}
    for key, value in KV.findall(text):
        try: out[key] = float(value) if "." in value else int(value)
        except ValueError: out[key] = value
    return out


def totals(root, variant):
    values = {}
    for file in sorted(root.glob(f"raw/attention/{variant}_q*_s*.log")):
        grouped = defaultdict(float)
        for line in file.read_text(errors="replace").splitlines():
            match = TIMER.search(line)
            if not match: continue
            d = fields(match.group(1))
            if d.get("phase") == "measure": grouped[int(d["iteration"])] += float(d["profiled_total"])
        for iteration, value in grouped.items(): values[(file.name, iteration)] = value
    return values


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); args = ap.parse_args()
    root = Path(args.run_dir)
    noinline, inline = totals(root, "pair_d8_fma_noinline"), totals(root, "pair_d8_fma_inline")
    keys = sorted(set(noinline) & set(inline))
    # filenames differ by variant: pair by q/session/iteration instead.
    def normalize(data, variant):
        return {(name.replace(variant, "POLICY"), iteration): value for (name, iteration), value in data.items()}
    noinline = normalize(noinline, "pair_d8_fma_noinline"); inline = normalize(inline, "pair_d8_fma_inline")
    keys = sorted(set(noinline) & set(inline))
    ratios = [noinline[k] / inline[k] for k in keys if inline[k] > 0]
    rng = random.Random(20260811)
    boots = sorted(math.exp(statistics.mean(math.log(rng.choice(ratios)) for _ in ratios)) for _ in range(5000)) if ratios else []
    geometric_ratio = math.exp(statistics.mean(map(math.log, ratios))) if ratios else None
    ci95 = [boots[124], boots[4874]] if boots else [None, None]
    winner = "inline" if geometric_ratio is not None and geometric_ratio >= 1.01 and ci95[0] > 1.0 else "noinline"
    result = {"winner": winner, "paired_samples": len(ratios), "noinline_over_inline_geomean": geometric_ratio,
              "paired_bootstrap_ci95": ci95, "rule": "inline only if >=1% faster and CI entirely >1.0; otherwise noinline"}
    (root/"inline_selection.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(winner)


if __name__ == "__main__": main()
