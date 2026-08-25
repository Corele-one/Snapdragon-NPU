#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

MEASUREMENT = re.compile(r'^\{"record":"measurement".*\}$')
CORRECTNESS = re.compile(r'^\{"record":"correctness".*\}$')
GEOMETRY = re.compile(r'(?P<kernel>hmx-pipe|hmx-seq|hvx) Br (?P<br>\d+) Bc (?P<bc>\d+) threads (?P<threads>\d+) vtcm (?P<vtcm>\d+)')


def bootstrap_median(values: list[float], draws: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    medians = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(draws))
    return statistics.median(values), medians[int(0.025 * draws)], medians[int(0.975 * draws)]


def records(path: Path, pattern: re.Pattern[str]) -> list[dict]:
    result = []
    for line in path.read_text(errors='replace').splitlines():
        if pattern.match(line):
            result.append(json.loads(line))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--draws', type=int, default=10_000)
    args = ap.parse_args()
    summary: dict = {'schema_version': 1, 'run_id': args.run_dir.name, 'performance': {}, 'correctness': {}, 'failures': []}
    for path in sorted((args.run_dir / 'raw/performance').glob('*.log')):
        match = re.match(r'(.+)_q(\d+)_s(\d+)\.log$', path.name)
        if not match:
            summary['failures'].append({'path': str(path), 'reason': 'bad filename'})
            continue
        config, q, session = match.group(1), match.group(2), int(match.group(3))
        recs = records(path, MEASUREMENT)
        if not recs:
            summary['failures'].append({'path': str(path), 'reason': 'no measurements'})
            continue
        bucket = summary['performance'].setdefault(config, {}).setdefault(q, {'host_us': [], 'dsp_op_us': [], 'sessions': [], 'kernels': {}, 'geometry': {}})
        bucket['host_us'].extend(float(r['host_us']) for r in recs)
        bucket['dsp_op_us'].extend(float(r['dsp_op_us']) for r in recs if r['dsp_op_us'] != 'N/A')
        bucket['sessions'].append(session)
        for r in recs:
            kernel = r['effective_kernel']
            bucket['kernels'][kernel] = bucket['kernels'].get(kernel, 0) + 1
            gm = GEOMETRY.search(r['profile'])
            if gm:
                key = f"Br={gm['br']},Bc={gm['bc']},threads={gm['threads']},vtcm={gm['vtcm']}"
                bucket['geometry'][key] = bucket['geometry'].get(key, 0) + 1
    for config, qmap in summary['performance'].items():
        for q, bucket in qmap.items():
            for metric in ('host_us', 'dsp_op_us'):
                vals = bucket[metric]
                if vals:
                    med, lo, hi = bootstrap_median(vals, args.draws, 0x25A0 + int(q) + len(config))
                    bucket[metric] = {'median': med, 'ci_low': lo, 'ci_high': hi, 'n': len(vals)}
            bucket['sessions'] = sorted(set(bucket['sessions']))
    for path in sorted((args.run_dir / 'raw/correctness').glob('*.log')):
        match = re.match(r'(.+)_q(\d+)_seed(.+)\.log$', path.name)
        if not match: continue
        measurements = records(path, MEASUREMENT)
        checks = records(path, CORRECTNESS)
        key = f"q{match.group(2)}_seed{match.group(3)}"
        summary['correctness'][key] = {
            'result': checks[-1] if checks else None,
            'effective_kernels': sorted(set(r['effective_kernel'] for r in measurements)),
            'pass': bool(checks and checks[-1].get('pass')),
        }
    summary['correctness_gate_pass'] = bool(summary['correctness']) and all(v['pass'] and v['effective_kernels'] and all(k.startswith('hmx') for k in v['effective_kernels']) for v in summary['correctness'].values())
    recovery = sorted(str(p.relative_to(args.run_dir)) for p in (args.run_dir / 'raw/recovery').glob('*.log'))
    summary['recovery_attempts'] = recovery
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
