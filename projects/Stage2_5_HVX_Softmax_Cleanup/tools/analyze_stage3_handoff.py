#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PAIR = re.compile(r'(\w+)=([^ ]+)')


def fields(line: str) -> dict[str, str]:
    return dict(PAIR.findall(line))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    per_iteration: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    host: dict[int, int] = {}
    record_counts: dict[int, int] = defaultdict(int)
    for line in args.input.read_text(errors='replace').splitlines():
        if 'phase=measure' not in line:
            continue
        data = fields(line)
        iteration = int(data.get('iteration', '-1'))
        if line.startswith('FIG8_ATTENTION_HOST_TIMING'):
            host[iteration] = int(data['host_elapsed_us'])
        elif line.startswith('FIG8_ATTENTION_TIMERS'):
            record_counts[iteration] += 1
            for key in ('profiled_total', 'q_load', 'k_load', 'qk_dot', 'safe_sm', 'v_load',
                        'core_acc', 'o_scale', 'o_store', 'scna_exp', 'state_update',
                        'rowmax_mem', 'rowsum_reduce'):
                per_iteration[iteration][key] += int(data[key])
    expected = sorted(host)
    if expected != list(range(5)) or any(record_counts[i] != 16 for i in expected):
        raise SystemExit(f'incomplete diagnostic records: host={expected} counts={dict(record_counts)}')
    rows = []
    for i in expected:
        d = per_iteration[i]
        row = {
            'iteration': i,
            'host_us': host[i],
            'T_QK_us': d['q_load'] + d['k_load'] + d['qk_dot'],
            'T_Softmax_us': d['safe_sm'],
            'T_V_load_us': d['v_load'],
            'T_PV_O_update_us': d['core_acc'] + d['o_scale'] + d['o_store'],
            'decode_mask_max_us': d['rowmax_mem'],
            'SCNA_us': d['scna_exp'],
            'state_us': d['state_update'],
            'rowsum_reduce_us': d['rowsum_reduce'],
            'softmax_residual_us': d['safe_sm'] - d['rowmax_mem'] - d['scna_exp'] - d['state_update'],
            'profiled_total_us': d['profiled_total'],
        }
        rows.append(row)
    metrics = {key: {'median': statistics.median(row[key] for row in rows),
                     'min': min(row[key] for row in rows),
                     'max': max(row[key] for row in rows), 'n': len(rows)}
               for key in rows[0] if key != 'iteration'}
    result = {
        'schema_version': 1,
        'artifact': args.input.stem + ' (fine timers)',
        'shape': {'q': 32, 'kv': 4096, 'q_heads': 12, 'kv_heads': 2, 'head_dim': 128, 'workers': 1},
        'iterations': rows,
        'metrics': metrics,
        'dma_wait_us': None,
        'dma_wait_status': 'N/A: production HMX path uses synchronous HVX/HMX loads and l2fetch hints, not Hexagon DMA submission',
        'barrier_wait_points': ['one worker_pool_synctoken_wait after all submitted tasks; duration not separately instrumented'],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
