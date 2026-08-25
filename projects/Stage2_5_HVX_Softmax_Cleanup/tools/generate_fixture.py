#!/usr/bin/env python3
"""Generate the canonical Stage 2.5 attention fixture format."""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
from pathlib import Path


MAGIC = b"S25FIX01"
VERSION = 1
HEADER = struct.Struct("<8s10I4Q")
MASK_IDS = {"full": 0, "causal": 1, "padding": 2}


def mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def f16(value: float) -> bytes:
    return struct.pack("<e", value)


def seed_value(label: str) -> int:
    if label == "figure8_fixed":
        return 0
    return int(label, 0) & 0xFFFFFFFF


def make_payload(qo: int, kv: int, n_heads: int, n_kv_heads: int, dim: int,
                 mask_mode: str, seed_label: str) -> tuple[bytes, bytes, bytes, bytes]:
    seed = seed_value(seed_label)
    q = bytearray()
    k = bytearray()
    v = bytearray()
    mask = bytearray()
    q_elems = qo * n_heads * dim
    kv_elems = kv * n_kv_heads * dim
    if seed == 0:
        for i in range(q_elems):
            q.extend(struct.pack("<f", (((i * 13 + 7) % 251) - 125) * 0.0078125))
        for i in range(kv_elems):
            k.extend(f16((((i * 17 + 3) % 257) - 128) * 0.00390625))
            v.extend(f16((((i * 19 + 5) % 263) - 131) * 0.00390625))
    else:
        for i in range(q_elems):
            value = (mix32(seed ^ i) & 0xFFFF) - 32768
            q.extend(struct.pack("<f", value / 32768.0))
        for i in range(kv_elems):
            kval = (mix32(seed ^ 0x51ED270B ^ i) & 0xFFFF) - 32768
            vval = (mix32(seed ^ 0x9E3779B9 ^ i) & 0xFFFF) - 32768
            k.extend(f16(kval / 65536.0))
            v.extend(f16(vval / 65536.0))
    kv_pad = math.ceil(kv / 64) * 64
    for q_idx in range(qo):
        absolute_q = kv - qo + q_idx
        for k_idx in range(kv_pad):
            masked = k_idx >= kv
            if mask_mode == "causal":
                masked = masked or k_idx > absolute_q
            elif mask_mode == "padding":
                masked = masked or k_idx >= kv - 3
            mask.extend(f16(-65504.0 if masked else 0.0))
    return bytes(q), bytes(k), bytes(v), bytes(mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qo-len", required=True, type=int)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--mask-mode", choices=MASK_IDS, default="full")
    parser.add_argument("--seed", default="figure8_fixed")
    args = parser.parse_args()
    if args.n_heads % args.n_kv_heads:
        parser.error("n-heads must be divisible by n-kv-heads")
    q, k, v, mask = make_payload(args.qo_len, args.kv_len, args.n_heads,
                                  args.n_kv_heads, args.head_dim, args.mask_mode, args.seed)
    header = HEADER.pack(
        MAGIC, VERSION, args.qo_len, args.kv_len, math.ceil(args.kv_len / 64) * 64,
        args.n_heads, args.n_kv_heads, args.head_dim, MASK_IDS[args.mask_mode],
        seed_value(args.seed), 0, len(q), len(k), len(v), len(mask),
    )
    payload = header + q + k + v + mask
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"{args.output} bytes={len(payload)} sha256={hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()

