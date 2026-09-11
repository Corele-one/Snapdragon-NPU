#!/usr/bin/env python3
"""Shared model-lineage primitives for the Stage 2.9.1 repair gate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MATMUL_SUFFIXES = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
)


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def command_output(argv: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(argv, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def git_commit(path: Path) -> str:
    return command_output(["git", "rev-parse", "HEAD"], path)


def git_lfs_oid(repo: Path, relative: str) -> tuple[str, int]:
    blob = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=repo, check=True,
                          stdout=subprocess.PIPE).stdout.decode("utf-8", "replace")
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", blob, re.MULTILINE)
    size = re.search(r"^size (\d+)$", blob, re.MULTILINE)
    if not oid or not size:
        raise RuntimeError(f"{relative} is not a Git-LFS pointer in HEAD")
    return oid.group(1), int(size.group(1))


def build_id(path: Path) -> str:
    text = command_output(["readelf", "-n", str(path)])
    match = re.search(r"Build ID:\s*([0-9a-f]+)", text)
    if not match:
        raise RuntimeError(f"ELF build-id missing: {path}")
    return match.group(1)


def is_hmx_matmul(name: str, suffixes: Iterable[str] = MATMUL_SUFFIXES) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


def hmx_permute(logical: np.ndarray) -> np.ndarray:
    """Map a logical [n,k] row-major FP tensor to the HMX tiled byte order."""
    if logical.ndim != 2:
        raise ValueError(f"expected rank 2, got {logical.shape}")
    n, k = logical.shape
    if n % 32 or k % 32:
        raise ValueError(f"HMX tensor dimensions must be 32-aligned: {logical.shape}")
    return (logical.reshape(n // 32, 32, k // 32, 32)
            .transpose(0, 2, 1, 3)
            .reshape(n // 32, k // 32, 32, 16, 2)
            .transpose(0, 1, 3, 2, 4)
            .reshape(n, k))


def hmx_inverse(permuted: np.ndarray) -> np.ndarray:
    """Invert :func:`hmx_permute` for a [n,k] tensor."""
    if permuted.ndim != 2:
        raise ValueError(f"expected rank 2, got {permuted.shape}")
    n, k = permuted.shape
    if n % 32 or k % 32:
        raise ValueError(f"HMX tensor dimensions must be 32-aligned: {permuted.shape}")
    return (permuted.reshape(n // 32, k // 32, 16, 32, 2)
            .transpose(0, 1, 3, 2, 4)
            .reshape(n // 32, k // 32, 32, 32)
            .transpose(0, 2, 1, 3)
            .reshape(n, k))


def unrepack_q8_0_hvx(data: np.ndarray) -> np.ndarray:
    """Convert packed 8-block HTP Q8_0 groups back to ordinary GGUF blocks."""
    flat = np.asarray(data, dtype=np.uint8).reshape(-1)
    if flat.size % 272:
        raise ValueError("repacked Q8_0 payload is not a multiple of 272 bytes")
    groups = flat.reshape(-1, 272)
    out = np.empty((groups.shape[0], 8, 34), dtype=np.uint8)
    out[:, :, :2] = groups[:, :16].reshape(-1, 8, 2)
    out[:, :, 2:] = groups[:, 16:].reshape(-1, 8, 32)
    return out.reshape(-1)


def unrepack_iq4_nl_hvx(data: np.ndarray) -> np.ndarray:
    """Invert repack_q4_0_super_block_hvx for IQ4_NL/Q4_0 payloads."""
    flat = np.asarray(data, dtype=np.uint8).reshape(-1)
    if flat.size % 144:
        raise ValueError("repacked IQ4_NL payload is not a multiple of 144 bytes")
    groups = flat.reshape(-1, 144)
    out = np.empty((groups.shape[0], 8, 18), dtype=np.uint8)
    out[:, :, :2] = groups[:, :16].reshape(-1, 8, 2)
    packed = groups[:, 16:]
    values = np.empty((groups.shape[0], 256), dtype=np.uint8)
    values[:, 0:64] = packed[:, 0::2] & 15
    values[:, 64:128] = packed[:, 1::2] & 15
    values[:, 128:192] = packed[:, 0::2] >> 4
    values[:, 192:256] = packed[:, 1::2] >> 4
    values = values.reshape(-1, 8, 32)
    out[:, :, 2:] = values[:, :, :16] | (values[:, :, 16:] << 4)
    return out.reshape(-1)


def tensor_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    got = np.asarray(actual, dtype=np.float32).reshape(-1)
    if ref.shape != got.shape:
        raise ValueError(f"metric shape mismatch: {ref.shape} != {got.shape}")
    finite = np.isfinite(got)
    nonfinite = int((~finite).sum())
    safe = np.where(finite, got, 0.0)
    delta = safe.astype(np.float64) - ref.astype(np.float64)
    ref64 = ref.astype(np.float64)
    sq = float(np.dot(delta, delta))
    ref_sq = float(np.dot(ref64, ref64))
    max_abs = float(np.max(np.abs(delta), initial=0.0))
    ref_max = float(np.max(np.abs(ref64), initial=0.0))
    denom = float(np.linalg.norm(ref64) * np.linalg.norm(safe.astype(np.float64)))
    return {
        "elements": int(ref.size),
        "rmse": float(np.sqrt(sq / max(1, ref.size))),
        "relative_l2": float(np.sqrt(sq / max(ref_sq, 1e-30))),
        "max_abs": max_abs,
        "normalized_max_abs": max_abs / max(ref_max, 1e-6),
        "cosine": float(np.dot(ref64, safe.astype(np.float64)) / denom) if denom else 1.0,
        "nonfinite_count": nonfinite,
    }


def stable_case_id(fields: dict[str, Any]) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def environment_snapshot(names: Iterable[str]) -> dict[str, str]:
    return {name: os.environ[name] for name in sorted(names) if name in os.environ}
