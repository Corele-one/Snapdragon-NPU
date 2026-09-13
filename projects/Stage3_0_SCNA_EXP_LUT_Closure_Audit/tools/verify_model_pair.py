#!/usr/bin/env python3
"""Stream and verify the logical correspondence of a CPU/HMX GGUF pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from model_lineage import (file_record, hmx_inverse, is_hmx_matmul, sha256,
                           tensor_metrics, unrepack_iq4_nl_hvx, unrepack_q8_0_hvx)


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "work/llama.cpp-stage29/gguf-py"))


def load_gguf():
    try:
        import gguf  # type: ignore
    except ImportError as exc:
        raise RuntimeError("run this verifier with generated/model_pair_venv/bin/python") from exc
    return gguf


def index(reader) -> dict[str, object]:
    result = {tensor.name: tensor for tensor in reader.tensors}
    if len(result) != len(reader.tensors):
        raise RuntimeError("duplicate GGUF tensor name")
    return result


def tensor_descriptor(tensor) -> dict[str, object]:
    return {"name": tensor.name, "type": tensor.tensor_type.name,
            "shape": [int(x) for x in tensor.shape], "offset": int(tensor.data_offset),
            "bytes": int(tensor.n_bytes)}


def dequant(gguf, tensor, *, hmx_repacked: bool) -> np.ndarray:
    data = tensor.data
    if hmx_repacked and tensor.tensor_type.name == "Q8_0":
        data = unrepack_q8_0_hvx(data).reshape(data.shape)
    elif hmx_repacked and tensor.tensor_type.name == "IQ4_NL":
        data = unrepack_iq4_nl_hvx(data).reshape(data.shape)
    return gguf.quants.dequantize(data, tensor.tensor_type)


def append(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=PROJECT / "model_repair_spec.json")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.resolve().read_text())
    model_dir = (args.model_dir or PROJECT / spec["outputs"]["root"]).resolve()
    manifest_path = (args.manifest or model_dir / spec["outputs"]["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repair_spec", {}).get("sha256") != sha256(args.spec.resolve()):
        raise RuntimeError("model manifest was created from a different repair spec hash")
    paths = {name: Path(value["path"]) for name, value in manifest["models"].items()}
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != manifest["models"][name]["sha256"]:
            raise RuntimeError(f"manifest hash mismatch for {name}: {path}")
    if manifest["lineage"]["cpu_quant_input_sha256"] != manifest["models"]["cpu_f16"]["sha256"]:
        raise RuntimeError("CPU quant lineage does not name the generated CPU F16 input")
    if manifest["lineage"]["hmx_quant_input_sha256"] != manifest["models"]["hmx_f16"]["sha256"]:
        raise RuntimeError("HMX quant lineage does not name the generated HMX F16 input")

    gguf = load_gguf()
    readers = {name: gguf.GGUFReader(path) for name, path in paths.items()}
    tensors = {name: index(reader) for name, reader in readers.items()}
    expected_names = set(tensors["cpu_f16"])
    for branch in tensors:
        if set(tensors[branch]) != expected_names:
            missing = sorted(expected_names - set(tensors[branch]))
            extra = sorted(set(tensors[branch]) - expected_names)
            raise RuntimeError(f"tensor set mismatch in {branch}; missing={missing}, extra={extra}")

    output = model_dir / spec["outputs"]["tensor_validation"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite validation evidence: {output}")
    suffixes = tuple(spec["hmx_matmul_tensor_suffixes"])
    failed: list[str] = []
    summaries: list[dict[str, object]] = []
    for name in sorted(expected_names):
        cpu = tensors["cpu_f16"][name]
        hmx = tensors["hmx_f16"][name]
        base = {"record_type": "f16_tensor", "tensor": name,
                "cpu": tensor_descriptor(cpu), "hmx": tensor_descriptor(hmx),
                "is_hmx_matmul": is_hmx_matmul(name, suffixes)}
        if list(cpu.shape) != list(hmx.shape) or cpu.tensor_type != hmx.tensor_type:
            base.update({"pass": False, "error": "shape_or_type_mismatch"})
        elif is_hmx_matmul(name, suffixes):
            logical_hmx = hmx_inverse(np.asarray(hmx.data))
            exact = np.array_equal(np.asarray(cpu.data).view(np.uint8), logical_hmx.view(np.uint8))
            base.update({"pass": bool(exact), "inverse_bitwise_equal": bool(exact)})
        else:
            exact = np.array_equal(np.asarray(cpu.data).view(np.uint8), np.asarray(hmx.data).view(np.uint8))
            base.update({"pass": bool(exact), "bitwise_equal": bool(exact)})
        append(output, base)
        if not base["pass"]:
            failed.append(name)

    for branch in ("cpu_quant", "hmx_quant"):
        quant_index = tensors[branch]
        f16_index = tensors["cpu_f16"]
        branch_source_name = "hmx_f16" if branch == "hmx_quant" else "cpu_f16"
        for name in sorted(expected_names):
            qt = quant_index[name]
            source = f16_index[name]
            record: dict[str, object] = {
                "record_type": "quant_tensor", "branch": branch, "tensor": name,
                "quant": tensor_descriptor(qt),
                "source_f16_sha256": manifest["models"][branch_source_name]["sha256"],
                "branch_input_f16_sha256": manifest["models"][branch_source_name]["sha256"],
                "logical_reference_f16_sha256": manifest["models"]["cpu_f16"]["sha256"],
                "is_hmx_matmul": is_hmx_matmul(name, suffixes),
            }
            if list(qt.shape) != list(source.shape):
                record.update({"pass": False, "error": "shape_mismatch"})
            elif qt.tensor_type != source.tensor_type:
                # Type 514 also quantizes the token embedding as Q6_K.  Treat
                # every genuinely quantized tensor as a logical-value check;
                # HMX byte unrepacking is only applicable to matmul IQ4/Q8.
                actual = dequant(gguf, qt, hmx_repacked=(branch == "hmx_quant" and is_hmx_matmul(name, suffixes)))
                if branch == "hmx_quant" and is_hmx_matmul(name, suffixes):
                    actual = hmx_inverse(actual)
                metrics = tensor_metrics(np.asarray(source.data, dtype=np.float32), actual)
                record.update(metrics)
                record["pass"] = metrics["nonfinite_count"] == 0 and metrics["cosine"] >= 0.90
            else:
                # Non-quantized tensors must remain byte-identical to their own F16 branch.
                branch_source = tensors["hmx_f16" if branch == "hmx_quant" else "cpu_f16"][name]
                exact = qt.tensor_type == branch_source.tensor_type and np.array_equal(
                    np.asarray(qt.data).view(np.uint8), np.asarray(branch_source.data).view(np.uint8))
                record.update({"pass": bool(exact), "bitwise_equal_to_branch_f16": bool(exact)})
            append(output, record)
            summaries.append(record)
            if not record["pass"]:
                failed.append(f"{branch}:{name}")

    summary = {"record_type": "model_pair_validation", "pass": not failed,
               "tensor_count": len(expected_names), "failed_count": len(failed),
               "failed": failed[:64]}
    append(output, summary)
    manifest["tensor_validation"] = {**summary, "validation_jsonl": file_record(output)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
