import hashlib
import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class ClosureAuditTests(unittest.TestCase):
    def test_gate0_cardinality(self):
        spec = json.loads((PROJECT / "experiment_spec.json").read_text())
        gate = spec["gate0"]
        count = len(gate["modes"]) * len(gate["masks"]) * len(gate["q"]) * len(gate["kv"]) * len(gate["head_dim"]) * len(gate["seeds"])
        self.assertEqual(count, 2160)
        self.assertEqual(count, gate["expected_cases"])

    def test_rowsum_helper_is_identical(self):
        paths = [PROJECT / "src/htp-ops-lib-main/include/dsp/flash_attn_rowsum.h",
                 PROJECT / "systems/exp_lut_system/htp-ops-lib-main/include/dsp/flash_attn_rowsum.h",
                 PROJECT / "systems/scna_system/htp-ops-lib-main/include/dsp/flash_attn_rowsum.h"]
        hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(len(hashes), 1)

    def test_numeric_block_parser(self):
        runner = load_module("run_closure_audit", PROJECT / "tools/run_closure_audit.py")
        line = "FIG8_NUMERIC_BLOCK seed=29101 mode=baseline block=0 rowsum0_bits=0x4a3c p_scalar_sum_bits=0x41478400"
        value = runner.records(line, {"kind": "matrix"})[0]
        self.assertEqual(value["record_type"], "attention_numeric_block")
        self.assertEqual(value["rowsum0_bits"], "0x4a3c")

    def test_fp16_ulp_reference(self):
        analyzer = load_module("analyze_gate0", PROJECT / "tools/analyze_gate0.py")
        value = struct.unpack("<f", struct.pack("<I", 0x41478400))[0]
        self.assertEqual(analyzer.fp16_bits(value), 0x4A3C)

    def test_decision_thresholds_are_preregistered(self):
        spec = json.loads((PROJECT / "experiment_spec.json").read_text())
        self.assertEqual(spec["decision"]["material_performance_ratio"], [0.98, 1.02])
        self.assertEqual(spec["decision"]["ppl_equivalence_ratio"], [0.995, 1.005])
        self.assertFalse(spec["thermal"]["energy_claims_allowed"])


if __name__ == "__main__":
    unittest.main()
