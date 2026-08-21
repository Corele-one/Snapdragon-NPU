import csv
import importlib.util
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT / "tools/create_artifact_manifest.py"
SPEC = importlib.util.spec_from_file_location("artifact_manifest", MANIFEST_PATH)
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


class PipelineContractTest(unittest.TestCase):
    def test_variant_mapping_matches_preregistered_spec_and_header(self):
        spec = json.loads((PROJECT / "experiment_spec.json").read_text())
        expected = {row["id"]: row["build_id"] for row in spec["kernel_implementations"]}
        header = (PROJECT / "src/htp-ops-lib-main/include/op_reg.h").read_text()
        observed = {}
        for name, value in re.findall(r"#define SCNA_KERNEL_IMPL_([A-Z0-9_]+) (\d+)", header):
            if name != "COUNT":
                observed[name.lower()] = int(value)
        self.assertEqual(expected, observed)

    def test_result_schema_v3_contains_pair_and_quad_fields(self):
        header = (PROJECT / "src/htp-ops-lib-main/include/op_reg.h").read_text()
        source = (PROJECT / "src/htp-ops-lib-main/src/dsp/ops/scna_exp2.c").read_text()
        for field in ("schema_version", "kernel_impl", "quad_elapsed_us", "quad_pair_mismatches"):
            self.assertIn(field, header)
        self.assertIn(".schema_version = 3", source)
        self.assertIn(".quad_pair_mismatches = quad_pair_mismatches", source)

    def test_static_gate_rejects_strict_caller_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = {
                "static_d8_ref": {"spill_memory": 0, "stack_frame_bytes": 0,
                                  "caller": {"spill_memory": 10, "stack_frame_bytes": 64}},
                "d7_pairret_inline": {"spill_memory": 0, "stack_frame_bytes": 0,
                                      "caller": {"spill_memory": 10, "stack_frame_bytes": 192}},
                "d7_prebroadcast": {"spill_memory": 0, "stack_frame_bytes": 0,
                                    "caller": {"spill_memory": 11, "stack_frame_bytes": 64}},
            }
            source = root / "metrics.json"; source.write_text(json.dumps(metrics))
            json_out = root / "gates.json"; csv_out = root / "gates.csv"
            subprocess.run(["python3", str(PROJECT / "tools/evaluate_static_gates.py"),
                            "--metrics", str(source), "--json-out", str(json_out),
                            "--csv-out", str(csv_out)], check=True)
            rows = {row["kernel_impl"]: row for row in json.loads(json_out.read_text())["rows"]}
            self.assertEqual(rows["static_d8_ref"]["static_pass"], 1)
            self.assertIn("caller_stack_increase", rows["d7_pairret_inline"]["reasons"])
            self.assertIn("caller_spill_increase", rows["d7_prebroadcast"]["reasons"])

    def test_artifact_manifest_validates_unique_hashes_and_mv79(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, name in enumerate(("static_d8_ref", "d7_serial")):
                directory = root / name; directory.mkdir()
                (directory / "libhtp_ops_skel.so").write_bytes(f"artifact-{index}".encode())
                (directory / "build_id.txt").write_text(
                    f"runtime_variant=pair_static_d8\nkernel_impl={name}\nkernel_impl_id={index}\n")
                (directory / "compile_flags.txt").write_text("FLAGS=-O2 -mv79\n")
            manifest = MANIFEST.build_manifest(root)
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual([row["kernel_impl_id"] for row in manifest["artifacts"]], [1, 0])
            self.assertEqual(len({row["sha256"] for row in manifest["artifacts"]}), 2)


if __name__ == "__main__":
    unittest.main()
