import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "roofline_report", PROJECT / "tools/generate_roofline_lut_scna_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class RooflineLutScnaTest(unittest.TestCase):
    def test_bootstrap_ratio_is_paired_and_deterministic(self):
        candidate = {(0, 0): 8.0, (0, 1): 12.0, (1, 0): 10.0}
        baseline = {(0, 0): 10.0, (0, 1): 15.0, (1, 0): 10.0}
        first = REPORT.bootstrap_ratio(candidate, baseline, draws=1000)
        second = REPORT.bootstrap_ratio(candidate, baseline, draws=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["pairs"], 3)
        self.assertAlmostEqual(first["ratio"], 1.0)

    def test_calibrated_bandwidth_supersedes_pilot_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            roof = Path(tmp) / "raw/roofline"
            roof.mkdir(parents=True)
            (roof / "bandwidth_sample1.log").write_text(
                "vtcm_bandwidth,vtcm_copy,1,1048576,200,1000,419430400,419.0,GB/s\n")
            (roof / "bandwidth_calibrated_sample1.log").write_text(
                "vtcm_bandwidth,vtcm_copy,1,1048576,6000,50000,12582912000,251.0,GB/s\n")
            rows = REPORT.parse_roofs(Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["iters"], 6000)
            self.assertEqual(rows[0]["metric"], 251.0)


if __name__ == "__main__":
    unittest.main()
