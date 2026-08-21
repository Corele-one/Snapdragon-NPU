import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = PROJECT / "tools/analyze_pipeline_experiment.py"
SPEC = importlib.util.spec_from_file_location("pipeline_analyzer", ANALYZER_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


class PipelineAnalyzerTest(unittest.TestCase):
    def test_bootstrap_ratio_is_paired_and_deterministic(self):
        candidate = {(1, 0): 9.0, (1, 1): 10.0, (2, 0): 11.0}
        baseline = {(1, 0): 10.0, (1, 1): 10.0, (2, 0): 10.0}
        first = ANALYZER.bootstrap_ratio(candidate, baseline, draws=1000)
        second = ANALYZER.bootstrap_ratio(candidate, baseline, draws=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["pairs"], 3)
        self.assertAlmostEqual(first["ratio"], 1.0)

    def test_empty_run_reports_unavailable_without_fabricated_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("raw/micro", "raw/accuracy", "raw/attention", "raw/confirm",
                           "raw/scaling", "raw/diagnostic", "static", "evidence"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            (root / "static/static_metrics.json").write_text("{}\n")
            (root / "static/static_gates.json").write_text('{"schema_version":3,"rows":[]}\n')
            subprocess.run(["python3", str(ANALYZER_PATH), "--run-dir", str(root)], check=True)
            report_path = root / "SCNA_HVX_D8_PIPELINE_V79_REPORT_ZH.md"
            report = report_path.read_text()
            summary = json.loads((root / "summary.json").read_text())
            self.assertIn("正式 Attention 数据不可用", report)
            self.assertIsNone(summary["latency"]["static_d8_ref"]["32"])
            self.assertTrue((root / "figures/01_attention_latency.svg").exists())
            first = (report_path.read_bytes(), (root / "summary.json").read_bytes(),
                     (root / "figures/01_attention_latency.svg").read_bytes())
            subprocess.run(["python3", str(ANALYZER_PATH), "--run-dir", str(root)], check=True)
            second = (report_path.read_bytes(), (root / "summary.json").read_bytes(),
                      (root / "figures/01_attention_latency.svg").read_bytes())
            self.assertEqual(first, second)

    def test_accuracy_gate_requires_every_case(self):
        rows = [{"label": "d7_serial", "pass": 1, "finite": True, "mask_zero": True,
                 "tail_zero": True, "rmse": 0.001, "max_abs": 0.009}]
        self.assertTrue(ANALYZER.accuracy_status(rows, "d7_serial"))
        rows[0]["max_abs"] = 0.02
        self.assertFalse(ANALYZER.accuracy_status(rows, "d7_serial"))


if __name__ == "__main__":
    unittest.main()
