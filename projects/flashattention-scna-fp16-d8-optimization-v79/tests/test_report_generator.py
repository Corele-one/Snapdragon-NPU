import json
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT / "tools" / "generate_optimization_report.py"
SPEC = PROJECT / "experiment_spec.json"


class ReportGeneratorTest(unittest.TestCase):
    def run_generator(self, root):
        subprocess.run(["python3", str(GENERATOR), "--run-dir", str(root), "--spec", str(SPEC)], check=True)
        return (root / "REPORT.md").read_bytes(), (root / "summary.json").read_bytes()

    def test_missing_and_failed_logs_are_not_fabricated_and_regeneration_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"raw/micro").mkdir(parents=True); (root/"static").mkdir()
            (root/"raw/micro/failed.log").write_text("SCNA Exp2 bench failed: 0x1\n")
            (root/"static/static_metrics.json").write_text("{}\n")
            first = self.run_generator(root); second = self.run_generator(root)
            self.assertEqual(first, second)
            self.assertIn("预注册工程假设，前置证据不足", first[0].decode())
            self.assertIn("#### 假设", first[0].decode())
            self.assertNotIn("Establish the vertical baseline", first[0].decode())
            self.assertGreaterEqual(first[0].decode().count("Key Finding"), 17)
            self.assertEqual(first[0].decode().count("Key Finding（表）："), 7)
            self.assertEqual(first[0].decode().count("Key Finding（图）："), 7)
            self.assertIn("figures/stages/01_stage1_dynamic_row_metrics.svg", first[0].decode())
            self.assertTrue((root / "figures/stages/07_optimized_metrics.svg").exists())
            self.assertTrue((root / "figures/02b_horizontal_baselines_detail.svg").exists())
            self.assertIn("Progressive SCNA Optimization", (root / "figures/01_optimization_ladder.svg").read_text())
            self.assertIn("Absolute Latency by Qo", (root / "figures/stages/01_stage1_dynamic_row_metrics.svg").read_text())
            self.assertIn("去除 Stage1 后的横向细节", first[0].decode())
            self.assertIn("Optimized vs. Origin", first[0].decode())
            self.assertIn("## 异常结果", first[0].decode())
            self.assertIn("## 局限性", first[0].decode())
            self.assertEqual(json.loads(first[1])["correctness"]["cases"], 0)

    def test_resume_logs_and_parent_evidence_are_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"raw/micro").mkdir(parents=True); (root/"static").mkdir()
            for i, ns in enumerate((12.0, 10.0), 1):
                (root/f"raw/micro/stage1_dynamic_row_sample{i}.log").write_text(
                    f"SCNA_EXP_BENCH variant=stage1_dynamic_row paired_ns_per_64={ns}\n")
            (root/"raw/micro/prepare_once_row_sample1.log").write_text(
                "SCNA_EXP_BENCH variant=prepare_once_row paired_ns_per_64=8.0\n")
            (root/"static/static_metrics.json").write_text(json.dumps({
                "stage1_dynamic_row": {"branches": 2, "calls": 8, "splat": 16, "qf16": 24,
                                       "spill_memory": 0, "stack_frame_bytes": 16, "instructions": 60,
                                       "packets": 20, "code_bytes": 240}
            }))
            report, summary = self.run_generator(root)
            data = json.loads(summary)
            self.assertEqual(data["micro"]["stage1_dynamic_row"]["n"], 2)
            self.assertIn("splat=16", report.decode())


if __name__ == "__main__":
    unittest.main()
