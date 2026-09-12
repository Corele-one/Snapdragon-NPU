from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools/analyze_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_results", MODULE_PATH)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


class AnalyzeResultsTest(unittest.TestCase):
    def make_tree(self, root: Path) -> None:
        for label, mode in ANALYZER.LABEL_MODE.items():
            directory = root / label
            directory.mkdir()
            for q in ANALYZER.Q_VALUES:
                for session in range(1, 6):
                    lines = [f"FIG8_ATTENTION_CONFIG mode={mode} qo_len={q}"]
                    lines += [
                        f"FIG8_ATTENTION_HOST_TIMING mode={mode} qo_len={q} phase=measure "
                        f"iteration={iteration} host_elapsed_us={100 + q + iteration} ret=0"
                        for iteration in range(20)
                    ]
                    (directory / f"q{q}_s{session}.log").write_text("\n".join(lines) + "\n")
        diagnostic = root / "diagnostic"
        diagnostic.mkdir()
        for label, mode in ANALYZER.LABEL_MODE.items():
            lines = []
            for iteration in range(3):
                lines.append(
                    f"FIG8_ATTENTION_TIMERS mode={mode} phase=measure iteration={iteration} "
                    + " ".join(f"{name}=1" for name in ANALYZER.COMPONENTS)
                )
                lines.append(
                    f"FIG8_ATTENTION_WORKERS mode={mode} phase=measure iteration={iteration} "
                    "active_workers=2 hvx_contexts=6 vtcm_worker_cap=6 tasks=2 q_task_rows=16"
                )
            (diagnostic / f"{label}_q32.log").write_text("\n".join(lines) + "\n")

    def test_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_tree(root)
            self.assertEqual(ANALYZER.analyze(root)["valid_host_samples"], 1200)

    def test_missing_log_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_tree(root)
            (root / "baseline/q4_s1.log").unlink()
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                ANALYZER.analyze(root)

    def test_nonzero_return_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_tree(root)
            path = root / "scna/q8_s2.log"
            path.write_text(path.read_text().replace("ret=0", "ret=7", 1))
            with self.assertRaisesRegex(ValueError, "nonzero return"):
                ANALYZER.analyze(root)

    def test_mode_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_tree(root)
            path = root / "lut_exp/q16_s3.log"
            path.write_text(path.read_text().replace("mode=lut-exp", "mode=baseline"))
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                ANALYZER.analyze(root)

    def test_incomplete_session_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_tree(root)
            path = root / "baseline/q32_s5.log"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete session"):
                ANALYZER.analyze(root)


if __name__ == "__main__":
    unittest.main()
