from pathlib import Path
import importlib.util
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("analyze_results", Path(__file__).parents[1] / "tools/analyze_results.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalyzeResultsTest(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(MODULE.scalar("0x10"), 16)
        self.assertAlmostEqual(MODULE.scalar("0.002"), 0.002)
        self.assertEqual(MODULE.scalar("PASS"), "PASS")

    def test_marker_and_total_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            raw = run / "raw/micro"
            raw.mkdir(parents=True)
            (raw / "one.log").write_text(
                "SCNA_SIM_RESULT status=PASS variant=optimized elapsed_us=2 pair_elapsed_us=1 prepare_elapsed_us=0\n"
                "\tTotal: Insns=123 Pcycles=456\n"
            )
            rows = MODULE.parse_logs(run / "raw")
            self.assertEqual(rows["SCNA_SIM_RESULT"][0]["source"], "raw/micro/one.log")
            self.assertEqual(rows["SIMULATOR_TOTAL"][0]["instructions"], 123)
            self.assertEqual(rows["SIMULATOR_TOTAL"][0]["pcycles"], 456)


if __name__ == "__main__":
    unittest.main()
