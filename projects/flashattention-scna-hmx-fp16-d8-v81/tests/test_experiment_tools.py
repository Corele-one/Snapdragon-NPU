from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


runner = load("runner", ROOT / "scripts/run_scna_hmx_v81_experiment.py")
analysis = load("analysis", ROOT / "tools/analyze_scna_hmx_v81.py")


class ExperimentToolsTest(unittest.TestCase):
    def test_mode_parser_and_profile_abi(self):
        line = ("FIG8_ATTENTION_TIMERS mode=scna-hmx-fp16-d8-two-pass scna_engine=3 "
                "profile_version=1 phase=measure iteration=7 scna_pack=1 "
                "scna_hmx_affine_relu=2 scna_reduction=3 scna_unpack=4")
        record = runner.parse_records(line)[0]
        self.assertEqual(record["fields"]["profile_version"], "1")
        self.assertEqual(record["fields"]["scna_engine"], "3")
        self.assertEqual(record["fields"]["scna_unpack"], "4")

    def test_latin_square(self):
        modes = ["a", "b", "c", "d", "e"]
        orders = [runner.latin_order(modes, index) for index in range(5)]
        for position in range(5):
            self.assertEqual({order[position] for order in orders}, set(modes))

    def test_mixed_binary_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "mixed/stale binary rejected"):
            runner.ensure_digest_match({"dsp.so": "expected"}, {"dsp.so": "wrong"})
        with self.assertRaisesRegex(RuntimeError, "mixed/stale binary rejected"):
            runner.ensure_digest_match({"dsp.so": "expected"}, {})

    def test_missing_and_failed_log(self):
        self.assertEqual(runner.parse_records("driver failed without structured output"), [])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(analysis.read_jsonl(Path(tmp) / "missing.jsonl"), [])

    def test_bootstrap_is_deterministic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(analysis.bootstrap_median(values, 10000, 8108),
                         analysis.bootstrap_median(values, 10000, 8108))

    def test_quick_report_rebuild_is_deterministic_when_fixture_exists(self):
        fixture = ROOT / "results/plumbing_all_quick"
        if not (fixture / "manifest.json").is_file():
            self.skipTest("quick device fixture not present")
        manifest = json.loads((fixture / "manifest.json").read_text())
        rows = analysis.read_jsonl(fixture / "raw/performance.jsonl")
        first = analysis.summarize_performance(rows, 1000, int(manifest["spec"]["seed"]))
        second = analysis.summarize_performance(rows, 1000, int(manifest["spec"]["seed"]))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
