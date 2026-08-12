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
pipeline_analysis = load("pipeline_analysis", ROOT / "tools/analyze_scna_hmx_bottleneck_pipeline.py")


class ExperimentToolsTest(unittest.TestCase):
    def test_mode_parser_and_profile_abi(self):
        line = ("FIG8_ATTENTION_TIMERS mode=scna-hmx-fp16-d8-two-pass-direct-p scna_engine=3 "
                "profile_version=2 phase=measure iteration=7 scna_pack=1 "
                "scna_hmx_affine_relu=2 scna_reduction=3 scna_unpack=4 scna_transpose=5 "
                "scna_p_store=6 scna_completion_fence=7 scna_hmx_commands=8 "
                "scna_physical_macs=9 scna_useful_macs=10 scna_pipeline_supported=0")
        record = runner.parse_records(line)[0]
        self.assertEqual(record["fields"]["profile_version"], "2")
        self.assertEqual(record["fields"]["scna_engine"], "3")
        self.assertEqual(record["fields"]["scna_unpack"], "4")
        self.assertEqual(record["fields"]["scna_transpose"], "5")
        self.assertEqual(record["fields"]["scna_pipeline_supported"], "0")

    def test_bottleneck_spec_has_all_versioned_modes(self):
        spec = json.loads((ROOT / "experiment_bottleneck_pipeline_spec.json").read_text())
        self.assertEqual(spec["schema_version"], 2)
        self.assertEqual(spec["performance"]["max_start_temperature_span_c"], 5.0)
        for suffix in ("vtranspose", "batch4", "direct-p", "attn-pipeline"):
            self.assertTrue(any(mode.endswith(suffix) for mode in spec["modes"]))

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

    def test_manifest_refresh_preserves_existing_evidence(self):
        previous = {"created_at": "fixed", "custom_gate": {"pass": True}}
        merged = runner.merge_manifest(previous, {"updated_at": "later"})
        self.assertEqual(merged["created_at"], "fixed")
        self.assertTrue(merged["custom_gate"]["pass"])
        self.assertEqual(merged["updated_at"], "later")

    def test_micro_speedup_does_not_pair_reused_sample_ids(self):
        def row(mode, sample, value):
            return {"mode": mode, "sample": sample, "pair_ns_per_vector": value,
                    "status": "pass", "duration_gate": True}
        rows = ([row("scna-hmx-fp16-d8-hybrid-direct-p", i, 2.0) for i in range(30)] +
                [row("scna-hvx-fp16-d8", i, 1.0) for i in range(30)] +
                [row("scna-hvx-fp16-d8", i, 1000.0) for i in range(30)])
        summary = pipeline_analysis.micro_paired_speedups(rows, 1000, 8108)
        comparison = next(x for x in summary if x["candidate"] == "scna-hmx-fp16-d8-hybrid-direct-p" and
                          x["baseline"] == "scna-hvx-fp16-d8")
        self.assertEqual(comparison["baseline_samples"], 60)
        self.assertGreater(comparison["speedup"], 100.0)

    def test_missing_and_failed_log(self):
        self.assertEqual(runner.parse_records("driver failed without structured output"), [])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(analysis.read_jsonl(Path(tmp) / "missing.jsonl"), [])

    def test_bootstrap_is_deterministic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(analysis.bootstrap_median(values, 10000, 8108),
                         analysis.bootstrap_median(values, 10000, 8108))
        self.assertEqual(pipeline_analysis.bootstrap_median(values, 10000, 8108),
                         pipeline_analysis.bootstrap_median(values, 10000, 8108))

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
