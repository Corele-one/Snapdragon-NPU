from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(value)
    return value


analyze = module("analyze_audit", ROOT / "tools/analyze_audit.py")
runner = module("run_device_audit", ROOT / "tools/run_device_audit.py")
model = module("run_model_audit", ROOT / "tools/run_model_audit.py")
lineage = module("model_lineage_test", ROOT / "tools/model_lineage.py")
matmul = module("run_matmul_audit_test", ROOT / "tools/run_matmul_audit.py")
diagnostic = module("summarize_model_chain_failure_test",
                    ROOT / "tools/summarize_model_chain_failure.py")
final_manifest = module("create_final_manifest_test", ROOT / "tools/create_final_manifest.py")


class AuditToolTests(unittest.TestCase):
    def test_preregistered_matrix_sizes(self):
        spec = json.loads((ROOT / "experiment_spec.json").read_text())
        c = spec["attention_correctness"]
        per_mode = len(c["masks"]) * len(c["q"]) * len(c["kv"]) * len(c["head_dim"]) * len(c["seeds"])
        self.assertEqual(per_mode, 180)
        self.assertEqual(per_mode + len(c["long_context_sentinels"]["masks"]) *
                         len(c["long_context_sentinels"]["q"]), 184)
        self.assertEqual(len(runner.Runner(Path("/tmp/dry"), True).performance_cases()), 21)

    def test_parse_device_records(self):
        text = ('VALUE_AUDIT_JSON {"schema_version":1,"evaluator":"lut-exp"}\n'
                'FIG8_ATTENTION_HOST_TIMING phase=measure iteration=7 host_elapsed_us=42\n'
                'FIG8_ATTENTION_TIMERS phase=measure iteration=7 profiled_total=31 worker=2\n'
                'FIG8_ATTENTION_COMPARE candidate_mode=scna-fp16 rmse=0.001 max_abs_error=0.002 pass=1\n')
        records = runner.parsed_records(text, {"phase": "attention_performance", "session": 2})
        self.assertEqual(records[0]["record_type"], "nonlinear")
        self.assertEqual(records[0]["audit_phase"], "attention_performance")
        self.assertEqual(records[1]["phase"], "measure")
        self.assertEqual(records[1]["audit_phase"], "attention_performance")
        self.assertEqual(records[2]["record_type"], "attention_timers")
        self.assertEqual(records[2]["profiled_total"], 31)
        self.assertEqual(records[3]["pass"], 1)
        self.assertEqual(records[3]["session"], 2)

    def test_model_bench_warmup_is_discarded_in_same_session(self):
        text = json.dumps({"samples_ts": [1, 2, 3, 4, 5, 10, 11, 12]})
        records = model.parse_bench(text, "lut-exp", "pp", 512, warmup=5, measure=3)
        self.assertEqual(records[0]["warmup_samples_ts"], [1, 2, 3, 4, 5])
        self.assertEqual(records[0]["samples_ts"], [10, 11, 12])

    def test_attention_performance_details_include_dsp_and_quantiles(self):
        records = []
        for mode, host, dsp in (("lut-exp", 100, 80), ("scna-fp16", 110, 88)):
            common = {"phase": "measure", "mask": "causal", "q": 4, "kv": 512,
                      "session": 0, "iteration": 0, "mode": mode}
            records.append({**common, "record_type": "attention_timing", "host_elapsed_us": host})
            records.append({**common, "record_type": "attention_timers", "profiled_total": dsp})
            records.append({**common, "record_type": "resource", "active_workers": 4, "tasks": 8})
        details = analyze.attention_performance_details(records, ("mask", "q", "kv"))
        self.assertEqual(details["host_quantiles"]["causal|4|512|lut-exp"]["p50_us"], 100)
        self.assertAlmostEqual(details["dsp_work_scna_over_lut"]["causal|4|512"]["estimate"], 1.1)
        self.assertEqual(details["worker_task_counts"]["causal|4|512|scna-fp16"]["tasks"], [8])

    def test_thermal_units_and_latency(self):
        records = [
            {"record_type": "thermal_run", "mode": mode, "event": "start", "temperature_c": 30.0}
            for mode in ("lut-exp", "scna-fp16")
        ]
        records += [
            {"record_type": "thermal_batch", "mode": mode, "requests": 1000, "elapsed_us": elapsed}
            for mode, elapsed in (("lut-exp", 2_000_000), ("scna-fp16", 2_500_000))
        ]
        summary = analyze.thermal_summary(records)
        self.assertEqual(summary["modes"]["lut-exp"]["first_throughput"], 500.0)
        self.assertEqual(summary["modes"]["lut-exp"]["p99_request_latency_us"], 2000.0)

    def test_ppl_chunk_reconstruction(self):
        text = "[1]4.0000,[2]8.0000,\nFinal estimate: PPL = 8.0000 +/- 1"
        ppl, chunks = model.extract_ppl(text)
        self.assertEqual(ppl, 8.0)
        self.assertAlmostEqual(chunks[0], math.log(4.0))
        self.assertAlmostEqual(chunks[1], 2 * math.log(8.0) - math.log(4.0))

    def test_model_metadata_pair_validation(self):
        template = ("llama_model_loader: loaded meta data with 2 key-value pairs and 3 tensors "
                    "from model.gguf (version GGUF V3 (latest))\n"
                    "llama_model_loader: - kv   0: general.architecture str = qwen2\n"
                    "llama_model_loader: - kv   1: qwen2.block_count u32 = 28\n"
                    "llama_model_loader: - type f16: 3 tensors\n")
        cpu = model.extract_model_metadata(template)
        hmx = model.extract_model_metadata(template)
        self.assertTrue(model.compare_model_metadata(cpu, hmx)["structural_match"])
        hmx["metadata"]["qwen2.block_count"] = "29"
        self.assertFalse(model.compare_model_metadata(cpu, hmx)["structural_match"])

    def test_bootstrap_is_deterministic(self):
        first = analyze.bootstrap([0.9, 1.0, 1.1], draws=100)
        second = analyze.bootstrap([0.9, 1.0, 1.1], draws=100)
        self.assertEqual(first, second)

    def test_incomplete_evidence_is_invalid(self):
        summary = analyze.evaluate([])
        self.assertEqual(summary["verdict"], "AUDIT_INVALID")
        self.assertFalse(summary["energy_claim_made"])

    def test_llama_bridge_is_exact_and_idempotent_from_archive(self):
        archive = ROOT.parent / "Archived/flashattention-scna-v81/src/llama.cpp-npu-htp-backend/ggml/src/ggml-htp"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "ggml/src/ggml-htp"
            target.mkdir(parents=True)
            (target / "htp-ops.cc").write_bytes((archive / "htp-ops.cc").read_bytes())
            (target / "op_reg.h").write_bytes((archive / "op_reg.h").read_bytes())
            subprocess.run(["python3", str(ROOT / "tools/patch_llama_backend.py"), "--tree", temp,
                            "--audit-op-reg", str(ROOT / "src/htp-ops-lib-main/include/op_reg.h")], check=True)
            patched = (target / "htp-ops.cc").read_text()
            self.assertIn("LLM_NPU_MODE_Q_TASK_ROWS_AUTO", patched)
            self.assertIn("SCNA_VARIANT_PAIR_STATIC_D8 << 10", patched)
            self.assertNotIn("LLM_NPU_MODE_SCNA_INT8", patched)
            self.assertEqual((target / "op_reg.h").read_bytes(),
                             (ROOT / "src/htp-ops-lib-main/include/op_reg.h").read_bytes())

    def test_c_abi_layout(self):
        binary = Path(tempfile.gettempdir()) / "stage29_abi_layout"
        subprocess.run(["cc", "-std=c11", "-I", str(ROOT / "src/htp-ops-lib-main/include"),
                        str(ROOT / "tests/abi_layout.c"), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)

    def test_hmx_permutation_inverse_is_bit_exact(self):
        import numpy as np
        source = np.arange(64 * 96, dtype=np.uint16).view(np.float16).reshape(64, 96)
        restored = lineage.hmx_inverse(lineage.hmx_permute(source))
        self.assertTrue(np.array_equal(source.view(np.uint16), restored.view(np.uint16)))

    def test_quant_hvx_repack_inverse(self):
        import numpy as np
        q8_standard = np.arange(8 * 34, dtype=np.uint8).reshape(8, 34)
        q8_hvx = np.concatenate((q8_standard[:, :2].reshape(-1), q8_standard[:, 2:].reshape(-1)))
        self.assertTrue(np.array_equal(lineage.unrepack_q8_0_hvx(q8_hvx), q8_standard.reshape(-1)))
        iq_standard = np.zeros((8, 18), dtype=np.uint8)
        iq_standard[:, :2] = np.arange(16, dtype=np.uint8).reshape(8, 2)
        values = np.arange(256, dtype=np.uint8).reshape(8, 32) & 15
        iq_standard[:, 2:] = values[:, :16] | (values[:, 16:] << 4)
        flat = values.reshape(-1)
        packed = np.empty(128, dtype=np.uint8)
        for x in range(64):
            packed[2*x] = flat[x] | (flat[x+128] << 4)
            packed[2*x+1] = flat[x+64] | (flat[x+192] << 4)
        iq_hvx = np.concatenate((iq_standard[:, :2].reshape(-1), packed))
        self.assertTrue(np.array_equal(lineage.unrepack_iq4_nl_hvx(iq_hvx), iq_standard.reshape(-1)))

    def test_three_dtype_oracle_paths_cover_qwen_matrix(self):
        repair = json.loads((ROOT / "model_repair_spec.json").read_text())
        self.assertEqual(repair["matmul_audit"]["dtypes"], ["f16", "iq4_nl", "q8_0"])
        self.assertEqual(repair["matmul_audit"]["single_tile_cases"], {
            "f16": [1, 32, 32], "iq4_nl": [1, 256, 32], "q8_0": [1, 256, 32]})
        for m in repair["matmul_audit"]["m"]:
            for k, n in repair["matmul_audit"]["kn"]:
                for dtype in repair["matmul_audit"]["dtypes"]:
                    self.assertTrue(matmul.compatible_paths(dtype, m, k, n))

    def test_failed_case_can_only_be_retried_once(self):
        records = [{"case_id": "x", "attempt": 1, "spec_sha256": "a"},
                   {"case_id": "x", "attempt": 2, "spec_sha256": "a"},
                   {"case_id": "x", "attempt": 2, "spec_sha256": "a"}]
        self.assertTrue(any("one-rerun" in reason for reason in analyze.validate_attempts(records)))
        changed = [{"case_id": "y", "attempt": 1, "model_sha256": "a"},
                   {"case_id": "y", "attempt": 2, "model_sha256": "b"}]
        self.assertTrue(any("frozen hash" in reason for reason in analyze.validate_attempts(changed)))

    def test_completeness_requires_cpu_and_baseline_ppl(self):
        records = [{"record_type": "model_sanity", "pass": True}]
        reasons = analyze.completeness(records)
        self.assertTrue(any("model PPL incomplete for cpu" in reason for reason in reasons))
        self.assertTrue(any("model PPL incomplete for baseline" in reason for reason in reasons))

    def test_model_chain_diagnostic_parsers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ppl = root / "ppl.log"
            ppl.write_text("Final estimate: PPL = 6.3753 +/- 1\n")
            self.assertEqual(diagnostic.parse_ppl(ppl), 6.3753)
            numeric = root / "numeric.log"
            numeric.write_text(
                "FIG8_NUMERIC score_count=32 rowsum0_bits=0x51d9 "
                "p_expected_sum_bits=0x41c8be48 masked_p_nonzero=0 tail_p_nonzero=0\n")
            parsed = diagnostic.parse_numeric(numeric)
            self.assertAlmostEqual(parsed["rowsum_fp16"], 46.78125)
            self.assertAlmostEqual(parsed["expected_probability_sum_fp32"], 25.0929107666)
            self.assertTrue(parsed["masked_and_tail_zero"])

    def test_final_manifest_follows_model_evidence_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "lineage.jsonl"
            evidence.write_text("{}\n")
            nested = {"tensor_validation": {"evidence": {"path": str(evidence), "sha256": "x"}},
                      "device": {"path": "/data/local/tmp/not-local"}}
            self.assertEqual(final_manifest.referenced_local_files(nested), [evidence])


if __name__ == "__main__":
    unittest.main()
