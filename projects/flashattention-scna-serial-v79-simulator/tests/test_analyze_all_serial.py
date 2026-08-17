import copy
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_all_serial import BUILD_IDS, QOS, VARIANTS, dedupe, evaluate_gates, item_id, summarize_group


def timer(scheme, qo, policy="w1", kv=64, heads=12, kvh=2, dim=128):
    mode = scheme if scheme in ("origin", "exp-lut") else "serial"
    row = {"mode": mode, "variant": "none" if mode != "serial" else scheme,
           "build_id": 6 if mode != "serial" else BUILD_IDS[scheme], "requested_workers": 0 if policy == "auto" else 1,
           "active_workers": min(6, ((qo + 3) // 4) * 2, 8) if policy == "auto" else 1,
           "qo": qo, "kv": kv, "heads": heads, "kv_heads": kvh, "head_dim": dim,
           "tail_nonzero": 0, "masked_nonzero": 0, "source": "raw/attention/x.log"}
    for field in ("kernel_us", "profiled_total_us", "q_load_us", "k_load_us", "v_load_us", "qk_dot_us",
                  "safe_sm_us", "core_acc_us", "o_scale_us", "o_store_us", "scna_exp_us", "param_prepare_us"):
        row[field] = 10
    row["safe_sm_us"], row["scna_exp_us"], row["kernel_us"] = 20, 5, 100
    return row


def fixture():
    rows = defaultdict(list); attention = {}; verifies = []
    for scheme in ["origin", "exp-lut", *VARIANTS]:
        for qo in QOS:
            samples = [timer(scheme, qo) for _ in range(5)]; key = (scheme, "w1", qo, 64, 12, 2, 128)
            attention[item_id(key)] = summarize_group(key, samples); rows["ATTENTION_TIMER"].extend(samples)
    for variant in VARIANTS:
        sample = timer(variant, 3, kv=65, heads=2, kvh=1, dim=64); key = (variant, "w1", 3, 65, 2, 1, 64)
        attention[item_id(key)] = summarize_group(key, [sample]); rows["ATTENTION_TIMER"].append(sample)
    for qo in QOS:
        samples = [timer("optimized", qo, "auto") for _ in range(5)]; key = ("optimized", "auto", qo, 64, 12, 2, 128)
        attention[item_id(key)] = summarize_group(key, samples); rows["ATTENTION_TIMER"].extend(samples)
    for variant in VARIANTS: rows["SCNA_SIM_RESULT"].append({"variant": variant, "status": "PASS", "source": "raw/micro/x.log"})
    rows["SIM_CAPABILITY"].append({"status": "PASS"}); rows["SIM_PROCESS_RESULT"].append({"exit_code": 0, "source": "raw/x.log"})
    for item in attention.values(): verifies.append({"status": "PASS", "rmse": .001, "max_abs": .005, "candidate_nonfinite": 0, "reference_nonfinite": 0})
    return rows, attention, verifies


def test_complete_fixture_passes_and_nested_share_is_not_added():
    rows, attention, verifies = fixture(); gates = evaluate_gates(rows, attention, verifies, [])
    assert all(gates.values())
    metric = attention["optimized_w1_q1_kv64_h12_kh2_d128"]["metrics"]
    assert metric["scna_share_safe_sm_percent"] == 25.0
    assert metric["profiled_total_us"]["median"] == 10


def test_missing_variant_fails():
    rows, attention, verifies = fixture()
    del attention["pair_static_d8_w1_q1_kv64_h12_kh2_d128"]
    assert not evaluate_gates(rows, attention, verifies, [])["attention_performance_matrix_complete"]


def test_duplicate_verification_is_reported():
    base = {"mode": "serial", "variant": "optimized", "requested_workers": 1, "qo": 1, "kv": 64,
            "heads": 12, "kv_heads": 2, "head_dim": 128, "tail_check": 0, "source": "a"}
    other = {**base, "source": "b"}; unique, duplicates = dedupe([base, other], "ATTENTION_VERIFY")
    assert len(unique) == 1 and len(duplicates) == 1


def test_wrong_build_id_and_failed_case_fail_gates():
    rows, attention, verifies = fixture(); broken = next(row for row in rows["ATTENTION_TIMER"] if row["mode"] == "serial")
    broken["build_id"] = 99; verifies[0] = {**verifies[0], "status": "FAIL"}
    gates = evaluate_gates(rows, attention, verifies, [])
    assert not gates["attention_variant_build_ids_match"]
    assert not gates["attention_all_verifications_pass"]
