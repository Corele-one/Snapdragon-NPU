from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_combined_resources import BUFFER_FIELDS, parse_resources, resource_workload_match, validate_native

def line(marker, **kw): return marker+" "+" ".join(f"{k}={v}" for k,v in kw.items())+"\n"
def fixture(tmp_path, duplicate=False, wrong_shape=False, zero_pmu=False):
    root=tmp_path/"raw/resource"; root.mkdir(parents=True); log=root/"serial_optimized_q32.log"
    stages=[("q_load","MEMORY"),("k_load","MEMORY"),("v_load","MEMORY"),("qk_dot","HMX"),("safe_sm","HVX"),("core_acc","HVX"),("core_acc","HMX"),("o_scale","HMX"),("o_store","STORE")]
    qo=31 if wrong_shape else 32
    text=line("ATTENTION_RESOURCE_HEADER",status="PASS",mode="serial",variant="optimized",build_id=6,iteration=0,qo=qo,kv=64,heads=12,kv_heads=2,head_dim=128,requested_workers=1,active_workers=1,hvx_contexts=6,vtcm_worker_cap=8,tasks=16,q_task_rows=4,figure_events=208,figure_overflow=0,llm_events=9,llm_overflow=0)
    if duplicate: text+=text
    for i,(s,u) in enumerate(stages): text+=line("ATTENTION_RESOURCE_EVENT",index=i,stage=s,stage_id=i,unit=u,unit_id=i,worker=0,block_r=0,block_c=0,chunk_r=1,chunk_c=1,value_kind="logical_ops" if u=="HMX" else "logical_bytes",value=128,t0_us=0,t1_us=1,duration_us=1)
    vals={f:0 for f in BUFFER_FIELDS}; vals["q_bytes"]=128
    text+=line("ATTENTION_VTCM_LAYOUT",worker=0,kv_head=0,blk_r=6,blk_c=64,g_br=64,**vals,used_bytes=128,reserved_bytes=1048576,vtcm_total_bytes=8388608)
    text+=line("SCNA_RESOURCE_SUMMARY",mode="serial",variant="optimized",build_id=6,figure_scna_events=208,pair_calls=192,online_calls=16,logical_calls=208,total_duration_us=116)
    text+=line("ATTENTION_TIMER",status="PASS",mode="serial",variant="optimized",qo=qo,kv=64,heads=12,kv_heads=2,head_dim=128,kernel_us=1)
    text+=line("ATTENTION_VERIFY",status="PASS",mode="serial",variant="optimized",qo=qo,kv=64,heads=12,kv_heads=2,head_dim=128,rmse=0,max_abs=0)
    text+=line("SIM_PROCESS_RESULT",exit_code=0,wall_ns=1)
    log.write_text(text); return tmp_path

def test_valid_resource_case(tmp_path):
    cases,_,_=parse_resources(fixture(tmp_path)); assert cases["optimized"]["pass"]
def test_duplicate_header_fails(tmp_path):
    cases,_,_=parse_resources(fixture(tmp_path,duplicate=True)); assert not cases["optimized"]["gates"]["single_header"]
def test_vtcm_sum_error_fails(tmp_path):
    root=fixture(tmp_path); p=root/"raw/resource/serial_optimized_q32.log"; p.write_text(p.read_text().replace("used_bytes=128","used_bytes=129")); cases,_,_=parse_resources(root); assert not cases["optimized"]["gates"]["vtcm_layout_valid"]
def test_scna_nested_is_not_added_to_stage_total(tmp_path):
    cases,_,_=parse_resources(fixture(tmp_path)); assert "scna_exp" not in cases["optimized"]["stages"]
def test_shape_is_preserved_for_outer_gate(tmp_path):
    cases,_,_=parse_resources(fixture(tmp_path,wrong_shape=True)); assert not resource_workload_match(cases)
def test_missing_case_fails_matrix_set(tmp_path):
    cases,_,_=parse_resources(fixture(tmp_path)); assert set(cases) != {"origin","exp-lut",*set(["stage1_dynamic_row","prepare_once_row","pair_shared_dynamic","pair_static_d8","pair_d8_fma_noinline","pair_d8_fma_inline","optimized"])}
def test_zero_pmu_is_unavailable_not_no_access(tmp_path):
    (tmp_path/"evidence").mkdir(); (tmp_path/"raw/native").mkdir(parents=True); (tmp_path/"metrics/native/pcfilter").mkdir(parents=True); (tmp_path/"metrics/native/unfiltered").mkdir(parents=True)
    (tmp_path/"evidence/native_pc_range.txt").write_text("pc_start=0x1000\npc_end=0x1fff\nexit_code=0\n")
    for name in ("memtrace","pctrace"): (tmp_path/f"metrics/native/pcfilter/{name}.txt").write_text("PC=00001000\n")
    (tmp_path/"metrics/native/pcfilter/coproctrace.txt").write_text("")
    (tmp_path/"raw/native/optimized_unfiltered_timing.log").write_text("SIM_PROCESS_RESULT exit_code=143 wall_ns=1\n")
    for name in ("pmu","packet_analyze"): (tmp_path/f"metrics/native/unfiltered/{name}.txt").write_text("")
    result=validate_native(tmp_path); assert result["status"]=="UNAVAILABLE" and not result["kernel_attributable"]
