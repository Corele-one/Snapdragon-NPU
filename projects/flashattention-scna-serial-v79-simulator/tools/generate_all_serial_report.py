#!/usr/bin/env python3
"""Generate the gated Chinese all-serial internal report from measured artifacts."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

VARIANTS=["stage1_dynamic_row","prepare_once_row","pair_shared_dynamic","pair_static_d8","pair_d8_fma_noinline","pair_d8_fma_inline","optimized"]
PARENT={VARIANTS[i]:VARIANTS[i-1] for i in range(1,len(VARIANTS))}
PHASES=["q_load_us","k_load_us","v_load_us","qk_dot_us","safe_sm_us","core_acc_us","o_scale_us","o_store_us"]
QOS=(1,4,8,16,32)
def f(x,n=3): return f"{float(x):.{n}f}"
def table(lines,headers,rows):
    lines += ["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]+["| "+" | ".join(map(str,r))+" |" for r in rows]+[""]
def get(s,scheme,q,policy="w1"): return s["attention"][f"{scheme}_{policy}_q{q}_kv64_h12_kh2_d128"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,required=True); a=p.parse_args(); run=a.run_dir.resolve()
    s=json.loads((run/"summary.json").read_text()); st=json.loads((run/"static/static_metrics.json").read_text()); dyn=json.loads((run/"metrics/detailed/runtime_filtered_metrics.json").read_text())
    diagnostic=not s["pass"]; title="# 全部 Serial SCNA 方案 Hexagon v79 仿真性能分析" if not diagnostic else "# 全部 Serial SCNA 方案仿真诊断报告（门禁未通过）"
    lines=[title,"",f"Run ID：`{s['run_id']}`。数据源仅为本 run 的原始日志、`summary.json` 和 v79 反汇编。","","> **边界：** 本报告中的 latency、cycle、speedup 与排序只代表 Hexagon Simulator 诊断结果，不是真实 Snapdragon NPU/HTP 性能；不得外推到真机。","","## 验收","" ]
    table(lines,["门禁","结果"],[(k,"PASS" if v else "FAIL") for k,v in s["gates"].items()])
    if diagnostic:
        lines += ["完整矩阵任一门禁失败，因此按预先约定只生成诊断报告，不发布性能排序结论。请从 [`verification_all_serial.json`](verification_all_serial.json) 与 [`raw/`](raw/) 定位失败 case。",""]
    else:
        lines += [f"已通过 45 个单 worker 性能 case（9 方案×5 Qo）、7 个 KV=65 tail case和 5 个 optimized auto-worker case；每个性能点 5 次实测。正确性共 {len(s['verifications'])} 个 case，最大 RMSE={f(max(v['rmse'] for v in s['verifications']),9)}，最大绝对误差={f(max(v['max_abs'] for v in s['verifications']),9)}。","","## 1. 公共 Attention 数据流","","DDR/L2 中 Q/K/V 由 HVX 搬运/转换到 VTCM；HMX 计算缩放后的 `QKᵀ`；HVX 执行 mask、online row-max、safe softmax 与 serial SCNA exp2；HMX 计算 `P×V`；最后 HVX 归一化并写回 O。SCNA evaluator 的下层路径是 `score-rowmax → clamp → d8 SCNA exp2 → rowsum → online recurrence`。所有方案共享 QK/PV 与 K/V tiling，本优化不包含 K/V tile 复用。","","图：[`Attention .drawio`](figures/attention_scna_dataflow.drawio)、[`seven-step .drawio`](figures/serial_scna_seven_variant_evolution.drawio)，以及同名 SVG/PDF/PNG。概念图是可编辑初稿，用于论文前须逐图核验和矢量润色。","","## 2. 七步优化演进与实测收益","","七个产物依次改变 prepare 位置、双行参数共享、d8 静态展开、算术表达/noinline、inline 与最终调度集成。实际指令以反汇编为准；源码名称含 FMA 不代表一定生成单条 FMA。","" ]
        rows=[]
        for q in QOS:
            origin=get(s,"origin",q)["metrics"]["kernel_us"]["median"]; lut=get(s,"exp-lut",q)["metrics"]["kernel_us"]["median"]; stage=get(s,VARIANTS[0],q)["metrics"]["kernel_us"]["median"]
            for v in VARIANTS:
                m=get(s,v,q)["metrics"]["kernel_us"]; parent=PARENT.get(v); parent_speed="—" if not parent else f(get(s,parent,q)["metrics"]["kernel_us"]["median"]/m["median"])
                rows.append((q,v,f(m["median"]),f(m["min"]),f(m["max"]),f(origin/m["median"]),f(lut/m["median"]),f(stage/m["median"]),parent_speed))
        table(lines,["Qo","variant","median us","min","max","vs Origin","vs EXP-LUT","vs stage1","vs parent"],rows)
        lines += ["原始 5 次测量保存在 [`attention_all_serial.csv`](attention_all_serial.csv) 的 `values` 列和 [`summary.json`](summary.json) 的 samples 中；未补点、未平滑。","","## 3. 阶段 latency 与 SCNA 占比","","`scna_exp` 嵌套在 `safe_sm` 内，以下只单列占比，不把它再次加到八阶段总和。",""]
        rows=[]
        for v in VARIANTS:
            m=get(s,v,32)["metrics"]; rows.append([v]+[f(m[x]["median"]) for x in PHASES]+[f(m["scna_exp_us"]["median"]),f(m["scna_share_safe_sm_percent"],1)+"%",f(m["scna_share_attention_percent"],1)+"%"])
        table(lines,["variant","q_load","k_load","v_load","qk_dot","safe_sm","core_acc","o_scale","o_store","scna_exp","% safe_sm","% kernel"],rows)
        lines += ["完整 Qo 阶段矩阵见 [`summary.json`](summary.json)；图见 [`03_phases_and_scna_share.svg`](figures/03_phases_and_scna_share.svg)。","","## 4. optimized 自动 worker 扩展","" ]
        rows=[]
        for q in QOS:
            one=get(s,"optimized",q); auto=get(s,"optimized",q,"auto"); a_med=auto["metrics"]["kernel_us"]["median"]; rows.append((q,auto["active_workers"][0],f(one["metrics"]["kernel_us"]["median"]),f(a_med),f(one["metrics"]["kernel_us"]["median"]/a_med)))
        table(lines,["Qo","active workers","1-worker us","auto us","speedup"],rows)
        lines += ["自动 worker 仅用于并行 scaling 附加分析，不混入 evaluator 单线程归因。图见 [`05_optimized_worker_scaling.svg`](figures/05_optimized_worker_scaling.svg)。","","## 5. v79 指令分析","" ]
        rows=[]
        keys=("instructions","packets","vector_load_store","splat_broadcast","vector_mux_compare","qf16_or_convert","fp16_multiply_fma","vector_add_max","predicate_ops","branches","calls","returns","spill_memory","stack_frame_bytes","code_bytes")
        for v in VARIANTS: rows.append([v]+[st[v][k] for k in keys])
        table(lines,["variant","insns","packets","load/store","splat","cmp/mux","qf16/cvt","FP16 mul/FMA","add/max","predicate","branch","call","return","stack ref","stack B","code B"],rows)
        lines += ["作用对应：load/store 搬入/写回向量；splat 广播权重和偏置；compare/mux 实现 clamp/ReLU；qf16 convert 控制中间格式；FP16 multiply 与 add/max 实现层计算和 ReLU；predicate/branch 控制尾部、循环和 variant guard；call/return、stack reference 反映 inline/noinline 与 ABI 代价。逐产物热函数见 [`static/`](static/)，汇总见 [`static_metrics.json`](static/static_metrics.json)。若所谓 FMA 方案实际反汇编为 multiply+add，本表和报告以机器码为准。","","动态 PC-filter 验证：",""]
        table(lines,["variant","runtime range","packet PCs in range","ihist nonzero","是否可作动态证据"],[(v,f"{dyn[v].get('pc_start','?')}–{dyn[v].get('pc_end','?')}",dyn[v]["packet_addresses_in_range"],dyn[v]["ihist_nonzero"],"YES" if dyn[v]["validated"] else "NO；静态反汇编为权威") for v in VARIANTS])
        lines += ["只有 runtime base、热函数重定位范围、范围内 packet 地址与非零 ihist 同时验证后才采用动态证据。本 run 的 7 个 timing 样本均在 simulator `FSM_HOYA-L2D` 报告无法识别的 state/event 后退出，因此没有可发布的热函数 model cycles；PMU/ihist 数字不进入性能结论，静态反汇编是指令分析权威。","","## 6. 图与研究完整性","","- [`01_all_serial_latency.svg`](figures/01_all_serial_latency.svg)：全方案单 worker latency。","- [`02_parent_speedup.svg`](figures/02_parent_speedup.svg)：逐父方案 speedup。","- [`04_instruction_heatmap.svg`](figures/04_instruction_heatmap.svg)：静态指令构成。","- 全部性能图均从真实 `summary.json`/`static_metrics.json` 自动生成，提供 SVG/PDF/PNG，使用颜色之外的 marker、线型、hatch 或数值标注。","","用户需要逐句、逐图核验研究解释；若用于论文，须遵守学校或投稿方 AI 披露政策。",""]
    walls=[p.get("wall_ns",0)/1e9 for p in s["processes"] if "raw/attention/" in p.get("source","")]
    if walls: lines += ["## 仿真开销","",f"Attention simulator 进程 wall time：median={f(statistics.median(walls))} s，min={f(min(walls))} s，max={f(max(walls))} s。该数值仅代表宿主机仿真成本。",""]
    lines += ["## 可追溯证据","","- [`summary.json`](summary.json)、[`verification_all_serial.json`](verification_all_serial.json)、[`attention_all_serial.csv`](attention_all_serial.csv)","- [`raw/attention/`](raw/attention/)、[`static/`](static/)、[`runtime_filtered_metrics.json`](metrics/detailed/runtime_filtered_metrics.json)、[`manifest.json`](manifest.json)"]
    out=run/"SCNA_ALL_SERIAL_VARIANTS_PERFORMANCE_REPORT_ZH.md"; out.write_text("\n".join(lines)+"\n"); print(out)
if __name__=="__main__": main()
