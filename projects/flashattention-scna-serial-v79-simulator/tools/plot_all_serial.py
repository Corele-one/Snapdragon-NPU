#!/usr/bin/env python3
"""Render all-serial figures solely from measured summary/static JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

VARIANTS = ["stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
            "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized"]
SCHEMES = ["origin", "exp-lut", *VARIANTS]
PARENT = dict(zip(VARIANTS[1:], VARIANTS[:-1])) | {VARIANTS[0]: "stage1_dynamic_row"}
QOS = (1, 4, 8, 16, 32)
PHASES = ["q_load_us", "k_load_us", "v_load_us", "qk_dot_us", "safe_sm_us", "core_acc_us", "o_scale_us", "o_store_us"]
INSTRUCTIONS = ["vector_load_store", "splat_broadcast", "vector_mux_compare", "qf16_or_convert",
                "fp16_multiply_fma", "vector_add_max", "predicate_ops", "branches", "calls", "spill_memory"]

def save(fig, out, stem):
    for ext in ("svg", "pdf", "png"): fig.savefig(out / f"{stem}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

def get(data, scheme, qo, policy="w1"):
    return data["attention"][f"{scheme}_{policy}_q{qo}_kv64_h12_kh2_d128"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); a=p.parse_args()
    data=json.loads((a.run_dir/"summary.json").read_text()); static=json.loads((a.run_dir/"static/static_metrics.json").read_text())
    out=a.run_dir/"figures"; out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0); plt.rcParams.update({"font.size":9,"axes.labelsize":9,"legend.fontsize":8})
    colors=sns.color_palette("colorblind", len(SCHEMES)); markers=("o","s","^","v","D","P","X","<",">"); lines=("-","--","-.",":","-","--","-.",":","-")
    fig,ax=plt.subplots(figsize=(8.2,4.6))
    for scheme,c,m,ls in zip(SCHEMES,colors,markers,lines):
        items=[get(data,scheme,q) for q in QOS]; y=[x["metrics"]["kernel_us"]["median"] for x in items]
        lo=[v-x["metrics"]["kernel_us"]["min"] for v,x in zip(y,items)]; hi=[x["metrics"]["kernel_us"]["max"]-v for v,x in zip(y,items)]
        ax.errorbar(QOS,y,yerr=[lo,hi],label=scheme,color=c,marker=m,linestyle=ls,capsize=2,linewidth=1.25)
    ax.set(xlabel="Qo",ylabel="DSP qtimer kernel latency (us)",title="All serial SCNA variants — one worker, KV=64, H=12/KVH=2/D=128"); ax.legend(ncol=3)
    save(fig,out,"01_all_serial_latency")

    fig,ax=plt.subplots(figsize=(8.0,4.1)); x=np.arange(len(QOS)); width=.11
    for i,variant in enumerate(VARIANTS):
        parent=PARENT[variant]; vals=[]
        for q in QOS:
            current=get(data,variant,q)["metrics"]["kernel_us"]["median"]
            base=get(data,parent,q)["metrics"]["kernel_us"]["median"] if parent != variant else current
            vals.append(base/current)
        ax.bar(x+(i-3)*width,vals,width,label=variant,color=colors[i+2],edgecolor="black",linewidth=.3,hatch=("/","\\",".","x","+","o","-")[i])
    ax.axhline(1,color="black",linewidth=.8); ax.set_xticks(x,QOS); ax.set(xlabel="Qo",ylabel="Parent / current speedup",title="Stepwise speedup against direct parent (simulator diagnostic)"); ax.legend(ncol=2)
    save(fig,out,"02_parent_speedup")

    fig,axes=plt.subplots(1,2,figsize=(11.0,4.2)); q=32; phase_colors=sns.color_palette("colorblind",len(PHASES)); bottom=np.zeros(len(SCHEMES))
    for phase,c,h in zip(PHASES,phase_colors,("/","\\",".","x","+","o","-","|")):
        vals=np.array([get(data,s,q)["metrics"][phase]["median"] for s in SCHEMES]); axes[0].bar(np.arange(len(SCHEMES)),vals,bottom=bottom,label=phase.removesuffix("_us"),color=c,hatch=h,edgecolor="white"); bottom+=vals
    axes[0].set_xticks(np.arange(len(SCHEMES)),[s.replace("_","\n") for s in SCHEMES],rotation=20,ha="right"); axes[0].set_ylabel("Phase latency (us)"); axes[0].set_title("Qo=32 phase breakdown"); axes[0].legend(ncol=2,fontsize=7)
    share=np.array([[get(data,s,q)["metrics"]["scna_share_safe_sm_percent"] for q in QOS] for s in VARIANTS])
    sns.heatmap(share,annot=True,fmt=".1f",cmap="cividis",xticklabels=QOS,yticklabels=[v.replace("_"," ") for v in VARIANTS],ax=axes[1],cbar_kws={"label":"% of safe_sm"}); axes[1].set(xlabel="Qo",title="SCNA exp share (nested in safe_sm)")
    save(fig,out,"03_phases_and_scna_share")

    matrix=np.array([[static[v].get(k,0) for k in INSTRUCTIONS] for v in VARIANTS],dtype=float)
    fig,ax=plt.subplots(figsize=(9.2,4.4)); sns.heatmap(matrix,annot=True,fmt=".0f",cmap="viridis",xticklabels=[x.replace("_","\n") for x in INSTRUCTIONS],yticklabels=[v.replace("_"," ") for v in VARIANTS],ax=ax); ax.set_title("Static hot-function instruction classes (v79 disassembly)")
    save(fig,out,"04_instruction_heatmap")

    single=[get(data,"optimized",q,"w1") for q in QOS]; auto=[get(data,"optimized",q,"auto") for q in QOS]
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.7))
    for items,label,c,m in ((single,"1 worker",colors[2],"o"),(auto,"auto",colors[5],"s")):
        axes[0].plot(QOS,[i["metrics"]["kernel_us"]["median"] for i in items],label=label,color=c,marker=m)
    axes[0].set(xlabel="Qo",ylabel="Kernel latency (us)",title="Optimized worker policy"); axes[0].legend()
    speed=[single[i]["metrics"]["kernel_us"]["median"]/auto[i]["metrics"]["kernel_us"]["median"] for i in range(len(QOS))]
    workers=[i["active_workers"][0] for i in auto]; axes[1].bar(QOS,speed,color=colors[5],hatch="//",edgecolor="black");
    for q,s,w in zip(QOS,speed,workers): axes[1].text(q,s,f"{w}w",ha="center",va="bottom",fontsize=8)
    axes[1].axhline(1,color="black",linewidth=.8); axes[1].set(xlabel="Qo",ylabel="1-worker / auto speedup",title="Auto-worker scaling")
    save(fig,out,"05_optimized_worker_scaling"); print(out)

if __name__=="__main__": main()
