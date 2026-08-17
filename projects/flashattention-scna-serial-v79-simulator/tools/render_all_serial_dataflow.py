#!/usr/bin/env python3
"""Render editable Attention pipeline and seven-step serial SCNA evolution."""
from __future__ import annotations
import argparse, html
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from render_dataflow import drawio, render

STEPS=[("1 stage1_dynamic_row","prepare per row\ndynamic loop"),("2 prepare_once_row","prepare once\nper evaluator"),("3 pair_shared_dynamic","two rows share\nweights/bias"),("4 pair_static_d8","static d8\nunroll"),("5 fma_noinline","arithmetic rewrite\nnoinline wrapper"),("6 fma_inline","inline hot\nevaluator"),("7 optimized","final scheduling\n+ integration")]

def evolution_drawio(path):
    mx=Element("mxfile",host="app.diagrams.net",version="24.7.17"); dg=SubElement(mx,"diagram",id="serial-evolution",name="Seven Serial SCNA Variants"); model=SubElement(dg,"mxGraphModel",page="1",pageWidth="1600",pageHeight="900"); root=SubElement(model,"root"); SubElement(root,"mxCell",id="0"); SubElement(root,"mxCell",id="1",parent="0")
    for i,(title,delta) in enumerate(STEPS,2):
        cell=SubElement(root,"mxCell",id=str(i),value=html.escape(title+"\n"+delta).replace("\n","&lt;br&gt;"),style="rounded=1;whiteSpace=wrap;html=1;fillColor=#D9EAF7;strokeColor=#24536B;fontSize=13;",vertex="1",parent="1"); SubElement(cell,"mxGeometry",x=str(40+(i-2)*205),y="180",width="175",height="95",**{"as":"geometry"})
        if i>2:
            edge=SubElement(root,"mxCell",id=f"e{i}",style="endArrow=block;strokeWidth=2;",edge="1",parent="1",source=str(i-1),target=str(i)); SubElement(edge,"mxGeometry",relative="1",**{"as":"geometry"})
    note=SubElement(root,"mxCell",id="20",value="All variants replace exp2 inside safe softmax; QKᵀ, P×V and K/V tiling are common.",style="shape=note;whiteSpace=wrap;html=1;fillColor=#FFF4CC;strokeColor=#B38F00;fontSize=12;",vertex="1",parent="1"); SubElement(note,"mxGeometry",x="390",y="350",width="760",height="70",**{"as":"geometry"}); ElementTree(mx).write(path,encoding="utf-8",xml_declaration=True)

def evolution_render(out):
    fig,ax=plt.subplots(figsize=(13.5,4.2)); ax.set_xlim(0,14); ax.set_ylim(0,4); ax.axis("off"); ax.set_title("Seven-step serial SCNA evolution (conceptual source-to-machine-code ladder)",fontsize=13,weight="bold")
    colors=plt.get_cmap("cividis")(list(i/7 for i in range(7)))
    for i,((title,delta),c) in enumerate(zip(STEPS,colors)):
        x=.2+i*1.95; ax.add_patch(FancyBboxPatch((x,1.65),1.65,1.05,boxstyle="round,pad=.04",facecolor=c,edgecolor="#263238")); ax.text(x+.825,2.18,title+"\n"+delta,ha="center",va="center",fontsize=8,color="white" if i<4 else "black")
        if i<6: ax.add_patch(FancyArrowPatch((x+1.65,2.18),(x+1.94,2.18),arrowstyle="-|>",mutation_scale=11,color="#263238"))
    ax.text(.3,.65,"Common scope: score−rowmax → clamp → d8 SCNA exp2 → rowsum → online recurrence. No K/V tile-reuse claim.",fontsize=9,bbox={"boxstyle":"round,pad=.35","facecolor":"#FFF4CC","edgecolor":"#B38F00"})
    for ext in ("svg","pdf","png"): fig.savefig(out/f"serial_scna_seven_variant_evolution.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out-dir",type=Path,required=True); a=p.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    drawio(a.out_dir/"attention_scna_dataflow.drawio"); render(a.out_dir); evolution_drawio(a.out_dir/"serial_scna_seven_variant_evolution.drawio"); evolution_render(a.out_dir); print(a.out_dir)
if __name__=="__main__": main()
