#!/usr/bin/env python3
"""Create editable draw.io sources and publication-scale first-draft flow diagrams."""
import argparse, html, json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

EVIDENCE={"qtimer":("#0072B2","solid"),"software":("#E69F00","dashed"),"static":("#CC79A7","dotted"),"native":("#009E73","dashdot"),"unavailable":("#777777","dashed")}

ATTN=[("External Q/K/V/mask","software"),("HVX load / convert / scatter","static"),("VTCM Q/K/V tiles","software"),("HMX QKᵀ","static"),("VTCM score S","software"),("HVX mask / rowmax / SCNA exp2\nrowsum / online recurrence","qtimer"),("VTCM probability P","software"),("HMX P×V + historical O","static"),("HMX/HVX normalize","static"),("HVX store O","qtimer")]
SCNA=[("score − rowmax","qtimer"),("clamp to SCNA range","static"),("pair two rows","software"),("broadcast 8 weight/bias groups","static"),("FP16/qf16 multiply + add\n+ max/ReLU","static"),("dual-row accumulate + reduction","static"),("exp2 approximation output","qtimer"),("rowsum + online recurrence","qtimer")]

def make_drawio(path,title,nodes,native_ok):
    mx=Element("mxfile",host="app.diagrams.net",version="24.7.17"); dg=SubElement(mx,"diagram",id=path.stem,name=title); model=SubElement(dg,"mxGraphModel",page="1",pageWidth="1800",pageHeight="1000"); root=SubElement(model,"root"); SubElement(root,"mxCell",id="0"); SubElement(root,"mxCell",id="1",parent="0")
    for i,(label,evidence) in enumerate(nodes,2):
        color,_=EVIDENCE[evidence]; style=f"rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor={color};strokeWidth=3;fontSize=13;"
        row=(i-2)//5; col=(i-2)%5 if row==0 else 4-(i-2)%5
        cell=SubElement(root,"mxCell",id=str(i),value=html.escape(label).replace("\n","&lt;br&gt;"),style=style,vertex="1",parent="1"); SubElement(cell,"mxGeometry",x=str(40+col*330),y=str(110+row*250),width="270",height="100",**{"as":"geometry"})
        if i>2:
            edge=SubElement(root,"mxCell",id=f"e{i}",style="endArrow=block;strokeWidth=2;",edge="1",parent="1",source=str(i-1),target=str(i)); SubElement(edge,"mxGeometry",relative="1",**{"as":"geometry"})
    note=f"Native PMU/trace: {'VALIDATED' if native_ok else 'UNAVAILABLE; not shown as measured access'}"
    cell=SubElement(root,"mxCell",id="note",value=note,style=f"shape=note;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor={EVIDENCE['native' if native_ok else 'unavailable'][0]};fontSize=12;",vertex="1",parent="1"); SubElement(cell,"mxGeometry",x="550",y="650",width="650",height="75",**{"as":"geometry"})
    ElementTree(mx).write(path,encoding="utf-8",xml_declaration=True)

def render(path,title,nodes,native_ok,subtitle):
    plt.rcParams.update({"font.family":["WenQuanYi Zen Hei","DejaVu Sans"],"font.size":9,"svg.fonttype":"none"})
    fig,ax=plt.subplots(figsize=(13.5,5.8)); ax.set_xlim(0,15); ax.set_ylim(0,6); ax.axis("off"); ax.set_title(title,fontsize=13,weight="bold"); ax.text(7.5,5.42,subtitle,ha="center",fontsize=9)
    for i,(label,evidence) in enumerate(nodes):
        row=i//5; col=i%5 if row==0 else 4-i%5; x=.35+col*2.95; y=3.65-row*2.0
        color,ls=EVIDENCE[evidence]; box=FancyBboxPatch((x,y),2.45,.85,boxstyle="round,pad=.04",facecolor="#FAFAFA",edgecolor=color,linewidth=2.3,linestyle=ls); ax.add_patch(box); ax.text(x+1.225,y+.425,label,ha="center",va="center",fontsize=8)
        if i<len(nodes)-1:
            nr=(i+1)//5; nc=(i+1)%5 if nr==0 else 4-(i+1)%5
            if nr==row and row==0: start=(x+2.45,y+.425); end=(.35+nc*2.95,y+.425)
            elif nr==row: start=(x,y+.425); end=(.35+nc*2.95+2.45,y+.425)
            else: start=(x+1.225,y); end=(.35+nc*2.95+1.225,3.65-nr*2.0+.85)
            ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=11,color="#333333",connectionstyle="arc3,rad=0.0"))
    legend=[("qtimer","实测 qtimer"),("software","软件逻辑计数/布局"),("static","静态 v79 反汇编"),("native" if native_ok else "unavailable","原生 PMU/trace " + ("通过门禁" if native_ok else "未通过门禁"))]
    for i,(key,label) in enumerate(legend):
        color,ls=EVIDENCE[key]; ax.plot([.5+i*3.6,1.1+i*3.6],[.45,.45],color=color,linestyle=ls,linewidth=2.5); ax.text(1.2+i*3.6,.45,label,va="center",fontsize=8)
    for ext in ("svg","pdf","png"): fig.savefig(path.with_suffix("."+ext),dpi=240,bbox_inches="tight")
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,required=True); a=p.parse_args(); out=a.run_dir/"figures"; out.mkdir(parents=True,exist_ok=True); d=json.loads((a.run_dir/"combined_summary.json").read_text()); ok=d["native_trace"]["kernel_attributable"]
    a11=out/"11_attention_hvx_hmx_vtcm_flow.drawio"; make_drawio(a11,"HVX/HMX/VTCM Attention",ATTN,ok); render(a11,"HVX/HMX/VTCM Attention dataflow",ATTN,ok,"Logical movement and compute annotations are software-derived; borders encode evidence authority.")
    a12=out/"12_serial_scna_hvx_flow.drawio"; make_drawio(a12,"Serial SCNA HVX evaluator",SCNA,ok); render(a12,"Serial SCNA HVX evaluator",SCNA,ok,"The evaluator replaces exp2 inside safe softmax; it does not add K/V tile reuse.")
if __name__=="__main__": main()
