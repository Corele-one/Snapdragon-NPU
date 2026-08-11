#!/usr/bin/env python3
"""Render the SCNA v79 lane8 report figures from checked-in result artifacts."""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/v79/scna-lane8/20260808_194500"
OUT = ROOT / "docs/figures"
W, H = 960, 540
COLORS = {"Origin-HVX": "#64748b", "LUT-EXP": "#0ea5a4", "SCNA serial d8": "#2563eb", "SCNA lane8 d8": "#dc2626"}

def tag(name, attrs="", text=""):
    return f"<{name}{(' ' + attrs) if attrs else ''}>{escape(str(text))}</{name}>"

def svg(name, body, title, desc):
    OUT.mkdir(parents=True, exist_ok=True)
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<style>text{{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#172033}}.title{{font-size:22px;font-weight:700}}.sub{{font-size:13px;fill:#526174}}.axis{{font-size:12px;fill:#526174}}.note{{font-size:12px;fill:#334155}}.value{{font-size:12px;font-weight:700}}.grid{{stroke:#dbe3ed;stroke-width:1}}.axisline{{stroke:#94a3b8;stroke-width:1.2}}.legend{{font-size:13px;font-weight:600}}</style>
<rect width="100%" height="100%" fill="#ffffff"/>{body}</svg>'''
    (OUT / name).write_text(content, encoding="utf-8")

def line(x1, y1, x2, y2, cls="grid", extra=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" {extra}/>'

def text(x, y, s, cls="axis", anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(str(s))}</text>'

def main():
    summary = json.loads((RUN / "analysis/summary.json").read_text())
    static = json.loads((RUN / "static/scna_disassembly_summary.json").read_text())
    trace = json.loads((RUN / "trace/event_trace_summary.json").read_text())
    qos = [4, 8, 16, 32]

    # 1. Attention latency
    left, right, top, bottom = 90, 900, 90, 440
    ymax = 3200
    body = text(left, 38, "Attention latency rises sharply for grouped lane8", "title") + text(left, 61, "Median headline latency; whiskers are 95% stratified-bootstrap CIs", "sub")
    for y in range(0, 3201, 800):
        yy = bottom - y / ymax * (bottom-top); body += line(left, yy, right, yy) + text(left-12, yy+4, f"{y:,}", "axis", "end")
    body += line(left, top, left, bottom, "axisline") + line(left, bottom, right, bottom, "axisline") + text(24, 270, "Latency (µs)", "axis")
    for i, q in enumerate(qos):
        x = left + i * (right-left) / 3; body += text(x, bottom+27, f"Qo={q}", "axis", "middle")
    for j, (name, color) in enumerate(COLORS.items()):
        pts=[]
        for i,q in enumerate(qos):
            d=summary["attention"][name][str(q)]; x=left+i*(right-left)/3; y=bottom-d["median_us"]/ymax*(bottom-top); lo=bottom-d["median_ci_low_us"]/ymax*(bottom-top); hi=bottom-d["median_ci_high_us"]/ymax*(bottom-top)
            body += line(x, lo, x, hi, "", f'stroke="{color}" stroke-width="2"') + line(x-5,lo,x+5,lo,"",f'stroke="{color}" stroke-width="2"') + line(x-5,hi,x+5,hi,"",f'stroke="{color}" stroke-width="2"'); pts.append(f"{x:.1f},{y:.1f}")
        body += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>'
        for p in pts: body += f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="4.5" fill="{color}"/>'
        body += f'<rect x="{left+j*190}" y="478" width="12" height="12" rx="2" fill="{color}"/>' + text(left+18+j*190,488,name,"legend")
    svg("attention_latency.svg", body, "Attention latency by Qo", "Median attention latency and 95 percent confidence intervals for four evaluators.")

    # 2. Speedup
    left, right, top, bottom = 95, 900, 95, 425; ymin, ymax = 0.4, 1.1
    body=text(left,38,"Grouped lane8 misses the speedup gate at every Qo", "title")+text(left,61,"Speedup = serial median / lane8 median; values below 1 mean lane8 is slower", "sub")
    for v in [0.4,0.6,0.8,1.0]:
        y=bottom-(v-ymin)/(ymax-ymin)*(bottom-top); body+=line(left,y,right,y,"grid" if v!=1 else "", 'stroke="#64748b" stroke-dasharray="5 4"' if v==1 else "")+text(left-12,y+4,f"{v:.1f}×","axis","end")
    for i,d in enumerate(summary["speedups"]):
        x=left+105+i*210; y=lambda v: bottom-(v-ymin)/(ymax-ymin)*(bottom-top)
        body+=f'<rect x="{x-42}" y="{y(d["median"]):.1f}" width="84" height="{bottom-y(d["median"]):.1f}" rx="5" fill="#dc2626"/>'
        body+=line(x,y(d["ci_low"]),x,y(d["ci_high"]),"",'stroke="#7f1d1d" stroke-width="2"')+line(x-8,y(d["ci_low"]),x+8,y(d["ci_low"]),"",'stroke="#7f1d1d" stroke-width="2"')+line(x-8,y(d["ci_high"]),x+8,y(d["ci_high"]),"",'stroke="#7f1d1d" stroke-width="2"')
        body+=text(x,y(d["median"])-12,f"{d['median']:.3f}×","value","middle")+text(x,bottom+28,d["label"],"axis","middle")
    body+=text(right,top+10,"Gate: > 1.0×", "note", "end")
    svg("lane8_speedup.svg",body,"Serial to lane8 speedup", "All four speedup confidence intervals lie below one.")

    # 3. micro + static categories
    body=text(70,38,"Microkernel slowdown is accompanied by shuffle and spill overhead", "title")+text(70,61,"Left: median latency per 64 scores. Right: static instruction categories in the measured binary.","sub")
    body+=text(70,96,"Paired evaluator latency", "legend")+text(540,96,"Static instruction composition", "legend")
    micro=summary["micro"]["items"]; maxlat=150
    for i,item in enumerate(micro):
        y=145+i*100; width=item["median"]/maxlat*350; body+=text(155,y+28,item["label"],"axis","end")+f'<rect x="170" y="{y}" width="{width:.1f}" height="42" rx="5" fill="{"#2563eb" if item["label"]=="serial" else "#dc2626"}"/>'+text(180+width,y+27,f"{item['median']:.3f} ns", "value")
    body+=text(170,385,"lane8 paired is 3.250× slower than serial", "note")
    cats=[("vector_compute","Vector compute","#2563eb"),("shuffle_reduction","Shuffle / reduction","#f59e0b"),("vector_load_store","Vector load/store","#0ea5a4"),("spill_memory_ops","Spill","#dc2626"),("scalar_other","Scalar / other","#94a3b8")]
    for i,row in enumerate(static):
        y=145+i*82; x=565; body+=text(555,y+25,row["variant"],"axis","end")
        for key,label,color in cats:
            w=row[key]/454*300; body+=f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="36" fill="{color}"/>'; x+=w
        body+=text(875,y+25,row["instructions"],"value","start")
    for i,(_,label,color) in enumerate(cats): body+=f'<rect x="{555+i%3*130}" y="{430+(i//3)*28}" width="12" height="12" fill="{color}"/>'+text(573+i%3*130,441+(i//3)*28,label,"axis")
    svg("microkernel_static_cost.svg",body,"Microkernel and static instruction costs", "Paired lane8 is slower and contains shuffle reduction and spill instructions absent from serial.")

    # 4. Correctness RMSE scatter
    left,right,top,bottom=100,900,105,430; lo,hi=-3,5.2
    body=text(left,38,"Attention correctness gate fails across every tested case", "title")+text(left,61,"RMSE by mask, KV length, and head dimension; logarithmic scale. Gate is RMSE ≤ 0.002.","sub")
    def sy(v):
        import math
        return bottom-(math.log10(v)-lo)/(hi-lo)*(bottom-top)
    for power in [-3,-1,1,3,5]:
        y=bottom-(power-lo)/(hi-lo)*(bottom-top); body+=line(left,y,right,y)+text(left-12,y+4,f"1e{power}","axis","end")
    body+=line(left,sy(.002),right,sy(.002),"",'stroke="#7f1d1d" stroke-width="2" stroke-dasharray="6 4"')+text(right,sy(.002)-7,"RMSE gate", "note","end")
    cases=summary["correctness"]["cases"]
    for i,c in enumerate(cases):
        x=left+30+i*(right-left-60)/23; color="#2563eb" if c["layout"]=="serial" else "#dc2626"; shape="circle" if c["head_dim"]==64 else "rect"
        if shape=="circle": body+=f'<circle cx="{x:.1f}" cy="{sy(c["rmse"]):.1f}" r="5" fill="{color}"/>'
        else: body+=f'<rect x="{x-4:.1f}" y="{sy(c["rmse"])-4:.1f}" width="8" height="8" fill="{color}"/>'
    body+=text(left,bottom+28,"24 deterministic cases: full / padding / causal × KV=4093 / 4096 × d=64 / 128 × layout", "axis")
    body+=f'<circle cx="{left}" cy="485" r="5" fill="#2563eb"/>'+text(left+10,489,"serial", "axis")+f'<circle cx="{left+100}" cy="485" r="5" fill="#dc2626"/>'+text(left+110,489,"lane8", "axis")+f'<rect x="{left+195}" y="481" width="8" height="8" fill="#334155"/>'+text(left+210,489,"d=128 (circle: d=64)", "axis")
    svg("attention_correctness.svg",body,"Attention correctness RMSE", "Every measured serial and lane8 attention case exceeds the RMSE gate.")

    # 5. Trace summary
    body=text(90,38,"DSP qtimer replay confirms lane8 SCNA-exp takes 2.34–2.43× longer", "title")+text(90,61,"Cumulative scna_exp duration across three diagnostic measured iterations; software event trace, not hardware PMU.","sub")
    left,right,top,bottom=105,885,110,420; ymin,ymax=0,13000
    for yv in range(0,13001,3000):
        y=bottom-yv/ymax*(bottom-top); body+=line(left,y,right,y)+text(left-12,y+4,f"{yv/1000:g}k", "axis","end")
    for i,q in enumerate(qos):
        x=left+110+i*210; s=trace["serial"]["by_qo_len"][str(q)]["component_event_duration_us"]["scna_exp"]; l=trace["lane8"]["by_qo_len"][str(q)]["component_event_duration_us"]["scna_exp"]
        for offset,val,color in [(-31,s,"#2563eb"),(9,l,"#dc2626")]:
            y=bottom-val/ymax*(bottom-top); body+=f'<rect x="{x+offset}" y="{y:.1f}" width="22" height="{bottom-y:.1f}" rx="3" fill="{color}"/>'
        body+=text(x,bottom+28,f"Qo={q}","axis","middle")+text(x, min(bottom-s/ymax*(bottom-top),bottom-l/ymax*(bottom-top))-11, f"{l/s:.3f}×", "value","middle")
    body+=text(38,270,"Cumulative scna_exp time (µs)","axis")+f'<rect x="{left}" y="478" width="12" height="12" fill="#2563eb"/>'+text(left+18,489,"serial", "legend")+f'<rect x="{left+105}" y="478" width="12" height="12" fill="#dc2626"/>'+text(left+123,489,"lane8", "legend")
    svg("trace_scna_exp_duration.svg",body,"SCNA exp trace duration", "Grouped bars compare serial and lane8 cumulative scna exp duration from DSP qtimer diagnostic replays.")

if __name__ == "__main__": main()
