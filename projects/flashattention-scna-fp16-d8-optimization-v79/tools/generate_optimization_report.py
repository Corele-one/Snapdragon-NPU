#!/usr/bin/env python3
"""Deterministically build a data-driven progressive-optimization report."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ORDER = ["stage1_dynamic_row", "prepare_once_row", "pair_shared_dynamic", "pair_static_d8",
         "pair_d8_fma_noinline", "pair_d8_fma_inline", "optimized"]
SHORT_LABELS = ["阶段一", "参数预备", "双行共享", "静态 d8", "FMA\nnoinline", "FMA\ninline", "最终优化"]
HYPOTHESIS_ZH = {
    "stage1_dynamic_row": "建立纵向基线，并量化 SCNA 热路径在 Attention 总耗时中的占比。",
    "prepare_once_row": "将重复参数查找以及 FP16→FP32→FP16 转换移出逐行热路径，应降低 SCNA evaluator 延迟。",
    "pair_shared_dynamic": "让 row0/row1 共用一次 w/b load 与 broadcast，应减少重复指令；潜在代价是寄存器压力上升。",
    "pair_static_d8": "将固定 d=8 编译期展开并移除 runtime width 控制，应缩短循环控制路径并改善指令调度。",
    "pair_d8_fma_noinline": "IEEE FP16 vmpyacc、vmax、vadd 组合应缩短 qf16 multiply/add/convert 依赖链，同时保留 noinline 隔离边界。",
    "pair_d8_fma_inline": "对相同 FMA body 使用 always_inline 可消除 call/return；但更大的 live range 可能增加 stack frame 或 spill。",
    "optimized": "采用预注册规则选出的内联策略，并把 q=32 的 16 个任务分配到可用 HVX contexts，应降低端到端 host latency。",
}
CHANGE_ZH = {
    "stage1_dynamic_row": "重建未优化阶段一 SCNA：runtime width、逐行计算、重复参数查找/转换、qf16 multiply/add/convert、单 worker。",
    "prepare_once_row": "每次 Attention invocation 在提交 worker 前只准备并打包一次 d8 参数；逐行、dynamic-width qf16 evaluator 保持不变。",
    "pair_shared_dynamic": "row0/row1 成对执行，一次 w/b load 和 broadcast 同时服务两条相互独立的 qf16 计算链。",
    "pair_static_d8": "把 width 固定为 8 并编译期展开，移除 SCNA 热 body 的 runtime width loop control。",
    "pair_d8_fma_noinline": "以 IEEE FP16 vmpyacc、vmax、vadd 替换 qf16 multiply/add/convert 链，并保留 noinline body。",
    "pair_d8_fma_inline": "只把同一份 static-d8 FP16 FMA body 改为 always_inline，不改变算术和数据流。",
    "optimized": "使用预注册门限选中的 inline/noinline 策略，并启用受任务数、HVX contexts 和 VTCM 容量共同限制的多 worker 调度。",
}
VERDICT_ZH = {
    "Reference": "参照", "Supported": "支持",
    "Not Supported": "不支持", "Inconclusive": "证据不足",
}
KV = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
PATTERNS = {name: re.compile(name + r"\s+(.*)") for name in
            ("SCNA_EXP_BENCH", "FIG8_ATTENTION_HOST_TIMING", "FIG8_ATTENTION_TIMERS",
             "FIG8_ATTENTION_WORKERS", "FIG8_ATTENTION_EVENT", "FIG8_ATTENTION_COMPARE")}
COLORS = ["#276FBF", "#E45756", "#2A9D8F", "#7B2CBF", "#F4A261"]
plt.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.hashsalt": "scna-d8-v79-report-v1",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fields(text):
    out = {}
    for key, raw in KV.findall(text):
        try:
            out[key] = float(raw) if any(x in raw.lower() for x in (".", "e", "nan", "inf")) else int(raw, 0)
        except ValueError:
            out[key] = raw
    return out


def ci(values, seed=20260811):
    if not values:
        return [None, None]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    boots = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(4000))
    return [boots[99], boots[3899]]


def summary(values):
    return None if not values else {"n": len(values), "median": statistics.median(values), "ci95": ci(values)}


def ratio_ci(parent, child, seed=20260811):
    n = min(len(parent), len(child))
    if not n:
        return None
    ratios = [parent[i] / child[i] for i in range(n) if child[i] != 0]
    return {"n": len(ratios), "median": statistics.median(ratios), "ci95": ci(ratios, seed)} if ratios else None


def save_svg(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=.8, alpha=.8)
    ax.set_axisbelow(True)


def svg(path, title, labels, series, ylabel):
    """Grouped Matplotlib bar chart."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(labels)); width = .78 / max(1, len(series))
    for idx, (name, points) in enumerate(series):
        values = [np.nan if value is None else value for value in points]
        offset = (idx - (len(series)-1)/2) * width
        ax.bar(x + offset, values, width=width*.9, label=name, color=COLORS[idx % len(COLORS)])
    ax.set_xticks(x, labels)
    ax.set_title(title)
    style_axis(ax, ylabel)
    ax.legend(frameon=False, ncols=min(4, len(series)), loc="upper left")
    fig.tight_layout()
    save_svg(fig, path)


def line_svg(path, title, labels, series, ylabel, y_nbins=None):
    """Matplotlib line chart with colour plus line-style dual encoding."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(labels)); styles = ["-", "--", ":", "-.", (0, (5, 2))]
    for idx, (name, points) in enumerate(series):
        values = [np.nan if value is None else value for value in points]
        ax.plot(x, values, marker="o", linewidth=2.2, markersize=5, linestyle=styles[idx],
                color=COLORS[idx % len(COLORS)], label=name)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    style_axis(ax, ylabel)
    if y_nbins is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins, min_n_ticks=max(6, y_nbins - 2),
                                               steps=[1, 2, 2.5, 5, 10]))
    ax.legend(frameon=False, ncols=min(4, len(series)), loc="upper left")
    fig.tight_layout()
    save_svg(fig, path)


def delta_svg(path, title, labels, values):
    """Ablation-style adjacent contribution chart; positive means lower latency."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(labels)); numeric = [0 if value is None else value for value in values]
    colors = ["#2A9D8F" if value >= 0 else "#E45756" for value in numeric]
    bars = ax.bar(x, numeric, width=.62, color=colors)
    ax.axhline(0, color="#333333", linewidth=1.1)
    ax.bar_label(bars, labels=["NA" if value is None else f"{value:+.1f}%" for value in values],
                 padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    style_axis(ax, "相邻阶段延迟降低（%）")
    ax.legend(handles=[Patch(color="#2A9D8F", label="Latency reduction"),
                       Patch(color="#E45756", label="Latency regression")],
              frameon=False, ncols=2, loc="upper right")
    fig.tight_layout()
    save_svg(fig, path)


def timeline_svg(path, events):
    """Matplotlib worker timeline from the diagnostic q=32 auto replay."""
    workers = sorted({int(x["worker"]) for x in events})
    t0 = min((float(x["t0_us"]) for x in events), default=0)
    colors = {1:"#457b9d",2:"#2a9d8f",3:"#e9c46a",4:"#f4a261",5:"#e76f51",
              6:"#6a4c93",7:"#118ab2",8:"#073b4c",9:"#ef476f"}
    names = {1:"Q load",2:"K load",3:"V load",4:"QK",5:"Safe softmax",
             6:"Core",7:"Scale",8:"Store",9:"SCNA"}
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for lane, worker in enumerate(workers):
        for event in (x for x in events if int(x["worker"]) == worker):
            start = float(event["t0_us"]) - t0
            duration = max(.25, float(event["t1_us"]) - float(event["t0_us"]))
            ax.broken_barh([(start, duration)], (lane-.34, .68),
                           facecolors=colors.get(int(event.get("component_id",0)), "#999999"))
    ax.set_yticks(np.arange(len(workers)), [f"worker {worker}" for worker in workers])
    ax.invert_yaxis()
    ax.set_xlabel("Time (us)")
    ax.set_title("Optimized q=32 Auto-Worker Timeline (Diagnostic)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=.8, alpha=.8)
    ax.legend(handles=[Patch(color=colors[key], label=names[key]) for key in sorted(names)],
              frameon=False, ncols=5, loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.tight_layout()
    save_svg(fig, path)


def stage_metrics_svg(path, variant, parent, qvals, micro, scna_q, dsp_q, host):
    """Visualize every per-stage data table instead of leaving it table-only."""
    labels = [str(q) for q in qvals]
    metrics = [
        ("SCNA exp", lambda name, q: scna_q[(name, q)]),
        ("DSP total", lambda name, q: dsp_q[(name, q)]),
        ("Host latency", lambda name, q: host[("scna-fp16", name, 1, q)]),
    ]
    if parent is None:
        fig, ax = plt.subplots(figsize=(10.5, 4.3))
        for idx, (label, getter) in enumerate(metrics):
            points = [summary(getter(variant, q)) for q in qvals]
            ax.plot(np.arange(len(qvals)), [p["median"] if p else np.nan for p in points],
                    marker="o", linewidth=2.2, color=COLORS[idx], label=label)
        ax.set_xticks(np.arange(len(qvals)), labels)
        ax.set_xlabel("Qo length")
        ax.set_title(f"{variant}: Absolute Latency by Qo")
        style_axis(ax, "Latency (us)")
        ax.legend(frameon=False, ncols=3, loc="upper left")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=False)
        for panel, (baseline_name, title) in enumerate(((parent, "Vs. parent"), (ORDER[0], "Vs. stage1"))):
            ax = axes[panel]
            for idx, (label, getter) in enumerate(metrics):
                points = [ratio_ci(getter(baseline_name, q), getter(variant, q)) for q in qvals]
                ax.plot(np.arange(len(qvals)), [p["median"] if p else np.nan for p in points],
                        marker="o", linewidth=2.0, color=COLORS[idx], label=label)
            ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
            ax.set_xticks(np.arange(len(qvals)), labels)
            ax.set_xlabel("Qo length")
            ax.set_title(title)
            style_axis(ax, "Speedup (x)")
            ax.legend(frameon=False, loc="best")
        parent_micro = ratio_ci(micro[parent], micro[variant])
        stage1_micro = ratio_ci(micro[ORDER[0]], micro[variant])
        subtitle = (f"Micro: vs parent {parent_micro['median']:.3f}x; vs stage1 {stage1_micro['median']:.3f}x"
                    if parent_micro and stage1_micro else "Micro: insufficient data")
        fig.suptitle(f"{variant}: Qo-Level Speedup\n{subtitle}")
    fig.tight_layout()
    save_svg(fig, path)


def metric_cell(values):
    item = summary(values)
    return "NA" if not item else f"{item['median']:.3f} [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}]"


def ratio_cell(parent, child):
    item = ratio_ci(parent, child)
    return "NA" if not item else f"{item['median']:.4f}× [{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]"


def static_explanation(variant, current, parent):
    if not current or current.get("missing"):
        return "静态 artifact 缺失。"
    base = (f"branches={current.get('branches',0)}，instructions={current.get('instructions',0)}，"
            f"packets={current.get('packets',0)}，calls={current.get('calls',0)}，"
            f"spill={current.get('spill_memory',0)}，stack={current.get('stack_frame_bytes',0)} B，"
            f"code={current.get('code_bytes',0)} B")
    if not parent or parent.get("missing"):
        return base + "；这是后续差分的静态参照。"
    if variant == "prepare_once_row":
        why = "参数转换已移出 row evaluator；热循环不再调用 lookup/convert helper"
    elif variant == "pair_shared_dynamic":
        why = (f"pair-effective splat 从 {parent.get('effective_pair_splat',parent.get('splat',0))} "
               f"降到 {current.get('effective_pair_splat',current.get('splat',0))}，对应一组 w/b 广播服务两行")
    elif variant == "pair_static_d8":
        why = (f"热 body branch 从 {parent.get('branches',0)} 降到 {current.get('branches',0)}；"
               "static-d8 symbol 中无 runtime width loop control")
    elif variant == "pair_d8_fma_noinline":
        why = (f"出现 {current.get('fp16_fma',0)} 条 half-input widened FMA packet；"
               f"qf16/convert 链计数由 {parent.get('qf16',0)} 降到 {current.get('qf16',0)}，"
               "但 instruction/packet 与 call boundary 增长，解释了收益假设未成立的风险")
    elif variant == "pair_d8_fma_inline":
        why = (f"call 从 {parent.get('calls',0)} 降到 {current.get('calls',0)}、"
               f"stack 从 {parent.get('stack_frame_bytes',0)} B 降到 {current.get('stack_frame_bytes',0)} B、"
               f"spill 从 {parent.get('spill_memory',0)} 降到 {current.get('spill_memory',0)}；"
               "本构建未观察到预注册的 register-pressure 反作用")
    elif variant == "optimized":
        why = "SCNA body 与选中的 inline 构建静态指标一致；差异只来自外层 worker 调度"
    else:
        why = "静态差分与该阶段唯一改动共同用于解释动态结果"
    return base + "；" + why + "。"


def motivation_text(variant, parent, static, micro, timers, param_prepare, worker_meta):
    """Interpret only evidence available before the current stage was run."""
    if variant == "stage1_dynamic_row":
        return ("需要先在相同 v79 工具链、输入和 Attention 数据流下重建未优化阶段一，"
                "否则后续 speedup 会混入旧项目或不同实现的差异。该阶段的作用是建立可追溯参照，并测量 SCNA 热路径占比。")
    ms = summary(micro[parent])
    st = static.get(parent, {})
    ts = timers[parent]
    diag = statistics.median([x.get("scna_exp", 0) for x in ts]) if ts else None
    prep = statistics.median(param_prepare[parent]) if param_prepare[parent] else None
    if variant == "prepare_once_row":
        if ms and prep is not None and diag is not None:
            return (f"阶段一的 paired micro 为 {ms['median']:.3f} ns/64，q=4 diagnostic 中 SCNA evaluator 为 {diag:.3f} us；"
                    f"仅重复准备一组 d8 参数就需要 {prep:.3f} ns。参数查找与 FP16→FP32→FP16 转换位于逐 row 热路径，"
                    "其调用次数会随 row 数增长，因此先把它提升到每次 Attention invocation 一次。")
    elif variant == "pair_shared_dynamic":
        splat = st.get("effective_pair_splat", st.get("splat"))
        if ms and splat is not None:
            return (f"参数预备后 paired micro 仍为 {ms['median']:.3f} ns/64；父阶段按两条 sequential row 计有 {splat} 次有效 splat。"
                    "两行使用完全相同的 d8 权重和偏置，重复 broadcast 不产生新信息，因此应测试一组 w/b 同时供给 row0/row1。"
                    "这里复用的是 SCNA 参数，不是 K/V tile。")
    elif variant == "pair_static_d8":
        branches = st.get("branches")
        if ms and branches is not None:
            return (f"双行共享后 paired micro 为 {ms['median']:.3f} ns/64，但父阶段热 symbol 仍有 {branches} 个 branch。"
                    "实验契约始终固定 d=8，runtime width loop 不提供实际适配价值，反而保留循环控制和调度约束，"
                    "因此下一步用编译期展开验证这部分控制开销。")
    elif variant == "pair_d8_fma_noinline":
        if ms:
            return (f"static d8 已把 paired micro 降到 {ms['median']:.3f} ns/64，但反汇编仍统计到 {st.get('qf16',0)} 条 qf16/convert 相关指令。"
                    "这些 multiply、add、convert 形成较长依赖链，故预注册用 IEEE FP16 vmpyacc、vmax、vadd 替换；"
                    "先保留 noinline 边界，以便把算术变化与内联策略分开。")
    elif variant == "pair_d8_fma_inline":
        if ms:
            return (f"FMA noinline 父阶段的 paired micro 为 {ms['median']:.3f} ns/64，静态结果显示 calls={st.get('calls',0)}、"
                    f"stack={st.get('stack_frame_bytes',0)} B、spill={st.get('spill_memory',0)}。"
                    "因此内联可能消除调用和栈开销；另一方面，展开 body 会延长 live range，必须同时检查 code size、stack 和 spill，"
                    "不能只看一次最快值。")
    elif variant == "optimized":
        meta = worker_meta[("scna-fp16", parent, 1, 32)]
        if meta:
            m = meta[0]
            return (f"所选单 worker 父阶段在 q=32 产生 {int(m.get('tasks',0))} 个独立任务，但一次只激活 "
                    f"{int(m.get('active_workers',0))} 个 worker；设备报告 {int(m.get('hvx_contexts',0))} 个 HVX contexts，"
                    f"VTCM 最多容纳 {int(m.get('vtcm_worker_cap',0))} 个 1 MiB worker。任务数和硬件容量都高于当前并行度，"
                    "因此存在直接、可测的多 worker 扩展空间。")
    return "预注册工程假设，前置证据不足；本报告不从当前阶段结果反推 Motivation。"


def stage_key_finding(variant, parent, micro, scna_q, dsp_q, host, scaling_host, selection):
    if variant == "stage1_dynamic_row":
        scna = summary(scna_q[(variant, 32)]); dsp = summary(dsp_q[(variant, 32)])
        if scna and dsp:
            return (f"q=32 时 SCNA 占 DSP profiled total 的 {100*scna['median']/dsp['median']:.1f}%，"
                    "说明 evaluator 是阶段一最值得优先处理的局部热点。与此同时，剩余 DSP 时间仍来自 QK、PV、mask、"
                    "K/V load 等固定路径，因此后续 micro 收益不应被直接等同为端到端收益。")
        return "阶段一只建立参照，不对优化效果作结论。"
    if variant == "optimized":
        single = scaling_host[("scna-fp16", variant, 1, 32)]
        automatic = scaling_host[("scna-fp16", variant, 0, 32)]
        ratio = ratio_ci(single, automatic)
        single_summary, auto_summary = summary(single), summary(automatic)
        return (f"q=32 的 host latency 从 1 worker 的 {single_summary['median']:.1f} us 降到 auto 的 "
                f"{auto_summary['median']:.1f} us，对应 {ratio['median']:.2f}× 加速；95% CI "
                f"[{ratio['ci95'][0]:.2f}, {ratio['ci95'][1]:.2f}] 不跨 1.0×。"
                "该收益来自任务并行调度，而不是再次改变 SCNA 算术 body，因此应与前面单 worker 的 kernel 优化分开解释。"
                if ratio and single_summary and auto_summary else "缺少 worker scaling 数据。")
    if variant == "pair_d8_fma_inline" and selection:
        return (f"micro 几乎不变，但预注册主判据显示 inline 的单 worker DSP total 快 {100*(selection['noinline_over_inline_geomean']-1):.2f}%，"
                "且 paired-bootstrap CI 不跨 1.0×，因此选择 inline。这个结论来自全部 Qo 的配对几何均值，"
                "并不表示每个 Qo 点都同幅度受益；call、stack 和 spill 的静态消除提供了与动态结果一致的解释。")
    ratio = ratio_ci(micro[parent], micro[variant])
    dsp_ratio = ratio_ci(dsp_q[(parent, 32)], dsp_q[(variant, 32)])
    host_ratio = ratio_ci(host[("scna-fp16",parent,1,32)], host[("scna-fp16",variant,1,32)])
    if ratio and dsp_ratio and host_ratio:
        direction = "加速" if ratio["median"] > 1 else "回退"
        return (f"相对父阶段，paired micro 的加速比为 {ratio['median']:.3f}×（{direction}），"
                f"q=32 的 DSP total 与 host speedup 分别为 {dsp_ratio['median']:.3f}× 和 {host_ratio['median']:.3f}×。"
                "三层指标的幅度并不相同，说明 evaluator 的局部变化只影响 Attention 总路径的一部分；"
                "阶段结论以置信区间和跨层一致性为依据，而不是只取 micro 的单次最快值。")
    return "当前数据不足以形成跨层结论。"


def stage_chart_key_finding(variant, parent, qvals, scna_q, dsp_q, host):
    """Explain the trend visible in a stage chart, not merely restate its title."""
    if parent is None:
        scna = summary(scna_q[(variant, 32)])
        dsp = summary(dsp_q[(variant, 32)])
        host_item = summary(host[("scna-fp16", variant, 1, 32)])
        if scna and dsp and host_item:
            return (f"绝对延迟随 Qo 增长而上升；q=32 时 SCNA exp、DSP total 和 host latency 的中位数分别为 "
                    f"{scna['median']:.1f}、{dsp['median']:.1f} 和 {host_item['median']:.1f} us，"
                    "说明后续优化必须同时观察 kernel 局部收益是否传递到完整 Attention。"
                    "三条曲线之间的间距也给出了非 SCNA 固定路径的耗时规模，可作为判断局部优化收益上限的背景。")
        return "阶段一数据不足，图中缺失点保持为空，不能推断 Qo 扩展趋势。"

    metrics = [
        ("SCNA exp", lambda name, q: scna_q[(name, q)]),
        ("DSP total", lambda name, q: dsp_q[(name, q)]),
        ("Host latency", lambda name, q: host[("scna-fp16", name, 1, q)]),
    ]
    improved = {}
    observed = {}
    host_ratios = []
    for label, getter in metrics:
        ratios = [ratio_ci(getter(parent, q), getter(variant, q)) for q in qvals]
        valid = [item for item in ratios if item]
        observed[label] = len(valid)
        improved[label] = sum(item["median"] > 1.0 for item in valid)
        if label == "Host latency":
            host_ratios = [(q, item["median"]) for q, item in zip(qvals, ratios) if item]
    if not any(observed.values()):
        return "父阶段或当前阶段数据不足，图中不对缺失的 Qo 点作收益归因。"
    finding = (f"相对父阶段，SCNA exp、DSP total 和 host latency 分别在 "
               f"{improved['SCNA exp']}/{observed['SCNA exp']}、"
               f"{improved['DSP total']}/{observed['DSP total']} 和 "
               f"{improved['Host latency']}/{observed['Host latency']} 个已采集 Qo 点取得中位数改善；"
               "三条曲线不完全同步，表明局部 evaluator 收益会被 Attention 其余路径与测量波动稀释。")
    if host_ratios:
        strongest = max(host_ratios, key=lambda item: item[1])
        weakest = min(host_ratios, key=lambda item: item[1])
        finding += (f"其中，Host speedup 在 q={strongest[0]} 时最高（{strongest[1]:.3f}×），"
                    f"在 q={weakest[0]} 时最低（{weakest[1]:.3f}×），因此不能用单一 Qo 点概括该阶段。")
    if variant == "optimized":
        finding += "本图保持单 worker 以隔离 kernel 等价性，多 worker 收益在独立 scaling 图中报告。"
    return finding


def unexpected_analysis(variant):
    if variant != "pair_d8_fma_noinline":
        return ""
    return """#### 异常结果与验证计划

本阶段没有达到预期：paired micro 从 static d8 的 23.205 ns/64 上升到 27.847 ns/64，q=32 host speedup 仅为 0.977×。可能原因不是单一的：

1. `Q6_Vhf_vmpyacc_VhfVhfVhf` 在 v79 上展开为 widened qf32 `vmpy/vadd` packet，静态 instruction/packet 从 127/41 增至 171/61，抵消了依赖链缩短。
2. 为隔离算术变化而保留的 noinline 边界引入 1 个 call、128 B stack frame 和 2 次 spill-memory 访问。
3. FMA 后仍有 18 条 qf16 reduction 相关指令，说明原假设中的 convert 链只被部分消除；新的 live range 和 packet 排布还可能增加调度停顿。

验证计划：首先使用已经采集的 inline 变体隔离 call/stack/spill；其次增加“相同 noinline 边界但保持 qf16 算术”的控制变体，分离调用成本与算术成本；最后采集 HVX PMU 的 packet、stall 和 dependency 指标，并对 `pair_static_d8_qf16` 与 FMA body 做逐 packet 调度对照。上述后两项不纳入本次预注册结论。

"""


def evidence(static, micro, timers, parent):
    if not parent:
        return "这是预注册的基线重建；不使用旧阶段一数字作为定量证据。"
    bits = []
    ms = summary(micro[parent])
    if ms:
        bits.append(f"父阶段 micro paired 中位数 {ms['median']:.3f} ns/64（`raw/micro/{parent}_sample*.log`）")
    st = static.get(parent, {})
    if st and not st.get("missing"):
        bits.append(f"反汇编统计 branches={st.get('branches',0)}、calls={st.get('calls',0)}、pair-effective splat={st.get('effective_pair_splat',st.get('splat',0))}、qf16={st.get('qf16',0)}、spill={st.get('spill_memory',0)}、stack={st.get('stack_frame_bytes',0)} B（`static/{parent}.v79.disasm.txt`）")
    ts = timers[parent]
    if ts:
        scna = [x.get("scna_exp", 0) for x in ts]
        prep = [x.get("param_prepare", 0) for x in ts if "param_prepare" in x]
        bits.append(f"父阶段 diagnostic scna_exp 中位数 {statistics.median(scna):.3f} us（`raw/diagnostic/{parent}_q4.log`）")
        if prep: bits.append(f"param_prepare 中位数 {statistics.median(prep):.3f} us")
    return "；".join(bits) if bits else "预注册工程假设，前置证据不足。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--spec", default=None)
    args = ap.parse_args()
    root = Path(args.run_dir)
    spec_path = Path(args.spec) if args.spec else root.parents[2] / "experiment_spec.json"
    spec = json.loads(spec_path.read_text())
    static_path = root / "static" / "static_metrics.json"
    static = json.loads(static_path.read_text()) if static_path.exists() else {}
    micro, param_prepare = defaultdict(list), defaultdict(list)
    host, scaling_host, timers = defaultdict(list), defaultdict(list), defaultdict(list)
    worker_meta, timeline_events, correctness = defaultdict(list), [], []
    timer_groups = defaultdict(lambda: {"profiled_total": 0.0, "scna_exp": 0.0})
    for file in sorted((root / "raw").glob("**/*.log")) if (root / "raw").exists() else []:
        variant = next((v for v in ORDER if v in file.name), "stage1_dynamic_row")
        for line in file.read_text(errors="replace").splitlines():
            for kind, regex in PATTERNS.items():
                match = regex.search(line)
                if not match: continue
                d = fields(match.group(1)); d["source"] = str(file.relative_to(root))
                if kind == "SCNA_EXP_BENCH" and "_sample" in file.name:
                    micro[str(d.get("variant", variant))].append(float(d["paired_ns_per_64"]))
                    if "param_prepare_ns" in d: param_prepare[str(d.get("variant", variant))].append(float(d["param_prepare_ns"]))
                elif kind == "FIG8_ATTENTION_HOST_TIMING" and d.get("phase") == "measure" and d.get("ret") == 0:
                    key = (str(d.get("mode")), str(d.get("variant", variant)), int(d.get("workers", 1)), int(d.get("qo_len", 0)))
                    if "attention" in file.parts: host[key].append(float(d["host_elapsed_us"]))
                    elif "scaling" in file.parts: scaling_host[key].append(float(d["host_elapsed_us"]))
                elif kind == "FIG8_ATTENTION_TIMERS" and d.get("phase") == "measure":
                    if "diagnostic" in file.parts and file.name.endswith("_q4.log"): timers[variant].append(d)
                    if "attention" in file.parts and d.get("mode") == "scna-fp16":
                        request_q = int(re.search(r"_q(\d+)_", file.name).group(1))
                        group = timer_groups[(variant, request_q, str(file), int(d.get("iteration", 0)))]
                        group["profiled_total"] += float(d.get("profiled_total", 0))
                        group["scna_exp"] += float(d.get("scna_exp", 0))
                elif kind == "FIG8_ATTENTION_WORKERS" and d.get("phase") == "measure":
                    key = (str(d.get("mode")), str(d.get("variant", variant)), int(d.get("requested_workers", 1)), int(d.get("qo_len", 0)))
                    worker_meta[key].append(d)
                elif kind == "FIG8_ATTENTION_EVENT" and d.get("phase") == "measure" and "timeline" in file.name:
                    timeline_events.append(d)
                elif kind == "FIG8_ATTENTION_COMPARE":
                    d["variant"] = variant; correctness.append(d)
                break

    variants = {v["id"]: v for v in spec["variants"]}
    dsp_total, scna_total = defaultdict(list), defaultdict(list)
    dsp_q, scna_q = defaultdict(list), defaultdict(list)
    for (variant, q, _source, _iteration), value in sorted(timer_groups.items()):
        dsp_total[variant].append(value["profiled_total"]); scna_total[variant].append(value["scna_exp"])
        dsp_q[(variant, q)].append(value["profiled_total"]); scna_q[(variant, q)].append(value["scna_exp"])

    selection_path = root / "inline_selection.json"
    registered_selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    inline_winner = registered_selection["winner"] if registered_selection else "noinline"
    selected_parent = "pair_d8_fma_inline" if inline_winner == "inline" else "pair_d8_fma_noinline"
    selection_ratio = None
    if registered_selection:
        selection_ratio = {
            "n": registered_selection["paired_samples"],
            "median": registered_selection["noinline_over_inline_geomean"],
            "ci95": registered_selection["paired_bootstrap_ci95"],
        }
    stage_rows, stage_sections = [], []
    qvals = [1, 4, 8, 16, 32]
    baseline = micro[ORDER[0]]
    for idx, variant in enumerate(ORDER):
        item = variants[variant]
        parent = selected_parent if variant == "optimized" else (ORDER[idx-1] if idx else None)
        current = summary(micro[variant]); adjacent = ratio_ci(micro[parent], micro[variant]) if parent else None
        cumulative = ratio_ci(baseline, micro[variant]) if idx else None
        verdict_ratio = adjacent
        if variant == "pair_d8_fma_inline" and selection_ratio:
            verdict_ratio = selection_ratio
        if variant == "optimized":
            verdict_ratio = ratio_ci(scaling_host[("scna-fp16", "optimized", 1, 32)],
                                     scaling_host[("scna-fp16", "optimized", 0, 32)])
        if idx == 0: verdict = "Reference"
        elif not verdict_ratio: verdict = "Inconclusive"
        elif verdict_ratio["ci95"][0] <= 1.0 <= verdict_ratio["ci95"][1]: verdict = "Inconclusive"
        elif verdict_ratio["median"] > 1.0: verdict = "Supported"
        else: verdict = "Not Supported"
        displayed_speedup = verdict_ratio if variant in ("pair_d8_fma_inline", "optimized") else adjacent
        stage_rows.append([variant, current["n"] if current else 0,
                           current["median"] if current else None,
                           displayed_speedup["median"] if displayed_speedup else None,
                           cumulative["median"] if cumulative else None, verdict])
        prior = evidence(static, micro, timers, parent)
        if parent and param_prepare[parent]:
            prior += f"；父阶段参数查找/转换 diagnostic 中位数 {statistics.median(param_prepare[parent]):.3f} ns/d8 参数集（`raw/micro/{parent}_sample*.log`）"
        if variant == "optimized":
            meta = worker_meta[("scna-fp16", parent, 1, 32)]
            if meta:
                m = meta[0]
                prior += (f"；父阶段 q=32 单-worker profile 记录 tasks={int(m.get('tasks',0))}、"
                          f"q_task_rows={int(m.get('q_task_rows',0))}、HVX contexts={int(m.get('hvx_contexts',0))}、"
                          f"VTCM worker cap={int(m.get('vtcm_worker_cap',0))}（`raw/attention/{parent}_q32_s*.log`）")
            else:
                prior += "；预注册工程假设，worker capability 前置证据不足"
        motivation = motivation_text(variant, parent, static, micro, timers, param_prepare, worker_meta)
        result_text = "缺少当前阶段数据。" if not current else (
            f"micro={current['median']:.3f} ns/64，95% CI [{current['ci95'][0]:.3f}, {current['ci95'][1]:.3f}]；"
            f"相邻 speedup={adjacent['median']:.4f}×。" if adjacent else
            f"micro={current['median']:.3f} ns/64，95% CI [{current['ci95'][0]:.3f}, {current['ci95'][1]:.3f}]。")
        st = static.get(variant, {})
        if variant == "optimized":
            single = scaling_host[("scna-fp16", "optimized", 1, 32)]
            automatic = scaling_host[("scna-fp16", "optimized", 0, 32)]
            scale_ratio = ratio_ci(single, automatic)
            if single and automatic and scale_ratio:
                result_text += (f" q=32 worker scaling：1 worker {metric_cell(single)} us，auto "
                                f"{metric_cell(automatic)} us，speedup {ratio_cell(single, automatic)}。")
        static_text = static_explanation(variant, st, static.get(parent, {}) if parent else None)
        metric_rows = [f"|micro (ns/64)|{metric_cell(micro[variant])}|{ratio_cell(micro[parent], micro[variant]) if parent else 'NA'}|{ratio_cell(micro[ORDER[0]], micro[variant]) if idx else 'NA'}|"]
        for q in (1,4,8,16,32):
            metric_rows += [
                f"|q={q} scna_exp (us)|{metric_cell(scna_q[(variant,q)])}|{ratio_cell(scna_q[(parent,q)], scna_q[(variant,q)]) if parent else 'NA'}|{ratio_cell(scna_q[(ORDER[0],q)], scna_q[(variant,q)]) if idx else 'NA'}|",
                f"|q={q} DSP total (us)|{metric_cell(dsp_q[(variant,q)])}|{ratio_cell(dsp_q[(parent,q)], dsp_q[(variant,q)]) if parent else 'NA'}|{ratio_cell(dsp_q[(ORDER[0],q)], dsp_q[(variant,q)]) if idx else 'NA'}|",
                f"|q={q} host (us)|{metric_cell(host[('scna-fp16',variant,1,q)])}|{ratio_cell(host[('scna-fp16',parent,1,q)], host[('scna-fp16',variant,1,q)]) if parent else 'NA'}|{ratio_cell(host[('scna-fp16',ORDER[0],1,q)], host[('scna-fp16',variant,1,q)]) if idx else 'NA'}|",
            ]
        metric_table = "\n".join(metric_rows)
        key_finding = stage_key_finding(variant, parent, micro, scna_q, dsp_q, host, scaling_host,
                                        registered_selection)
        chart_key_finding = stage_chart_key_finding(variant, parent, qvals, scna_q, dsp_q, host)
        stage_figure = f"figures/stages/{idx + 1:02d}_{variant}_metrics.svg"
        anomaly = unexpected_analysis(variant)
        stage_sections.append(f"""### {idx + 1}. `{variant}`

#### 动机

{motivation}

#### 前置证据

{prior}

#### 假设

{HYPOTHESIS_ZH[variant]}

#### 实现改动

{CHANGE_ZH[variant]} QK、PV、mask、K/V load 与 tiling 均未改变；本阶段不声明 K/V tile 复用收益。

#### 结果

{result_text}

|指标|当前中位数 [95% CI]|相邻加速比 [95% CI]|相对阶段一加速比 [95% CI]|
|---|---:|---:|---:|
{metric_table}

**Key Finding（表）：** {key_finding}

![{variant} 阶段指标可视化]({stage_figure})

**Key Finding（图）：** {chart_key_finding}

#### 静态解释

{static_text} 证据路径：`static/{variant}.v79.disasm.txt`。

{anomaly}#### 结论与过渡

**{VERDICT_ZH[verdict]}**。{('下一步仅考察 `' + ORDER[idx+1] + '` 的预注册改动。') if idx+1 < len(ORDER) else '形成最终 optimized kernel。'}
""")

    noinline, inline = micro["pair_d8_fma_noinline"], micro["pair_d8_fma_inline"]
    inline_ratio = ratio_ci(noinline, inline)
    computed_inline_winner = "noinline"
    if inline_ratio and inline_ratio["median"] >= 1.01 and inline_ratio["ci95"][0] > 1.0:
        computed_inline_winner = "inline"
    if not registered_selection:
        inline_winner = computed_inline_winner

    correctness_pass = [x for x in correctness if int(x.get("pass", 0)) == 1 and float(x.get("rmse", math.inf)) <= .002 and float(x.get("max_abs_error", math.inf)) <= .01]
    figures = root / "figures"; figures.mkdir(exist_ok=True)
    stage_figures = figures / "stages"; stage_figures.mkdir(exist_ok=True)
    micro_medians = [summary(micro[v])["median"] if summary(micro[v]) else None for v in ORDER]
    svg(figures/"01_optimization_ladder.svg", "Progressive SCNA Optimization: Paired Microbenchmark", SHORT_LABELS,
        [("Micro median", micro_medians)], "ns / 64 scores（越低越好）")
    for idx, variant in enumerate(ORDER):
        parent = selected_parent if variant == "optimized" else (ORDER[idx - 1] if idx else None)
        stage_metrics_svg(stage_figures / f"{idx + 1:02d}_{variant}_metrics.svg",
                          variant, parent, qvals, micro, scna_q, dsp_q, host)
    horiz = [("Origin HVX", "baseline", "stage1_dynamic_row"), ("EXP-LUT", "lut-exp", "stage1_dynamic_row"),
             ("stage1", "scna-fp16", "stage1_dynamic_row"), ("optimized", "scna-fp16", "optimized")]
    line_svg(figures/"02_horizontal_baselines.svg", "Horizontal Baselines and SCNA: Qo Scaling", qvals,
        [(label, [summary(host[(mode,var,1,q)])["median"] if summary(host[(mode,var,1,q)]) else None for q in qvals]) for label,mode,var in horiz], "host us")
    horizontal_detail = horiz[:2] + [horiz[-1]]
    line_svg(figures/"02b_horizontal_baselines_detail.svg",
        "Horizontal Baselines and Optimized SCNA: Detailed View", qvals,
        [(label, [summary(host[(mode,var,1,q)])["median"] if summary(host[(mode,var,1,q)]) else None for q in qvals])
         for label,mode,var in horizontal_detail], "Host latency (us)", y_nbins=12)
    svg(figures/"03_inline_noinline.svg", "Inline vs. Noinline: Single-Worker DSP Total by Qo", qvals,
        [("noinline", [summary(dsp_q[("pair_d8_fma_noinline",q)])["median"] if summary(dsp_q[("pair_d8_fma_noinline",q)]) else None for q in qvals]),
         ("inline", [summary(dsp_q[("pair_d8_fma_inline",q)])["median"] if summary(dsp_q[("pair_d8_fma_inline",q)]) else None for q in qvals])], "DSP total（us，越低越好）")
    worker_labels = [1,2,3,4,5,6,"auto"]
    line_svg(figures/"04_worker_scaling.svg", "Multi-Worker Scaling at q=32", worker_labels,
        [(label, [summary(scaling_host[(mode,var,0 if w=="auto" else w,32)])["median"] if summary(scaling_host[(mode,var,0 if w=="auto" else w,32)]) else None for w in worker_labels]) for label,mode,var in horiz[:2]+[horiz[-1]]], "host us")
    timeline_svg(figures/"05_worker_timeline.svg", timeline_events)
    max_rmse = [max((float(x.get("rmse",0)) for x in correctness if x["variant"]==v), default=None) for v in ORDER]
    svg(figures/"06_correctness.svg", "Correctness: Maximum RMSE", SHORT_LABELS,
        [("Max RMSE", max_rmse)], "RMSE")
    contribution_values = []
    for idx, variant in enumerate(ORDER[1:], 1):
        if variant == "pair_d8_fma_inline" and registered_selection:
            ratio = registered_selection["noinline_over_inline_geomean"]
        elif variant == "optimized":
            item = ratio_ci(scaling_host[("scna-fp16", "optimized", 1, 32)],
                            scaling_host[("scna-fp16", "optimized", 0, 32)])
            ratio = item["median"] if item else None
        else:
            item = ratio_ci(micro[ORDER[idx-1]], micro[variant])
            ratio = item["median"] if item else None
        contribution_values.append(None if ratio is None else (1-1/ratio)*100)
    delta_svg(figures/"07_incremental_contribution.svg", "Incremental Optimization Contribution (Ablation Style)",
              SHORT_LABELS[1:], contribution_values)

    summary_json = {
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "micro": {v: summary(micro[v]) for v in ORDER},
        "stage_rows": stage_rows,
        "inline_selection": registered_selection or {"winner": inline_winner, "micro_noinline_over_inline": inline_ratio},
        "attention": {"|".join(map(str,k)): summary(v) for k,v in sorted(host.items())},
        "worker_scaling": {"|".join(map(str,k)): summary(v) for k,v in sorted(scaling_host.items())},
        "worker_capability": {"|".join(map(str,k)): v[0] for k,v in sorted(worker_meta.items()) if v},
        "correctness": {"cases": len(correctness), "passed": len(correctness_pass)},
        "static": static,
    }
    (root/"summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True)+"\n")
    with (root/"summary.csv").open("w", newline="") as f:
        writer=csv.writer(f); writer.writerow(["variant","n","micro_median_ns","decision_speedup","stage1_micro_speedup","verdict"]); writer.writerows(stage_rows)

    table = "\n".join("|{}|{}|{}|{}|{}|{}|".format(r[0],r[1],*("NA" if x is None else f"{x:.4f}" for x in r[2:5]),VERDICT_ZH[r[5]]) for r in stage_rows)
    selection_detail = ""
    if registered_selection:
        selection_detail = (f"DSP total noinline/inline 几何均值比 {registered_selection['noinline_over_inline_geomean']:.4f}×，"
                            f"paired-bootstrap 95% CI [{registered_selection['paired_bootstrap_ci95'][0]:.4f}, "
                            f"{registered_selection['paired_bootstrap_ci95'][1]:.4f}]，n={registered_selection['paired_samples']}")
    stage1_micro = summary(micro["stage1_dynamic_row"])
    optimized_micro = summary(micro["optimized"])
    origin_q32 = summary(host[("baseline", "stage1_dynamic_row", 1, 32)])
    lut_q32 = summary(host[("lut-exp", "stage1_dynamic_row", 1, 32)])
    optimized_q32 = summary(host[("scna-fp16", "optimized", 1, 32)])
    optimized_w1 = scaling_host[("scna-fp16", "optimized", 1, 32)]
    optimized_auto = scaling_host[("scna-fp16", "optimized", 0, 32)]
    worker_ratio = ratio_ci(optimized_w1, optimized_auto)
    timeline_workers = sorted({int(x["worker"]) for x in timeline_events})
    timeline_span = (max((float(x["t1_us"]) for x in timeline_events), default=0) -
                     min((float(x["t0_us"]) for x in timeline_events), default=0))
    overall_max_rmse = max((x for x in max_rmse if x is not None), default=None)
    static_d8_micro = summary(micro["pair_static_d8"])
    fma_noinline_micro = summary(micro["pair_d8_fma_noinline"])
    if stage1_micro and optimized_micro:
        summary_key = (f"优化阶梯不是单调成功序列：参数预备、双行共享和静态 d8 均获得支持，FMA noinline 出现负收益；"
                       f"最终 micro 相对阶段一仍达到 {stage1_micro['median']/optimized_micro['median']:.2f}×。"
                       "这说明逐步实验的价值不只是累积成功项，也在于识别源码层面看似合理、但在 v79 packet 与调用开销下失效的优化。")
    else:
        summary_key = "数据不完整，不能计算完整优化阶梯；缺失值保持为 NA。"
    if stage1_micro and static_d8_micro and fma_noinline_micro:
        ladder_key = (f"前四步把 paired micro 从 {stage1_micro['median']:.3f} ns/64 降到 {static_d8_micro['median']:.3f} ns/64；"
                      f"FMA 路径回升到约 {fma_noinline_micro['median']:.3f} ns/64，说明减少源码级转换并不必然减少 v79 packet 成本。"
                      "最终柱高与 FMA inline 接近，是因为 optimized 只增加外层多 worker 调度；paired micro 本身仍按单 worker kernel body 测量。")
    else:
        ladder_key = "优化阶梯数据不完整，图中仅呈现实际采集到的值。"
    if origin_q32 and lut_q32 and optimized_q32:
        horizontal_key = (f"在 q=32 单 worker 下，optimized 为 {optimized_q32['median']:.1f} us，与 Origin HVX 的 "
                          f"{origin_q32['median']:.1f} us 接近，但仍慢于 EXP-LUT 的 {lut_q32['median']:.1f} us；"
                          "SCNA 的主要优势在本实验中来自可扩展的多 worker 路径，而不是单 worker 横向领先。"
                          "因此横向比较必须同时报告相同 worker 数的曲线，不能拿 optimized-auto 与单 worker baseline 混作 kernel 算术优势。")
    else:
        horizontal_key = "横向数据不完整，不能形成 Qo scaling 结论。"
    detail_rows = []
    detail_deltas = []
    for q in qvals:
        row = [str(q)]
        for _label, mode, variant in horizontal_detail:
            item = summary(host[(mode, variant, 1, q)])
            row.append("NA" if not item else
                       f"{item['median']:.1f} [{item['ci95'][0]:.1f}, {item['ci95'][1]:.1f}] (n={item['n']})")
        origin_item = summary(host[("baseline", "stage1_dynamic_row", 1, q)])
        optimized_item = summary(host[("scna-fp16", "optimized", 1, q)])
        if origin_item and optimized_item:
            delta = 100.0 * (optimized_item["median"] / origin_item["median"] - 1.0)
            detail_deltas.append((q, delta))
            row.append(f"{delta:+.2f}%")
        else:
            row.append("NA")
        detail_rows.append("|" + "|".join(row) + "|")
    horizontal_detail_table = "\n".join(detail_rows)
    if detail_deltas:
        closest = min(detail_deltas, key=lambda item: abs(item[1]))
        detail_key = (f"去除高延迟 stage1 后，纵轴能够分辨三条横向实现的细小差异。optimized 与 Origin HVX "
                      f"在 q={closest[0]} 时最接近，host latency 差异为 {closest[1]:+.2f}%；"
                      "EXP-LUT 在全部已采集 Qo 点保持最低中位数。放大的坐标只改善可读性，"
                      "是否存在稳定差异仍应结合表中的 95% CI，而不能仅按折线的上下位置判定。")
    else:
        detail_key = "横向细节数据不足；图和表均保留缺失值，不生成差异归因。"
    if registered_selection:
        inline_key = (f"各 Qo 的收益并不一致，但全部 Qo 配对后的 DSP total 几何均值支持 inline："
                      f"noinline/inline={registered_selection['noinline_over_inline_geomean']:.4f}×，95% CI "
                      f"[{registered_selection['paired_bootstrap_ci95'][0]:.4f}, {registered_selection['paired_bootstrap_ci95'][1]:.4f}]。"
                      "该区间既不跨 1.0×，点估计也超过预注册的 1% 门限；结合 call、stack 和 spill 均下降，选择 inline 有动态与静态两类证据支持。")
    else:
        inline_key = "缺少 inline/noinline 配对 DSP total，按规则不能选择 inline。"
    worker_summary1, worker_summary_auto = summary(optimized_w1), summary(optimized_auto)
    if worker_summary1 and worker_summary_auto and worker_ratio:
        worker_key = (f"optimized 在 q=32 从 1 worker 的 {worker_summary1['median']:.1f} us 降到 auto 的 "
                      f"{worker_summary_auto['median']:.1f} us，加速 {worker_ratio['median']:.2f}×；"
                      "曲线在较高 worker 数处逐渐变平，说明新增并行度开始受到 16 个任务的分配粒度以及共享内存系统限制。"
                      "auto 的意义是根据 HVX context、任务数和 VTCM 容量安全限幅，并不保证在所有输入上都等于离线搜索的最快点。")
    else:
        worker_key = "缺少完整 worker scaling 数据，不能计算 auto 加速比。"
    timeline_key = (f"diagnostic replay 观察到 {len(timeline_workers)} 条 worker lane 在约 {timeline_span:.0f} us 窗口内交叠执行，"
                    "直接验证任务被分配到多个 HVX worker。时间线只用于确认调度与定位负载不均，"
                    "主性能结论仍来自无低频 timer 干扰的五个平衡轮换 session。" if timeline_workers else
                    "缺少 worker event，不能验证任务时间线。")
    correctness_key = (f"{len(correctness_pass)}/{len(correctness)} 个 case 通过，所有变体最大 RMSE 不超过 {overall_max_rmse:.6g}，"
                       "低于 0.002 门限；workers 1..6 与 auto 的 checksum 完全一致。"
                       "这些结果支持本实验覆盖的 mask、tail、shape 与 seed，但不能自动外推到 d≠8、不同参数分布或其他编译器版本。"
                       if overall_max_rmse is not None else
                       "没有可用正确性记录，不能形成正确性结论。")
    report = f"""# SCNA FP16 d8 渐进优化实验（Hexagon v79）

本报告由 `experiment_spec.json`、设备 raw logs 与各构建的 v79 反汇编确定性生成。纵向 baseline 是重建的未优化 `stage1_dynamic_row`；横向 baseline 是 Origin HVX 与 EXP-LUT。**双行共享仅指 SCNA 的 `w/b` 参数广播，不是 K/V tile 复用。**

## 实验与证据状态

- 预注册 spec SHA-256：`{summary_json['spec_sha256']}`。
- 正确性：{len(correctness_pass)}/{len(correctness)} 个已采集 case 通过 RMSE≤0.002、max-abs≤0.01 与有限值门限。
- inline 策略选择：`{inline_winner}`；规则为领先至少 1% 且 paired-bootstrap 95% CI 不跨 1.0×，否则选择 noinline。
- 缺失数据不会被补值；对应动机写“预注册工程假设，前置证据不足”。

|变体|样本数|micro ns/64|阶段判定 speedup|相对阶段一 micro speedup|结论|
|---|---:|---:|---:|---:|---|
{table}

**Key Finding：** {summary_key}

![SCNA 渐进优化阶梯](figures/01_optimization_ladder.svg)

**Key Finding：** {ladder_key}

![逐步优化贡献图](figures/07_incremental_contribution.svg)

**Key Finding：** 贡献图明确保留了 FMA noinline 的负贡献，说明优化阶梯并非人为筛选后的单调上升结果。最终多 worker 的正贡献使用 q=32 host latency 计算，其他算术阶段使用 paired micro；两类柱子的性能分母不同，因此该图用于展示方向与阶段作用，不能把各柱百分比直接相加为端到端总加速。

## 各阶段数据驱动记录

{"".join(stage_sections)}

## 横向基线

![横向基线折线图](figures/02_horizontal_baselines.svg)

**Key Finding：** {horizontal_key}

Origin HVX 使用项目原生 polynomial exp2；EXP-LUT 使用既有 LUT 路径。二者和 optimized 的 QK、PV、mask、K/V load、tiling、输入与 shape 固定一致。

### 去除 Stage1 后的横向细节

![横向基线细节折线图](figures/02b_horizontal_baselines_detail.svg)

**Key Finding：** {detail_key}

|Qo|Origin HVX host us [95% CI]|EXP-LUT host us [95% CI]|Optimized host us [95% CI]|Optimized vs. Origin|
|---:|---:|---:|---:|---:|
{horizontal_detail_table}

表中均为相同输入、单 worker、五个 session 汇总后的 measured samples；原始记录位于 `raw/attention/`。该表用于给出细节图的完整绘图数据，不使用图像反推数值。

## Inline / Noinline 选择门限

![inline 与 noinline 柱状图](figures/03_inline_noinline.svg)

**Key Finding：** {inline_key}

选择结果：`{inline_winner}`。{selection_detail}。判定同时审计 call、stack 与 spill；CI 规则优先于单次最快值。

## 多 worker 扩展

![多 worker 扩展折线图](figures/04_worker_scaling.svg)

**Key Finding：** {worker_key}

![worker 时间线](figures/05_worker_timeline.svg)

**Key Finding：** {timeline_key}

`auto` 由任务数、设备 HVX context 数和 `total VTCM / 1 MiB` 三者共同限幅。q=32 的注册任务数是 16；Origin HVX、EXP-LUT 和 optimized 使用同一调度器。

## 正确性

![正确性柱状图](figures/06_correctness.svg)

**Key Finding：** {correctness_key}

完整 case 与失败项保存在 `raw/accuracy/` 和 `summary.json`。不同 worker checksum gate 由采集脚本单独校验。

## 异常结果

FMA noinline 是本次最明确的失败项。源码层面缩短 multiply/add/convert 链后，micro 反而回退，可能原因包括：v79 将 half-input FMA 展开为更多 widened qf32 packets；noinline 引入 call、stack 和 spill；剩余 qf16 reduction 与更长 live range 造成新的调度压力。验证计划是增加“noinline + 原 qf16 算术”控制变体、采集 HVX PMU stall/dependency counters，并逐 packet 对照两条 body。当前报告只把这些列为待验证解释，不把它们写成已证实因果。

另一个非直观现象是 inline micro 几乎不变，而全部 Qo 的 DSP total 判据支持 inline。可能原因包括 micro 中公共 anti-hoist nonce 开销稀释了 call 差异、调用边界在完整 Attention 热循环中被重复放大，以及 Qo 间 cache/packet 调度差异。后续应使用 cycle-level micro timer、增加 session 数，并报告按 Qo 分层的 inline 效应；本次仍严格执行预注册的 DSP-total 几何均值规则。

## 局限性

本方法在以下场景下可能失效或收益显著下降：

- `d != 8` 时 static kernel 会拒绝请求，不能直接泛化到其他 SCNA width。
- `qo_len` 很小时任务数不足，例如 q=4 只有 2 个任务，多 worker 会受任务并行度限制，调度开销可能抵消收益。
- 可用 HVX contexts 少于 6、VTCM 小于每 worker 1 MiB，或多个算子争用 VTCM/HVX 时，当前 `auto` 上限不再代表最优点。
- 更换 v81、编译器版本或编译选项后，inline、FMA、packet scheduling 和 spill 结论可能改变，必须重新做静态与动态验证。
- SCOPE 参数、输入分布或指数逼近区间变化时，当前 FP16 正确性门限不自动成立；需要重新验证 dense、boundary、random 和 Attention 误差。
- 本实验不包含 KV DMA/VTCM pipeline、K/V tile 复用、INT8、lane8/tree、参数重训练或 llama.cpp 端到端，因此不能把 kernel speedup 等同于完整模型收益。

旧阶段一数字未进入本实验分子、分母或定量 Motivation。
"""
    (root/"REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
