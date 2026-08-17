#!/usr/bin/env python3
"""Generate the Chinese internal report strictly from measured JSON and disassembly metrics."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

MODES = ("origin", "exp-lut", "stage1", "optimized")
PHASES = ("q_load_us", "k_load_us", "v_load_us", "qk_dot_us", "safe_sm_us", "core_acc_us", "o_scale_us", "o_store_us")
TOTAL_RE = re.compile(r"Total:\s+Insns=(\d+)\s+Pcycles=(\d+)")


def f(value, digits=3):
    return f"{float(value):.{digits}f}"


def table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    summary = json.loads((run / "summary.json").read_text())
    static = json.loads((run / "static/static_metrics.json").read_text())
    if not summary["pass"]:
        raise SystemExit("Refusing full performance report: acceptance gates did not all pass")

    lines = [
        "# Serial SCNA 在 Hexagon v79 仿真器上的性能分析",
        "",
        f"Run ID：`{summary['run_id']}`。本报告由 `summary.json`、原始日志与 v79 反汇编自动生成。",
        "",
        "> **结论边界：** 这些数字只描述 Hexagon Simulator 的诊断趋势，不是真实 Snapdragon NPU/HTP 性能。官方手册明确模拟器不是 cycle-accurate；未启用 `--timing` 的功能扫描更不能用于预测真机 latency、吞吐或版本排序。",
        "",
        "## 验收结果",
        "",
    ]
    table(lines, ["门禁", "结果"], [(k, "PASS" if v else "FAIL") for k, v in summary["gates"].items()])
    cap = summary["capability"][0]
    lines += [
        f"QuRT/动态加载链路成功；模型 `{cap.get('model')}`，VTCM {cap.get('vtcm_total')} bytes，"
        f"HVX128 contexts={cap.get('hvx128_contexts')}，HMX legacy adapter={cap.get('hmx_lock2_adapter')}。",
        "",
        "正确性共读取到 %d 条 `ATTENTION_VERIFY`；最大 RMSE=%s，最大绝对误差=%s，均无 NaN/Inf。"
        % (len(summary["verifications"]), f(max(v["rmse"] for v in summary["verifications"]), 9),
           f(max(v["max_abs"] for v in summary["verifications"]), 9)),
        "tail case `Qo=3, KV=65` 的 padding/mask 非零计数均为 0。",
        "",
        "## 1. Serial SCNA 微基准",
        "",
        "每个版本为两个独立模拟器进程；每进程 warmup=5、measurement=1000。下表的 pair 时间是 1000 次双行 evaluator 的 DSP qtimer 累计值。",
        "",
    ]
    baseline = summary["micro"]["stage1_dynamic_row"]["pair_elapsed_us"]["median"]
    rows = []
    for name, item in summary["micro"].items():
        metric = item["pair_elapsed_us"]
        rows.append((name, int(item["samples"][0]["build_id"]), f(metric["median"]), f(metric["min"]),
                     f(metric["max"]), f(baseline / metric["median"], 3)))
    table(lines, ["版本", "build ID", "median us", "min", "max", "相对 stage1"], rows)
    fastest = min(summary["micro"], key=lambda n: summary["micro"][n]["pair_elapsed_us"]["median"])
    lines += [
        "微基准数值门限为 RMSE≤0.003、max_abs≤0.02、dense/random RMSE≤0.011、dense/random max_abs≤0.16；"
        "允许 canonical scalar oracle 最多 1 个 FP16 bit-exact lane 差异（前四个 arithmetic-order 版本实测为 1），但 paired/single 必须 bit-exact，且 monotonic/negative/NaN/nonfinite 必须为 0。完整 Attention 仍使用更严格的 RMSE≤0.002、max_abs≤0.01。",
        "",
        f"`optimized` 相对 stage1 为 {f(baseline / summary['micro']['optimized']['pair_elapsed_us']['median'], 3)}×；"
        f"本次模拟器微基准的最低值是 `{fastest}`，不是 `optimized`。因此不能宣称最终版本在仿真器上是全局最快。",
        "",
        "## 2. 完整 Attention latency",
        "",
        "工作量固定为 KV=64、heads=12、kv_heads=2、head_dim=128、full mask、单 worker；每点 warmup=1、measurement=5。`kernel_us` 是 DSP 入口外层 qtimer，`profiled_total_us` 是 8 个阶段计时之和。",
        "",
    ]
    lookup = {(v["mode"], v["qo"]): v for v in summary["attention"].values() if v["kv"] == 64 and v["heads"] == 12}
    rows = []
    for qo in (1, 4, 8, 16, 32):
        bases = {mode: lookup[(mode, qo)]["metrics"]["kernel_us"]["median"] for mode in MODES}
        for mode in MODES:
            item = lookup[(mode, qo)]["metrics"]["kernel_us"]
            rows.append((qo, mode, f(item["median"]), f(item["min"]), f(item["max"]),
                         f(bases["origin"] / item["median"], 3), f(bases["exp-lut"] / item["median"], 3),
                         f(bases["stage1"] / item["median"], 3)))
    table(lines, ["Qo", "模式", "kernel median us", "min", "max", "vs Origin", "vs EXP-LUT", "vs stage1"], rows)
    lines += [
        "在这组 KV=64 功能仿真中，optimized 没有超过 Origin/EXP-LUT 的完整 kernel latency；它只稳定快于 stage1。差异很小且外层 latency 被 QuRT 调度、worker、动态加载后的调用路径等开销显著稀释。该排序不得外推到真机。",
        "",
        "性能图：[`micro ladder SVG`](figures/01_scna_micro_ladder.svg)、[`Attention latency SVG`](figures/02_attention_latency.svg)、[`component breakdown SVG`](figures/03_attention_components.svg)；同目录提供 PDF 与 PNG。",
        "",
        "### optimized 阶段分解",
        "",
        "`scna_exp` 已包含在 `safe_sm` 中，下面单列占比但不重复累加。qtimer 的微秒转换在小 case 上有明显量化。",
        "",
    ]
    rows = []
    for qo in (1, 4, 8, 16, 32):
        m = lookup[("optimized", qo)]["metrics"]
        rows.append([qo] + [f(m[p]["median"]) for p in PHASES] + [f(m["scna_exp_us"]["median"]),
                    f(m["scna_share_safe_sm_percent"], 1) + "%", f(m["scna_share_profiled_percent"], 1) + "%",
                    f(m["scna_share_attention_percent"], 1) + "%"])
    table(lines, ["Qo", "q_load", "k_load", "v_load", "qk_dot", "safe_sm", "core_acc", "o_scale", "o_store", "scna_exp", "% safe_sm", "% profiled", "% kernel"], rows)

    process_walls = [p["wall_ns"] / 1e9 for p in summary["processes"] if "raw/attention/" in p["source"]]
    lines += [
        "### 模拟器 wall time 与 model cycles",
        "",
        f"Attention 模拟器进程 wall time：median={f(statistics.median(process_walls))} s，min={f(min(process_walls))} s，max={f(max(process_walls))} s。"
        "它表示宿主机仿真成本，不是 DSP latency。",
        "",
    ]
    detailed_log = run / "raw/detailed/optimized_micro_timing.log"
    if detailed_log.exists():
        match = TOTAL_RE.search(detailed_log.read_text(errors="replace"))
        if match:
            lines += [f"单独的 `--timing` 样本记录全进程 `Insns={match.group(1)}`、`Pcycles={match.group(2)}`。"
                      "虽然命令请求了 SCNA 链接地址的 PC filter，但 PMU 文件明确写着 `full run simulation cycles`，且 ihist 为空；"
                      "shared object 运行时重定位使该过滤范围未得到可靠验证。因此 PMU/packet 仍按包含 QuRT/loader/harness 的全进程数据解释。",
                      ""]
    lines += [
        "功能扫描日志末尾的全进程 `Total: Insns/Pcycles` 包含 QuRT、loader 与 harness，且没有 `--timing`，只作为可复现运行痕迹，不作为 kernel PMU 证据。",
        "",
        "## 3. 指令证据与作用",
        "",
    ]
    opt = static["optimized"]
    attn = static["attention_hmx_hvx"]
    table(lines, ["范围", "instructions", "packets", "branches", "calls", "stack refs*", "stack bytes", "code bytes"], [
        ("optimized SCNA 热符号", opt["instructions"], opt["packets"], opt["branches"], opt["calls"], opt["spill_memory"], opt["stack_frame_bytes"], opt["code_bytes"]),
        ("Attention/HMX 选定符号", attn["instructions"], attn["packets"], attn["branches"], attn["calls"], attn["spill_memory"], attn["stack_frame_bytes"], attn["code_bytes"]),
    ])
    lines += [
        "`*` stack refs 是对 `r29/r30` memory operand 的静态启发式计数，包含显式 frame 访问，不等价于编译器优化报告中的动态 spill 次数。",
        "",
        "反汇编中实际出现的关键指令如下（静态出现次数，不是动态执行次数）：",
        "",
        f"- `vmem`/标量 `mem*`：HVX 与标量 load/store；SCNA 热范围 vector load/store={opt['vector_load_store']}。",
        f"- `vsplat`：把 FP16 权重/偏置广播到 128-byte HVX 向量；出现 {opt['splat_broadcast']} 次。",
        f"- `vmux`/`vcmp.gt`：clamp 与 ReLU predicate 选择；出现 {opt['vector_mux_compare']} 次。",
        f"- `vmpy(...hf...)` 生成 qf32 中间量，`vadd.qf32/qf16` 累加，`vmax.hf` 做 ReLU；FP16/qf16 multiply={opt['fp16_multiply_fma']}，add/max={opt['vector_add_max']}。本 optimized 热函数没有编译成单条 FMA，也没有 shuffle/permute。",
        f"- HMX `activation.hf = mxmem`、`weight.hf = mxmem` 装入矩阵操作数，`mxclracc.hf` 清 accumulator，`mxmem(...)=cvt` 转换并写回；所选 Attention/HMX 范围 mxmem={attn['hmx_load_store']}、accumulator control={attn['hmx_accumulator_control']}。",
        "- `call`/`jump`/predicate branch/`jumpr`/`dealloc_return` 实现 helper 调用、variant guard、循环与返回；noinline wrapper 因 ABI 对齐产生 128-byte frame 和 2 次栈访存。",
        "",
        "权威静态证据：[`optimized.hot.disasm.txt`](static/optimized.hot.disasm.txt)、[`attention_hmx_hvx.hot.disasm.txt`](static/attention_hmx_hvx.hot.disasm.txt)、[`static_metrics.json`](static/static_metrics.json)。动态 ihist 若无法严格覆盖重复调用，不替代静态反汇编。",
        "",
        "## 4. Attention / serial SCNA 数据流",
        "",
        "上层：DDR/L2 的 Q/K/V 经 HVX 转换、scatter/deal 搬入 VTCM；HMX 计算缩放后的 `QKᵀ`；HVX 做 mask、online row-max 与 safe softmax；HMX 计算 `P×V` 并累加历史 O；HVX 归一化并写回 O。",
        "",
        "下层：`score-rowmax → clamp → d8 SCNA exp2 → rowsum → online recurrence`。双行 evaluator 共享同一组 8 层权重/偏置广播。此优化只替换 softmax 内的 exp2 evaluator，不包含 K/V tile 复用。",
        "",
        "可编辑图与导出图：[`attention_scna_dataflow.drawio`](figures/attention_scna_dataflow.drawio)、[`SVG`](figures/attention_scna_dataflow.svg)、[`PDF`](figures/attention_scna_dataflow.pdf)、[`PNG`](figures/attention_scna_dataflow.png)。",
        "",
        "## 5. 结论",
        "",
        "- 已验证：不能把 Android Host/rpcmem/FastRPC/驱动链放进 `hexagon-sim`；但 QuRT 可加载带 DSP `main()` 的 Hexagon `.so`，因此能在 DSP 侧直接仿真 HTP kernel。",
        "- 7 个 serial d8 版本、smoke、KV=65 tail 与 20 点扫描全部通过；这证明可执行性与数值正确性，不证明真机性能。",
        "- optimized SCNA 微基准比 stage1 快，但完整 Attention 在 KV=64 模拟器诊断中未超过 Origin/EXP-LUT；需要真机恢复后用 FastRPC 端到端与 DSP kernel profiler 复测。",
        "",
        "## 证据索引",
        "",
        "- [`summary.json`](summary.json)、[`attention_summary.csv`](attention_summary.csv)、[`verification.json`](verification.json)",
        "- [`raw/`](raw/)、[`evidence/fastrpc_skel_load.log`](evidence/fastrpc_skel_load.log)、[`manifest.json`](manifest.json)",
        "- 源码：[`sim_main.c`](../../../src/htp-ops-lib-main/src/dsp/sim_main.c)、[`flash_attn.c`](../../../src/htp-ops-lib-main/src/dsp/ops/flash_attn.c)、[`scna_exp2.c`](../../../src/htp-ops-lib-main/src/dsp/ops/scna_exp2.c)",
    ]
    out = run / "SCNA_SERIAL_PERFORMANCE_REPORT_ZH.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
