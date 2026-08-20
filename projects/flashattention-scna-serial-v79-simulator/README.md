# flashattention-scna-serial-v79-simulator

独立的 Hexagon v79 simulator 项目，用 DSP-side harness 复现 serial SCNA 与 FlashAttention。源代码从原实验快照复制；不通过软链接依赖原目录，也不复制 Qualcomm SDK、QuRT 或工具二进制。

`experiment_spec.source.json` 保存原真机实验预注册规格；本轮实际执行参数固定在 `experiment_spec.simulator.json`，两者不混用。

## 当前验证结论

FastRPC Host 链路不能由 `hexagon-sim` 仿真；DSP `.so` 可以经 `hexagon-sim → QuRT runelf.pbn → run_main_on_hexagon_sim → DSP main()` 直接执行 HTP kernel。正式 run `scna_sim_v79_20260813_r1` 的 capability、7 个 micro variant、Attention smoke/tail 和 20 点扫描均 PASS。

所有 latency/cycle 仅为 simulator diagnostic，不能表述为真实 Snapdragon NPU 性能。

## 使用

```bash
./scripts/build_all.sh
SCNA_RUN_ID=my_run ./scripts/probe_simulator.sh
SCNA_RUN_ID=my_run ./scripts/run_micro.sh --all
SCNA_RUN_ID=my_run ./scripts/run_attention.sh --smoke
SCNA_RUN_ID=my_run ./scripts/run_attention.sh --sweep
SCNA_RUN_ID=my_all_serial_run ./scripts/run_attention.sh --all-serial
python3 tools/analyze_results.py --run-dir results/runs/my_run
```

一键运行：

```bash
SCNA_RUN_ID=my_run ./scripts/run_all.sh
SCNA_RUN_ID=my_all_serial_run ./scripts/run_all_serial.sh
SCNA_RUN_ID=scna_combined_resource_v79_$(date +%Y%m%d_%H%M%S) ./scripts/run_combined_resource.sh
```

默认 SDK 是 `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0`。可覆盖 `HEXAGON_SDK_ROOT`、`HEXAGON_SIM_CORE`；默认单 case timeout 600 s，可用 `SCNA_SIM_CASE_TIMEOUT_SEC` 覆盖。

## 输出与文档

- [模拟器中文手册](docs/HEXAGON_SIMULATOR_GUIDE_ZH.md)
- [FastRPC 边界验证](docs/FASTRPC_BOUNDARY_ZH.md)
- `results/runs/<run-id>/summary.json`、原始日志、PMU/ihist、反汇编、图与自动报告
- Attention 数据流图同时输出可编辑 `.drawio` 和 SVG/PDF/PNG
- 全 serial 匹配实验生成 `SCNA_ALL_SERIAL_VARIANTS_PERFORMANCE_REPORT_ZH.md`、`attention_all_serial.csv`、7 份热函数反汇编、5 组性能图和两份可编辑 `.drawio`
- 综合资源审计生成 `SCNA_COMBINED_PERFORMANCE_AND_RESOURCE_REPORT_ZH.md`、`combined_summary.json`、4 份 CSV、12 组 SVG/PDF/PNG 图和两份可编辑 `.drawio`。原生 trace 仅在 exit/file/PC/nonzero 四门禁全部通过时作为 kernel 动态证据；资源审计插桩时间不参与性能排序。
- 同一综合 run 的 `tutorial/SCNA_TUTORIAL_INDEX_ZH.md` 是教程式入口：五篇文档逐步解释 Attention/SCNA 基础、七步优化、图 1–12、资源证据、复现与排障；`tutorial_manifest.json` 记录主要数字的 JSON 路径、原始证据和计算公式。

原生 `memtrace`/`pctrace_min` 文件仍可能较大。脚本保留原始文件，不会自动删除；运行前应确认 `results/` 所在文件系统有足够空间。

`Tools/vscode` 不是 VS Code 本体；详见手册第 8 节。本项目只提供 `.vscode` 模板，不自动安装或修改用户扩展。

已有综合资源 run 可额外生成 LUT-vs-SCNA 同口径诊断表：

```bash
python3 tools/generate_lut_scna_roofline_diagnostic.py \
  --run-dir results/runs/scna_combined_resource_v79_20260818_164158
```

输出仅描述 simulator diagnostic；不会生成或声称 Snapdragon 真机 Roofline。

Qo=32 下重试 KV=64 与受控长 KV=512：

```bash
SCNA_RUN_ID=lut_scna_kv_v79_retry ./scripts/run_lut_scna_kv_diagnostic.sh
```

脚本分别运行 Origin、EXP-LUT、stage1 和 optimized，要求 `ATTENTION_VERIFY=PASS` 与正常退出，并输出 CSV/JSON/中文表。动态 trace/PMU 的 authority 从综合资源 run 重新执行四门禁：正常退出、文件非空、PC 命中目标范围、非零样本必须同时成立；任一失败均报告 `UNAVAILABLE`，不会用零值代替。
