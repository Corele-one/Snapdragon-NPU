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
```

默认 SDK 是 `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0`。可覆盖 `HEXAGON_SDK_ROOT`、`HEXAGON_SIM_CORE`；默认单 case timeout 600 s，可用 `SCNA_SIM_CASE_TIMEOUT_SEC` 覆盖。

## 输出与文档

- [模拟器中文手册](docs/HEXAGON_SIMULATOR_GUIDE_ZH.md)
- [FastRPC 边界验证](docs/FASTRPC_BOUNDARY_ZH.md)
- `results/runs/<run-id>/summary.json`、原始日志、PMU/ihist、反汇编、图与自动报告
- Attention 数据流图同时输出可编辑 `.drawio` 和 SVG/PDF/PNG
- 全 serial 匹配实验生成 `SCNA_ALL_SERIAL_VARIANTS_PERFORMANCE_REPORT_ZH.md`、`attention_all_serial.csv`、7 份热函数反汇编、5 组性能图和两份可编辑 `.drawio`

`Tools/vscode` 不是 VS Code 本体；详见手册第 8 节。本项目只提供 `.vscode` 模板，不自动安装或修改用户扩展。
