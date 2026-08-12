# FlashAttention SCNA HMX FP16 d8（Hexagon v81）

这是从 `flashattention-scna-v79-framework` 独立迁移的 v81 工程；原 v79 工程不被修改。项目在 FlashAttention 的两个 online safe-softmax 指数位置实现 SCOPE `exp2`、FP16、8 神经元 SCNA，并在同一 DSP skeleton 中提供五个无 pipeline 模式。

| CLI mode | 指数实现 | SCNA engine |
|---|---|---|
| `baseline` | Origin-HVX | — |
| `lut-exp` | 64 KiB×4 VTCM 表 + HVX gather | — |
| `scna-hvx-fp16-d8` | FP16 direct d8 | HVX |
| `scna-hmx-fp16-d8-hybrid` | HMX affine/bias/ReLU + HVX 8→1 reduction | HMX/HVX |
| `scna-hmx-fp16-d8-two-pass` | HMX affine/bias/ReLU + HMX 8→1 reduction | HMX |

HMX 映射每批处理 32 个标量：输入位于 32 个 spatial position 的 channel 0，第一遍输出 channel 0–7；two-pass 将第一遍 crouton 直接作为第二遍 activation。activation/output、weight、bias 在 VTCM 中分别满足 2 KiB、128 B、256 B 对齐。输入在 kernel 边界限制为 `[-256, 0]`。

## 构建与单次运行

工程只接受 v81，并固定编译参数 `-mv81 -mhmx -mhvx -mhvx-length=128B`：

```bash
./scripts/build.sh --dsp-arch v81
./scripts/deploy_and_smoke.sh --mode ping
./scripts/deploy_and_smoke.sh --mode scna-hmx-fp16-d8-two-pass --qo-len 4 --kv-len 4096
```

依赖 Hexagon SDK 6.6.0.0、Tools 19.0.07、Android NDK、CMake、Ninja、Python 3、Matplotlib，以及已授权的 `adb` 设备。可用 `SCNA_HEXAGON_SDK_ROOT` 覆盖 SDK 根目录。

## 正式实验

版本化参数在 [`experiment_spec.json`](experiment_spec.json)。正式采集会执行 HMX 反汇编门禁、SCNA 微核门禁、36 项 FlashAttention 正确性矩阵，以及五模式循环 Latin-square 配对性能实验：

```bash
./scripts/run_scna_hmx_v81_experiment.py \
  --run-id 20260812_scna_hmx_fp16_d8_v81

./tools/analyze_scna_hmx_v81.py \
  results/20260812_scna_hmx_fp16_d8_v81
```

采集器在每个 performance session 前校验远端三项 artifact SHA256，不一致即拒绝；每个 mode 含 5 次 warmup 与 20 次测量，每个 shape 需要 5 个有效 session。汇总使用 session 中位数与固定种子的 10,000 次配对 bootstrap 95% CI。

若温控导致部分 session 无效，可保持同一 binary 并仅补齐缺失 session：

```bash
./scripts/run_scna_hmx_v81_experiment.py \
  --skip-build --skip-deploy --phase performance --resume-performance \
  --run-id 20260812_scna_hmx_fp16_d8_v81
```

自动化测试：

```bash
python3 -m unittest discover -s tests -v
```

## 本次实测产物

- 中文报告：[`reports/SCNA_HMX_FP16_D8_V81_REPORT.md`](reports/SCNA_HMX_FP16_D8_V81_REPORT.md)
- 原始数据：[`results/20260812_scna_hmx_fp16_d8_v81/raw/`](results/20260812_scna_hmx_fp16_d8_v81/raw/)
- 汇总与图表：[`results/20260812_scna_hmx_fp16_d8_v81/summary/`](results/20260812_scna_hmx_fp16_d8_v81/summary/)、[`figures/`](results/20260812_scna_hmx_fp16_d8_v81/figures/)
- 构建/设备/参数哈希：[`manifest.json`](results/20260812_scna_hmx_fp16_d8_v81/manifest.json)
- named-symbol HMX 证据：[`scna_hmx_symbols.disasm.txt`](results/20260812_scna_hmx_fp16_d8_v81/verification/scna_hmx_symbols.disasm.txt)
- Naive→局部最优→稳定最终版的优化 checkpoint：[`HMX_SCNA_OPTIMIZATION_HISTORY.json`](docs/HMX_SCNA_OPTIMIZATION_HISTORY.json)

报告中的所有数值和图均由 run 目录中的 JSONL 确定性生成；失败与缺失值不会被插值。
