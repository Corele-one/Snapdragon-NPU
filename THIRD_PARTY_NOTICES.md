# Third-party notices and provenance

本文件记录仓库各部分的来源和许可边界。它不是法律意见，也不会替代各上游项目自己的许可证。

## Repository-level license

仓库根目录包含 MIT `LICENSE`。该文件仅适用于仓库维护者有权以 MIT 许可发布的原创整合、说明和脚本，不会覆盖或重新许可第三方代码。

## llama.cpp / GGML snapshots

以下目录保留各自源树中的 MIT `LICENSE`：

- `projects/w4-w8-a16/src/llama.cpp-npu-htp-backend/`
- `projects/pure-fp16/src/llama.cpp-npu-htp-backend/`
- `projects/flashattention/src/llama.cpp-npu-htp-backend/`

研究分支来源：

- <https://github.com/haozixu/llama.cpp-npu>
- llama.cpp upstream: <https://github.com/ggerganov/llama.cpp>

这些本地研究快照没有可验证的完整上游 Git 历史，因此本仓库不声称它们对应某个精确上游 commit。

## HTP operator snapshots

以下目录来自 HTP operator 研究实现：

- `projects/w4-w8-a16/src/htp-ops-lib-main/`
- `projects/pure-fp16/src/htp-ops-lib-main/`
- `projects/flashattention/src/htp-ops-lib-main/`

来源与论文：

- <https://github.com/haozixu/htp-ops-lib>
- Zixu Hao et al., *Scaling LLM Test-Time Compute with Mobile NPU on Smartphones*, arXiv:2509.23324.

本地快照和检查到的上游均未随附明确的 `LICENSE`。因此：

- 不要推断这些文件属于 MIT；
- 在公开仓库发布、复制或再分发前，应取得原作者/权利人的许可；
- 本仓库应保持私有，直至许可状态得到确认。

两份明确带有 Qualcomm proprietary/confidential 标记的头文件没有进入仓库：

```text
include/dsp/hvx_internal.h
include/dsp/worker_pool.h
```

构建者必须从其获授权的 Qualcomm Hexagon/QAIRT SDK 环境提供所需接口或等价实现。仓库也不包含 Qualcomm SDK、QAIC 生成物、PDF 或其他受限材料。

## MLLM snapshot

`reference/w4-w8-a16-mllm/` 来源：

- repository: <https://github.com/zcy-001-001/mllm>
- commit: `729ca4c9f28dae314beee77e15362be159918827`
- license: MIT（见该目录的 `LICENSE`）

快照没有携带嵌套 `.git`。其 12 个上游 submodule 未初始化，详情见 `reference/w4-w8-a16-mllm/SNAPSHOT.md`。一份 Qualcomm 保密 XML 没有进入快照：

```text
mllm/backends/qnn/custom-op-package/LLaMAPackage/config/LLaMAOpPackageHtp.xml
```

## Models and generated artifacts

模型权重、GGUF、共享库、可执行文件、目标文件、CMake cache、完整设备日志和大型派生 trace 都不属于源码发布内容，已通过 `.gitignore` 和上传前审计排除。
