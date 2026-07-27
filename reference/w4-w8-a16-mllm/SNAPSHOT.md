# MLLM W4A16/W8A16 snapshot

该目录用于补充 `projects/w4-w8-a16/` 的 llama.cpp HTP weight-only 路径，提供字面意义上的 QNN graph W4A16/W8A16 参考。

## Provenance

- repository: <https://github.com/zcy-001-001/mllm>
- commit: `729ca4c9f28dae314beee77e15362be159918827`
- license: MIT（见本目录 `LICENSE`）
- snapshot policy: source-only，未携带嵌套 `.git`、模型或构建输出

未上传的 Qualcomm confidential 文件：

```text
mllm/backends/qnn/custom-op-package/LLaMAPackage/config/LLaMAOpPackageHtp.xml
```

## Relevant implementation

- `mllm/backends/qnn/op/QNNLinearOp.cpp`：QNN linear/W8A16 相关实现。
- `mllm/backends/qnn/aot/passes/LLMQuantRecipePass.cpp`：W4A16 LPBQ recipe/pass。
- `mllm/backends/qnn/QNNUtils.cpp`：UInt16 与 QNN UFIXED16 类型映射。
- `examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B.json`：W4A16 配置示例。
- `examples/qwen3_qnn_aot/qnn_aot_cfg_4B.json`：W4A16 + W8A16 混合配置示例。
- `docs/qnn_backend/aot_execute.rst`：QNN AOT 流程。

## Submodules

上游 commit 声明了 12 个 submodule，但本快照中的对应 gitlink 内容没有初始化：

```text
third_party/fmt
third_party/googletest
third_party/benchmark
mllm/backends/cpu/vendors/kleidiai
mllm/backends/cuda/vendors/cccl
mllm/backends/cuda/vendors/cutlass
mllm/ffi/vendors/tvm-ffi
mllm/ext/vendors/llvm-project
mllm/ext/vendors/tokenizers
mllm-chat
mllm/backends/ascend/vendors/catlass
third_party/flatbuffers
```

因此该 vendored snapshot 适合源码阅读和定位 QNN W4A16/W8A16 实现，不保证单独 checkout 后直接完成全功能构建。需要完整 MLLM 构建时，推荐直接 clone 上述仓库并 checkout 固定 commit：

```bash
git clone --recurse-submodules https://github.com/zcy-001-001/mllm.git
cd mllm
git checkout 729ca4c9f28dae314beee77e15362be159918827
git submodule update --init --recursive
```

随后根据上游 `README.md` 和 `docs/qnn_backend/aot_execute.rst` 配置 QNN/QAIRT 环境。不要从本仓库恢复或传播已排除的 Qualcomm confidential XML。
