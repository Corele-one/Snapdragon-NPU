# htp-ops-lib-main: 从 Android 调用到 HTP FlashAttention

这份笔记以本仓库的代码为准。目标不是把每条 HVX/HMX 指令背下来，而是能回答三个问题：数据如何从 Android 到 cDSP，`flash_attn.c` 如何在有限 VTCM 中计算 attention，以及出错时先检查哪一层。

## 1. 这个目录实际提供什么

`htp-ops-lib-main` 是一个自定义 FastRPC 算子库，不是完整的 llama.cpp 程序。它会构建三件东西：

| 产物 | 运行位置 | 作用 |
| --- | --- | --- |
| `htp_ops_test` | Android CPU | 独立测试、分配共享内存、发请求 |
| `libhtp_ops.so` | Android CPU | FastRPC stub，负责把普通 C 调用封装成 RPC |
| `libhtp_ops_skel.so` | cDSP/HTP | 真正执行 HVX/HMX kernel 的动态库 |

后面的 `llama.cpp-npu-htp-backend` 是上层调用者。它会把 GGML 的 RMSNorm、MatMul、FlashAttention 等任务打包后交给这里。因此先让本目录的 `htp_ops_test` 成功，是正确的自底向上验证顺序。

```text
htp_ops_test / llama.cpp backend
        |
        |  libhtp_ops.so + FastRPC
        v
Android FastRPC driver (/dev/fastrpc-cdsp)
        |
        v
libhtp_ops_skel.so on cDSP
        |
        +-- init_backend: power, VTCM, HMX
        +-- message channel: receive requests from rpcmem
        +-- execute_op_simple: map buffers, run an op, flush results
        +-- flash_attn.c: HMX/HVX FlashAttention kernel
```

两个 RPC 通路要区分开：

1. `include/htp_ops.idl` 定义的 FastRPC 调用只用于启动后端、创建/销毁 channel 和少量直接 RPC。
2. 高频算子走自定义 message channel。host 将一个 `rpcmem` 缓冲区交给 DSP；之后 host 与 DSP 通过其中的 `MessageHeader`、`RequestHeader` 和 payload 通信。定义在 `include/message.h`。

这样设计的原因是：逐次 FastRPC 调用的调度开销对 LLM 小算子很显眼，而共享 message buffer 可在一次会话中提交很多请求。

## 2. 初始化顺序

Host 的会话逻辑在 `src/host/session.c`：

1. `open_dsp_session()` 通过 `htp_ops_open()` 打开 cDSP 动态库。
2. `init_htp_backend()` 调用 DSP 的 `htp_ops_init_backend()`。
3. `create_htp_message_channel()` 将 `rpcmem` 的 fd 和 buffer 大小交给 DSP。

DSP 端在 `src/dsp/commu.c` 的 `htp_ops_init_backend()` 依次做：

1. `power_setup()`：请求 DCVS performance mode，并请求 HMX 上电。
2. `vtcm_manager_setup()`：用 `HAP_compute_res_*` 申请 VTCM，记录 `vtcm_base` 与大小。
3. `hmx_manager_setup()`：申请 HMX 资源，选择 shared/non-shared lock 模式。
4. `init_precomputed_tables()`：仅在启用 LUT exp 时预留并生成 softmax 的 `exp2` 查表。

`create_channel()` 随后在 DSP 建立 `hops-msg-recv` QuRT 线程。该线程循环读取共享内存中的状态位，先 invalidate cache，执行每个请求，flush payload，再以 release store 标记完成。这里的 cache 操作不是多余的：CPU 与 DSP 不是同一个 cache coherence 域。

## 3. 一个 FlashAttention 请求如何落到 kernel

`src/host/test.c` 的 Figure 8 测试会：

1. 分配 Q、K、V、mask、O 和 profile 的 rpcmem buffer。
2. 构造 `FlashAttnProfileParams`，把每个 buffer 的 `{fd, offset}` 和 shape 写进 message。
3. 将 `msg->state.v[0] = 1`，轮询 DSP 将 `v[1]` 写为完成。

DSP `msg_receiver_loop()` 收到 `REQUEST_TYPE_OP_COMPUTE` 后，进入 `execute_op_simple()`（`src/dsp/op_executor.cc`）。针对 `HTP_OPS_FLASH_ATTN_PROFILE_QO_F32_KV_F16`，它会：

1. 用 `HAP_mmap_get()` 将每个 rpcmem fd 映射到 DSP 虚拟地址。
2. 对输入做 cache invalidate。
3. 调用 `simple_flash_attn_profiled()`。
4. 对 O 和 profile 做 cache flush，使 Android CPU 能读到结果。

这也是调试时“no-op 成功”很有价值的原因：它证明 skel 加载、FastRPC、rpcmem、message channel、状态机和 cache 往返都已经正常；若 attention 仍超时，问题就已缩小到算子执行或相关硬件资源。

## 4. FlashAttention 的数学与数据形状

本实现的输入约定写在 `simple_flash_attn_impl()` 上方：

```text
Q: [qo_len, n_heads,    head_dim]
K: [kv_len, n_kv_heads, head_dim]
V: [kv_len, n_kv_heads, head_dim]
O: [qo_len, n_heads,    head_dim]
```

它支持 GQA。`G = n_heads / n_kv_heads`，即一个 KV head 对应 G 个 Q head。以这次的 `n_heads=12, n_kv_heads=2` 为例，`G=6`。

对每个 KV head，理论上的计算是：

```text
S = Q * K^T / sqrt(D) + mask
P = softmax(S)
O = P * V
```

直接把完整 `S=[qo_len, kv_len]` 放入内存会很大，因此 `flash_attn.c` 使用 online softmax。它在每个 K/V 列块处理后维护每行的：

```text
m = 已处理分块的最大 score
l = exp(score - m) 的累计和
O = 已归一化前的 V 加权累计
```

当新块到来时，旧累计先按新旧最大值的差缩放，再加上新块。这使得无需保存完整 attention matrix，并保持 softmax 数值稳定。

代码中将 `exp(x)` 写作 `exp2(x * log2(e))`，所以 `qk_scale` 是 `log2(e) / sqrt(head_dim)`；这正是 `simple_flash_attn_f16_core()` 中的常数来源。

## 5. `flash_attn.c` 是怎样分块的

入口为 `simple_flash_attn_impl()`：

1. `head_dim % 64 != 0` 时转到 `flash_attn_sp_hdim.c` 的通用路径。
2. 正常 HMX FP16 路径为每个 KV head 创建任务。
3. 每项任务由 `simple_flash_attn_worker()` 执行，并取得一段独占 VTCM。
4. worker 调用 `simple_flash_attn_f16_core()`。

`fa_f16_find_chunk_size()` 会根据该 worker 可用的 VTCM 决定 `Br`（query 行块）与 `Bc`（KV 列块）。`fa_f16_compute_vtcm_usage()` 是它的空间模型，主要包括：

| VTCM 区域 | 逻辑形状 | 用途 |
| --- | --- | --- |
| `q_tile`, `o_tile0`, `o_tile1` | `[Br' , D]` | Q 与两份 O 累计，便于 ping-pong |
| `k_tile`, `v_tile` | `[Bc, D]` | 当前 K/V 分块 |
| `s_tile`, `p_tile` | `[Br', Bc]` | score 与 softmax 概率 |
| `d_tile` | `[Br', Br']` | HMX/中间转换辅助 |
| `mvec_*`, `row_buffer*` | 向量 | row max、sum、转置和 softmax 辅助 |
| `hmx_output_scales_*` | 256 B 对齐 | HMX 输出 scale/bias 配置 |

`Br' = align_up(G * Br, 32)`，因为 HMX FP16 tile 的逻辑行列是 32 x 32。许多分配还按 4 KiB 或 256 B 对齐，既满足 HMX/HVX 的内存访问约束，也让不同工作线程在 VTCM 中有清晰边界。

核心循环的阅读顺序建议如下：

```text
for each Q row block (ir)
  load/convert Q -> q_tile
  initialize m, l, O accumulator
  for each KV column block (ic)
    load/repack K -> k_tile
    HMX: q_tile * k_tile^T -> s_tile
    add mask, compute block max and online softmax -> p_tile
    load/repack V -> v_tile
    HMX: p_tile * v_tile -> partial O
    combine partial O with old accumulator using m/l
  convert/store O tile to shared output
```

`hmx_utils.h` 里的几个函数很小却很关键：`hmx_init_column_scales()` 向 VTCM 写 HMX scale；`hmx_load_tiles_fp16()` 用内联汇编把 activation 与 weight tile 同一 packet 载入 HMX；`hmx_consume_accumulator_fp16()` 将 accumulator 写回 VTCM。它们要求的数据布局和对齐比普通 C 矩阵乘更严格。

## 6. 并行模型

默认设计是把 `n_kv_heads` 个任务分给 QuRT worker pool。每个 worker：

1. 用原子递增领取一个 KV head index。
2. 以 `worker_index * vtcm_size_per_thread` 取得自己的 VTCM 区域。
3. 申请 HMX execution lock，执行 kernel，释放 lock。
4. 用 `worker_pool_synctoken_jobdone()` 通知 host-side task coordinator。

HMX 是共享硬件资源，所以仅有多线程并不代表所有线程能随意同时发矩阵指令；`hmx_mgr.c` 负责使用 `HAP_compute_res_hmx_lock2()` 协调。新设备上 shared/non-shared 语义可能改变，不能把旧设备的锁策略当作硬件通用事实。

## 7. 本次 SM8750P 调试是如何收敛的

现象和结论按排障层次排列：

| 观察 | 排除/确认的范围 | 结论 |
| --- | --- | --- |
| `fastrpc_mmap failed, err: 1`，同时 cDSP 有 `TLBMISS` | FastRPC 不一定是根因 | DSP 在 `init_precomputed_tables()` 崩溃后，host 才看到 mmap 失败 |
| 禁用未启用的 LUT 预计算后，mmap 成功 | 初始化阶段 | VTCM LUT 预计算在此设备/旧构建组合下不安全，baseline 不应预计算它 |
| no-op channel request 成功 | 传输层 | skel、FastRPC、rpcmem、message channel 都正常 |
| attention 超时，profile breadcrumb 停在 VTCM allocation 之后、HMX scale 初始化之前 | kernel/hardware 资源层 | 不是 `adb shell` 权限，也不是 message channel；应检查目标 ISA、VTCM/HMX runtime compatibility |
| `DSP_ARCH=v81` 的实际编译命令包含 `-mv68` | 构建层 | 当前 SDK 6.3 的 CMake 不支持 v81，目录名中的 v81 不代表二进制真的是 v81 |

所以 root 或 `setenforce 0` 没有改变最后的超时是预期的：它们解决的是 Android 权限/SELinux，无法让 v68 代码变成 v81 HMX ABI。

## 8. SM8750P 的构建要求

`SM8750P` 是 Snapdragon 8 Elite，目标为 Hexagon v81。原来的 HAP SDK 6.3.0.0 包含的 CMake 映射只到 v79，并且缺少：

```text
rtos/qurt/computev81
Tools/target/hexagon/lib/v81
.../remote/.../hexagon_toolv*_v81
```

因此它不能完整构建 v81 skel。项目的 `CMakeLists.txt` 已增加 fail-fast 检查：请求 `DSP_ARCH=v81` 却解析到其他架构时立即停止。升级到包含 v81 QuRT、target libraries 和对应 Hexagon Tools 的 SDK 后，正确构建必须同时满足：

```text
CMake configure: HEXAGON_ARCH=v81
compile commands: -mv81
link inputs: .../computev81/... and .../lib/v81/...
output: hexagon_ReleaseG_toolv*_v81/ship/libhtp_ops_skel.so
```

不能只替换 `hexagon-clang`：QuRT headers/libraries、FastRPC remote prebuilts 与编译器必须来自同一代 SDK。

本机现已安装 HAP SDK 6.6.0.0 与 Hexagon Tools 19.0.07；它包含 v81 CMake 映射、`rtos/qurt/computev81` 和 v81 target runtime，v81 skel 的实际编译参数为 `-mv81`。在 WSL 中执行：

```bash
source scripts/use_hexagon_sdk_6_6.sh
cd src/htp-ops-lib-main
build_cmake android
build_cmake hexagon DSP_ARCH=v81 \
  FIGURE8_ENABLE_PROFILE_TIMERS=ON \
  FIGURE8_ENABLE_LUT_EXP=OFF
```

该脚本临时复用 SDK 6.3 的 CMake 3.28.3 host generator，因为当前 6.6 安装包的 CMake 目录没有此用户的执行权限；这不影响 DSP 的编译器、QuRT 或运行时库，它们仍全部来自 6.6。

### 当前设备的运行时资源状态

在这台手机上，`HAP_compute_res_query_VTCM()` 报告 VTCM 总量为 8192 KiB，但当前可用量为 0 KiB，HMX resource acquire 也超时。自定义 FastRPC PD 因此不能运行 FlashAttention；这与 SDK/ISA 无关。代码现在会返回 `-1`，不会再用空 VTCM 指针触发 cDSP TLBMISS。

设备上可见的 `com.xiaomi.aicr`、`com.xiaomi.aiservice` 等系统 AI 服务是优先排查对象。临时停止它们可能释放资源，但会影响语音、搜索和系统 AI 功能，必须确认后再执行。

## 9. 下次调试的最短路径

1. 用新版 SDK 构建后，先检查 `build.ninja` 或 verbose build 中是否真的出现 `-mv81`。
2. 先跑 no-op；失败就查 skel 搜索路径和 FastRPC。
3. 再跑 VTCM bandwidth/HMX smoke；失败先查 `HAP_compute_res_query_VTCM`、HMX lock 与设备 runtime 日志。
4. 最后用最小 attention shape，再增加 `kv_len`、head 数和迭代次数。
5. 出现 host 的 `fastrpc_mmap` 错误时，同时抓 cDSP 崩溃日志；它常常只是 DSP 侧更早失败后的二次症状。

这条顺序能避免在完整 llama.cpp 上调一个底层 skel 兼容问题。底层 smoke 未通过前，llama.cpp 路径只会把同一问题包装成更长的调用栈。
