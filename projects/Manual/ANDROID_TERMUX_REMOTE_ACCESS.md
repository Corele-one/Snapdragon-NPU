# 通过 VPS 远程部署与调试 Android NPU

本文适用于以下场景：WSL2 笔记本随开发者出差，Android NPU 平板留在远端且不再与 WSL2 有线连接。Google VPS 只作为公网 SSH 中继；构建、`adb` 命令和结果分析仍在 WSL2 完成，NPU 程序在 Android 的 `adb shell` 域运行。

## 给后续 Agent 的强制执行规则

后续 Agent 在部署、测试或调试 Android NPU 前必须先阅读本节，并遵守以下规则：

1. 当前有效设备固定为 `127.0.0.1:15555`。所有设备操作必须显式使用 `adb -s 127.0.0.1:15555 ...`，不得依赖 adb 自动选择 transport。
2. 首先运行 `/home/corleone/code/server/scripts/open_remote_adb.sh`，然后检查 `get-state`、`id` 和 `u:r:shell:s0`；预检失败时不得部署。
3. 不得恢复旧的 VPS 5037 → WSL USB ADB bridge。当前方案不依赖 USB，也不在 VPS 上执行 adb client。
4. 不得在 Termux 或 Termux `su` 中直接运行 FastRPC/HTP/NPU 二进制；Termux 只负责 SSH 管理与 Android → VPS 反向隧道。
5. 部署前确认本地产物存在；部署后对主程序、CPU stub 和 DSP skeleton 计算 SHA-256，并与 Android 端结果逐项比较。
6. 调试时优先使用新的独立目录，避免覆盖仍在使用的设备端产物；除非用户明确要求，不得删除旧结果或修改持久化 adbd、防火墙和隧道配置。
7. 至少运行一次最小 NPU smoke test；更改 Attention Kernel 后还应运行 reference/correctness test。必须报告命令、设备目录、退出状态和关键结果。

Agent 开始工作的标准预检：

```sh
/home/corleone/code/server/scripts/open_remote_adb.sh
adb devices -l
adb -s 127.0.0.1:15555 get-state
adb -s 127.0.0.1:15555 shell 'id; getenforce; getprop ro.board.platform'
```

验收条件：目标状态为 `device`，身份包含 `uid=2000(shell)` 和 `context=u:r:shell:s0`，平台为 `sun`。出现 `offline`、`unauthorized`、`error: closed` 或 connection refused 时，先执行本文“故障恢复”，不得绕过检查继续部署。

## 当前已验证状态（2026-08-19）

- 已在物理 USB 断开后确认 adb 列表仅保留远程 transport `127.0.0.1:15555`。
- 已通过该 transport 将全新的 FlashAttention v81 产物部署到 `/data/local/tmp/scna_hmx_fp16_d8_v81_remote_test`。
- WSL2 与 Android 端的 `htp_ops_test`、`libhtp_ops.so` 和 `libhtp_ops_skel.so` SHA-256 完全一致。
- 已成功运行 SCNA/HVX smoke test，并成功启动 `scna-hmx-fp16-d8-hybrid-attn-pipeline` Figure-8 Attention 测试。
- Android 重启后，adbd 5555、防火墙规则、Termux 8022/15555 反向隧道均可自动恢复。

## 已配置架构

```text
WSL2 adb client/server
  -> WSL2 127.0.0.1:15555
  -> SSH local forward
  -> VPS 127.0.0.1:15555
  -> Termux SSH reverse forward
  -> Android 127.0.0.1:5555
  -> adbd (u:r:shell:s0)
  -> FastRPC / HTP / NPU
```

管理终端使用独立通道：

```text
WSL2 -- mosh --> VPS -- VPS 127.0.0.1:8022 --> Termux sshd
```

端口均只绑定回环地址，不向 VPS 公网开放：

| 端口 | 位置 | 用途 |
| --- | --- | --- |
| `8022` | VPS `127.0.0.1` | SSH 登录 Termux |
| `15555` | VPS `127.0.0.1` | 中继 Android TCP adbd |
| `5555` | Android | adbd；防火墙仅允许 Android 本机回环访问 |

## 为什么不能直接在 Termux 中运行 NPU 程序

Termux SSH 默认位于 `u:r:untrusted_app:s0`。即使 `su` 得到 `uid=0`，进程仍位于 `u:r:ksu:s0`，无法等价替代 Android 的 `u:r:shell:s0`，并可能出现 FastRPC 会话失败或 Binder 库符号错误。

本方案通过真正的 `adbd` 获取 `u:r:shell:s0`，因此可以继续使用原项目的 `adb push`、`adb shell`、`adb pull`、FastRPC 和 HTP 调试流程。

## 平板端持久化配置

平板已安装以下 KernelSU 开机服务：

```text
/data/adb/service.d/99-remote-adbd.sh
```

它会在开机时：

1. 将 adbd TCP 端口设为 5555；
2. 启动 adbd；
3. 添加 IPv4/IPv6 防火墙规则，拒绝所有非回环接口访问 5555。

Termux 的 `~/.local/bin/termux-vps-tunnel` 同时建立两条反向隧道：

```text
VPS 127.0.0.1:8022  -> Android 127.0.0.1:8022
VPS 127.0.0.1:15555 -> Android 127.0.0.1:5555
```

Termux 已通过系统设置允许开机启动，其 `.bashrc` 会调用 `~/.local/bin/ensure-vps-tunnel`，使用 `autossh` 自动恢复隧道。

## WSL2 建立远程 ADB

每次 WSL2 重启或 SSH 本地转发断开后，在 WSL2 执行：

```sh
/home/corleone/code/server/scripts/open_remote_adb.sh
```

等价的手动命令为：

```sh
ssh -fN \
  -L 127.0.0.1:15555:127.0.0.1:15555 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  wzliao@35.209.202.207

adb start-server
adb connect 127.0.0.1:15555
adb -s 127.0.0.1:15555 get-state
```

预期最后输出：

```text
device
```

确认权限域：

```sh
adb -s 127.0.0.1:15555 shell id
```

输出应包含：

```text
uid=2000(shell) ... context=u:r:shell:s0
```

## 远程部署 NPU Kernel

所有命令均在 WSL2 项目目录执行，无需把构建产物复制到 VPS。示例部署目录：

```sh
export ANDROID_REMOTE=127.0.0.1:15555
export NPU_REMOTE_DIR=/data/local/tmp/scna_hmx_fp16_d8_v81

adb -s "$ANDROID_REMOTE" shell \
  "mkdir -p $NPU_REMOTE_DIR/cdsp $NPU_REMOTE_DIR/dsp"

adb -s "$ANDROID_REMOTE" push \
  src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/htp_ops_test \
  "$NPU_REMOTE_DIR/"

adb -s "$ANDROID_REMOTE" push \
  src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/libhtp_ops.so \
  "$NPU_REMOTE_DIR/"

adb -s "$ANDROID_REMOTE" push \
  src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so \
  "$NPU_REMOTE_DIR/cdsp/"

adb -s "$ANDROID_REMOTE" push \
  src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so \
  "$NPU_REMOTE_DIR/dsp/"

adb -s "$ANDROID_REMOTE" shell "chmod 755 $NPU_REMOTE_DIR/htp_ops_test"
```

标准项目脚本如未显式指定设备，应在只连接远程 transport 时运行，或将其中的 `adb` 调用改为 `adb -s "$ANDROID_REMOTE"`，避免误选其他设备。

部署后必须做完整性校验。示例：

```sh
sha256sum \
  src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/htp_ops_test \
  src/htp-ops-lib-main/android_ReleaseG_aarch64/ship/libhtp_ops.so \
  src/htp-ops-lib-main/hexagon_ReleaseG_toolv19_v81/ship/libhtp_ops_skel.so

adb -s "$ANDROID_REMOTE" shell \
  "cd $NPU_REMOTE_DIR && sha256sum \
    htp_ops_test libhtp_ops.so \
    cdsp/libhtp_ops_skel.so dsp/libhtp_ops_skel.so"
```

## 运行 FlashAttention NPU 测试

```sh
adb -s 127.0.0.1:15555 shell '
  cd /data/local/tmp/scna_hmx_fp16_d8_v81 &&
  LD_LIBRARY_PATH=. DSP_LIBRARY_PATH="./cdsp;./dsp;." \
  ./htp_ops_test --scna-exp-bench --mode scna-hvx-fp16-d8 \
    --scna-width 8 --warmup 1 --iters 1
'
```

该链路已在平板重启后显式指定远程 transport `127.0.0.1:15555` 验证成功，并返回正常 `SCNA_EXP_BENCH` 结果；测试命令不使用 USB 序列号。

## 调试与结果回传

实时日志：

```sh
adb -s 127.0.0.1:15555 logcat
```

清空日志后重新采集：

```sh
adb -s 127.0.0.1:15555 logcat -c
adb -s 127.0.0.1:15555 logcat
```

将设备结果直接拉回 WSL2：

```sh
adb -s 127.0.0.1:15555 pull \
  /data/local/tmp/scna_hmx_fp16_d8_v81/results \
  ./remote-npu-results
```

进入 Termux 做辅助检查：

```sh
mosh -p 60000:60010 wzliao@35.209.202.207
ssh termux-android
```

## 故障恢复

| 现象 | 检查与处理 |
| --- | --- |
| `adb connect` 显示 connection refused | 在 VPS 检查 `ss -ltn 'sport = :15555'`；无监听说明 Termux 隧道未恢复。 |
| WSL2 本地 15555 无监听 | 重新运行 `open_remote_adb.sh`。 |
| ADB 显示旧 transport 或 `error: closed` | 执行 `adb disconnect 127.0.0.1:15555`，再运行连接脚本。 |
| VPS 15555 不存在但 8022 存在 | 登录 Termux，检查 `~/.local/state/termux-vps-tunnel.log` 并重启 `ensure-vps-tunnel`。 |
| 8022 和 15555 都不存在 | Termux 未开机启动或网络不可用，需要现场恢复 Termux。 |
| `unauthorized` | WSL2 的 ADB 密钥未被平板授权；出差前必须完成授权验证。 |

Agent 遇到 `error: closed` 或重启后的旧 transport 时，应执行：

```sh
adb disconnect 127.0.0.1:15555
/home/corleone/code/server/scripts/open_remote_adb.sh
adb -s 127.0.0.1:15555 get-state
```

若仍无法连接，在 WSL2 检查本地端口，在 VPS 检查反向端口：

```sh
ss -ltn 'sport = :15555'
ssh wzliao@35.209.202.207 "ss -ltn 'sport = :15555 or sport = :8022'"
```

## 安全要求

- 不得把 VPS 的 8022 或 15555 改为 `0.0.0.0`。
- 不得删除平板 INPUT 链中针对非回环 5555 的 DROP 规则。
- 不要把 Android 5555 直接暴露到 Wi-Fi、互联网或路由器端口转发。
- 不要关闭 SELinux；NPU 调试应始终走受认证的 `adb shell`。
- 出差前至少完成一次平板重启测试，并确认 USB transport 消失后 `127.0.0.1:15555` 仍能运行 NPU smoke test。
