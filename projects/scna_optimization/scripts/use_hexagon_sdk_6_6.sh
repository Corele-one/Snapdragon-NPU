#!/usr/bin/env bash
# Source before invoking build_cmake manually.
set -eo pipefail

scna_sdk_root="${SCNA_HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0}"
scna_setup="$scna_sdk_root/setup_sdk_env.source"
if [[ ! -f "$scna_setup" ]]; then
  echo "Hexagon SDK setup script not found: $scna_setup" >&2
  echo "Set SCNA_HEXAGON_SDK_ROOT to an authorized SDK 6.6 installation." >&2
  return 1 2>/dev/null || exit 1
fi

unset HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT SDK_SETUP_ENV
unset DEFAULT_HEXAGON_TOOLS_ROOT DEFAULT_DSP_ARCH DEFAULT_BUILD DEFAULT_HLOS_ARCH
unset DEFAULT_TOOLS_VARIANT DEFAULT_NO_QURT_INC DEFAULT_TREE DEFAULT_QURT_PATH
unset ANDROID_ROOT_DIR QNX_BIN_DIR
# shellcheck source=/dev/null
source "$scna_setup"

