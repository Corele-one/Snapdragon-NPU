#!/usr/bin/env bash
# Source this file before invoking build_cmake manually.
# SCNA_HEXAGON_SDK_ROOT may be set to use another authorized SDK installation.

# Qualcomm's setup script itself reads a few optional, undefined variables, so
# this helper must remain compatible with shells that do not enable `nounset`.
set -eo pipefail

scna_sdk_root="${SCNA_HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0}"
scna_setup="$scna_sdk_root/setup_sdk_env.source"

if [[ ! -f "$scna_setup" ]]; then
  echo "Hexagon SDK setup script not found: $scna_setup" >&2
  echo "Set SCNA_HEXAGON_SDK_ROOT to the authorized SDK directory." >&2
  return 1 2>/dev/null || exit 1
fi

# Qualcomm's setup script keeps an already-active SDK environment.  Clear it so
# the selected SDK, its QAIC, and its target runtime always match.
unset HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT SDK_SETUP_ENV
unset DEFAULT_HEXAGON_TOOLS_ROOT DEFAULT_DSP_ARCH DEFAULT_BUILD DEFAULT_HLOS_ARCH
unset DEFAULT_TOOLS_VARIANT DEFAULT_NO_QURT_INC DEFAULT_TREE DEFAULT_QURT_PATH
unset ANDROID_ROOT_DIR QNX_BIN_DIR LV_TOOLS_DIR LRH_TOOLS_DIR QCL_TOOLS_DIR
unset CMAKE_ROOT_PATH DEBUGGER_UTILS HEXAGONSDK_TELEMATICS_ROOT

# shellcheck source=/dev/null
source "$scna_setup"
