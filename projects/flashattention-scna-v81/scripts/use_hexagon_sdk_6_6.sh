#!/usr/bin/env bash

# Source this file from a WSL shell before invoking build_cmake.
# The setup script does not replace an already active Hexagon SDK environment.
unset HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT SDK_SETUP_ENV
unset DEFAULT_HEXAGON_TOOLS_ROOT DEFAULT_DSP_ARCH DEFAULT_BUILD DEFAULT_HLOS_ARCH
unset DEFAULT_TOOLS_VARIANT DEFAULT_NO_QURT_INC DEFAULT_TREE DEFAULT_QURT_PATH
unset ANDROID_ROOT_DIR QNX_BIN_DIR LV_TOOLS_DIR LRH_TOOLS_DIR QCL_TOOLS_DIR
unset CMAKE_ROOT_PATH DEBUGGER_UTILS HEXAGONSDK_TELEMATICS_ROOT

source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0/setup_sdk_env.source

# The SDK 6.6 package's bundled CMake directory is currently inaccessible to
# this user. CMake/Ninja are host-only generators; reuse the executable 3.28.3
# bundle from the readable 6.3 installation while retaining all 6.6 target
# tools, QuRT headers, and FastRPC libraries.
export CMAKE_ROOT_PATH=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/tools/cmake-3.28.3-linux-x86_64
