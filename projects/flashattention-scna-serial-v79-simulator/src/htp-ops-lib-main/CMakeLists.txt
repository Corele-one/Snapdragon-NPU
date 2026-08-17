cmake_minimum_required(VERSION 3.14.3)
project(htp_ops C CXX ASM)
# cmake_policy(SET CMP0115 OLD)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(FIGURE8_ENABLE_LUT_EXP "Use LUT/vgather exp in the Figure 8 profiled flash attention path" OFF)
option(FIGURE8_ENABLE_PROFILE_TIMERS "Enable Figure 8 flash attention profiling timers/logging" ON)
set(SCNA_BUILD_VARIANT "0" CACHE STRING "SCNA optimization build id (0..6)")
set(SCNA_OPTIMIZED_INLINE "0" CACHE STRING "Final optimized call policy: 0=noinline, 1=always_inline")
option(SCNA_SIMULATOR "Build the QuRT run-main simulator entry point" ON)
option(HTP_ENABLE_OUT_STATIONARY_PREFILL "Enable the repaired output-stationary prefill matmul path" OFF)
option(HTP_OS_DEBUG_COMPARE "Enable output-stationary debug accumulator checks" OFF)
set(HTP_MATMUL_PIPELINE_MODE "0" CACHE STRING "Matmul qk pipeline mode: 0=auto original 4-stage path, 1=force sequential safe path")
set(HTP_MATMUL_WEIGHT_AREA_KB "1024" CACHE STRING "VTCM KiB reserved for matmul weights")
set(HTP_MATMUL_ACTIVATION_AREA_KB "2048" CACHE STRING "VTCM KiB reserved for matmul activations")
set(HTP_MATMUL_OUTPUT_AREA_KB "1536" CACHE STRING "VTCM KiB reserved for matmul outputs")
set(HTP_MATMUL_SCRATCH_AREA_KB "1024" CACHE STRING "VTCM KiB reserved for each matmul scratch buffer")

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

set(ignoreVar ${DEPENDENCIES}${V})

# NOTE(hzx): QAIC generated code are located in ${CMAKE_CURRENT_BINARY_DIR}
set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_BINARY_DIR}
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)
include_directories(${common_incs})

if(${OS_TYPE} MATCHES "HLOS")
    # HLOS dynamic binary
    add_library(htp_ops SHARED 
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/op_export.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/session.c
    )

    # Build interface files
    build_idl(include/htp_ops.idl htp_ops)

    # HLOS executable
    add_executable(htp_ops_test
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/test.c
    )
    
    # Add common HLOS-specific linker options
    set_common_compile_and_link_options(htp_ops_test)

    target_compile_definitions(htp_ops_test PUBLIC VERIFY_PRINT_ERROR)
    if(${CMAKE_SYSTEM_NAME} MATCHES "UbuntuARM")
        target_link_options(htp_ops_test PUBLIC -lpthread -ldl)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        target_link_options(htp_ops_test PUBLIC -llog -ldl)
    endif()

    # Find the DSP from provided domain value, default is CDSP
    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(htp_ops ${dsprpc})

    add_dependencies(htp_ops_test htp_ops)
    target_link_libraries(htp_ops_test htp_ops)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops_test htp_ops)

else()

    set(scna_sim_sources)
    if(SCNA_SIMULATOR)
        list(APPEND scna_sim_sources
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/sim_main.c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/sim_hmx_compat.c
        )
    endif()

    # Shared skel library
    add_library(htp_ops_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_skel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/commu.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/power.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/hmx_mgr.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/worker_pool.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/mmap_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/op_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/op_tests.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/vtcm_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/flash_attn_sp_hdim.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/flash_attn.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/mat_mul.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/precompute_table.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/rms_norm.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/scna_exp2.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/mm_benchmark.c
        ${scna_sim_sources}
    )
    build_idl(include/htp_ops.idl htp_ops_skel)

    # Enable HMX inline assembly
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mhmx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mhmx")
    # V81 BF16/FP8 HVX intrinsics require the IEEE floating-point lowering
    # path.  Without this flag the declarations parse, but backend selection
    # fails for the real 128-byte vmpyacc instructions.  Do not apply the flag
    # target-wide: existing qf16 helpers intentionally use the qf16 lowering.
    if(HEXAGON_ARCH STREQUAL "v81")
        # target_compile_options(htp_ops_skel PRIVATE -mhvx-ieee-fp)
        # The target-wide form above is retained as a warning: SDK 19 reports
        # "hf type used as qf16" in hvx_math.h when it reaches op_tests.cc.
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/op_executor.cc
            PROPERTIES COMPILE_OPTIONS "-mhvx-ieee-fp"
        )
    endif()
    target_compile_definitions(htp_ops_skel PRIVATE
        FIGURE8_ENABLE_LUT_EXP=$<IF:$<BOOL:${FIGURE8_ENABLE_LUT_EXP}>,1,0>
        FIGURE8_ENABLE_PROFILE_TIMERS=$<IF:$<BOOL:${FIGURE8_ENABLE_PROFILE_TIMERS}>,1,0>
        SCNA_BUILD_VARIANT=${SCNA_BUILD_VARIANT}
        SCNA_OPTIMIZED_INLINE=${SCNA_OPTIMIZED_INLINE}
        HTP_ENABLE_OUT_STATIONARY_PREFILL=$<IF:$<BOOL:${HTP_ENABLE_OUT_STATIONARY_PREFILL}>,1,0>
        HTP_OS_DEBUG_COMPARE=$<IF:$<BOOL:${HTP_OS_DEBUG_COMPARE}>,1,0>
        HTP_MATMUL_PIPELINE_MODE=${HTP_MATMUL_PIPELINE_MODE}
        HTP_MATMUL_WEIGHT_AREA_KB=${HTP_MATMUL_WEIGHT_AREA_KB}
        HTP_MATMUL_ACTIVATION_AREA_KB=${HTP_MATMUL_ACTIVATION_AREA_KB}
        HTP_MATMUL_OUTPUT_AREA_KB=${HTP_MATMUL_OUTPUT_AREA_KB}
        HTP_MATMUL_SCRATCH_AREA_KB=${HTP_MATMUL_SCRATCH_AREA_KB}
        SCNA_SIMULATOR=$<IF:$<BOOL:${SCNA_SIMULATOR}>,1,0>
    )

    # Link C++ libraries
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.so.1)
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.so.1)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops_skel)

endif()

# Keep the original environment-only ping target alongside the migrated HTP
# operator library. It deliberately does not initialize VTCM or HMX, so it can
# isolate FastRPC deployment failures from the FlashAttention baselines.
if(${OS_TYPE} MATCHES "HLOS")
    add_library(scna_env SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/scna_env_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
    )
    build_idl(include/scna_env.idl scna_env)

    add_executable(scna_env_smoke
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/scna_env_smoke.c
    )
    set_common_compile_and_link_options(scna_env_smoke)
    if(${CMAKE_SYSTEM_NAME} MATCHES "UbuntuARM")
        target_link_options(scna_env_smoke PRIVATE -lpthread -ldl)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        target_link_options(scna_env_smoke PRIVATE -llog -ldl)
    endif()
    choose_dsprpc(${DSP_TYPE} scna_env_dsprpc)
    link_custom_library(scna_env ${scna_env_dsprpc})
    add_dependencies(scna_env_smoke scna_env)
    target_link_libraries(scna_env_smoke PRIVATE scna_env)
    copy_binaries(scna_env_smoke scna_env)
else()
    add_library(scna_env_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/scna_env_skel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/scna_env_skel.c
    )
    build_idl(include/scna_env.idl scna_env_skel)
    copy_binaries(scna_env_skel)
endif()
