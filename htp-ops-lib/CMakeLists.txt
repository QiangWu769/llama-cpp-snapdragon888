cmake_minimum_required(VERSION 3.14.3)
project(htp_ops C CXX ASM)
# cmake_policy(SET CMP0115 OLD)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

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

# Keep the established v68 HVX path available; HMX uses a legacy ISA branch.
set(HTP_HMX_DEFAULT ON)
if(HEXAGON_ARCH STREQUAL "v68")
    set(HTP_HMX_DEFAULT OFF)
endif()
option(HTP_USE_HMX "Build FP16 HMX kernels (legacy ISA on v68)" ${HTP_HMX_DEFAULT})

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

    add_executable(htp_v68_test src/host/v68_test.c)
    set_common_compile_and_link_options(htp_v68_test)
    target_link_libraries(htp_v68_test htp_ops m)
    copy_binaries(htp_v68_test)

else()

    # Shared skel library
    add_library(htp_ops_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_skel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/commu.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/power.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/worker_pool.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/mmap_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/op_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/rms_norm.c
    )
    build_idl(include/htp_ops.idl htp_ops_skel)

    # Enable HMX inline assembly
    if(HTP_USE_HMX)
        target_compile_definitions(htp_ops_skel PRIVATE HTP_USE_HMX=1)
        target_compile_options(htp_ops_skel PRIVATE -mhmx)
        if(HEXAGON_ARCH STREQUAL "v68")
            target_compile_definitions(htp_ops_skel PRIVATE HTP_HMX_V68=1)
            message(STATUS "HTP backend: v68 legacy FP16 HMX kernels")
        endif()
        target_sources(htp_ops_skel PRIVATE
            src/dsp/hmx_mgr.c src/dsp/vtcm_mgr.cc
            src/dsp/ops/flash_attn.c
            src/dsp/ops/mat_mul.c)
        if(NOT HEXAGON_ARCH STREQUAL "v68")
            target_sources(htp_ops_skel PRIVATE src/dsp/ops/flash_attn_sp_hdim.c
                src/dsp/op_tests.cc src/dsp/ops/precompute_table.c src/dsp/ops/mm_benchmark.c)
        endif()
    else()
        target_compile_definitions(htp_ops_skel PRIVATE HTP_USE_HMX=0)
        target_sources(htp_ops_skel PRIVATE
            src/dsp/ops/mat_mul_v68.c src/dsp/ops/flash_attn_v68.c)
        message(STATUS "HTP backend: v68 HVX kernels (no FP16 HMX)")
    endif()

    # Link C++ libraries
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.so.1)
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.so.1)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops_skel)

endif()
