cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(dh_comms LANGUAGES HIP CXX)
find_package(hip REQUIRED)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
message(STATUS "CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})

set(DH_COMMS_SOURCES
    src/dh_comms.cpp
    src/memory_heatmap.cpp
    src/data_headers.cpp
    src/message.cpp
    src/message_handlers.cpp
    src/time_interval_handler.cpp
    src/memory_analysis_handler.cpp
)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
include(env)

link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/kernelDB/lib)
set_source_files_properties(${DH_COMMS_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(dh_comms SHARED ${DH_COMMS_SOURCES})
target_compile_options(dh_comms PRIVATE -fgpu-rdc -Werror -Wall -Wextra)

target_include_directories (
    dh_comms
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_ROOT_DIR}/include
    ${ROOT_DIR}
    ${HSA_RUNTIME_INC_PATH}
    ${HSA_KMT_LIB_PATH}/..
    ${KERNELDB_INCLUDE_DIR}
)
target_link_libraries(
    dh_comms
PRIVATE
    kernelDB64
)

target_link_options(dh_comms PRIVATE -fgpu-rdc)
set_target_properties(dh_comms PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

file(GLOB_RECURSE DH_COMMS_INCLUDE_FILES "include/*.h")
set_target_properties(dh_comms PROPERTIES PUBLIC_HEADER "${DH_COMMS_INCLUDE_FILES}")
include(GNUInstallDirs)
install(TARGETS dh_comms
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/dh_comms)

message(STATUS "==> {Library,include} install dirs: ${CMAKE_INSTALL_PREFIX}/{${CMAKE_INSTALL_LIBDIR},${CMAKE_INSTALL_INCLUDEDIR}/dh_comms}")
message(STATUS "==> To change install dirs, rerun cmake with argument \"-DCMAKE_INSTALL_PREFIX=/path/to/install/dir\"")

add_subdirectory(examples)
add_subdirectory(doc)

add_custom_command(
    OUTPUT dh_comms_bc.stamp
    COMMAND mkdir -p lib
    # Generate co4 version (code object version 4)
    COMMAND ${CMAKE_HIP_COMPILER} -Wno-pragma-once-outside-header -mcode-object-version=4
        -O3 -I${ROCM_ROOT_DIR}/include -I${HSA_RUNTIME_INC_PATH}
        $<$<BOOL:${KERNELDB_INCLUDE_DIR}>:-I${KERNELDB_INCLUDE_DIR}>
        $<$<BOOL:${LIBDWARF_INCLUDE_DIR}>:-I${LIBDWARF_INCLUDE_DIR}> -x hip
        --cuda-device-only -emit-llvm -S ${CMAKE_CURRENT_SOURCE_DIR}/include/dh_comms_dev.h
        -o lib/dh_comms_dev_co4.ll
    COMMAND sed -i 's/\"target-cpu\"=\"[^\"]*\" //g' lib/dh_comms_dev_co4.ll
    COMMAND ${ROCM_ROOT_DIR}/llvm/bin/llvm-as lib/dh_comms_dev_co4.ll -o lib/dh_comms_dev_co4.bc
    # Generate co5 version (default code object version 5)
    COMMAND ${CMAKE_HIP_COMPILER} -Wno-pragma-once-outside-header
        -O3 -I${ROCM_ROOT_DIR}/include -I${HSA_RUNTIME_INC_PATH}
        $<$<BOOL:${KERNELDB_INCLUDE_DIR}>:-I${KERNELDB_INCLUDE_DIR}>
        $<$<BOOL:${LIBDWARF_INCLUDE_DIR}>:-I${LIBDWARF_INCLUDE_DIR}> -x hip
        --cuda-device-only -emit-llvm -S ${CMAKE_CURRENT_SOURCE_DIR}/include/dh_comms_dev.h
        -o lib/dh_comms_dev_co5.ll
    COMMAND sed -i 's/\"target-cpu\"=\"[^\"]*\" //g' lib/dh_comms_dev_co5.ll
    COMMAND ${ROCM_ROOT_DIR}/llvm/bin/llvm-as lib/dh_comms_dev_co5.ll -o lib/dh_comms_dev_co5.bc
    COMMAND touch dh_comms_bc.stamp
    DEPENDS include/dh_comms_dev.h
)

add_custom_target(dh_comms_bc ALL
    DEPENDS dh_comms_bc.stamp
)

set_property(TARGET dh_comms_bc APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${DH_COMMS_BITCODE_FILES})

install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/lib/dh_comms_dev_co4.bc
    ${CMAKE_CURRENT_BINARY_DIR}/lib/dh_comms_dev_co5.bc
    DESTINATION lib
)
