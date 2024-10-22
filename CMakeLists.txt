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
)
set_source_files_properties(${DH_COMMS_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(dh_comms SHARED ${DH_COMMS_SOURCES})
target_compile_options(dh_comms PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_include_directories(dh_comms PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_options(dh_comms PRIVATE -fgpu-rdc)
set_target_properties(dh_comms PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

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

# Ugly hack to create bitcode file(s) from one of the headers with inline device functions
# To be changed into a proper CMake approach after attending to higher-priority tasks
add_custom_target(dh_comms_bc ALL
    COMMAND ${CMAKE_HIP_COMPILER} -Wno-pragma-once-outside-header -O3 -x hip -c -save-temps ${CMAKE_SOURCE_DIR}/include/dh_comms_dev.h
    COMMAND mkdir -p lib
# ignore multiple GPU archs for now, and assume a single one
    COMMAND bash -c \'mv dh_comms*-hip-*.bc lib/dh_comms_dev.bc\'
    COMMAND rm -f dh_comms_dev-hip* dh_comms_dev-host* dh_comms_dev*hipfb dh_comms_dev.o
)

install(PROGRAMS
    ${CMAKE_BINARY_DIR}/lib/dh_comms_dev.bc
    DESTINATION lib)
