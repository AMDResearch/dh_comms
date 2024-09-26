cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(dh_comms LANGUAGES HIP CXX)
find_package(hip REQUIRED)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
MESSAGE(STATUS "CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})

set(DH_COMMS_SOURCES
    src/dh_comms.cpp
    src/memory_heatmap.cpp
)
set_source_files_properties(${DH_COMMS_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(dh_comms SHARED ${DH_COMMS_SOURCES})
target_compile_options(dh_comms PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_include_directories(dh_comms PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_options(dh_comms PRIVATE -fgpu-rdc)
set_target_properties(dh_comms PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(examples)
add_subdirectory(doc)
