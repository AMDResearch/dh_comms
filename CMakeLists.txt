cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(dh_comms LANGUAGES HIP CXX)
find_package(hip REQUIRED)


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

set(HEATMAP_EXAMPLE_SOURCES
    examples/src/heatmap_example.cpp
)
set_source_files_properties(${HEATMAP_EXAMPLE_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(heatmap_example ${HEATMAP_EXAMPLE_SOURCES})
target_compile_options(heatmap_example PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_include_directories(heatmap_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_options(heatmap_example PRIVATE -fgpu-rdc)
target_link_libraries(heatmap_example PRIVATE dh_comms)
set_target_properties(heatmap_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
