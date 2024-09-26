set(HEATMAP_EXAMPLE_SOURCES
    src/heatmap_example.cpp
)
set_source_files_properties(${HEATMAP_EXAMPLE_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(heatmap_example ${HEATMAP_EXAMPLE_SOURCES})
target_compile_options(heatmap_example PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_include_directories(heatmap_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_options(heatmap_example PRIVATE -fgpu-rdc)
target_link_libraries(heatmap_example PRIVATE dh_comms)
set_target_properties(heatmap_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin)