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

set(BANK_CONFLICTS_EXAMPLE_SOURCES
    src/bank_conflicts_example.cpp
)
set_source_files_properties(${BANK_CONFLICTS_EXAMPLE_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(bank_conflicts_example ${BANK_CONFLICTS_EXAMPLE_SOURCES})
target_compile_options(bank_conflicts_example PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_include_directories(bank_conflicts_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_options(bank_conflicts_example PRIVATE -fgpu-rdc)
target_link_libraries(bank_conflicts_example PRIVATE dh_comms)
set_target_properties(bank_conflicts_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin)

set(MEMORY_ALIGNMENT_EXAMPLE_SOURCES
    src/memory_alignment_example.cpp
)
set_source_files_properties(${MEMORY_ALIGNMENT_EXAMPLE_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(memory_alignment_example ${MEMORY_ALIGNMENT_EXAMPLE_SOURCES})
target_compile_options(memory_alignment_example PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_include_directories(memory_alignment_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_options(memory_alignment_example PRIVATE -fgpu-rdc)
target_link_libraries(memory_alignment_example PRIVATE dh_comms)
set_target_properties(memory_alignment_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin)