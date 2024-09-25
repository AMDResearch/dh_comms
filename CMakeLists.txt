cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(dh_comms LANGUAGES HIP CXX)
find_package(hip REQUIRED)

# single executable target
#[[
set(DH_COMMS_SOURCES
    dh_comms.cpp
    memory_heatmap.cpp
    test.cpp
)
set_source_files_properties(${DH_COMMS_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(test ${DH_COMMS_SOURCES})
target_compile_options(test PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_link_options(test PRIVATE -fgpu-rdc)
]]

# ideally, we'd have a shared lib with the functionality in buffer.cpp
# that can be used by application code. Currently doesn't work, giving
# "undefined hidden symbol" errors while linking

##[[
set(DH_COMMS_SOURCES
    dh_comms.cpp
    memory_heatmap.cpp
)
set_source_files_properties(${DH_COMMS_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(dh_comms SHARED ${DH_COMMS_SOURCES})
target_compile_options(dh_comms PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_link_options(dh_comms PRIVATE -fgpu-rdc)

set(TEST_SOURCES
    test.cpp
)
set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(test ${TEST_SOURCES})
target_compile_options(test PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_link_options(test PRIVATE -fgpu-rdc)
target_link_libraries(test PRIVATE dh_comms)
#]]
