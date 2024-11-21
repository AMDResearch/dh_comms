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
    src/dummy_kernel.cpp
)
set_source_files_properties(${DH_COMMS_SOURCES} PROPERTIES LANGUAGE HIP)
add_library(dh_comms SHARED ${DH_COMMS_SOURCES})
target_compile_options(dh_comms PRIVATE -fgpu-rdc -Werror -Wall -Wextra)
target_include_directories(dh_comms PRIVATE ${CMAKE_SOURCE_DIR}/include)
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

# Create bitcode files for use in kernel instrumentation

set(DH_COMMS_SUPPORTED_ARCHS gfx906 gfx908 gfx90a gfx940 gfx941 gfx942)
foreach(arch ${DH_COMMS_SUPPORTED_ARCHS})
    list(APPEND DH_COMMS_BITCODE_FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/dh_comms_dev.co4.${arch}.bc)
    list(APPEND DH_COMMS_BITCODE_FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/dh_comms_dev.co5.${arch}.bc)
    list(APPEND DH_COMMS_OFFLOAD_ARCHS "--offload-arch=${arch}")
endforeach()

add_custom_command(
    OUTPUT dh_comms_bc.stamp
    COMMAND mkdir -p lib
    COMMAND ${CMAKE_HIP_COMPILER} -Wno-pragma-once-outside-header ${DH_COMMS_OFFLOAD_ARCHS}
            -O3 -mcode-object-version=4 -x hip -c -save-temps ${CMAKE_CURRENT_SOURCE_DIR}/include/dh_comms_dev.h
    COMMAND for f in dh_comms*-hip-*[0-9a].bc \; do mv $$f `echo $$f | sed 's/^.*-gfx/lib\\/dh_comms_dev.co4.gfx/'` \; done
    COMMAND ${CMAKE_HIP_COMPILER} -Wno-pragma-once-outside-header ${DH_COMMS_OFFLOAD_ARCHS}
            -O3 -mcode-object-version=5 -x hip -c -save-temps ${CMAKE_CURRENT_SOURCE_DIR}/include/dh_comms_dev.h
    COMMAND for f in dh_comms*-hip-*[0-9a].bc \; do mv $$f `echo $$f | sed 's/^.*-gfx/lib\\/dh_comms_dev.co5.gfx/'` \; done
    COMMAND rm -f dh_comms_dev-hip* dh_comms_dev-host* dh_comms_dev*hipfb dh_comms_dev.o
    COMMAND touch dh_comms_bc.stamp
    DEPENDS include/dh_comms_dev.h
)

add_custom_target(dh_comms_bc ALL
    DEPENDS dh_comms_bc.stamp
)

set_property(TARGET dh_comms_bc APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${DH_COMMS_BITCODE_FILES})

install(PROGRAMS
    ${DH_COMMS_BITCODE_FILES}
    DESTINATION lib
)
