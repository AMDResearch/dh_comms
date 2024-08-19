cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(dh_comms LANGUAGES HIP CXX)
find_package(hip REQUIRED)
set(DH_COMMS_SOURCES
    buffer.cpp
    test.cpp
)
set_source_files_properties(${DH_COMMS_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(test ${DH_COMMS_SOURCES})
