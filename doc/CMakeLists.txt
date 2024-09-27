find_package(Doxygen)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_STATIC YES)
if(DOXYGEN_FOUND)
    doxygen_add_docs(doc
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include)
 else()
    message(WARNING "Building HTML documentation requires Doxygen; build target \"doc\" not supported on this machine.")
endif()