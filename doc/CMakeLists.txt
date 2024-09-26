find_package(Doxygen)
if(DOXYGEN_FOUND)
    doxygen_add_docs(doc
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include)
else()
    message(WARNING "Building HTML documentation requires Doxygen.")
endif()