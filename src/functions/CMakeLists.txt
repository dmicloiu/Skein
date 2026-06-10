add_subdirectory(scalar)
add_subdirectory(aggregate)
add_subdirectory(operator)

set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/input_parser.cpp
    PARENT_SCOPE)
