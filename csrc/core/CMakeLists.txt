add_subdirectory(dialect)

file(GLOB CORE_SRCS p2c.cpp)
add_library(p2c_core ${CORE_SRCS})
target_link_libraries(p2c_core ${PYTHON_LIBRARIES})