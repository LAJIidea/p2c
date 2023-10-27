add_subdirectory(dialect)
add_subdirectory(codegen)
add_subdirectory(jit)
add_subdirectory(runtime)

file(GLOB CORE_SRCS p2c.cpp)
add_library(p2c_core ${CORE_SRCS})
target_link_libraries(p2c_core ${PYTHON_LIBRARIES})