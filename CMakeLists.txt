cmake_minimum_required(VERSION 3.13.4)
project(p2c)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)

# # Python Config (Need 3.9.8)
# find_package(PythonInterp REQUIRED)
# find_package(PythonLibs 3 REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})

# Thread
if(UNIX)
    find_package(Threads REQUIRED)
endif(UNIX)

# Pybind11
# Python, numpy, and pybind11
# execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmake
#                 OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
# execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy;print(numpy.get_include())"
#                 OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

# message("-- Python: Using ${PYTHON_EXECUTABLE} as the interpreter")
# message("    version: ${PYTHON_VERSION_STRING}")
# message("    include: ${PYTHON_INCLUDE_DIR}")
# message("    library: ${PYTHON_LIBRARY}")
# message("    numpy include: ${NUMPY_INCLUDE_DIR}")

find_package(pybind11 CONFIG REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/csrc)

add_subdirectory(csrc)
