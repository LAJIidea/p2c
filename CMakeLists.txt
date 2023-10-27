cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(p2c)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)

if(NOT MLIR_DIR)
  find_package(LLVM REQUIRED)
  if(LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

    set(LLVM_LIBRARY_DIR ${LLVM_LIBRARY_DIRS})
    set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
  endif() 
endif()

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
if(MLIR_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
endif()

# Thread
if(UNIX)
    find_package(Threads REQUIRED)
endif(UNIX)

# Pybind11
# Python Config (Need 3.9.8)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/csrc)
include_directories(${PROJECT_BINARY_DIR}/csrc)

add_subdirectory(csrc)
