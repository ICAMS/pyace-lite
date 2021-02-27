cmake_minimum_required(VERSION 3.7)  # CMake version check
project(ace)
set(CMAKE_CXX_STANDARD 11)            # Enable c++11 standard

set(EVALUATOR_PATH ace-evaluator)
include(${EVALUATOR_PATH}/CMakeLists.txt)
message("EVALUATOR_PATH=${EVALUATOR_PATH}")

set(FITTING_PATH src/fitting)
FILE(GLOB FITTING_SOURCE_FILES ${FITTING_PATH}/*.cpp)
set(FITTING_INCLUDE_DIR ${FITTING_PATH})

FILE(GLOB UTILS_SOURCE_FILES src/utils/*.cpp)
set(UTILS_INCLUDE_DIR src/utils)
FILE(GLOB TEST_FITTING test/fitting/test_*.cpp)
FILE(GLOB TEST_UTILS test/utils/test_*.cpp)

set(TEST_DIR test)

set(YAML_PATH utils/yaml-cpp)
set(YAML_INCLUDE_PATH ${YAML_PATH}/include)
set(YAML_LIB_PATH ${YAML_PATH}/build)

set(WIGNER_PATH utils/wigner-cpp)
set(WIGNER_INCLUDE_PATH ${WIGNER_PATH}/include/wigner)

add_subdirectory(${YAML_PATH})