cmake_minimum_required(VERSION 3.2)

project(RLLTEDeployer_test)

if(CMAKE_BUILD_TYPE STREQUAL "")
set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})

add_library(RLLTEDeployer SHARED RLLTEDeployer.cpp ./common/logger.cpp)
target_link_libraries(RLLTEDeployer nvinfer nvonnxparser ${CUDA_LIBRARIES})


add_executable(DeployerTest DeployerTest.cpp)
target_link_libraries(DeployerTest RLLTEDeployer)

message("Build type: " ${CMAKE_BUILD_TYPE})

