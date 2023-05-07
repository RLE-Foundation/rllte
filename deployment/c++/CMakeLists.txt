cmake_minimum_required(VERSION 3.2)

project(HsuanwuDeployer_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

add_library(HsuanwuDeployer SHARED HsuanwuDeployer.cpp ./common/logger.cpp)
target_link_libraries(HsuanwuDeployer nvinfer nvonnxparser ${CUDA_LIBRARIES})

add_executable(HsuanwuDeployer_test HsuanwuDeployer_test.cpp)
target_link_libraries(HsuanwuDeployer_test HsuanwuDeployer)

