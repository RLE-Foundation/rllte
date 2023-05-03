cmake_minimum_required(VERSION 3.2)

project(HsuanwuDeployer_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})

add_library(HsuanwuDeployer SHARED HsuanwuDeployer.cpp)
target_link_libraries(HsuanwuDeployer nvinfer nvonnxparser ${CUDA_LIBRARIES})

add_executable(HsuanwuDeployer_test HsuanwuDeployer_test.cpp)
target_link_libraries(HsuanwuDeployer_test HsuanwuDeployer)

