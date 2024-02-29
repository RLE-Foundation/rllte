cmake_minimum_required(VERSION 3.2)

project(NCNNDeploy_test)

if(CMAKE_BUILD_TYPE STREQUAL "")
set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

include_directories("~/Documents/Hsuanwu/deployment/ncnn/ncnn/build/install/include")
set(ncnn_DIR "~/Documents/Hsuanwu/deployment/ncnn/ncnn/build/install/lib/cmake/ncnn" 
    CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

add_executable(NCNNDeployTest NCNNDeployeTest.cpp)
target_link_libraries(NCNNDeployTest ncnn)

message("Build type: " ${CMAKE_BUILD_TYPE})

