
cmake_minimum_required (VERSION 2.8)
project (superpixel_benchmark)

find_package(OpenCV REQUIRED)

include_directories(../lib_eval/ ${OpenCV_INCLUDE_DIRS})

#set(CMAKE_CXX_FLAGS "-g -O0 -std=c++0x")
set(CMAKE_CXX_FLAGS "-O4 -std=c++0x")

add_library(adaptive_slic
    adaptive_slic.cpp
    slic.cpp
    slic-utils.cpp
    opencl-slic.cpp
)
target_link_libraries(adaptive_slic eval ${OpenCV_LIBRARIES})
