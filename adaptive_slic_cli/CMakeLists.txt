
cmake_minimum_required (VERSION 2.8)
project (superpixel_benchmark)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options iostreams REQUIRED)

set(CMAKE_CXX_FLAGS "-g -O0 -std=c++0x")
set(CMAKE_CXX_FLAGS "-O4 -std=c++0x")

include_directories(../lib_eval/ ../lib_adaptive_slic/ ${OpenCV_INCLUDE_DIRS} 
        ${Boost_INCLUDE_DIRS})
add_executable(adaptive_slic_cli main.cpp)
target_link_libraries(adaptive_slic_cli eval adaptive_slic ${Boost_LIBRARIES} ${OpenCV_LIBS})
