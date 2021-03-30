cmake_minimum_required(VERSION 3.8)

project(mep-gl LANGUAGES CXX CUDA)

set(
    CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -arch=sm_61"
)

set(
    CMAKE_CUDA_FLAGS_DEBUG
    "${CMAKE_CUDA_FLAGS_DEBUG} -G --compiler-options -Wall"
)

find_package(ZLIB REQUIRED)
find_package(Boost REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})

link_libraries(stdc++fs ${ZLIB_LIBRARIES} boost_program_options cusparse)

add_executable(
    mep-gl
    mep-gl.cu
    nlcg.cu
    interpolation.cu
    parallel_sum.cu
    io.cu
    cnpy.cpp
)

target_compile_definitions(mep-gl PRIVATE)
