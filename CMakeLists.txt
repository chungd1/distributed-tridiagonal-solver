cmake_minimum_required(VERSION 3.24)

project(distributed_tridiagonal_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dtdma INTERFACE)
target_compile_features(dtdma INTERFACE cxx_std_20)
target_include_directories(
    dtdma
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(dtdma INTERFACE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
