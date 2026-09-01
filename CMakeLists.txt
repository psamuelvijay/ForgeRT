cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(ForgeRT 
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Resource-Aware Neural Inference Runtime"
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architecture - Target Pascal (GTX 1050, compute capability 6.1)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2)
    else()
        add_compile_options(/Zi /Od)
    endif()
else()
    add_compile_options(-Wall -Wextra -Werror)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
    else()
        add_compile_options(-g)
    endif()
endif()

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -use_fast_math")

# Project directories
set(FORGERT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FORGERT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(FORGERT_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
set(FORGERT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(FORGERT_BENCHMARK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)

# Include directories
include_directories(${FORGERT_INCLUDE_DIR})

# Enable testing
enable_testing()

# Subdirectories
add_subdirectory(src)
add_subdirectory(cuda)
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(examples)

# Print configuration summary
message(STATUS "")
message(STATUS "ForgeRT Configuration Summary")
message(STATUS "=============================")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard:     ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA Compiler:     ${CMAKE_CUDA_COMPILER}")
message(STATUS "Install Prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
