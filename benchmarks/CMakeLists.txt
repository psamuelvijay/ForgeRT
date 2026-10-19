# ForgeRT Benchmarks

# CPU operator benchmarks
add_executable(benchmark_operators benchmark_operators.cpp)
target_include_directories(benchmark_operators PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(benchmark_operators PRIVATE forgert)

message(STATUS "ForgeRT benchmarks configured")

# CUDA operator benchmark (requires CUDA-capable GPU at runtime)
add_executable(benchmark_cuda benchmark_cuda.cu)
target_include_directories(benchmark_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(benchmark_cuda PRIVATE forgert forgert_cuda)
set_target_properties(benchmark_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
)
message(STATUS "  - benchmark_cuda")
