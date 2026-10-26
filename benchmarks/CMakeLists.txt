# ForgeRT Benchmarks

# CPU operator benchmarks
add_executable(benchmark_operators benchmark_operators.cpp)
target_include_directories(benchmark_operators PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(benchmark_operators PRIVATE forgert forgert_cpu_stub)

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

# Launch overhead measurement tool
add_executable(measure_launch_overhead measure_launch_overhead.cu)
target_include_directories(measure_launch_overhead PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(measure_launch_overhead PRIVATE forgert_cuda)
set_target_properties(measure_launch_overhead PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
)
message(STATUS "  - measure_launch_overhead")

# Scheduler calibration profiler (Phase: measured profiling).
# Emits machine-readable JSON consumed by tools/calibrate_profiles.py.
# Separate from CTest correctness gates -- never registered as a test.
add_executable(benchmark_profile benchmark_profile.cu)
target_include_directories(benchmark_profile PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(benchmark_profile PRIVATE forgert forgert_cuda)
set_target_properties(benchmark_profile PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
)
message(STATUS "  - benchmark_profile")

# Bounded GPU stress/verification workload (MatMulOp on Backend::CUDA).
# Never registered as a CTest case: it is a live-observation tool.
add_executable(gpu_stress gpu_stress.cu)
target_include_directories(gpu_stress PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(gpu_stress PRIVATE forgert forgert_cuda)
set_target_properties(gpu_stress PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
)
message(STATUS "  - gpu_stress")

# Conv2D benchmark (Phase 8)
add_executable(benchmark_conv2d benchmark_conv2d.cu)
target_include_directories(benchmark_conv2d PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(benchmark_conv2d PRIVATE forgert forgert_cuda)
set_target_properties(benchmark_conv2d PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
)
message(STATUS "  - benchmark_conv2d")