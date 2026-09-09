# ForgeRT CUDA Library

set(FORGERT_CUDA_SOURCES
    kernels/relu_kernel.cu
    kernels/matmul_kernel.cu
    kernels/softmax_kernel.cu
    kernels/cuda_memory.cu
)

add_library(forgert_cuda STATIC ${FORGERT_CUDA_SOURCES})
target_include_directories(forgert_cuda PUBLIC
    ${FORGERT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(forgert_cuda PUBLIC FORGERT_CUDA_AVAILABLE)
set_target_properties(forgert_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
)

message(STATUS "ForgeRT CUDA library configured")
