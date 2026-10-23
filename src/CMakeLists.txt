# ForgeRT Core Library

set(FORGERT_SOURCES
    # Memory (Phase 4)
    memory/memory_stats.cpp

    # Tensor (Phase 4 - instrumented for tracking)
    tensor/tensor.cpp

    # Graph optimizer (Phase 5 - dead-node elimination)
    graph/optimizer.cpp

    # Resource-aware scheduler (Phase 6 - CPU/GPU placement)
    scheduler/scheduler.cpp
)

# Create core library (CPU-only: no CUDA dependency)
# CUDA tests link forgert_cuda separately.
# The forgert_cuda target adds FORGERT_CUDA_AVAILABLE when linked,
# which is needed for CUDA tensor tracking — but the tracking
# infrastructure itself is pure C++ and needs no CUDA headers.
add_library(forgert ${FORGERT_SOURCES})
target_include_directories(forgert PUBLIC ${FORGERT_INCLUDE_DIR})

message(STATUS "ForgeRT core library configured")

# CPU-only fallback definition of Tensor::cudaTransferImpl, as its OWN static
# library. Tensor::copyTo dispatches CUDA transfers to this out-of-line symbol
# so the inline body is macro-independent and identical in every TU.
#
# Link-time selection is made UNAMBIGUOUS by keeping the two providers in
# separate libraries and never linking both into one binary:
#   - CPU-only executables link forgert_cpu_stub (this lib)  -> throwing stub.
#   - CUDA executables link forgert_cuda (tensor_cuda_impl.cu) -> real memcpy,
#     and do NOT link forgert_cpu_stub.
# forgert itself does NOT depend on either provider, so the choice is made per
# executable and is independent of library link order.
add_library(forgert_cpu_stub STATIC tensor/tensor_cuda_stub.cpp)
target_include_directories(forgert_cpu_stub PUBLIC ${FORGERT_INCLUDE_DIR})

message(STATUS "ForgeRT CPU transfer-stub library configured")

# ONNX importer (Phase 9): strict .fgx reader. CPU-only and dependency-free;
# links only the core library. Do not link forgert_import and forgert_cuda into
# the same executable: operator headers compile differently with
# FORGERT_CUDA_AVAILABLE, which would violate the one-definition rule.
add_library(forgert_import
    import/fgx_reader.cpp
    import/importer.cpp
)
target_include_directories(forgert_import PUBLIC ${FORGERT_INCLUDE_DIR})
target_link_libraries(forgert_import PUBLIC forgert forgert_cpu_stub)


