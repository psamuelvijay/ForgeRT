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

# ONNX importer (Phase 9): strict .fgx reader. CPU-only and dependency-free;
# links only the core library. Do not link forgert_import and forgert_cuda into
# the same executable: operator headers compile differently with
# FORGERT_CUDA_AVAILABLE, which would violate the one-definition rule.
add_library(forgert_import
    import/fgx_reader.cpp
)
target_include_directories(forgert_import PUBLIC ${FORGERT_INCLUDE_DIR})
target_link_libraries(forgert_import PUBLIC forgert)


