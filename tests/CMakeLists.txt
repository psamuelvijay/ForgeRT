# ForgeRT Tests

# Tensor tests
add_executable(test_tensor test_tensor.cpp)
target_include_directories(test_tensor PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_tensor PRIVATE forgert)

# Operator tests
add_executable(test_operator test_operator.cpp)
target_include_directories(test_operator PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_operator PRIVATE forgert)

# ReLU operator tests
add_executable(test_relu test_relu.cpp)
target_include_directories(test_relu PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_relu PRIVATE forgert)

# MatMul operator tests
add_executable(test_matmul test_matmul.cpp)
target_include_directories(test_matmul PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_matmul PRIVATE forgert)

# Softmax operator tests
add_executable(test_softmax test_softmax.cpp)
target_include_directories(test_softmax PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_softmax PRIVATE forgert)

# LayerNorm operator tests
add_executable(test_layernorm test_layernorm.cpp)
target_include_directories(test_layernorm PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_layernorm PRIVATE forgert)


# AveragePool2D CPU tests (Phase 8)
add_executable(test_averagepool2d test_averagepool2d.cpp)
target_include_directories(test_averagepool2d PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_averagepool2d PRIVATE forgert)

# Dropout CPU tests (Phase 8)
add_executable(test_dropout test_dropout.cpp)
target_include_directories(test_dropout PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_dropout PRIVATE forgert)

# Reshape CPU tests (Phase 8)
add_executable(test_reshape test_reshape.cpp)
target_include_directories(test_reshape PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_reshape PRIVATE forgert)
# Conv2D CPU tests (Phase 8)
add_executable(test_conv2d test_conv2d.cpp)
target_include_directories(test_conv2d PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_conv2d PRIVATE forgert)
# Concat CPU tests (Phase 8)
add_executable(test_concat test_concat.cpp)
target_include_directories(test_concat PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_concat PRIVATE forgert)

# MaxPool2D CPU tests (Phase 8)
add_executable(test_maxpool2d test_maxpool2d.cpp)
target_include_directories(test_maxpool2d PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_maxpool2d PRIVATE forgert)

# Graph tests
add_executable(test_graph test_graph.cpp)
target_include_directories(test_graph PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_graph PRIVATE forgert)

# Graph integration tests (MatMul)
add_executable(test_graph_matmul test_graph_matmul.cpp)
target_include_directories(test_graph_matmul PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_graph_matmul PRIVATE forgert)

# Graph integration tests (Softmax)
add_executable(test_graph_softmax test_graph_softmax.cpp)
target_include_directories(test_graph_softmax PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_graph_softmax PRIVATE forgert)

# Memory stats tests (Phase 4)
add_executable(test_memory_stats test_memory_stats.cpp)
target_include_directories(test_memory_stats PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_memory_stats PRIVATE forgert)

# Graph optimizer tests (Phase 5)
add_executable(test_graph_optimizer test_graph_optimizer.cpp)
target_include_directories(test_graph_optimizer PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_graph_optimizer PRIVATE forgert)

# Resource-aware scheduler tests (Phase 6)
add_executable(test_scheduler test_scheduler.cpp)
target_include_directories(test_scheduler PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_scheduler PRIVATE forgert)

# --- CUDA tests ---
# Helpers used by all CUDA test targets
set(CUDA_TEST_PROPS
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
)

# CUDA graph integration tests -- uses forgert::Tensor + Graph + CUDA operators
add_executable(test_graph_cuda test_graph_cuda.cu)
target_include_directories(test_graph_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_graph_cuda PRIVATE forgert forgert_cuda)
set_target_properties(test_graph_cuda PROPERTIES ${CUDA_TEST_PROPS})

# Tensor CUDA memory tests -- allocation, RAII, copyTo, graph execution on GPU
add_executable(test_tensor_cuda test_tensor_cuda.cu)
target_include_directories(test_tensor_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_tensor_cuda PRIVATE forgert forgert_cuda)
set_target_properties(test_tensor_cuda PROPERTIES ${CUDA_TEST_PROPS})

# ReLU CUDA kernel tests -- calls kernel via extern "C", no Tensor dependency
add_executable(test_relu_cuda test_relu_cuda.cu)
target_include_directories(test_relu_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_relu_cuda PRIVATE forgert_cuda)
set_target_properties(test_relu_cuda PROPERTIES ${CUDA_TEST_PROPS})

# MatMul CUDA kernel + operator tests -- uses forgert::Tensor, requires forgert
add_executable(test_matmul_cuda test_matmul_cuda.cu)
target_include_directories(test_matmul_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_matmul_cuda PRIVATE forgert forgert_cuda)
set_target_properties(test_matmul_cuda PROPERTIES ${CUDA_TEST_PROPS})

# Softmax CUDA kernel + operator tests -- uses forgert::Tensor, requires forgert
add_executable(test_softmax_cuda test_softmax_cuda.cu)
target_include_directories(test_softmax_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_softmax_cuda PRIVATE forgert forgert_cuda)
set_target_properties(test_softmax_cuda PROPERTIES ${CUDA_TEST_PROPS})

# LayerNorm CUDA kernel + operator tests -- uses forgert::Tensor, requires forgert
add_executable(test_layernorm_cuda test_layernorm_cuda.cu)
target_include_directories(test_layernorm_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_layernorm_cuda PRIVATE forgert forgert_cuda)
set_target_properties(test_layernorm_cuda PROPERTIES ${CUDA_TEST_PROPS})

# ReLU operator CUDA dispatch tests -- uses forgert::Tensor, requires forgert
add_executable(test_relu_op_cuda test_relu_op_cuda.cu)
target_include_directories(test_relu_op_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_relu_op_cuda PRIVATE forgert forgert_cuda)
set_target_properties(test_relu_op_cuda PROPERTIES ${CUDA_TEST_PROPS})

# Add CUDA operator tests -- uses forgert::Tensor, requires forgert
add_executable(test_add_cuda test_add_cuda.cu)
target_include_directories(test_add_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_add_cuda PRIVATE forgert forgert_cuda)
set_target_properties(test_add_cuda PROPERTIES ${CUDA_TEST_PROPS})

# LayerNorm CUDA stress test -- memory leak detection
add_executable(test_layernorm_stress test_layernorm_stress.cu)
target_include_directories(test_layernorm_stress PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_layernorm_stress PRIVATE forgert forgert_cuda)
set_target_properties(test_layernorm_stress PROPERTIES ${CUDA_TEST_PROPS})

# Conv2D CUDA tests (Phase 8)
add_executable(test_conv2d_cuda test_conv2d_cuda.cu)
target_include_directories(test_conv2d_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_conv2d_cuda PRIVATE forgert forgert_cuda)
set_target_properties(test_conv2d_cuda PROPERTIES ${CUDA_TEST_PROPS})

# Register with CTest
add_test(NAME TensorTest          COMMAND test_tensor)
add_test(NAME TensorCUDATest      COMMAND test_tensor_cuda)
add_test(NAME OperatorTest        COMMAND test_operator)
add_test(NAME ReLUTest            COMMAND test_relu)
add_test(NAME MatMulTest          COMMAND test_matmul)
add_test(NAME SoftmaxTest         COMMAND test_softmax)
add_test(NAME LayerNormTest       COMMAND test_layernorm)
add_test(NAME Conv2DTest          COMMAND test_conv2d)
add_test(NAME ConcatTest          COMMAND test_concat)
add_test(NAME MaxPool2DTest       COMMAND test_maxpool2d)
add_test(NAME AveragePool2DTest   COMMAND test_averagepool2d)
add_test(NAME DropoutTest        COMMAND test_dropout)
add_test(NAME ReshapeTest        COMMAND test_reshape)
add_test(NAME GraphTest           COMMAND test_graph)
add_test(NAME GraphMatMulTest     COMMAND test_graph_matmul)
add_test(NAME GraphSoftmaxTest    COMMAND test_graph_softmax)
add_test(NAME MemoryStatsTest     COMMAND test_memory_stats)
add_test(NAME GraphOptimizerTest  COMMAND test_graph_optimizer)
add_test(NAME SchedulerTest       COMMAND test_scheduler)
add_test(NAME ReLUCUDATest        COMMAND test_relu_cuda)
add_test(NAME MatMulCUDATest      COMMAND test_matmul_cuda)
add_test(NAME SoftmaxCUDATest     COMMAND test_softmax_cuda)
add_test(NAME LayerNormCUDATest   COMMAND test_layernorm_cuda)
add_test(NAME ReLUOpCUDATest      COMMAND test_relu_op_cuda)
add_test(NAME AddCUDATest         COMMAND test_add_cuda)
add_test(NAME Conv2DCUDATest      COMMAND test_conv2d_cuda)
add_test(NAME LayerNormStressTest COMMAND test_layernorm_stress)

# --- Python-based importer tests (Phase 9) ---
# Exit code 77 means a prerequisite (Python package or model file) is missing;
# CTest reports those runs as Skipped, never Passed.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME OnnxToFgxConverterTest
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_onnx_to_fgx.py)
    set_tests_properties(OnnxToFgxConverterTest PROPERTIES SKIP_RETURN_CODE 77)
else()
    message(STATUS "Python3 not found: Python-based importer tests not registered")
endif()

message(STATUS "ForgeRT tests configured")