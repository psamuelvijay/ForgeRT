# ForgeRT Tests

# Tensor tests
add_executable(test_tensor test_tensor.cpp)
target_include_directories(test_tensor PRIVATE ${FORGERT_INCLUDE_DIR})

# Operator tests
add_executable(test_operator test_operator.cpp)
target_include_directories(test_operator PRIVATE ${FORGERT_INCLUDE_DIR})

# ReLU operator tests
add_executable(test_relu test_relu.cpp)
target_include_directories(test_relu PRIVATE ${FORGERT_INCLUDE_DIR})

# MatMul operator tests
add_executable(test_matmul test_matmul.cpp)
target_include_directories(test_matmul PRIVATE ${FORGERT_INCLUDE_DIR})

# Softmax operator tests
add_executable(test_softmax test_softmax.cpp)
target_include_directories(test_softmax PRIVATE ${FORGERT_INCLUDE_DIR})

# LayerNorm operator tests
add_executable(test_layernorm test_layernorm.cpp)
target_include_directories(test_layernorm PRIVATE ${FORGERT_INCLUDE_DIR})

# Graph tests
add_executable(test_graph test_graph.cpp)
target_include_directories(test_graph PRIVATE ${FORGERT_INCLUDE_DIR})

# Graph integration tests (MatMul)
add_executable(test_graph_matmul test_graph_matmul.cpp)
target_include_directories(test_graph_matmul PRIVATE ${FORGERT_INCLUDE_DIR})

# Graph integration tests (Softmax)
add_executable(test_graph_softmax test_graph_softmax.cpp)
target_include_directories(test_graph_softmax PRIVATE ${FORGERT_INCLUDE_DIR})

# Tensor CUDA memory tests (Phase 4) — allocation, RAII, transfers, graph execution
add_executable(test_tensor_cuda test_tensor_cuda.cu)
target_include_directories(test_tensor_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_tensor_cuda PRIVATE forgert_cuda)
set_target_properties(test_tensor_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
)

# ReLU CUDA kernel tests (Phase 3) — standalone kernel correctness
add_executable(test_relu_cuda test_relu_cuda.cu)
target_include_directories(test_relu_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_relu_cuda PRIVATE forgert_cuda)
set_target_properties(test_relu_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
)

# ReLU operator CUDA integration tests (Phase 3) — exercises the full
# ReLUOp::execute(Backend::CUDA) dispatch path through the operator abstraction
add_executable(test_relu_op_cuda test_relu_op_cuda.cu)
target_include_directories(test_relu_op_cuda PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(test_relu_op_cuda PRIVATE forgert_cuda)
set_target_properties(test_relu_op_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61
)

# Register with CTest
add_test(NAME TensorTest       COMMAND test_tensor)
add_test(NAME TensorCUDATest   COMMAND test_tensor_cuda)
add_test(NAME OperatorTest     COMMAND test_operator)
add_test(NAME ReLUTest COMMAND test_relu)
add_test(NAME MatMulTest COMMAND test_matmul)
add_test(NAME SoftmaxTest COMMAND test_softmax)
add_test(NAME LayerNormTest COMMAND test_layernorm)
add_test(NAME GraphTest COMMAND test_graph)
add_test(NAME GraphMatMulTest COMMAND test_graph_matmul)
add_test(NAME GraphSoftmaxTest COMMAND test_graph_softmax)
add_test(NAME ReLUCUDATest    COMMAND test_relu_cuda)
add_test(NAME ReLUOpCUDATest  COMMAND test_relu_op_cuda)

message(STATUS "ForgeRT tests configured")
