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

# Graph tests
add_executable(test_graph test_graph.cpp)
target_include_directories(test_graph PRIVATE ${FORGERT_INCLUDE_DIR})

# Graph integration tests (MatMul)
add_executable(test_graph_matmul test_graph_matmul.cpp)
target_include_directories(test_graph_matmul PRIVATE ${FORGERT_INCLUDE_DIR})

# Graph integration tests (Softmax)
add_executable(test_graph_softmax test_graph_softmax.cpp)
target_include_directories(test_graph_softmax PRIVATE ${FORGERT_INCLUDE_DIR})

# Register with CTest
add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME OperatorTest COMMAND test_operator)
add_test(NAME ReLUTest COMMAND test_relu)
add_test(NAME MatMulTest COMMAND test_matmul)
add_test(NAME SoftmaxTest COMMAND test_softmax)
add_test(NAME GraphTest COMMAND test_graph)
add_test(NAME GraphMatMulTest COMMAND test_graph_matmul)
add_test(NAME GraphSoftmaxTest COMMAND test_graph_softmax)

message(STATUS "ForgeRT tests configured")
