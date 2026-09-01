# ForgeRT Tests

# Tensor tests
add_executable(test_tensor test_tensor.cpp)
target_include_directories(test_tensor PRIVATE ${FORGERT_INCLUDE_DIR})

# Operator tests
add_executable(test_operator test_operator.cpp)
target_include_directories(test_operator PRIVATE ${FORGERT_INCLUDE_DIR})

# Register with CTest
add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME OperatorTest COMMAND test_operator)

message(STATUS "ForgeRT tests configured")
