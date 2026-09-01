# ForgeRT Tests

# Tensor tests
add_executable(test_tensor test_tensor.cpp)
target_include_directories(test_tensor PRIVATE ${FORGERT_INCLUDE_DIR})

# Register with CTest
add_test(NAME TensorTest COMMAND test_tensor)

message(STATUS "ForgeRT tests configured")
