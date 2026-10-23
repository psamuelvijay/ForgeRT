# ForgeRT Examples

# MNIST Inference Demo
add_executable(mnist_inference
    mnist/mnist_inference.cpp
)

target_link_libraries(mnist_inference PRIVATE
    forgert
    forgert_cpu_stub
)

target_include_directories(mnist_inference PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Copy to build directory for easier execution
set_target_properties(mnist_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/mnist"
)

message(STATUS "ForgeRT examples configured:")
message(STATUS "  - mnist_inference")

