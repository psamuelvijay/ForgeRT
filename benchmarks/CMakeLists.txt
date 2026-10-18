# ForgeRT Benchmarks

# CPU operator benchmarks
add_executable(benchmark_operators benchmark_operators.cpp)
target_include_directories(benchmark_operators PRIVATE ${FORGERT_INCLUDE_DIR})
target_link_libraries(benchmark_operators PRIVATE forgert)

message(STATUS "ForgeRT benchmarks configured")
