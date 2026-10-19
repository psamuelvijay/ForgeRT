#include "forgert/operator/layernorm.h"
#include "forgert/tensor/tensor.h"
#include "forgert/memory/memory_stats.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace forgert;

int main() {
    std::cout << "LayerNorm CUDA Memory Stress Test\n";
    std::cout << "===================================\n\n";
    
    // Get initial GPU memory
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);
    std::cout << "Initial GPU free memory: " << (free_before / 1024 / 1024) << " MB\n";
    
    TensorShape shape({128, 768});  // Typical transformer size
    LayerNormOp op(1e-5f);
    
    const int ITERATIONS = 10000;
    std::cout << "Running " << ITERATIONS << " LayerNorm operations...\n";
    
    for (int i = 0; i < ITERATIONS; ++i) {
        Tensor input(shape, DataType::Float32, Device::CUDA);
        Tensor output(shape, DataType::Float32, Device::CUDA);
        
        op.execute(Backend::CUDA, {&input}, {&output});
        
        // Report every 1000 iterations
        if ((i + 1) % 1000 == 0) {
            size_t free_current, total_current;
            cudaMemGetInfo(&free_current, &total_current);
            long long delta_mb = static_cast<long long>(free_before - free_current) / 1024 / 1024;
            std::cout << "  Iteration " << (i + 1) << ": GPU free memory delta = " 
                      << delta_mb << " MB\n";
        }
    }
    
    // Final GPU memory check
    size_t free_after, total_after;
    cudaMemGetInfo(&free_after, &total_after);
    std::cout << "\nFinal GPU free memory: " << (free_after / 1024 / 1024) << " MB\n";
    long long total_delta_mb = static_cast<long long>(free_before - free_after) / 1024 / 1024;
    std::cout << "Total memory delta: " << total_delta_mb << " MB\n";
    
    if (total_delta_mb > 100) {
        std::cerr << "\n❌ FAIL: Memory leak detected (> 100 MB growth)\n";
        return 1;
    }
    
    std::cout << "\n✓ PASS: No memory leak detected\n";
    return 0;
}
