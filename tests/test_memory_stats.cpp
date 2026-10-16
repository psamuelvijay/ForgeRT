/*
 * test_memory_stats.cpp - Memory tracking tests (Phase 4)
 *
 * Tests the MemoryStats singleton and Tensor instrumentation.
 * Validates that allocations, deallocations, and transfers are tracked correctly.
 */

#include "forgert/tensor/tensor.h"
#include "forgert/memory/memory_stats.h"
#include <cassert>
#include <cstdio>
#include <memory>

using namespace forgert;

void test_cpu_allocation_tracking() {
    printf("Test 1: CPU allocation tracking\n");
    
    MemoryStats::instance().reset();
    
    size_t before = MemoryStats::instance().getAllocatedBytes(Device::CPU);
    
    {
        // Allocate a 10×10 float32 tensor = 400 bytes
        Tensor t(TensorShape({10, 10}), DataType::Float32, Device::CPU);
        
        size_t during = MemoryStats::instance().getAllocatedBytes(Device::CPU);
        assert(during == before + 400 && "CPU allocation should be tracked");
        
        // t goes out of scope here
    }
    
    size_t after = MemoryStats::instance().getAllocatedBytes(Device::CPU);
    assert(after == before && "CPU deallocation should be tracked");
    
    printf("  \xE2\x9C\x93 CPU allocation tracking works\n");
}

void test_cpu_peak_tracking() {
    printf("Test 2: CPU peak memory tracking\n");
    
    MemoryStats::instance().reset();
    
    size_t peak_before = MemoryStats::instance().getPeakAllocatedBytes(Device::CPU);
    assert(peak_before == 0);
    
    {
        Tensor t1(TensorShape({100}), DataType::Float32, Device::CPU);  // 400 bytes
        size_t peak1 = MemoryStats::instance().getPeakAllocatedBytes(Device::CPU);
        assert(peak1 == 400);
        
        {
            Tensor t2(TensorShape({200}), DataType::Float32, Device::CPU);  // 800 bytes
            size_t peak2 = MemoryStats::instance().getPeakAllocatedBytes(Device::CPU);
            assert(peak2 == 1200);  // 400 + 800
            
            // t2 destroyed
        }
        
        // After t2 destroyed, peak should still be 1200
        size_t peak3 = MemoryStats::instance().getPeakAllocatedBytes(Device::CPU);
        assert(peak3 == 1200);
        
        // Current should be 400
        size_t current = MemoryStats::instance().getAllocatedBytes(Device::CPU);
        assert(current == 400);
    }
    
    printf("  \xE2\x9C\x93 CPU peak memory tracking works\n");
}

void test_cpu_to_cpu_transfer_tracking() {
    printf("Test 3: CPU->CPU transfer tracking\n");
    
    MemoryStats::instance().reset();
    
    Tensor src(TensorShape({50}), DataType::Float32, Device::CPU);  // 200 bytes
    Tensor dst(TensorShape({50}), DataType::Float32, Device::CPU);
    
    size_t before_bytes = MemoryStats::instance().getTransferBytes(Device::CPU, Device::CPU);
    size_t before_count = MemoryStats::instance().getTransferCount(Device::CPU, Device::CPU);
    
    src.copyTo(dst);
    
    size_t after_bytes = MemoryStats::instance().getTransferBytes(Device::CPU, Device::CPU);
    size_t after_count = MemoryStats::instance().getTransferCount(Device::CPU, Device::CPU);
    
    assert(after_bytes == before_bytes + 200 && "CPU->CPU transfer bytes should be tracked");
    assert(after_count == before_count + 1 && "CPU->CPU transfer count should be tracked");
    
    printf("  \xE2\x9C\x93 CPU->CPU transfer tracking works\n");
}

void test_multiple_allocations() {
    printf("Test 4: Multiple allocations tracking\n");
    
    MemoryStats::instance().reset();
    
    std::vector<std::unique_ptr<Tensor>> tensors;
    
    // Allocate 5 tensors
    for (int i = 0; i < 5; ++i) {
        tensors.emplace_back(std::make_unique<Tensor>(
            TensorShape({100}), DataType::Float32, Device::CPU));  // 400 bytes each
    }
    
    size_t allocated = MemoryStats::instance().getAllocatedBytes(Device::CPU);
    assert(allocated == 2000 && "Multiple allocations should sum correctly");
    
    // Deallocate 3 tensors
    for (int i = 0; i < 3; ++i) {
        tensors.pop_back();
    }
    
    allocated = MemoryStats::instance().getAllocatedBytes(Device::CPU);
    assert(allocated == 800 && "Partial deallocation should work correctly");
    
    tensors.clear();
    
    allocated = MemoryStats::instance().getAllocatedBytes(Device::CPU);
    assert(allocated == 0 && "Full deallocation should return to zero");
    
    printf("  \xE2\x9C\x93 Multiple allocations tracking works\n");
}

void test_reset() {
    printf("Test 5: Stats reset\n");
    
    // Allocate some memory and perform transfers
    {
        Tensor t(TensorShape({500}), DataType::Float32, Device::CPU);   // 2000 bytes
        Tensor t2(TensorShape({500}), DataType::Float32, Device::CPU);  // 2000 bytes
        
        t.copyTo(t2);  // CPU->CPU transfer: 2000 bytes
        
        assert(MemoryStats::instance().getAllocatedBytes(Device::CPU) > 0);
        assert(MemoryStats::instance().getTransferCount(Device::CPU, Device::CPU) > 0);
        
        // t, t2 destroyed here; deallocation is recorded before reset
    }
    
    // Reset should clear all counters
    MemoryStats::instance().reset();
    
    assert(MemoryStats::instance().getAllocatedBytes(Device::CPU) == 0);
    assert(MemoryStats::instance().getPeakAllocatedBytes(Device::CPU) == 0);
    assert(MemoryStats::instance().getTransferBytes(Device::CPU, Device::CPU) == 0);
    assert(MemoryStats::instance().getTransferCount(Device::CPU, Device::CPU) == 0);
    
    printf("  \xE2\x9C\x93 Stats reset works\n");
}

void test_print() {
    printf("Test 6: Stats print\n");
    
    MemoryStats::instance().reset();
    
    {
        Tensor t1(TensorShape({256, 256}), DataType::Float32, Device::CPU);  // 256 KB
        Tensor t2(TensorShape({256, 256}), DataType::Float32, Device::CPU);
        
        t1.copyTo(t2);
        t2.copyTo(t1);
        
        MemoryStats::instance().print();
    }
    
    printf("  \xE2\x9C\x93 Stats print works (visual inspection above)\n");
}

int main() {
    printf("ForgeRT Memory Stats Tests\n");
    printf("==========================\n\n");
    
    test_cpu_allocation_tracking();
    test_cpu_peak_tracking();
    test_cpu_to_cpu_transfer_tracking();
    test_multiple_allocations();
    test_reset();
    test_print();
    
    printf("\n=== All memory stats tests passed ===\n");
    return 0;
}
