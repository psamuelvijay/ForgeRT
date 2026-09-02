#include "forgert/operator/add.h"
#include "forgert/operator/relu.h"
#include "forgert/operator/matmul.h"
#include "forgert/operator/softmax.h"
#include "forgert/operator/layernorm.h"
#include "forgert/tensor/tensor.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

using namespace forgert;

struct BenchmarkResult {
    std::string op_name;
    std::string shape_desc;
    size_t num_elements;
    double avg_time_ms;
    double throughput_gflops;
    double bandwidth_gbps;
};

class BenchmarkRunner {
public:
    static constexpr int WARMUP_ITERS = 10;
    static constexpr int BENCH_ITERS = 100;

    template<typename OpType>
    static BenchmarkResult benchmarkUnaryOp(
        const std::string& op_name,
        const TensorShape& shape,
        size_t flops_per_element = 1) {
        
        Tensor input(shape, DataType::Float32, Device::CPU);
        Tensor output(shape, DataType::Float32, Device::CPU);
        
        // Initialize with random-ish data
        float* data = static_cast<float*>(input.data());
        for (size_t i = 0; i < shape.numElements(); ++i) {
            data[i] = static_cast<float>(i % 100) * 0.01f;
        }
        
        OpType op;
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            op.execute(Backend::CPU, {&input}, {&output});
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITERS; ++i) {
            op.execute(Backend::CPU, {&input}, {&output});
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_time_ms = elapsed_ms / BENCH_ITERS;
        
        size_t num_elements = shape.numElements();
        size_t total_flops = num_elements * flops_per_element;
        double gflops = (total_flops / 1e9) / (avg_time_ms / 1000.0);
        
        // Bandwidth: read + write
        size_t bytes_transferred = num_elements * sizeof(float) * 2;
        double gbps = (bytes_transferred / 1e9) / (avg_time_ms / 1000.0);
        
        return {op_name, shapeToString(shape), num_elements, avg_time_ms, gflops, gbps};
    }
    
    static BenchmarkResult benchmarkMatMul(const TensorShape& shape_a, const TensorShape& shape_b) {
        Tensor a(shape_a, DataType::Float32, Device::CPU);
        Tensor b(shape_b, DataType::Float32, Device::CPU);
        
        size_t M = shape_a.dim(0);
        size_t K = shape_a.dim(1);
        size_t N = shape_b.dim(1);
        
        TensorShape shape_c({M, N});
        Tensor c(shape_c, DataType::Float32, Device::CPU);
        
        // Initialize
        float* a_data = static_cast<float*>(a.data());
        float* b_data = static_cast<float*>(b.data());
        for (size_t i = 0; i < shape_a.numElements(); ++i) {
            a_data[i] = static_cast<float>(i % 100) * 0.01f;
        }
        for (size_t i = 0; i < shape_b.numElements(); ++i) {
            b_data[i] = static_cast<float>(i % 100) * 0.01f;
        }
        
        MatMulOp op;
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            op.execute(Backend::CPU, {&a, &b}, {&c});
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITERS; ++i) {
            op.execute(Backend::CPU, {&a, &b}, {&c});
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_time_ms = elapsed_ms / BENCH_ITERS;
        
        // MatMul FLOPs: 2*M*N*K (multiply-add)
        size_t total_flops = 2 * M * N * K;
        double gflops = (total_flops / 1e9) / (avg_time_ms / 1000.0);
        
        // Bandwidth: read A, read B, write C
        size_t bytes = (M*K + K*N + M*N) * sizeof(float);
        double gbps = (bytes / 1e9) / (avg_time_ms / 1000.0);
        
        std::string shape_desc = "[" + std::to_string(M) + "," + std::to_string(K) + "] @ [" +
                                std::to_string(K) + "," + std::to_string(N) + "]";
        
        return {"MatMul", shape_desc, M*N, avg_time_ms, gflops, gbps};
    }

private:
    static std::string shapeToString(const TensorShape& shape) {
        std::string result = "[";
        for (size_t i = 0; i < shape.ndim(); ++i) {
            if (i > 0) result += ",";
            result += std::to_string(shape.dim(i));
        }
        result += "]";
        return result;
    }
};

void printHeader() {
    std::cout << std::left
              << std::setw(15) << "Operator"
              << std::setw(25) << "Shape"
              << std::setw(12) << "Elements"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOP/s"
              << std::setw(15) << "GB/s"
              << std::endl;
    std::cout << std::string(97, '-') << std::endl;
}

void printResult(const BenchmarkResult& result) {
    std::cout << std::left
              << std::setw(15) << result.op_name
              << std::setw(25) << result.shape_desc
              << std::setw(12) << result.num_elements
              << std::fixed << std::setprecision(4)
              << std::setw(15) << result.avg_time_ms
              << std::setw(15) << result.throughput_gflops
              << std::setw(15) << result.bandwidth_gbps
              << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ForgeRT CPU Operator Benchmarks" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Warmup iterations: " << BenchmarkRunner::WARMUP_ITERS << std::endl;
    std::cout << "  Benchmark iterations: " << BenchmarkRunner::BENCH_ITERS << std::endl;
    std::cout << "  Data type: Float32" << std::endl;
    std::cout << "  Backend: CPU" << std::endl;
    std::cout << std::endl;
    
    std::vector<BenchmarkResult> results;
    
    printHeader();
    
    // ReLU benchmarks
    {
        auto r1 = BenchmarkRunner::benchmarkUnaryOp<ReLUOp>("ReLU", TensorShape({1024}), 2);
        printResult(r1);
        results.push_back(r1);
        
        auto r2 = BenchmarkRunner::benchmarkUnaryOp<ReLUOp>("ReLU", TensorShape({32, 512}), 2);
        printResult(r2);
        results.push_back(r2);
        
        auto r3 = BenchmarkRunner::benchmarkUnaryOp<ReLUOp>("ReLU", TensorShape({128, 768}), 2);
        printResult(r3);
        results.push_back(r3);
    }
    
    // Softmax benchmarks
    {
        auto r1 = BenchmarkRunner::benchmarkUnaryOp<SoftmaxOp>("Softmax", TensorShape({1024}), 10);
        printResult(r1);
        results.push_back(r1);
        
        auto r2 = BenchmarkRunner::benchmarkUnaryOp<SoftmaxOp>("Softmax", TensorShape({32, 1000}), 10);
        printResult(r2);
        results.push_back(r2);
        
        auto r3 = BenchmarkRunner::benchmarkUnaryOp<SoftmaxOp>("Softmax", TensorShape({128, 512}), 10);
        printResult(r3);
        results.push_back(r3);
    }
    
    // LayerNorm benchmarks
    {
        auto r1 = BenchmarkRunner::benchmarkUnaryOp<LayerNormOp>("LayerNorm", TensorShape({1024}), 10);
        printResult(r1);
        results.push_back(r1);
        
        auto r2 = BenchmarkRunner::benchmarkUnaryOp<LayerNormOp>("LayerNorm", TensorShape({32, 768}), 10);
        printResult(r2);
        results.push_back(r2);
        
        auto r3 = BenchmarkRunner::benchmarkUnaryOp<LayerNormOp>("LayerNorm", TensorShape({128, 1024}), 10);
        printResult(r3);
        results.push_back(r3);
    }
    
    // MatMul benchmarks
    {
        auto r1 = BenchmarkRunner::benchmarkMatMul(TensorShape({32, 128}), TensorShape({128, 256}));
        printResult(r1);
        results.push_back(r1);
        
        auto r2 = BenchmarkRunner::benchmarkMatMul(TensorShape({64, 512}), TensorShape({512, 512}));
        printResult(r2);
        results.push_back(r2);
        
        auto r3 = BenchmarkRunner::benchmarkMatMul(TensorShape({128, 768}), TensorShape({768, 768}));
        printResult(r3);
        results.push_back(r3);
    }
    
    std::cout << std::string(97, '=') << std::endl;
    std::cout << "\nCPU Baseline Performance Established" << std::endl;
    std::cout << "These baselines will be used for Phase 3 CUDA comparison." << std::endl;
    
    return 0;
}
