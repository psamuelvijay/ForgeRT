/*
 * benchmark_cuda.cu  --  CPU vs CUDA operator benchmarks for ForgeRT
 *
 * Measures:
 *   cpu_ms  : CPU operator execution, pre-allocated tensors
 *   gpu_kernel_ms : GPU kernel only, tensors already on device
 *   e2e_ms  : H2D copy + GPU kernel + D2H copy, pre-allocated device tensors
 *             (does NOT include cudaMalloc -- scheduler cares about steady-state
 *              transfer overhead, not allocation overhead)
 *
 * GPU timing uses cudaEvent_t for accuracy.
 * CPU timing uses std::chrono::high_resolution_clock.
 *
 * Hardware: GTX 1050 (sm_61, 4 GB VRAM), i5-9300H
 * Build:    CUDA 12.6, MSVC 14.39, Release (-O3 -use_fast_math)
 */

#include "forgert/operator/relu.h"
#include "forgert/operator/add.h"
#include "forgert/operator/matmul.h"
#include "forgert/operator/softmax.h"
#include "forgert/operator/layernorm.h"
#include "forgert/tensor/tensor.h"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <functional>

using namespace forgert;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << "  " << cudaGetErrorString(_e) << "\n";               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static void fill(Tensor& t, float val = 0.1f) {
    float* p = static_cast<float*>(t.data());
    for (size_t i = 0; i < t.shape().numElements(); ++i) p[i] = val;
}

static double timeCPU(std::function<void()> fn,
                      int warmup = 10, int iters = 100) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static double timeGPU(std::function<void()> fn,
                      int warmup = 10, int iters = 100) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

struct Row {
    std::string op, shape;
    size_t elems;
    double cpu_ms, kernel_ms, e2e_ms;
};

static void printHeader() {
    std::cout << "\n"
              << std::left
              << std::setw(12) << "Operator"
              << std::setw(24) << "Shape"
              << std::setw(12) << "Elems"
              << std::setw(12) << "CPU (ms)"
              << std::setw(14) << "Kernel (ms)"
              << std::setw(14) << "E2E (ms)"
              << "CPU/E2E speedup\n"
              << std::string(92, '-') << "\n";
}

static void printRow(const Row& r) {
    double sp = (r.e2e_ms > 0.0) ? r.cpu_ms / r.e2e_ms : 0.0;
    std::cout << std::left
              << std::setw(12) << r.op
              << std::setw(24) << r.shape
              << std::setw(12) << r.elems
              << std::setw(12) << std::fixed << std::setprecision(4) << r.cpu_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << r.kernel_ms
              << std::setw(14) << std::fixed << std::setprecision(4) << r.e2e_ms
              << std::fixed << std::setprecision(2) << sp << "x\n";
}

// Benchmark a unary operator (one input, one output, same shape)
template<typename Op>
static Row benchUnary(const std::string& name, TensorShape shape) {
    Tensor h_in(shape, DataType::Float32, Device::CPU);
    Tensor h_out(shape, DataType::Float32, Device::CPU);
    fill(h_in);

    Tensor d_in(shape, DataType::Float32, Device::CUDA);
    Tensor d_out(shape, DataType::Float32, Device::CUDA);
    h_in.copyTo(d_in);

    Op op;

    double cpu_ms = timeCPU([&]() {
        std::vector<const Tensor*> ins{&h_in};
        std::vector<Tensor*> outs{&h_out};
        op.execute(Backend::CPU, ins, outs);
    });

    double kernel_ms = timeGPU([&]() {
        std::vector<const Tensor*> ins{&d_in};
        std::vector<Tensor*> outs{&d_out};
        op.execute(Backend::CUDA, ins, outs);
    });

    // E2E: H2D + kernel + D2H, using pre-allocated device tensors
    double e2e_ms = timeGPU([&]() {
        h_in.copyTo(d_in);
        std::vector<const Tensor*> ins{&d_in};
        std::vector<Tensor*> outs{&d_out};
        op.execute(Backend::CUDA, ins, outs);
        d_out.copyTo(h_out);
    });

    size_t N = shape.numElements();
    return {name, "[" + std::to_string(N) + "]", N, cpu_ms, kernel_ms, e2e_ms};
}

// Benchmark Add (two inputs)
static Row benchAdd(size_t N) {
    TensorShape shape({N});
    Tensor h_a(shape, DataType::Float32, Device::CPU);
    Tensor h_b(shape, DataType::Float32, Device::CPU);
    Tensor h_c(shape, DataType::Float32, Device::CPU);
    fill(h_a); fill(h_b, 0.2f);

    Tensor d_a(shape, DataType::Float32, Device::CUDA);
    Tensor d_b(shape, DataType::Float32, Device::CUDA);
    Tensor d_c(shape, DataType::Float32, Device::CUDA);
    h_a.copyTo(d_a); h_b.copyTo(d_b);

    AddOp op;

    double cpu_ms = timeCPU([&]() {
        std::vector<const Tensor*> ins{&h_a, &h_b};
        std::vector<Tensor*> outs{&h_c};
        op.execute(Backend::CPU, ins, outs);
    });

    double kernel_ms = timeGPU([&]() {
        std::vector<const Tensor*> ins{&d_a, &d_b};
        std::vector<Tensor*> outs{&d_c};
        op.execute(Backend::CUDA, ins, outs);
    });

    double e2e_ms = timeGPU([&]() {
        h_a.copyTo(d_a); h_b.copyTo(d_b);
        std::vector<const Tensor*> ins{&d_a, &d_b};
        std::vector<Tensor*> outs{&d_c};
        op.execute(Backend::CUDA, ins, outs);
        d_c.copyTo(h_c);
    });

    return {"Add", "[" + std::to_string(N) + "]", N, cpu_ms, kernel_ms, e2e_ms};
}

// Benchmark Softmax/LayerNorm (2D input, varying batch)
template<typename Op>
static Row benchBatched(const std::string& name, size_t B, size_t F) {
    TensorShape shape({B, F});
    Tensor h_in(shape, DataType::Float32, Device::CPU);
    Tensor h_out(shape, DataType::Float32, Device::CPU);
    fill(h_in);

    Tensor d_in(shape, DataType::Float32, Device::CUDA);
    Tensor d_out(shape, DataType::Float32, Device::CUDA);
    h_in.copyTo(d_in);

    Op op;

    double cpu_ms = timeCPU([&]() {
        std::vector<const Tensor*> ins{&h_in};
        std::vector<Tensor*> outs{&h_out};
        op.execute(Backend::CPU, ins, outs);
    });

    double kernel_ms = timeGPU([&]() {
        std::vector<const Tensor*> ins{&d_in};
        std::vector<Tensor*> outs{&d_out};
        op.execute(Backend::CUDA, ins, outs);
    });

    double e2e_ms = timeGPU([&]() {
        h_in.copyTo(d_in);
        std::vector<const Tensor*> ins{&d_in};
        std::vector<Tensor*> outs{&d_out};
        op.execute(Backend::CUDA, ins, outs);
        d_out.copyTo(h_out);
    });

    std::string sh = "[" + std::to_string(B) + "," + std::to_string(F) + "]";
    return {name, sh, B * F, cpu_ms, kernel_ms, e2e_ms};
}

// Benchmark MatMul
static Row benchMatMul(size_t M, size_t K, size_t N) {
    TensorShape sa({M, K}), sb({K, N}), sc({M, N});
    Tensor h_a(sa, DataType::Float32, Device::CPU);
    Tensor h_b(sb, DataType::Float32, Device::CPU);
    Tensor h_c(sc, DataType::Float32, Device::CPU);
    fill(h_a); fill(h_b, 0.2f);

    Tensor d_a(sa, DataType::Float32, Device::CUDA);
    Tensor d_b(sb, DataType::Float32, Device::CUDA);
    Tensor d_c(sc, DataType::Float32, Device::CUDA);
    h_a.copyTo(d_a); h_b.copyTo(d_b);

    MatMulOp op;

    double cpu_ms = timeCPU([&]() {
        std::vector<const Tensor*> ins{&h_a, &h_b};
        std::vector<Tensor*> outs{&h_c};
        op.execute(Backend::CPU, ins, outs);
    }, 5, 20);  // fewer CPU iters for large MatMul

    double kernel_ms = timeGPU([&]() {
        std::vector<const Tensor*> ins{&d_a, &d_b};
        std::vector<Tensor*> outs{&d_c};
        op.execute(Backend::CUDA, ins, outs);
    });

    double e2e_ms = timeGPU([&]() {
        h_a.copyTo(d_a); h_b.copyTo(d_b);
        std::vector<const Tensor*> ins{&d_a, &d_b};
        std::vector<Tensor*> outs{&d_c};
        op.execute(Backend::CUDA, ins, outs);
        d_c.copyTo(h_c);
    });

    std::string sh = "[" + std::to_string(M) + "," + std::to_string(K) +
                     "] @ [" + std::to_string(K) + "," + std::to_string(N) + "]";
    return {"MatMul", sh, M * N, cpu_ms, kernel_ms, e2e_ms};
}

int main() {
    // Warm up CUDA context before any measurement
    {
        Tensor dummy(TensorShape({1}), DataType::Float32, Device::CUDA);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "ForgeRT CUDA Benchmark\n";
    std::cout << "======================\n";
    std::cout << "GPU : " << prop.name << "  sm_" << prop.major << prop.minor
              << "  " << (prop.totalGlobalMem >> 20) << " MB VRAM\n";
    std::cout << "CPU : i5-9300H (benchmark machine)\n";
    std::cout << "Build: CUDA 12.6, MSVC 14.39, Release\n";
    std::cout << "Warmup: 10 iters, Measure: 100 iters\n";
    std::cout << "E2E: H2D + kernel + D2H (pre-allocated device tensors)\n";

    printHeader();

    // ReLU
    for (size_t n : {1024UL, 65536UL, 1048576UL})
        printRow(benchUnary<ReLUOp>("ReLU", TensorShape({n})));

    // Add
    for (size_t n : {1024UL, 65536UL, 1048576UL})
        printRow(benchAdd(n));

    // Softmax [B, 1024]
    for (size_t b : {1UL, 64UL, 512UL})
        printRow(benchBatched<SoftmaxOp>("Softmax", b, 1024));

    // LayerNorm [B, 768]
    for (size_t b : {1UL, 64UL, 512UL})
        printRow(benchBatched<LayerNormOp>("LayerNorm", b, 768));

    // MatMul
    for (auto [m, k, n] : std::vector<std::tuple<size_t,size_t,size_t>>{
            {64,   64,   64},
            {256,  256,  256},
            {512,  512,  512},
            {1024, 1024, 1024}})
        printRow(benchMatMul(m, k, n));

    std::cout << "\nDone.\n";
    return 0;
}
