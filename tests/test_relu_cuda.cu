/*
 * test_relu_cuda.cu  –  CUDA ReLU kernel correctness tests (Phase 3)
 *
 * Compiled by nvcc so cuda_runtime.h is available automatically.
 * Tests:
 *   1. Mixed positive/negative 1-K element array
 *   2. 1-M element array (exercises grid-stride loop)
 *   3. CPU vs CUDA numerical equivalence on 10K elements
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// Kernel launcher (defined in relu_kernel.cu, same CUDA library)
extern "C" int forgert_relu_cuda(const float* input, float* output, size_t n);

// ------------------------------------------------------------------ helpers --

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static const float kTol = 1e-5f;

// -------------------------------------------------------------------- tests --

static void test_relu_cuda_small() {
    printf("  Testing ReLU CUDA small array (1024 elements)...\n");

    const int n     = 1024;
    const size_t nb = n * sizeof(float);

    float* h_in  = new float[n];
    float* h_out = new float[n];
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i) - 512.0f;

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  nb));
    CUDA_CHECK(cudaMalloc(&d_out, nb));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, nb, cudaMemcpyHostToDevice));

    int err = forgert_relu_cuda(d_in, d_out, static_cast<size_t>(n));
    assert(err == 0 && "ReLU CUDA kernel returned error");

    CUDA_CHECK(cudaMemcpy(h_out, d_out, nb, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        float expected = (h_in[i] > 0.0f) ? h_in[i] : 0.0f;
        if (fabsf(h_out[i] - expected) > kTol) {
            fprintf(stderr, "  FAIL at i=%d: expected %.6f got %.6f\n",
                    i, expected, h_out[i]);
            assert(false);
        }
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;

    printf("  \xE2\x9C\x93 Small array passed\n");
}

static void test_relu_cuda_large() {
    printf("  Testing ReLU CUDA large array (1M elements, grid-stride)...\n");

    const size_t n  = 1024u * 1024u;
    const size_t nb = n * sizeof(float);

    float* h_in  = new float[n];
    float* h_out = new float[n];
    for (size_t i = 0; i < n; ++i)
        h_in[i] = static_cast<float>(i % 2000) - 1000.0f;

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  nb));
    CUDA_CHECK(cudaMalloc(&d_out, nb));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, nb, cudaMemcpyHostToDevice));

    int err = forgert_relu_cuda(d_in, d_out, n);
    assert(err == 0 && "ReLU CUDA kernel returned error");

    CUDA_CHECK(cudaMemcpy(h_out, d_out, nb, cudaMemcpyDeviceToHost));

    // Spot-check every 4096th element
    for (size_t i = 0; i < n; i += 4096) {
        float expected = (h_in[i] > 0.0f) ? h_in[i] : 0.0f;
        assert(fabsf(h_out[i] - expected) <= kTol);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;

    printf("  \xE2\x9C\x93 Large array (grid-stride) passed\n");
}

static void test_relu_cuda_vs_cpu() {
    printf("  Testing ReLU CUDA vs CPU numerical equivalence (10K elements)...\n");

    const int n     = 10000;
    const size_t nb = static_cast<size_t>(n) * sizeof(float);

    float* h_in       = new float[n];
    float* h_cpu_out  = new float[n];
    float* h_cuda_out = new float[n];

    // Values that exercise both branches: positive, negative, exact zero
    for (int i = 0; i < n; ++i)
        h_in[i] = static_cast<float>((i * 17) % 2001) - 1000.0f;

    // CPU reference
    for (int i = 0; i < n; ++i)
        h_cpu_out[i] = (h_in[i] > 0.0f) ? h_in[i] : 0.0f;

    // CUDA
    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  nb));
    CUDA_CHECK(cudaMalloc(&d_out, nb));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, nb, cudaMemcpyHostToDevice));

    int err = forgert_relu_cuda(d_in, d_out, static_cast<size_t>(n));
    assert(err == 0 && "ReLU CUDA kernel returned error");

    CUDA_CHECK(cudaMemcpy(h_cuda_out, d_out, nb, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        if (fabsf(h_cuda_out[i] - h_cpu_out[i]) > kTol) {
            fprintf(stderr, "  FAIL at i=%d: CPU=%.6f CUDA=%.6f\n",
                    i, h_cpu_out[i], h_cuda_out[i]);
            assert(false);
        }
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_cpu_out;
    delete[] h_cuda_out;

    printf("  \xE2\x9C\x93 CPU vs CUDA equivalence passed\n");
}

// --------------------------------------------------------------------- main --
int main() {
    printf("========================================\n");
    printf("ForgeRT ReLU CUDA Kernel Tests\n");
    printf("========================================\n");

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    test_relu_cuda_small();
    test_relu_cuda_large();
    test_relu_cuda_vs_cpu();

    printf("========================================\n");
    printf("All CUDA tests PASSED\n");
    printf("========================================\n");
    return 0;
}
