# ForgeRT Development Guide

## Prerequisites

### Required
- **Windows 10/11** (primary development platform)
- **MSVC 2019 or later** (Visual Studio Build Tools)
- **CMake 3.18+**
- **NVIDIA CUDA Toolkit 12.x**
- **Python 3.8+** (for tooling and benchmarks)
- **Git**

### Recommended
- **Visual Studio Code** or **Visual Studio 2022**
- **NVIDIA Nsight Compute** (kernel profiling)
- **NVIDIA Nsight Systems** (system-wide profiling)

## Environment Setup

1. **Install CUDA Toolkit**:
   - Download from NVIDIA website
   - Ensure `nvcc` is in PATH
   - Verify: `nvcc --version`

2. **Verify GPU**:
   ```powershell
   nvidia-smi
   ```
   Should show GTX 1050 with compute capability 6.1

3. **Clone Repository**:
   ```powershell
   git clone https://github.com/psamuelvijay/ForgeRT.git
   cd ForgeRT
   ```

4. **Build**:
   ```powershell
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

## Development Workflow

### Adding a New Operator

1. **Define interface** in `include/forgert/graph/operator.h`
2. **Implement CPU version** in `src/backend/cpu/`
3. **Write unit tests** in `tests/`
4. **Verify correctness** against reference implementation
5. **Implement CUDA version** in `cuda/kernels/`
6. **Profile both implementations**
7. **Update scheduler cost model**

### Testing

```powershell
cd build
ctest --output-on-failure
```

For specific tests:
```powershell
ctest -R tensor_test --verbose
```

### Benchmarking

```powershell
cd build/benchmarks
./matmul_benchmark.exe
```

Best practices:
- Warm up GPU before measurement
- Run multiple iterations
- Record min/max/median/mean
- Document system state (GPU clocks, CPU frequency)

### Profiling CUDA Kernels

**Nsight Compute**:
```powershell
ncu --set full -o profile ./benchmark.exe
ncu-ui profile.ncu-rep
```

**Nsight Systems**:
```powershell
nsys profile -o timeline ./benchmark.exe
nsys-ui timeline.nsys-rep
```

## Code Style

### C++
- Use `clang-format` with provided `.clang-format`
- Modern C++17 features encouraged
- Prefer RAII over manual resource management
- Use `const` wherever possible
- Smart pointers over raw pointers for ownership

### CUDA
- Always check return values: `CUDA_CHECK(cudaMemcpy(...))`
- Synchronize after kernel launches when debugging
- Document kernel launch parameters
- Explain shared memory usage
- Comment on why specific block/grid sizes chosen

### Naming Conventions
- Classes: `PascalCase`
- Functions: `camelCase`
- Variables: `snake_case`
- Constants: `UPPER_CASE`
- Namespaces: `lowercase`

Example:
```cpp
namespace forgert {

class TensorShape {
public:
    size_t getDimension(int index) const;
    
private:
    std::vector<size_t> dimensions_;
};

constexpr size_t MAX_DIMENSIONS = 8;

} // namespace forgert
```

## Git Workflow

### Branch Naming
- `feature/tensor-abstraction`
- `fix/cuda-memory-leak`
- `perf/matmul-optimization`
- `docs/architecture-guide`

### Commit Messages
```
<type>: <short summary>

<detailed description>

<breaking changes if any>
```

Types: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `style`, `chore`

Example:
```
feat: add CUDA matrix multiplication kernel

Implement tiled matrix multiplication targeting Pascal sm_61.
Uses 32x32 tiles with shared memory for coalesced access.

Benchmark shows 2.3x speedup over naive CPU implementation
for matrices larger than 512x512.
```

## Performance Engineering Guidelines

### Measurement Methodology

1. **Isolate measurement**:
   - Measure only the kernel/operation
   - Exclude setup and teardown
   - Exclude memory allocation from timing

2. **Warm-up**:
   - Run operation 10 times before measurement
   - Allows GPU clocks to ramp up
   - Warms instruction cache

3. **Statistical rigor**:
   - Collect at least 100 samples
   - Report min, median, mean, max, stddev
   - Check for outliers

4. **Reproducibility**:
   - Document GPU clocks (base vs boost)
   - Record CPU frequency
   - Note background processes
   - Record CUDA driver version

### Optimization Process

1. **Measure baseline**: Get reference performance
2. **Identify bottleneck**: CPU? Memory bandwidth? Kernel?
3. **Hypothesize**: What is limiting performance?
4. **Optimize**: Make ONE change
5. **Measure again**: Did it improve?
6. **Document**: Why did it improve (or not)?

**Never optimize without measurements.**

### CUDA Optimization Checklist

- [ ] Coalesced memory access
- [ ] Shared memory usage for reuse
- [ ] Occupancy analysis
- [ ] Bank conflict avoidance
- [ ] Warp divergence minimization
- [ ] Register pressure check
- [ ] Grid/block size tuning

## Debugging

### CUDA Errors

Enable synchronous execution:
```cpp
cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
```

Check after every kernel:
```cpp
kernel<<<grid, block>>>();
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

### Memory Issues

Use `cuda-memcheck`:
```powershell
cuda-memcheck --leak-check full ./test.exe
```

### Correctness

Compare against reference:
```cpp
// Reference CPU implementation
auto expected = reference_matmul(A, B);

// ForgeRT implementation
auto result = forgert_matmul(A, B);

// Compare with tolerance
ASSERT_NEAR(expected, result, 1e-5);
```

## Common Issues

### Issue: `nvcc not found`
**Solution**: Add CUDA bin directory to PATH

### Issue: Kernel launch fails with `cudaErrorInvalidConfiguration`
**Solution**: Check grid/block dimensions don't exceed hardware limits

### Issue: Performance worse than expected
**Solution**: Profile with Nsight Compute, check occupancy and memory throughput

### Issue: Results differ between CPU and GPU
**Solution**: Check floating-point precision, operation order, reduction algorithm

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Pascal Architecture Whitepaper](https://www.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## Questions?

Open an issue on GitHub with:
- Problem description
- Environment details (OS, CUDA version, GPU)
- Minimal reproducible example
- What you've tried
