# ForgeRT - CPU Build and Test Container
#
# NOTE: ForgeRT CMakeLists.txt unconditionally requires CUDA toolkit (LANGUAGES CUDA).
# This image uses nvidia/cuda base to provide nvcc and headers, but runtime execution
# is CPU-only and no GPU is needed to run tests or the MNIST demo.

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /forgert

# Copy source tree (excludes build/, models/data/, *.fgwt per .dockerignore)
COPY CMakeLists.txt ./
COPY cmake/      ./cmake/
COPY include/    ./include/
COPY src/        ./src/
COPY cuda/       ./cuda/
COPY tests/      ./tests/
COPY examples/   ./examples/
COPY benchmarks/ ./benchmarks/

# Configure and build
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CUDA_ARCHITECTURES=61 && \
    cmake --build build --config Release

# Run tests during build to verify correctness
RUN cd build && ctest --output-on-failure

WORKDIR /forgert

CMD ["bash"]
