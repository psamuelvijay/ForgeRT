# ForgeRT - Build and Test Container
#
# Base: nvidia/cuda 12.6 devel on Ubuntu 22.04. The CUDA *toolkit* (nvcc +
# headers) is required because the top-level CMakeLists.txt declares
# `LANGUAGES CXX CUDA` and targets compute capability sm_61 (GTX 1050-class
# Pascal). CUDA 12.x is the correct toolkit generation for sm_61.
#
# What the default image DOES:
#   - Configures and builds the whole project (CPU + CUDA objects compile).
#   - Runs the CPU-safe CTest subset at build time as a correctness gate.
#
# What the default image does NOT do:
#   - It does NOT run CUDA tests at build time. Those need a real GPU at
#     runtime; `docker build` normally has no GPU, so they are excluded from
#     the build-time gate. Run them separately with `docker run --gpus all`
#     (see README "Build with Docker").
#   - It does NOT install Python by default, so the Python converter /
#     end-to-end tests are not registered. Enable them with
#     `--build-arg WITH_PYTHON=1` (adds numpy/onnx/onnxruntime, CPU-only).
#
# See README.md -> "Build with Docker" for full usage.

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Optional Python toolchain for the converter / end-to-end tests (CPU-only).
# 0 (default) = C++ only image. 1 = install python3 + pinned requirements and
# register the Python-based CTest cases.
ARG WITH_PYTHON=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
    && if [ "$WITH_PYTHON" = "1" ]; then \
         apt-get install -y --no-install-recommends python3 python3-pip ; \
       fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /forgert

# Python dependencies first (only when enabled) so the layer caches well.
COPY requirements.txt ./
RUN if [ "$WITH_PYTHON" = "1" ]; then \
        python3 -m pip install --no-cache-dir -r requirements.txt ; \
    fi

# Copy the source tree. build/, .git/, model weights, and Context/ are excluded
# via .dockerignore. models/ is copied for the CPU end-to-end test's ONNX input
# (the ~5 MB squeezenet1.1-7.onnx); large *.fgwt/*.npz/data are excluded.
COPY CMakeLists.txt ./
COPY include/    ./include/
COPY src/        ./src/
COPY cuda/       ./cuda/
COPY tests/      ./tests/
COPY examples/   ./examples/
COPY benchmarks/ ./benchmarks/
COPY apps/       ./apps/
COPY tools/      ./tools/
COPY models/     ./models/

# Configure and build the whole project from a clean checkout.
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CUDA_ARCHITECTURES=61 && \
    cmake --build build

# CPU-safe build-time test gate.
#
# IMPORTANT: ctest -L/-LE match a REGEX against individual labels, and only
# ScheduledCUDATest/SqueezeNetE2ETest actually carry labels. The 9 pure CUDA
# tests and LayerNormStressTest carry NO labels, so label filtering alone would
# let them run and fail without a GPU. We therefore exclude by NAME regex: every
# GPU test name contains "CUDA" except LayerNormStressTest. This keeps all CPU
# C++ tests (and, when WITH_PYTHON=1, the CPU-only Python tests) and preserves
# real failure propagation (no masking).
RUN cd build && ctest --output-on-failure -E "CUDA|LayerNormStressTest"

WORKDIR /forgert

CMD ["bash"]
