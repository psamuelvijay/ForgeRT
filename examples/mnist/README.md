# MNIST Inference Demo

This example demonstrates real neural network inference using ForgeRT's computation graph and operators.

## Architecture

**Simple 2-Layer MLP:**
```
Input [1, 784]
  ↓
MatMul [784, 128] + Bias [1, 128]
  ↓
ReLU
  ↓
MatMul [128, 10] + Bias [1, 10]
  ↓
Softmax
  ↓
Output [1, 10] (class probabilities)
```

**Operators Used:**
- `MatMul` - Matrix multiplication for fully connected layers
- `Add` - Bias addition
- `ReLU` - Activation function
- `Softmax` - Output normalization

## Setup

### 1. Train and Export Model Weights

```bash
cd models
pip install torch torchvision numpy
python train_mnist.py
```

This will:
- Download MNIST dataset (~11MB to `models/data/`)
- Train the model for 5 epochs (~2-3 minutes on CPU)
- Export weights to `mnist_mlp.fgwt` (~200KB)
- Export 10 test samples to `mnist_test_samples.fgwt`
- Validate exported weights (should get 100% on 10 samples)

**Expected training accuracy: 97-98%**

### 2. Build ForgeRT

```bash
# From project root
cmake --build build --config Release
```

### 3. Run Inference Demo

```bash
# From project root
build/examples/mnist/mnist_inference
```

or with custom paths:

```bash
build/examples/mnist/mnist_inference path/to/weights.fgwt path/to/samples.fgwt
```

## Expected Output

```
=== ForgeRT MNIST Inference Demo ===

Backend: CPU

Loading weights from ../../models/mnist_mlp.fgwt...
  fc1_weight: [784, 128]
  fc1_bias: [1, 128]
  fc2_weight: [128, 10]
  fc2_bias: [1, 10]
✓ Weights loaded successfully

Loading test samples from ../../models/mnist_test_samples.fgwt...
✓ Loaded 10 test samples

Building computation graph...
✓ Graph built with 7 nodes

Running inference...
------------------------------------------------------------
Sample  0: Pred=7 True=7 Conf=0.9998 Time=1.234ms ✓
Sample  1: Pred=2 True=2 Conf=0.9995 Time=0.876ms ✓
Sample  2: Pred=1 True=1 Conf=0.9999 Time=0.845ms ✓
...
------------------------------------------------------------
Results:
  Accuracy: 10/10 (100%)
  Avg latency: 1.234 ms
  Total time: 12.34 ms

Memory Statistics:
  Peak allocated: 45.6 KB
  Current allocated: 38.2 KB
```

## Model File Format

The `.fgwt` (FORGERT_WEIGHTS) format is a simple binary format:

```
Header:
  uint32_t magic = 0x46475754 ('FGWT')
  uint32_t num_arrays

For each array:
  uint32_t name_length
  char[name_length] name
  uint32_t ndim
  uint64_t[ndim] shape
  float[product(shape)] data (row-major, float32)
```

This format is:
- Simple to parse in C++
- Language-agnostic
- Includes shape metadata
- Compact (no compression overhead)

## Implementation Notes

### Weight Layout

PyTorch `nn.Linear` stores weights as `[out_features, in_features]`, but matrix multiplication requires `[in_features, out_features]`. The export script transposes weights automatically.

### Graph Structure

The computation graph treats weights as inputs (not learned parameters), allowing them to be loaded once and reused across multiple inferences. This is efficient for inference workloads.

### Memory Management

All tensors are managed through ForgeRT's `Tensor` class, which:
- Automatically tracks memory allocations
- Supports both CPU and CUDA devices
- Reports peak and current memory usage

## Extending This Example

To adapt for other models:

1. **Add more layers**: Insert additional `MatMul + Add + ReLU` blocks
2. **Different activations**: Use `LayerNorm` or other operators
3. **Batch inference**: Change input shape from `[1, 784]` to `[batch_size, 784]`
4. **Other datasets**: Replace MNIST with CIFAR-10, Fashion-MNIST, etc.
5. **CUDA acceleration**: Build with CUDA support and compare CPU vs GPU latency

## Troubleshooting

**Missing weights file:**
```
Error: Failed to open weights file: ../../models/mnist_mlp.fgwt
```
→ Run `python models/train_mnist.py` first

**Shape mismatch errors:**
```
MatMulOp: Inner dimensions must match
```
→ Verify weight transpose in export script

**Low accuracy:**
```
Accuracy: 3/10 (30%)
```
→ Train model for more epochs or check data preprocessing
