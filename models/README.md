# ForgeRT Models

This directory contains pretrained model weights for ForgeRT inference demonstrations.

## MNIST MLP

A simple 2-layer MLP trained on MNIST digit classification.

**Architecture:**
- Input: 784 (28×28 flattened)
- Hidden: 128 neurons with ReLU
- Output: 10 classes with Softmax

**Obtaining the Model:**

### Option 1: Train from Scratch (Recommended)

```bash
# Install dependencies
pip install torch torchvision numpy

# Train the model (takes ~2-3 minutes on CPU)
cd models
python train_mnist.py
```

This will:
- Download the MNIST dataset (~11MB)
- Train the model for 5 epochs
- Export weights to `mnist_mlp.npz` (~200KB)
- Export 10 test samples to `mnist_test_samples.npz`
- Validate the exported weights

Expected accuracy: **97-98%** on MNIST test set

### Option 2: Download Pretrained Weights

If you prefer to download pretrained weights instead of training:

```bash
# TODO: Add download link once model is uploaded to releases
```

## File Format

The exported `.npz` file contains:
- `fc1_weight`: [128, 784] - First layer weights
- `fc1_bias`: [128] - First layer bias
- `fc2_weight`: [10, 128] - Second layer weights
- `fc2_bias`: [10] - Second layer bias

All weights are stored as `float32` in row-major order.

## Usage with ForgeRT

See `examples/mnist/mnist_inference.cpp` for how to load and run inference with these weights.

## Note

⚠️ **Do not commit the generated `.npz` files or `data/` directory to Git.**

These files are listed in `.gitignore` and should be generated locally or downloaded separately.
