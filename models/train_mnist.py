"""
Train a simple MNIST MLP and export weights for ForgeRT inference demo.

Architecture: 784 -> 128 -> 10 (input -> hidden -> output)
Activations: ReLU, Softmax
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import struct

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Logits (softmax applied during loss)

def train_model(epochs=5):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create model
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)] '
                      f'Loss: {loss.item():.6f}')
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    
    return model

def write_fgwt_format(filename, arrays):
    """
    Write weights in FORGERT_WEIGHTS format (.fgwt)
    
    Format:
      uint32_t magic = 0x46475754 ('FGWT')
      uint32_t num_arrays
      For each array:
        uint32_t name_length
        char[name_length] name
        uint32_t ndim
        uint64_t[ndim] shape
        float[product(shape)] data
    """
    with open(filename, 'wb') as f:
        # Write magic number
        f.write(struct.pack('<I', 0x46475754))  # 'FGWT'
        
        # Write number of arrays
        f.write(struct.pack('<I', len(arrays)))
        
        for name, array in arrays.items():
            # Write name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write shape
            f.write(struct.pack('<I', len(array.shape)))
            for dim in array.shape:
                f.write(struct.pack('<Q', dim))
            
            # Write data (flatten and convert to float32)
            data = array.astype(np.float32).flatten()
            f.write(data.tobytes())

def export_weights(model, output_dir='.'):
    """Export model weights to ForgeRT format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract weights and biases (PyTorch stores as [out_features, in_features])
    fc1_weight = model.fc1.weight.detach().cpu().numpy()  # [128, 784]
    fc1_bias = model.fc1.bias.detach().cpu().numpy()      # [128]
    fc2_weight = model.fc2.weight.detach().cpu().numpy()  # [10, 128]
    fc2_bias = model.fc2.bias.detach().cpu().numpy()      # [10]
    
    # Transpose weights for ForgeRT (needs [in_features, out_features])
    fc1_weight_t = fc1_weight.T  # [784, 128]
    fc2_weight_t = fc2_weight.T  # [128, 10]
    
    # Reshape biases to [1, out_features] for broadcasting
    fc1_bias_reshaped = fc1_bias.reshape(1, 128)
    fc2_bias_reshaped = fc2_bias.reshape(1, 10)
    
    print("\nWeight shapes (transposed for ForgeRT):")
    print(f"  fc1_weight: {fc1_weight_t.shape}")
    print(f"  fc1_bias: {fc1_bias_reshaped.shape}")
    print(f"  fc2_weight: {fc2_weight_t.shape}")
    print(f"  fc2_bias: {fc2_bias_reshaped.shape}")
    
    # Export to FGWT format
    output_file = os.path.join(output_dir, 'mnist_mlp.fgwt')
    write_fgwt_format(output_file, {
        'fc1_weight': fc1_weight_t,
        'fc1_bias': fc1_bias_reshaped,
        'fc2_weight': fc2_weight_t,
        'fc2_bias': fc2_bias_reshaped
    })
    
    print(f"\n✓ Weights exported to {output_file}")
    
    # Save a few test samples for validation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Save first 10 test samples
    test_images = []
    test_labels = []
    for i in range(10):
        img, label = test_dataset[i]
        test_images.append(img.numpy())
        test_labels.append(label)
    
    images_array = np.array(test_images).reshape(10, 784)  # [10, 784]
    labels_array = np.array(test_labels, dtype=np.float32)  # [10]
    
    test_file = os.path.join(output_dir, 'mnist_test_samples.fgwt')
    write_fgwt_format(test_file, {
        'images': images_array,
        'labels': labels_array
    })
    
    print(f"✓ Test samples exported to {test_file}")
    
    return fc1_weight_t, fc1_bias_reshaped, fc2_weight_t, fc2_bias_reshaped

def validate_exported_model(fc1_w, fc1_b, fc2_w, fc2_b, output_dir='.'):
    """Validate that exported weights produce correct predictions"""
    # Load test samples
    test_file = os.path.join(output_dir, 'mnist_test_samples.fgwt')
    
    # Read the file manually
    with open(test_file, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == 0x46475754, "Invalid magic number"
        
        num_arrays = struct.unpack('<I', f.read(4))[0]
        
        images = None
        labels = None
        
        for _ in range(num_arrays):
            # Read name
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            
            # Read shape
            ndim = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
            
            # Read data
            num_elements = 1
            for d in shape:
                num_elements *= d
            data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
            data = data.reshape(shape)
            
            if name == 'images':
                images = data
            elif name == 'labels':
                labels = data
    
    # Convert to tensors
    fc1_w_t = torch.from_numpy(fc1_w.T)  # [128, 784]
    fc1_b_t = torch.from_numpy(fc1_b.flatten())  # [128]
    fc2_w_t = torch.from_numpy(fc2_w.T)  # [10, 128]
    fc2_b_t = torch.from_numpy(fc2_b.flatten())  # [10]
    
    test_images = torch.from_numpy(images)
    test_labels = labels.astype(np.int32)
    
    print("\nValidating exported weights on test samples:")
    correct = 0
    for i in range(len(test_images)):
        x = test_images[i:i+1]  # [1, 784]
        
        # Forward pass (matching ForgeRT graph)
        x = torch.matmul(x, fc1_w_t.T) + fc1_b_t  # [1, 128]
        x = torch.relu(x)
        x = torch.matmul(x, fc2_w_t.T) + fc2_b_t  # [1, 10]
        x = torch.softmax(x, dim=1)
        
        pred = x.argmax(dim=1).item()
        label = int(test_labels[i])
        confidence = x[0, pred].item()
        
        match = "✓" if pred == label else "✗"
        print(f"  Sample {i}: Pred={pred} True={label} Conf={confidence:.4f} {match}")
        
        if pred == label:
            correct += 1
    
    print(f"\n✓ Validation: {correct}/{len(test_images)} correct ({100*correct/len(test_images):.1f}%)")

if __name__ == '__main__':
    print("Training MNIST MLP for ForgeRT demo...\n")
    model = train_model(epochs=5)
    
    print("\nExporting weights to ForgeRT format...")
    fc1_w, fc1_b, fc2_w, fc2_b = export_weights(model)
    
    print("\nValidating exported weights...")
    validate_exported_model(fc1_w, fc1_b, fc2_w, fc2_b)
    
    print("\n" + "="*60)
    print("✓ Training and export complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Build ForgeRT: cmake --build build")
    print("  2. Run demo: build/examples/mnist/mnist_inference")
