// MNIST Inference Demo with Trained Weights
// Demonstrates real neural network inference using ForgeRT

#include <forgert/forgert.h>
#include <forgert/graph/graph.h>
#include <forgert/operator/matmul.h>
#include <forgert/operator/add.h>
#include <forgert/operator/relu.h>
#include <forgert/operator/softmax.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <iomanip>

using namespace forgert;

// FGWT file format loader
struct WeightArray {
    std::string name;
    std::vector<size_t> shape;
    std::vector<float> data;
};

class FGWTLoader {
public:
    static std::vector<WeightArray> load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open: " + filename);
        }

        // Read magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x46475754) {  // 'FGWT'
            throw std::runtime_error("Invalid file format (bad magic)");
        }

        // Read number of arrays
        uint32_t num_arrays;
        file.read(reinterpret_cast<char*>(&num_arrays), sizeof(num_arrays));

        std::vector<WeightArray> arrays;
        for (uint32_t i = 0; i < num_arrays; ++i) {
            WeightArray array;
            
            // Read name
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            array.name.resize(name_len);
            file.read(&array.name[0], name_len);
            
            // Read shape
            uint32_t ndim;
            file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
            array.shape.resize(ndim);
            for (uint32_t j = 0; j < ndim; ++j) {
                uint64_t dim;
                file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                array.shape[j] = static_cast<size_t>(dim);
            }
            
            // Read data
            size_t num_elements = 1;
            for (size_t dim : array.shape) {
                num_elements *= dim;
            }
            array.data.resize(num_elements);
            file.read(reinterpret_cast<char*>(array.data.data()), 
                     num_elements * sizeof(float));
            
            if (!file) {
                throw std::runtime_error("Failed to read data for " + array.name);
            }
            
            arrays.push_back(std::move(array));
        }
        
        return arrays;
    }
};

int main(int argc, char** argv) {
    std::cout << "ForgeRT MNIST Inference Demo\n";
    std::cout << "============================\n\n";
    
    // File paths
    std::string weights_file = "../../models/mnist_mlp.fgwt";
    std::string samples_file = "../../models/mnist_test_samples.fgwt";
    
    if (argc > 1) weights_file = argv[1];
    if (argc > 2) samples_file = argv[2];
    
    try {
        // Load weights
        std::cout << "Loading weights from " << weights_file << "...\n";
        auto weight_arrays = FGWTLoader::load(weights_file);
        
        std::cout << "Loaded " << weight_arrays.size() << " weight arrays:\n";
        for (const auto& arr : weight_arrays) {
            std::cout << "  " << arr.name << ": [";
            for (size_t i = 0; i < arr.shape.size(); ++i) {
                std::cout << arr.shape[i];
                if (i < arr.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        
        // Extract weights
        std::vector<float> fc1_w, fc1_b, fc2_w, fc2_b;
        for (const auto& arr : weight_arrays) {
            if (arr.name == "fc1_weight") fc1_w = arr.data;
            else if (arr.name == "fc1_bias") fc1_b = arr.data;
            else if (arr.name == "fc2_weight") fc2_w = arr.data;
            else if (arr.name == "fc2_bias") fc2_b = arr.data;
        }
        
        if (fc1_w.empty() || fc1_b.empty() || fc2_w.empty() || fc2_b.empty()) {
            throw std::runtime_error("Missing weight arrays");
        }
        
        // Create weight tensors
        Tensor fc1_weight(TensorShape({784, 128}), DataType::Float32, Device::CPU);
        Tensor fc1_bias(TensorShape({1, 128}), DataType::Float32, Device::CPU);
        Tensor fc2_weight(TensorShape({128, 10}), DataType::Float32, Device::CPU);
        Tensor fc2_bias(TensorShape({1, 10}), DataType::Float32, Device::CPU);
        
        std::memcpy(fc1_weight.data(), fc1_w.data(), fc1_w.size() * sizeof(float));
        std::memcpy(fc1_bias.data(), fc1_b.data(), fc1_b.size() * sizeof(float));
        std::memcpy(fc2_weight.data(), fc2_w.data(), fc2_w.size() * sizeof(float));
        std::memcpy(fc2_bias.data(), fc2_b.data(), fc2_b.size() * sizeof(float));
        
        std::cout << "[OK] Weights loaded\n\n";
        
        // Load test samples
        std::cout << "Loading test samples from " << samples_file << "...\n";
        auto sample_arrays = FGWTLoader::load(samples_file);
        
        std::vector<float> images, labels_float;
        for (const auto& arr : sample_arrays) {
            if (arr.name == "images") images = arr.data;
            else if (arr.name == "labels") labels_float = arr.data;
        }
        
        if (images.empty() || labels_float.empty()) {
            throw std::runtime_error("Missing test data");
        }
        
        size_t num_samples = labels_float.size();
        std::cout << "[OK] Loaded " << num_samples << " test samples\n\n";
        
        // Build computation graph
        std::cout << "Building computation graph...\n";
        Graph graph;
        
        size_t input_id = graph.addInput(TensorShape({1, 784}), DataType::Float32);
        size_t fc1_w_id = graph.addInput(TensorShape({784, 128}), DataType::Float32);
        size_t fc1_b_id = graph.addInput(TensorShape({1, 128}), DataType::Float32);
        
        auto fc1_out = graph.addNode(std::make_unique<MatMulOp>(), {input_id, fc1_w_id});
        auto fc1_bias_out = graph.addNode(std::make_unique<AddOp>(), {fc1_out[0], fc1_b_id});
        auto relu_out = graph.addNode(std::make_unique<ReLUOp>(), {fc1_bias_out[0]});
        
        size_t fc2_w_id = graph.addInput(TensorShape({128, 10}), DataType::Float32);
        size_t fc2_b_id = graph.addInput(TensorShape({1, 10}), DataType::Float32);
        
        auto fc2_out = graph.addNode(std::make_unique<MatMulOp>(), {relu_out[0], fc2_w_id});
        auto fc2_bias_out = graph.addNode(std::make_unique<AddOp>(), {fc2_out[0], fc2_b_id});
        auto softmax_out = graph.addNode(std::make_unique<SoftmaxOp>(), {fc2_bias_out[0]});
        
        graph.markOutput(softmax_out[0]);
        graph.validate();
        
        std::cout << "[OK] Graph built with " << graph.numNodes() << " nodes\n\n";
        
        // Run inference on test samples
        std::cout << "Running inference on " << num_samples << " samples...\n";
        std::cout << std::string(60, '-') << "\n";
        
        int correct = 0;
        for (size_t i = 0; i < num_samples; ++i) {
            // Create input tensor
            Tensor input(TensorShape({1, 784}), DataType::Float32, Device::CPU);
            std::memcpy(input.data(), &images[i * 784], 784 * sizeof(float));
            
            // Execute graph
            auto outputs = graph.execute({&input, &fc1_weight, &fc1_bias, 
                                         &fc2_weight, &fc2_bias}, Backend::CPU);
            
            // Get prediction
            const float* probs = static_cast<const float*>(outputs[0]->data());
            int prediction = 0;
            float max_prob = probs[0];
            for (int j = 1; j < 10; ++j) {
                if (probs[j] > max_prob) {
                    max_prob = probs[j];
                    prediction = j;
                }
            }
            
            int true_label = static_cast<int>(labels_float[i]);
            bool is_correct = (prediction == true_label);
            if (is_correct) correct++;
            
            std::cout << "Sample " << std::setw(2) << i 
                      << ": Pred=" << prediction 
                      << " True=" << true_label
                      << " Conf=" << std::fixed << std::setprecision(4) << max_prob
                      << " " << (is_correct ? "[OK]" : "[FAIL]") << "\n";
        }
        
        std::cout << std::string(60, '-') << "\n";
        std::cout << "Results:\n";
        std::cout << "  Accuracy: " << correct << "/" << num_samples 
                  << " (" << (100.0 * correct / num_samples) << "%)\n";
        
        std::cout << "\n[OK] Demo complete!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
