#include <neuronet/core/tensor.h>
#include <neuronet/core/ops.h>
#include <neuronet/nn/layers.h>
#include <neuronet/utils/logging.h>
#include <iostream>
#include <vector>

using namespace neuronet;
using namespace neuronet::ops;

// Simple feed-forward neural network for illustration
class SimpleNN {
public:
    SimpleNN(DeviceType device_type = DeviceType::CPU) : device_type_(device_type) {
        // Initialize layers
        fc1_ = std::make_shared<nn::Linear>(784, 128);     // Input -> Hidden
        fc2_ = std::make_shared<nn::Linear>(128, 10);      // Hidden -> Output
        
        // Move to specified device
        fc1_->to(device_type_);
        fc2_->to(device_type_);
        
        log_info("Initialized SimpleNN on {}", Device(device_type_).toString());
    }
    
    Tensor forward(const Tensor& x) {
        // x shape: [batch_size, 784]
        Tensor h1 = fc1_->forward(x);        // [batch_size, 128]
        Tensor h1_act = ops::relu(h1);       // Apply ReLU activation
        Tensor logits = fc2_->forward(h1_act); // [batch_size, 10]
        return logits;
    }
    
private:
    DeviceType device_type_;
    std::shared_ptr<nn::Linear> fc1_;
    std::shared_ptr<nn::Linear> fc2_;
};

int main() {
    // Initialize the operations backends
    ops::initialize_backends();
    
    // Set log level
    set_log_level(LogLevel::Info);
    
    log_info("NeuroNet Simple Example");
    
    // Choose the available device with highest priority: CUDA > Metal > CPU
    DeviceType device_type = DeviceType::CPU;
    
    if (Device::isCudaAvailable()) {
        device_type = DeviceType::CUDA;
        log_info("Using CUDA device");
    } else if (Device::isMetalAvailable()) {
        device_type = DeviceType::Metal;
        log_info("Using Metal device");
    } else {
        log_info("Using CPU device");
    }
    
    // Create a simple neural network
    SimpleNN model(device_type);
    
    // Create a batch of random input data (simulating MNIST images)
    // 10 samples of 784 features (28x28 images)
    std::vector<float> input_data(10 * 784);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Create input tensor on the selected device
    Tensor input({10, 784}, input_data.data(), DType::Float32, device_type);
    
    // Forward pass
    log_info("Performing forward pass...");
    Tensor output = model.forward(input);
    
    log_info("Output tensor shape: [{}x{}]", std::to_string(output.shape()[0]), std::to_string(output.shape()[1]));
    
    // Move output to CPU for display
    Tensor cpu_output = output.cpu();
    
    // Print first sample prediction
    log_info("First sample predictions:");
    const float* output_data = cpu_output.data<float>();
    for (int i = 0; i < 10; ++i) {
        std::cout << "Class " << i << ": " << output_data[i] << std::endl;
    }
    
    // Clean up
    ops::cleanup_backends();
    
    return 0;
}
