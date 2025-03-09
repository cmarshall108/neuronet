# NeuroNet

A C++ tensor computation library optimized for older GPUs (like Nvidia Tesla K80) and modern macOS devices with Metal support. NeuroNet provides a PyTorch-like API with hardware acceleration capabilities for both CUDA and Metal backends, along with a CPU fallback implementation.

## Features

- Cross-platform tensor operations (CPU, CUDA, Metal)
- Neural network primitives (layers, activations, optimizers)
- HuggingFace model loading capability
- Optimized for older NVIDIA GPUs (Tesla K80) and Apple Silicon
- PyTorch-like API for easy adoption

## Hardware Support

- **CPU**: Generic multi-threaded implementation
- **CUDA**: Support for NVIDIA GPUs (optimized for Tesla K80)
- **Metal**: Support for Apple GPUs (M1/M2/M3 chips and AMD GPUs in Intel Macs)

## Prerequisites

- CMake 3.18+
- C++17 compatible compiler
- For CUDA backend: CUDA Toolkit 10.0+ and compatible NVIDIA GPU
- For Metal backend: macOS 10.15+ with compatible GPU
- libcurl for HuggingFace model downloading

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Options

- `NEURONET_USE_CUDA=ON/OFF` - Enable CUDA support (default: ON)
- `NEURONET_USE_METAL=ON/OFF` - Enable Metal support on macOS (default: ON)
- `NEURONET_BUILD_TESTS=ON/OFF` - Build tests (default: ON)
- `NEURONET_BUILD_EXAMPLES=ON/OFF` - Build examples (default: ON)

## Basic Usage

### Tensor Creation and Operations

```cpp
#include <neuronet/core/tensor.h>
#include <neuronet/core/ops.h>
#include <iostream>

using namespace neuronet;

int main() {
    // Initialize backends
    ops::initialize_backends();
    
    // Create tensors on CPU
    Tensor a({2, 3}, DType::Float32);
    Tensor b({2, 3}, DType::Float32);
    
    // Move to available GPU if present
    DeviceType device = Device::isCudaAvailable() ? DeviceType::CUDA : 
                       (Device::isMetalAvailable() ? DeviceType::Metal : DeviceType::CPU);
    
    a = a.to(device);
    b = b.to(device);
    
    // Perform operations
    Tensor c = a + b;
    Tensor d = a.matmul(b.transpose(0, 1));
    
    std::cout << c << std::endl;
    std::cout << d << std::endl;
    
    // Clean up
    ops::cleanup_backends();
    
    return 0;
}
```

### Neural Network Example

```cpp
#include <neuronet/nn/layers.h>
#include <neuronet/core/tensor.h>

using namespace neuronet;
using namespace neuronet::nn;

// Create a simple neural network
auto model = Sequential();
model.add_module("fc1", std::make_shared<Linear>(784, 128));
model.add_module("relu1", std::make_shared<ReLU>());
model.add_module("fc2", std::make_shared<Linear>(128, 10));

// Forward pass
Tensor input({32, 784}, DType::Float32);  // 32 samples of 784 features (e.g., MNIST)
Tensor output = model.forward(input);
```

### Loading a HuggingFace Model

```cpp
#include <neuronet/models/huggingface.h>
#include <neuronet/core/tensor.h>

using namespace neuronet;

// Choose the best available device
DeviceType device_type = Device::isCudaAvailable() ? DeviceType::CUDA : 
                       (Device::isMetalAvailable() ? DeviceType::Metal : DeviceType::CPU);

// Load a pre-trained BERT model
auto model = models::HuggingFaceModel::from_pretrained("bert-base-uncased", "", device_type);

// Prepare input (simplified example)
Tensor input({1, 128}, DType::Int64);  // Batch size 1, sequence length 128
// ... fill input with token IDs ...

// Run inference
Tensor output = model->forward(input);
```

## Architecture

NeuroNet is designed with a layered architecture:

1. **Core Layer**: Tensor operations, memory management, device abstraction
2. **Backend Layer**: Implementations for CPU, CUDA, and Metal
3. **Neural Network Layer**: Common neural network building blocks
4. **Models Layer**: Pre-trained model loading and execution

## Optimizations

- Multi-threaded CPU operations
- cuBLAS integration for CUDA matrix operations
- Metal Performance Shaders (MPS) for optimized Metal operations
- Memory pooling to reduce allocation overhead
- Kernel fusion when possible

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
