#include <neuronet/nn/activations.h>
#include <neuronet/core/ops.h>
#include <neuronet/utils/logging.h>
#include <cmath>

namespace neuronet {
namespace nn {

// ReLU activation
Tensor relu(const Tensor& input) {
    return ops::relu(input);
}

// Sigmoid activation
Tensor sigmoid(const Tensor& input) {
    // Check if we have a built-in sigmoid operation for the current device
    DeviceType device_type = input.device().type();
    
    if (device_type == DeviceType::CPU) {
        return ops::cpu::sigmoid(input);
    }
#ifdef NEURONET_USE_CUDA
    else if (device_type == DeviceType::CUDA) {
        return ops::cuda::sigmoid(input);
    }
#endif
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    else if (device_type == DeviceType::Metal) {
        // If Metal sigmoid not implemented, fall back to CPU
        Tensor cpu_input = input.cpu();
        Tensor cpu_result = ops::cpu::sigmoid(cpu_input);
        return cpu_result.to(device_type);
    }
#endif

    // Fallback: use CPU implementation and move back to original device
    Tensor cpu_input = input.cpu();
    Tensor cpu_result = ops::cpu::sigmoid(cpu_input);
    return cpu_result.to(device_type);
}

// Tanh activation
Tensor tanh(const Tensor& input) {
    DeviceType device_type = input.device().type();
    
    if (device_type == DeviceType::CPU) {
        // Implement using the CPU backend if available
        // For simplicity, we'll create a custom implementation
        Tensor result(input.shape(), input.dtype(), device_type);
        const float* input_data = input.data<float>();
        float* result_data = result.data<float>();
        
        for (int64_t i = 0; i < input.size(); ++i) {
            // Use standard tanh implementation
            result_data[i] = std::tanh(input_data[i]);
        }
        
        return result;
    }
#ifdef NEURONET_USE_CUDA
    else if (device_type == DeviceType::CUDA) {
        // Use CUDA implementation if available
        if (ops::cuda::tanh != nullptr) {
            return ops::cuda::tanh(input);
        }
    }
#endif
    
    // Fallback: use CPU implementation and move back to original device
    Tensor cpu_input = input.cpu();
    Tensor cpu_result = tanh(cpu_input); // Use the CPU implementation
    return cpu_result.to(device_type);
}

// Softmax activation
Tensor softmax(const Tensor& input, int dim) {
    DeviceType device_type = input.device().type();
    
    if (device_type == DeviceType::CPU) {
        return ops::cpu::softmax(input, dim);
    }
#ifdef NEURONET_USE_CUDA
    else if (device_type == DeviceType::CUDA) {
        return ops::cuda::softmax(input, dim);
    }
#endif
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    else if (device_type == DeviceType::Metal) {
        // If Metal softmax not implemented, fall back to CPU
        Tensor cpu_input = input.cpu();
        Tensor cpu_result = ops::cpu::softmax(cpu_input, dim);
        return cpu_result.to(device_type);
    }
#endif

    // Fallback: use CPU implementation and move back to original device
    Tensor cpu_input = input.cpu();
    Tensor cpu_result = ops::cpu::softmax(cpu_input, dim);
    return cpu_result.to(device_type);
}

// GELU activation (Gaussian Error Linear Unit)
Tensor gelu(const Tensor& input) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    DeviceType device_type = input.device().type();
    
    // For now, we implement GELU on CPU and then transfer back if needed
    Tensor cpu_input = input.device().type() == DeviceType::CPU ? input : input.cpu();
    
    // Get input data pointer
    const float* input_data = cpu_input.data<float>();
    
    // Create output tensor on CPU
    Tensor result(cpu_input.shape(), cpu_input.dtype(), DeviceType::CPU);
    float* result_data = result.data<float>();
    
    // Calculate GELU for each element
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    for (int64_t i = 0; i < cpu_input.size(); ++i) {
        float x = input_data[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        result_data[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    
    // Transfer back to original device if necessary
    return device_type == DeviceType::CPU ? result : result.to(device_type);
}

// LeakyReLU activation
Tensor leaky_relu(const Tensor& input, float negative_slope) {
    DeviceType device_type = input.device().type();
    
    if (device_type == DeviceType::CPU) {
        // Custom CPU implementation
        Tensor result(input.shape(), input.dtype(), device_type);
        const float* input_data = input.data<float>();
        float* result_data = result.data<float>();
        
        for (int64_t i = 0; i < input.size(); ++i) {
            float val = input_data[i];
            result_data[i] = val > 0 ? val : (negative_slope * val);
        }
        
        return result;
    }
    
    // Fallback: use CPU implementation and move back to original device
    Tensor cpu_input = input.cpu();
    Tensor cpu_result = leaky_relu(cpu_input, negative_slope); // Use the CPU implementation
    return cpu_result.to(device_type);
}

// ELU activation (Exponential Linear Unit)
Tensor elu(const Tensor& input, float alpha) {
    DeviceType device_type = input.device().type();
    
    if (device_type == DeviceType::CPU) {
        // Custom CPU implementation
        Tensor result(input.shape(), input.dtype(), device_type);
        const float* input_data = input.data<float>();
        float* result_data = result.data<float>();
        
        for (int64_t i = 0; i < input.size(); ++i) {
            float val = input_data[i];
            result_data[i] = val > 0 ? val : (alpha * (std::exp(val) - 1.0f));
        }
        
        return result;
    }
    
    // Fallback: use CPU implementation and move back to original device
    Tensor cpu_input = input.cpu();
    Tensor cpu_result = elu(cpu_input, alpha); // Use the CPU implementation
    return cpu_result.to(device_type);
}

// SiLU/Swish activation (x * sigmoid(x))
Tensor silu(const Tensor& input) {
    // First compute sigmoid
    Tensor sig = sigmoid(input);
    
    // Then multiply with input
    return input * sig;
}

} // namespace nn
} // namespace neuronet
