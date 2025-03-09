#pragma once

#include <neuronet/backends/cpu/cpu_ops.h>

// Include CUDA ops if enabled
#ifdef NEURONET_USE_CUDA
#include <neuronet/backends/cuda/cuda_ops.h>
#endif

// Include Metal ops if enabled
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
#include <neuronet/backends/metal/metal_ops.h>
#endif

namespace neuronet {
namespace ops {

bool initialize_backends();
void cleanup_backends();

// Device-agnostic operations that dispatch to the appropriate backend

// Addition
inline Tensor add(const Tensor& a, const Tensor& b) {
    DeviceType device = a.device().type();
    
    if (device == DeviceType::CPU) {
        return cpu::add(a, b);
    }
#ifdef NEURONET_USE_CUDA
    else if (device == DeviceType::CUDA) {
        return cuda::add(a, b);
    }
#endif
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    else if (device == DeviceType::Metal) {
        return metal::add(a, b);
    }
#endif
    
    // Fall back to CPU if device not supported
    return cpu::add(a.cpu(), b.cpu()).to(device);
}

// Matrix multiplication
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    DeviceType device = a.device().type();
    
    if (device == DeviceType::CPU) {
        return cpu::matmul(a, b);
    }
#ifdef NEURONET_USE_CUDA
    else if (device == DeviceType::CUDA) {
        return cuda::matmul(a, b);
    }
#endif
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    else if (device == DeviceType::Metal) {
        return metal::matmul(a, b);
    }
#endif
    
    // Fall back to CPU if device not supported
    return cpu::matmul(a.cpu(), b.cpu()).to(device);
}

// ReLU activation
inline Tensor relu(const Tensor& input) {
    DeviceType device = input.device().type();
    
    if (device == DeviceType::CPU) {
        return cpu::relu(input);
    }
#ifdef NEURONET_USE_CUDA
    else if (device == DeviceType::CUDA) {
        return cuda::relu(input);
    }
#endif
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    else if (device == DeviceType::Metal) {
        return metal::relu(input);
    }
#endif
    
    // Fall back to CPU if device not supported
    return cpu::relu(input.cpu()).to(device);
}

// Add maximum operation
inline Tensor maximum(const Tensor& a, const Tensor& b) {
    DeviceType device = a.device().type();
    
    if (device == DeviceType::CPU) {
        return cpu::maximum(a, b);
    }
#ifdef NEURONET_USE_CUDA
    else if (device == DeviceType::CUDA) {
        return cuda::maximum(a, b);
    }
#endif
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    else if (device == DeviceType::Metal) {
        // Metal doesn't have maximum implemented, use CPU version and transfer back
        Tensor cpu_a = a.cpu();
        Tensor cpu_b = b.cpu();
        Tensor cpu_result = cpu::maximum(cpu_a, cpu_b);
        return cpu_result.to(device);
    }
#endif
    
    // Fall back to CPU if device not supported
    return cpu::maximum(a.cpu(), b.cpu()).to(device);
}

// Add log operation
inline Tensor log(const Tensor& input) {
    DeviceType device = input.device().type();
    
    if (device == DeviceType::CPU) {
        return cpu::log(input);
    }
#ifdef NEURONET_USE_CUDA
    else if (device == DeviceType::CUDA) {
        return cuda::log(input);
    }
#endif
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    else if (device == DeviceType::Metal) {
        // Metal doesn't have log implemented, use CPU version and transfer back
        Tensor cpu_input = input.cpu();
        Tensor cpu_result = cpu::log(cpu_input);
        return cpu_result.to(device);
    }
#endif
    
    // Fall back to CPU if device not supported
    return cpu::log(input.cpu()).to(device);
}

// Add tensor-scalar multiplication
inline Tensor mul_scalar(const Tensor& tensor, float scalar) {
    DeviceType device = tensor.device().type();
    
    if (device == DeviceType::CPU) {
        return cpu::mul_scalar(tensor, scalar);
    }
#ifdef NEURONET_USE_CUDA
    else if (device == DeviceType::CUDA) {
        return cuda::mul_scalar(tensor, scalar);
    }
#endif
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    else if (device == DeviceType::Metal) {
        return metal::mul_scalar(tensor, scalar);
    }
#endif
    
    // Fall back to CPU if device not supported
    return cpu::mul_scalar(tensor.cpu(), scalar).to(device);
}

// Add negate operation
inline Tensor negate(const Tensor& input) {
    return mul_scalar(input, -1.0f);
}

} // namespace ops
} // namespace neuronet
