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

// Add more operations as needed, following the same pattern...

} // namespace ops
} // namespace neuronet
