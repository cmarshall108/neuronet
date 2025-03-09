#include <neuronet/core/memory.h>
#include <neuronet/utils/logging.h>
#include <cstdlib>
#include <cstring>

#ifdef NEURONET_USE_CUDA
#include <cuda_runtime.h>
#endif

#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
#include <neuronet/backends/metal/metal_wrapper.h>
#endif

namespace neuronet {
namespace memory {

void* allocate(const Device& device, size_t size) {
    void* ptr = nullptr;
    
    switch (device.type()) {
        case DeviceType::CPU: {
            ptr = std::malloc(size);
            if (!ptr) {
                log_error("Failed to allocate CPU memory of size {}", std::to_string(size));
            }
            break;
        }
        
        case DeviceType::CUDA: {
#ifdef NEURONET_USE_CUDA
            cudaError_t status = cudaMalloc(&ptr, size);
            if (status != cudaSuccess) {
                log_error("Failed to allocate CUDA memory of size {}: {}", 
                          std::to_string(size), cudaGetErrorString(status));
                return nullptr;
            }
#else
            log_error("CUDA support not enabled");
#endif
            break;
        }
        
        case DeviceType::Metal: {
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
            ptr = metal::metal_allocate_buffer(size);
#else
            log_error("Metal support not enabled");
#endif
            break;
        }
    }
    
    return ptr;
}

void free(const Device& device, void* ptr) {
    if (!ptr) return;
    
    switch (device.type()) {
        case DeviceType::CPU: {
            std::free(ptr);
            break;
        }
        
        case DeviceType::CUDA: {
#ifdef NEURONET_USE_CUDA
            cudaError_t status = cudaFree(ptr);
            if (status != cudaSuccess) {
                log_error("Failed to free CUDA memory: {}", cudaGetErrorString(status));
            }
#endif
            break;
        }
        
        case DeviceType::Metal: {
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
            metal::metal_free_buffer(ptr);
#endif
            break;
        }
    }
}

void copy_to_device(const Device& device, void* dst, const void* src, size_t size) {
    switch (device.type()) {
        case DeviceType::CPU: {
            std::memcpy(dst, src, size);
            break;
        }
        
        case DeviceType::CUDA: {
#ifdef NEURONET_USE_CUDA
            cudaError_t status = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                log_error("Failed to copy to CUDA device: {}", cudaGetErrorString(status));
            }
#endif
            break;
        }
        
        case DeviceType::Metal: {
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
            metal::metal_copy_to_device(dst, src, size);
#endif
            break;
        }
    }
}

void copy_from_device(const Device& device, void* dst, const void* src, size_t size) {
    switch (device.type()) {
        case DeviceType::CPU: {
            std::memcpy(dst, src, size);
            break;
        }
        
        case DeviceType::CUDA: {
#ifdef NEURONET_USE_CUDA
            cudaError_t status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
            if (status != cudaSuccess) {
                log_error("Failed to copy from CUDA device: {}", cudaGetErrorString(status));
            }
#endif
            break;
        }
        
        case DeviceType::Metal: {
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
            metal::metal_copy_from_device(dst, src, size);
#endif
            break;
        }
    }
}

void copy_between_devices(const Device& src_device, const Device& dst_device,
                         const void* src, void* dst, size_t size) {
    // If same device type, use direct copy if possible
    if (src_device.type() == dst_device.type()) {
        switch (src_device.type()) {
            case DeviceType::CPU:
                std::memcpy(dst, src, size);
                return;
                
            case DeviceType::CUDA:
#ifdef NEURONET_USE_CUDA
                cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
                return;
#endif
                break;
                
            case DeviceType::Metal:
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
                // Direct metal-to-metal copy
                metal::metal_copy_to_device(dst, src, size);
                return;
#endif
                break;
        }
    }
    
    // Different device types: use CPU as intermediate
    void* cpu_buffer = std::malloc(size);
    if (!cpu_buffer) {
        log_error("Failed to allocate CPU buffer for device transfer");
        return;
    }
    
    copy_from_device(src_device, cpu_buffer, src, size);
    copy_to_device(dst_device, dst, cpu_buffer, size);
    
    std::free(cpu_buffer);
}

} // namespace memory
} // namespace neuronet
