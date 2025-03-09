#include <neuronet/core/device.h>

#ifdef NEURONET_USE_CUDA
#include <cuda_runtime.h>
#endif

// Fix the Metal/Foundation imports
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#else
// For C++ code, we just need the forward declarations
extern "C" {
    // Forward declarations of required Objective-C types
    typedef struct objc_object *id;
    bool MTLCreateSystemDefaultDevice(void);
}
#endif
#endif

namespace neuronet {

Device::Device(DeviceType type, int index) 
    : type_(type), index_(index) {}

DeviceType Device::type() const {
    return type_;
}

int Device::index() const {
    return index_;
}

std::string Device::toString() const {
    switch (type_) {
        case DeviceType::CPU:
            return "cpu";
        case DeviceType::CUDA:
            return "cuda:" + std::to_string(index_);
        case DeviceType::Metal:
            return "metal:" + std::to_string(index_);
        default:
            return "unknown";
    }
}

Device Device::cpu() {
    return Device(DeviceType::CPU);
}

Device Device::cuda(int index) {
    return Device(DeviceType::CUDA, index);
}

Device Device::metal(int index) {
    return Device(DeviceType::Metal, index);
}

bool Device::isCudaAvailable() {
#ifdef NEURONET_USE_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

bool Device::isMetalAvailable() {
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    // Simple check without using Objective-C directly in C++ file
    return MTLCreateSystemDefaultDevice();
#else
    return false;
#endif
}

} // namespace neuronet
