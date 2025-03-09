#include <neuronet/backends/metal/metal_wrapper.h>
#include <neuronet/backends/metal/metal_backend.h>
#include <neuronet/utils/logging.h>

#if defined(__APPLE__) && defined(NEURONET_USE_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace neuronet {
namespace metal {

// Use fully qualified names for metal logging functions
using ::neuronet::metal::metal_log_debug;
using ::neuronet::metal::metal_log_error;
using ::neuronet::metal::metal_log_warn;
using ::neuronet::metal::metal_log_info;

void* metal_allocate_buffer(size_t size) {
    @autoreleasepool {
        // Get the Metal device without bridging cast since the function already returns the right type
        id<MTLDevice> metal_device = get_metal_device();
        id<MTLBuffer> buffer = [metal_device newBufferWithLength:size 
                                                options:MTLResourceStorageModeShared];
        if (!buffer) {
            metal_log_error("Failed to allocate buffer of size " + std::to_string(size));
            return nullptr;
        }
        
        metal_log_debug("Allocated buffer of size " + std::to_string(size));
        
        // Store the buffer pointer. This needs proper cleanup.
        return (__bridge_retained void*)buffer;
    }
}

void metal_free_buffer(void* buffer) {
    if (!buffer) return;
    
    @autoreleasepool {
        // Release the Metal buffer
        metal_log_debug("Freeing Metal buffer");
        CFBridgingRelease(buffer);
    }
}

void metal_copy_to_device(void* dst, const void* src, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)dst;
        void* buffer_ptr = [metal_buffer contents];
        std::memcpy(buffer_ptr, src, size);
        metal_log_debug("Copied " + std::to_string(size) + " bytes to device");
    }
}

void metal_copy_from_device(void* dst, const void* src, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> metal_buffer = (__bridge id<MTLBuffer>)src;
        void* buffer_ptr = [metal_buffer contents];
        std::memcpy(dst, buffer_ptr, size);
        metal_log_debug("Copied " + std::to_string(size) + " bytes from device");
    }
}

} // namespace metal
} // namespace neuronet

#endif // defined(__APPLE__) && defined(NEURONET_USE_METAL)
