#include <neuronet/backends/metal/metal_backend.h>
#include <neuronet/utils/logging.h>

#if defined(__APPLE__) && defined(NEURONET_USE_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace neuronet {
namespace metal {

// Global Metal device
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> command_queue = nil;
static id<MTLLibrary> library = nil;

NSString* get_metal_shader_code() {
    return @"
    #include <metal_stdlib>
    using namespace metal;

    kernel void add_kernel(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* result [[buffer(2)]],
        uint id [[thread_position_in_grid]])
    {
        result[id] = a[id] + b[id];
    }

    kernel void mul_kernel(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* result [[buffer(2)]],
        uint id [[thread_position_in_grid]])
    {
        result[id] = a[id] * b[id];
    }

    kernel void relu_kernel(
        device const float* input [[buffer(0)]],
        device float* result [[buffer(1)]],
        uint id [[thread_position_in_grid]])
    {
        result[id] = max(input[id], 0.0f);
    }
    ";
}

bool initialize() {
    @autoreleasepool {
        // Get default Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            log_error("Failed to create Metal device");
            return false;
        }
        
        // Check if device supports unified memory
        if (![device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1]) {
            log_warn("Metal device may not support all required features");
        }
        
        // Create command queue
        command_queue = [device newCommandQueue];
        if (!command_queue) {
            log_error("Failed to create Metal command queue");
            return false;
        }
        
        // Create Metal library with kernel functions
        NSError* error = nil;
        library = [device newLibraryWithSource:get_metal_shader_code() options:nil error:&error];
        if (!library) {
            log_error("Failed to create Metal library: {}", error ? [[error localizedDescription] UTF8String] : "unknown error");
            return false;
        }
        
        log_info("Metal backend initialized successfully with device: {}", [[device name] UTF8String]);
        return true;
    }
}

void cleanup() {
    @autoreleasepool {
        [command_queue release];
        command_queue = nil;
        
        [library release];
        library = nil;
        
        [device release];
        device = nil;
    }
}

id<MTLDevice> get_metal_device() {
    if (!device) {
        initialize();
    }
    return device;
}

id<MTLCommandQueue> get_command_queue() {
    if (!command_queue) {
        initialize();
    }
    return command_queue;
}

id<MTLLibrary> get_metal_library() {
    if (!library) {
        initialize();
    }
    return library;
}

} // namespace metal
} // namespace neuronet

#endif // defined(__APPLE__) && defined(NEURONET_USE_METAL)
