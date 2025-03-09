#include <neuronet/backends/metal/metal_backend.h>
#include <neuronet/utils/logging.h>

#if defined(__APPLE__) && defined(NEURONET_USE_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <sys/sysctl.h>

namespace neuronet {
namespace metal {

// Metal-specific logging utilities
void metal_log_info(const std::string& message) {
    log_info("[Metal] {}", message);
}

void metal_log_warn(const std::string& message) {
    log_warn("[Metal] {}", message);
}

void metal_log_error(const std::string& message) {
    log_error("[Metal] {}", message);
}

void metal_log_debug(const std::string& message) {
    log_debug("[Metal] {}", message);
}

// Global Metal device
static id<MTLDevice> device = nil;
static id<MTLCommandQueue> command_queue = nil;
static id<MTLLibrary> library = nil;

// Helper function to check if running on Apple Silicon
bool isAppleSilicon() {
    char buffer[256];
    size_t length = sizeof(buffer);
    int result = sysctlbyname("machdep.cpu.brand_string", &buffer, &length, NULL, 0);
    
    if (result == 0) {
        NSString* cpuString = [NSString stringWithUTF8String:buffer];
        return [cpuString containsString:@"Apple"];
    }
    return false;
}

// Shader code with optimizations for Apple Silicon
NSString* get_metal_shader_code() {
    bool apple_silicon = isAppleSilicon();
    
    // Basic shader functions - note proper string concatenation with @ symbols
    NSString* shaderCode = @"#include <metal_stdlib>\n"
                          "using namespace metal;\n"
                          "\n"
                          "kernel void add_kernel(\n"
                          "    device const float* a [[buffer(0)]],\n"
                          "    device const float* b [[buffer(1)]],\n"
                          "    device float* result [[buffer(2)]],\n"
                          "    uint id [[thread_position_in_grid]])\n"
                          "{\n"
                          "    result[id] = a[id] + b[id];\n"
                          "}\n"
                          "\n"
                          "kernel void mul_kernel(\n"
                          "    device const float* a [[buffer(0)]],\n"
                          "    device const float* b [[buffer(1)]],\n"
                          "    device float* result [[buffer(2)]],\n"
                          "    uint id [[thread_position_in_grid]])\n"
                          "{\n"
                          "    result[id] = a[id] * b[id];\n"
                          "}\n"
                          "\n"
                          "kernel void relu_kernel(\n"
                          "    device const float* input [[buffer(0)]],\n"
                          "    device float* result [[buffer(1)]],\n"
                          "    uint id [[thread_position_in_grid]])\n"
                          "{\n"
                          "    result[id] = max(input[id], 0.0f);\n"
                          "}\n";
    
    // Add optimized functions for Apple Silicon if available
    if (apple_silicon) {
        NSString* appleCode = @"\n"
                             "// Optimized for Apple Silicon\n"
                             "kernel void simd_add_kernel(\n"
                             "    device const float4* a [[buffer(0)]],\n"
                             "    device const float4* b [[buffer(1)]],\n"
                             "    device float4* result [[buffer(2)]],\n"
                             "    uint id [[thread_position_in_grid]])\n"
                             "{\n"
                             "    result[id] = a[id] + b[id];\n"
                             "}\n"
                             "\n"
                             "kernel void simd_relu_kernel(\n"
                             "    device const float4* input [[buffer(0)]],\n"
                             "    device float4* result [[buffer(1)]],\n"
                             "    uint id [[thread_position_in_grid]])\n"
                             "{\n"
                             "    result[id] = max(input[id], float4(0.0f));\n"
                             "}\n";
        
        shaderCode = [shaderCode stringByAppendingString:appleCode];
    }
    
    return shaderCode;
}

bool initialize() {
    @autoreleasepool {
        // Get default Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            metal_log_error("Failed to create Metal device");
            return false;
        }
        
        // Check device capabilities
        bool apple_silicon = isAppleSilicon();
        NSString* deviceName = [device name];
        
        // Update deprecated code - check for GPU family instead of feature set
        #if __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
        // macOS 13.0 and later
        if (![device supportsFamily:MTLGPUFamilyMac2]) {
            metal_log_warn("Device may not support all required features");
        }
        #else
        // For earlier macOS versions
        if (![device supportsFeatureSet:MTLFeatureSet_macOS_GPUFamily2_v1]) {
            metal_log_warn("Device may not support all required features");
        }
        #endif
        
        // Create command queue
        command_queue = [device newCommandQueue];
        if (!command_queue) {
            metal_log_error("Failed to create command queue");
            return false;
        }
        
        // Create Metal library with kernel functions
        NSError* error = nil;
        library = [device newLibraryWithSource:get_metal_shader_code() options:nil error:&error];
        if (!library) {
            std::string errorMsg = error ? [[error localizedDescription] UTF8String] : "unknown error";
            metal_log_error("Failed to create library: " + errorMsg);
            return false;
        }
        
        metal_log_info("Backend initialized successfully with device: " + std::string([deviceName UTF8String]));
        if (apple_silicon) {
            metal_log_info("Running on Apple Silicon - enabling optimized kernels");
        } else {
            metal_log_info("Running on Intel Mac");
        }
        
        return true;
    }
}

void cleanup() {
    @autoreleasepool {
        metal_log_info("Cleaning up resources");
        
        // Remove explicit releases - ARC will handle this
        command_queue = nil;
        library = nil;
        device = nil;
    }
}

// Fix return type to match declarations in header
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

// Get Metal GPU information
MetalDeviceInfo getMetalDeviceInfo() {
    MetalDeviceInfo info;
    
    @autoreleasepool {
        if (!device) {
            initialize();
        }
        
        if (device) {
            info.name = [[device name] UTF8String];
            info.isAppleSilicon = isAppleSilicon();
            info.isLowPower = [device isLowPower];
            info.registryID = [device registryID];
            info.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup.width;
            
            metal_log_debug("Device info: " + info.name + 
                          ", Apple Silicon: " + (info.isAppleSilicon ? "Yes" : "No") +
                          ", Low Power: " + (info.isLowPower ? "Yes" : "No"));
        }
    }
    
    return info;
}

} // namespace metal
} // namespace neuronet

#endif // defined(__APPLE__) && defined(NEURONET_USE_METAL)
