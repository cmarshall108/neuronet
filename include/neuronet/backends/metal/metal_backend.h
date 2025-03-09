#pragma once

#include <string>

// Forward declarations for Objective-C types when compiled as C++
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLLibrary;
#endif

namespace neuronet {
namespace metal {

// Metal device information structure
struct MetalDeviceInfo {
    std::string name;
    bool isAppleSilicon = false;
    bool isLowPower = false;
    unsigned long long registryID = 0;
    int maxThreadsPerThreadgroup = 0;
};

// Metal-specific logging utilities
void metal_log_info(const std::string& message);
void metal_log_warn(const std::string& message);
void metal_log_error(const std::string& message);
void metal_log_debug(const std::string& message);

// Initialize Metal backend
bool initialize();

// Cleanup Metal resources
void cleanup();

// Helper function to check if running on Apple Silicon
bool isAppleSilicon();

// Get Metal device information
MetalDeviceInfo getMetalDeviceInfo();

// Get Metal device, command queue, and library (defined in .mm file)
// When compiling with Objective-C, use proper protocol type declarations
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
#ifdef __OBJC__
// Return actual types when included from Objective-C/C++ file
id<MTLDevice> get_metal_device();
id<MTLCommandQueue> get_command_queue();
id<MTLLibrary> get_metal_library();
#else
// Opaque pointer return for regular C++ code
void* get_metal_device();
void* get_command_queue();
void* get_metal_library();
#endif
#endif

} // namespace metal
} // namespace neuronet
