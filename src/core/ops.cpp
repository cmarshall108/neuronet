#include <neuronet/core/ops.h>
#include <neuronet/utils/logging.h>

namespace neuronet {
namespace ops {

// This file is primarily for implementing operations that are 
// shared across backends or need special device handling.
// Most device-specific implementations are in their respective backend files.

// Initialize all backends
bool initialize_backends() {
    // Initialize CPU backend (always available)
    bool cpu_success = cpu::initialize();
    if (!cpu_success) {
        log_error("Failed to initialize CPU backend");
        return false;
    }
    
#ifdef NEURONET_USE_CUDA
    // Initialize CUDA backend if available
    if (Device::isCudaAvailable()) {
        bool cuda_success = cuda::initialize();
        if (!cuda_success) {
            log_warn("Failed to initialize CUDA backend");
        } else {
            log_info("CUDA backend initialized successfully");
        }
    } else {
        log_info("CUDA not available on this system");
    }
#endif

#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    // Initialize Metal backend if available
    if (Device::isMetalAvailable()) {
        bool metal_success = metal::initialize();
        if (!metal_success) {
            log_warn("Failed to initialize Metal backend");
        } else {
            log_info("Metal backend initialized successfully");
        }
    } else {
        log_info("Metal not available on this system");
    }
#endif

    return true;
}

// Clean up all backends
void cleanup_backends() {
    // Clean up CPU backend
    cpu::cleanup();
    
#ifdef NEURONET_USE_CUDA
    // Clean up CUDA backend
    cuda::cleanup();
#endif

#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    // Clean up Metal backend
    metal::cleanup();
#endif

    log_info("All backends cleaned up");
}

} // namespace ops
} // namespace neuronet
