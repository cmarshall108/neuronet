#pragma once

// Core functionality
#include <neuronet/core/tensor.h>
#include <neuronet/core/device.h>
#include <neuronet/core/memory.h>
#include <neuronet/core/ops.h>

// Neural network primitives
#include <neuronet/nn/layers.h>
#include <neuronet/nn/activations.h>
#include <neuronet/nn/loss.h>
#include <neuronet/nn/optimizer.h>

// Models
#include <neuronet/models/model.h>
#include <neuronet/models/huggingface.h>

// Utilities
#include <neuronet/utils/logging.h>
#include <neuronet/utils/json.h>

#include <string>

/**
 * @namespace neuronet
 * @brief Main namespace for the NeuroNet library
 * 
 * NeuroNet is a PyTorch alternative library that provides efficient tensor operations
 * and neural network primitives for CPU, CUDA (Nvidia Tesla K80), and Metal (macOS).
 */

namespace neuronet {

/**
 * @brief Initialize the NeuroNet library
 * 
 * This function initializes all backend systems (CPU, CUDA if available, Metal if available).
 * It should be called before using any NeuroNet functionality.
 * 
 * @return bool True if initialization was successful
 */
inline bool initialize() {
    return ops::initialize_backends();
}

/**
 * @brief Clean up and release resources used by the NeuroNet library
 * 
 * This function should be called before program exit to ensure proper
 * cleanup of all allocated resources.
 */
inline void cleanup() {
    ops::cleanup_backends();
}

/**
 * @brief Get library version string
 * 
 * @return std::string Version string
 */
inline std::string version() {
#ifdef NEURONET_VERSION
    return NEURONET_VERSION;
#else
    return "0.1.0"; // Fallback if not defined during compilation
#endif
}

/**
 * @brief Check if Tesla K80 GPU was detected during initialization
 * 
 * @return bool True if Tesla K80 is available and CUDA support is enabled
 */
inline bool isTeslaK80Available() {
#if defined(NEURONET_USE_CUDA) && defined(NEURONET_TESLA_K80_AVAILABLE)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if Apple Silicon processor was detected
 * 
 * @return bool True if running on Apple Silicon (M1, M2, etc.)
 */
inline bool isAppleSiliconAvailable() {
#if defined(__APPLE__) && defined(NEURONET_APPLE_SILICON)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Returns information about the library and available hardware
 * 
 * @return std::string Detailed information about the library build and hardware support
 */
inline std::string libraryInfo() {
    std::string info = "NeuroNet v" + version() + "\n";
    info += "Supported backends:\n";
    info += "- CPU: YES\n";

#ifdef NEURONET_USE_CUDA
    info += "- CUDA: YES";
    info += isTeslaK80Available() ? " (Tesla K80 detected)\n" : "\n";
#else
    info += "- CUDA: NO\n";
#endif

#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
    info += "- Metal: YES";
    info += isAppleSiliconAvailable() ? " (Apple Silicon detected)\n" : "\n";
#else
    info += "- Metal: NO\n";
#endif

    return info;
}

} // namespace neuronet
