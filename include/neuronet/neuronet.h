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
 * @brief Get library version
 * 
 * @return const char* Version string
 */
inline const char* version() {
    return "0.1.0";
}

} // namespace neuronet
