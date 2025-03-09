#pragma once

#include <cublas_v2.h>

namespace neuronet {
namespace cuda {

// Initialize CUDA backend
bool initialize();

// Cleanup CUDA resources
void cleanup();

// Get cuBLAS handle
cublasHandle_t get_cublas_handle();

} // namespace cuda
} // namespace neuronet
