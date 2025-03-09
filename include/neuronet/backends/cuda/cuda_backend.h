#pragma once

#include <cublas_v2.h>
#include <string>

namespace neuronet {
namespace cuda {

// Initialize CUDA backend
bool initialize();

// Cleanup CUDA resources
void cleanup();

// Get cuBLAS handle
cublasHandle_t get_cublas_handle();

// Helper function to determine if the GPU is Tesla K80
bool isTeslaK80();

// Get CUDA capabilities
void getDeviceCapabilities(int* major, int* minor);

// Get device name
std::string getDeviceName();

// Get total device memory in bytes
size_t getTotalMemory();

} // namespace cuda
} // namespace neuronet
