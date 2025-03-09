#include <neuronet/backends/cuda/cuda_backend.h>
#include <neuronet/core/tensor.h>
#include <neuronet/utils/logging.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace neuronet {
namespace cuda {

// Global cublas handle
static cublasHandle_t cublas_handle = nullptr;

// Helper function to determine if the GPU is Tesla K80
bool isTeslaK80() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        // Tesla K80 has compute capability 3.7
        if (prop.major == 3 && prop.minor == 7 && 
            std::string(prop.name).find("Tesla K80") != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool initialize() {
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        log_error("Failed to set CUDA device: {}", cudaGetErrorString(cuda_status));
        return false;
    }

    // Create cuBLAS handle
    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        log_error("Failed to create cuBLAS handle");
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Log device information
    log_info("CUDA device: {} (compute capability {}.{})", prop.name, prop.major, prop.minor);
    log_info("Total global memory: {} MB", prop.totalGlobalMem / (1024 * 1024));
    log_info("Multiprocessors: {}", prop.multiProcessorCount);
    
    // Apply Tesla K80 specific optimizations if detected
    if (isTeslaK80()) {
        log_info("Tesla K80 detected, applying specific optimizations");
        
        // For K80, prefer L1 cache over shared memory
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        
        // Set cuBLAS math mode to prefer throughput
        cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
    } else {
        log_info("Generic CUDA device, using standard configuration");
    }

    log_info("CUDA backend initialized successfully");
    return true;
}

void cleanup() {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
    cudaDeviceReset();
}

// CUDA kernel for element-wise addition
__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for element-wise multiplication 
__global__ void mul_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

cublasHandle_t get_cublas_handle() {
    if (!cublas_handle) {
        initialize();
    }
    return cublas_handle;
}

// Get CUDA capabilities
void getDeviceCapabilities(int* major, int* minor) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    *major = prop.major;
    *minor = prop.minor;
}

// Get device name
std::string getDeviceName() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return std::string(prop.name);
}

// Get total device memory in bytes
size_t getTotalMemory() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.totalGlobalMem;
}

} // namespace cuda
} // namespace neuronet
