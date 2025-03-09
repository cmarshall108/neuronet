#include <neuronet/backends/cuda/cuda_backend.h>
#include <neuronet/core/tensor.h>
#include <neuronet/utils/logging.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace neuronet {
namespace cuda {

// Global cublas handle
static cublasHandle_t cublas_handle = nullptr;

bool initialize() {
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        log_error("Failed to set CUDA device: {}", cudaGetErrorString(cuda_status));
        return false;
    }

    cublasStatus_t cublas_status = cublasCreate(&cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        log_error("Failed to create cuBLAS handle");
        return false;
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

} // namespace cuda
} // namespace neuronet
