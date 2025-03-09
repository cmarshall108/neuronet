#include <neuronet/backends/cuda/cuda_ops.h>
#include <neuronet/backends/cuda/cuda_backend.h>
#include <neuronet/core/tensor.h>
#include <neuronet/utils/logging.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace neuronet {
namespace ops {
namespace cuda {

// Define CUDA kernels for basic operations
__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void subtract_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void multiply_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void divide_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void relu_kernel(const float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = a[idx] > 0 ? a[idx] : 0;
    }
}

__global__ void sigmoid_kernel(const float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = 1.0f / (1.0f + expf(-a[idx]));
    }
}

// Helper function to calculate CUDA grid dimensions
void get_grid_block_dims(int64_t size, dim3& grid_dim, dim3& block_dim) {
    block_dim.x = 256;
    block_dim.y = 1;
    block_dim.z = 1;

    grid_dim.x = (size + block_dim.x - 1) / block_dim.x;
    grid_dim.y = 1;
    grid_dim.z = 1;
}

Tensor add(const Tensor& a, const Tensor& b) {
    // Check if shapes match
    if (a.shape() != b.shape()) {
        log_error("Tensor shapes must match for addition");
        return Tensor();
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype(), DeviceType::CUDA);
    
    // Get raw pointers
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* result_data = result.data<float>();
    
    // Calculate dimensions
    int64_t size = a.size();
    dim3 grid_dim, block_dim;
    get_grid_block_dims(size, grid_dim, block_dim);
    
    // Launch kernel
    add_kernel<<<grid_dim, block_dim>>>(a_data, b_data, result_data, size);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_error("CUDA add kernel failed: {}", cudaGetErrorString(err));
    }
    
    return result;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    // Check if shapes match
    if (a.shape() != b.shape()) {
        log_error("Tensor shapes must match for subtraction");
        return Tensor();
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype(), DeviceType::CUDA);
    
    // Get raw pointers
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* result_data = result.data<float>();
    
    // Calculate dimensions
    int64_t size = a.size();
    dim3 grid_dim, block_dim;
    get_grid_block_dims(size, grid_dim, block_dim);
    
    // Launch kernel
    subtract_kernel<<<grid_dim, block_dim>>>(a_data, b_data, result_data, size);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_error("CUDA subtract kernel failed: {}", cudaGetErrorString(err));
    }
    
    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    // Check if shapes match
    if (a.shape() != b.shape()) {
        log_error("Tensor shapes must match for multiplication");
        return Tensor();
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype(), DeviceType::CUDA);
    
    // Get raw pointers
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* result_data = result.data<float>();
    
    // Calculate dimensions
    int64_t size = a.size();
    dim3 grid_dim, block_dim;
    get_grid_block_dims(size, grid_dim, block_dim);
    
    // Launch kernel
    multiply_kernel<<<grid_dim, block_dim>>>(a_data, b_data, result_data, size);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_error("CUDA multiply kernel failed: {}", cudaGetErrorString(err));
    }
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Check dimensions
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    
    if (a_shape.size() != 2 || b_shape.size() != 2 || a_shape[1] != b_shape[0]) {
        log_error("Invalid shapes for matrix multiplication");
        return Tensor();
    }
    
    // Create output tensor
    std::vector<int64_t> result_shape = {a_shape[0], b_shape[1]};
    Tensor result(result_shape, a.dtype(), DeviceType::CUDA);
    
    // Get cuBLAS handle
    cublasHandle_t handle = neuronet::cuda::get_cublas_handle();
    
    // Get raw pointers
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* result_data = result.data<float>();
    
    // Matrix dimensions
    int m = a_shape[0];
    int n = b_shape[1];
    int k = a_shape[1];
    
    // SGEMM parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform matrix multiplication using cuBLAS
    // Note: cuBLAS uses column-major format, so we compute B^T * A^T = (A * B)^T
    cublasStatus_t status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        b_data, n,
        a_data, k,
        &beta,
        result_data, n
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        log_error("cuBLAS matrix multiplication failed with status {}", (int)status);
    }
    
    return result;
}

Tensor relu(const Tensor& input) {
    // Create output tensor
    Tensor result(input.shape(), input.dtype(), DeviceType::CUDA);
    
    // Get raw pointers
    const float* input_data = input.data<float>();
    float* result_data = result.data<float>();
    
    // Calculate dimensions
    int64_t size = input.size();
    dim3 grid_dim, block_dim;
    get_grid_block_dims(size, grid_dim, block_dim);
    
    // Launch kernel
    relu_kernel<<<grid_dim, block_dim>>>(input_data, result_data, size);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_error("CUDA ReLU kernel failed: {}", cudaGetErrorString(err));
    }
    
    return result;
}

Tensor sigmoid(const Tensor& input) {
    // Create output tensor
    Tensor result(input.shape(), input.dtype(), DeviceType::CUDA);
    
    // Get raw pointers
    const float* input_data = input.data<float>();
    float* result_data = result.data<float>();
    
    // Calculate dimensions
    int64_t size = input.size();
    dim3 grid_dim, block_dim;
    get_grid_block_dims(size, grid_dim, block_dim);
    
    // Launch kernel
    sigmoid_kernel<<<grid_dim, block_dim>>>(input_data, result_data, size);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_error("CUDA sigmoid kernel failed: {}", cudaGetErrorString(err));
    }
    
    return result;
}

} // namespace cuda
} // namespace ops
} // namespace neuronet
