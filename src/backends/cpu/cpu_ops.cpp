#include <neuronet/backends/cpu/cpu_ops.h>
#include <neuronet/core/tensor.h>
#include <neuronet/utils/logging.h>
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>
#include <functional>

namespace neuronet {
namespace ops {
namespace cpu {

// Helper function for parallel execution
void parallel_for(int64_t start, int64_t end, const std::function<void(int64_t, int64_t)>& func) {
    int num_threads = neuronet::cpu::get_num_threads();
    
    if (num_threads <= 1 || end - start <= 1000) {
        // Sequential execution for small workloads or single thread
        func(start, end);
        return;
    }
    
    // Calculate work per thread
    int64_t items_per_thread = (end - start) / num_threads;
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int64_t thread_start = start + t * items_per_thread;
        int64_t thread_end = (t == num_threads - 1) ? end : thread_start + items_per_thread;
        
        threads.emplace_back([&func, thread_start, thread_end]() {
            func(thread_start, thread_end);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

Tensor add(const Tensor& a, const Tensor& b) {
    // Check that shapes are compatible
    if (a.shape() != b.shape()) {
        log_error("Tensor shapes must match for addition");
        return Tensor();
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype(), DeviceType::CPU);
    
    // Perform element-wise addition
    int64_t size = a.size();
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* result_data = result.data<float>();
    
    parallel_for(0, size, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
    });
    
    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    // Check that shapes are compatible
    if (a.shape() != b.shape()) {
        log_error("Tensor shapes must match for element-wise multiplication");
        return Tensor();
    }
    
    // Create output tensor
    Tensor result(a.shape(), a.dtype(), DeviceType::CPU);
    
    // Perform element-wise multiplication
    int64_t size = a.size();
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* result_data = result.data<float>();
    
    parallel_for(0, size, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            result_data[i] = a_data[i] * b_data[i];
        }
    });
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Check dimensions for matrix multiplication
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    
    if (a_shape.size() != 2 || b_shape.size() != 2 || a_shape[1] != b_shape[0]) {
        log_error("Invalid shapes for matrix multiplication: [{} x {}] * [{} x {}]",
                 std::to_string(a_shape[0]), std::to_string(a_shape[1]),
                 std::to_string(b_shape[0]), std::to_string(b_shape[1]));
        return Tensor();
    }
    
    // Output shape: [a.rows, b.cols]
    std::vector<int64_t> result_shape = {a_shape[0], b_shape[1]};
    Tensor result(result_shape, a.dtype(), DeviceType::CPU);
    
    // Get raw data pointers
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* result_data = result.data<float>();
    
    int64_t m = a_shape[0];
    int64_t n = b_shape[1];
    int64_t k = a_shape[1];
    
    // Perform matrix multiplication
    parallel_for(0, m, [&](int64_t start_i, int64_t end_i) {
        for (int64_t i = start_i; i < end_i; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }
    });
    
    return result;
}

Tensor relu(const Tensor& input) {
    Tensor result(input.shape(), input.dtype(), DeviceType::CPU);
    
    int64_t size = input.size();
    const float* input_data = input.data<float>();
    float* result_data = result.data<float>();
    
    parallel_for(0, size, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            result_data[i] = std::max(0.0f, input_data[i]);
        }
    });
    
    return result;
}

Tensor sigmoid(const Tensor& input) {
    Tensor result(input.shape(), input.dtype(), DeviceType::CPU);
    
    int64_t size = input.size();
    const float* input_data = input.data<float>();
    float* result_data = result.data<float>();
    
    parallel_for(0, size, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            result_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
        }
    });
    
    return result;
}

Tensor softmax(const Tensor& input, int dim) {
    // For simplicity, implement only for 2D tensors with dim=1 (softmax over rows)
    const auto& shape = input.shape();
    if (shape.size() != 2 || dim != 1) {
        log_error("CPU softmax currently only supported for 2D tensors with dim=1");
        return input;
    }
    
    Tensor result(shape, input.dtype(), DeviceType::CPU);
    
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    
    const float* input_data = input.data<float>();
    float* result_data = result.data<float>();
    
    parallel_for(0, rows, [&](int64_t start_row, int64_t end_row) {
        for (int64_t i = start_row; i < end_row; ++i) {
            // Find max value for numerical stability
            float max_val = input_data[i * cols];
            for (int64_t j = 1; j < cols; ++j) {
                max_val = std::max(max_val, input_data[i * cols + j]);
            }
            
            // Compute exp(x - max) and sum
            float sum = 0.0f;
            for (int64_t j = 0; j < cols; ++j) {
                float exp_val = std::exp(input_data[i * cols + j] - max_val);
                result_data[i * cols + j] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (int64_t j = 0; j < cols; ++j) {
                result_data[i * cols + j] /= sum;
            }
        }
    });
    
    return result;
}

} // namespace cpu
} // namespace ops
} // namespace neuronet
