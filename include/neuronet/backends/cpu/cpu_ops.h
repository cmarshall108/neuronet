#pragma once

#include <neuronet/core/tensor.h>
#include <neuronet/backends/cpu/cpu_backend.h>

namespace neuronet {
namespace ops {
namespace cpu {

// Basic operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);

// Matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& input, int dim0, int dim1);

// Activation functions
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor softmax(const Tensor& input, int dim);

// Reduction operations
Tensor sum(const Tensor& input, int dim, bool keepdim);
Tensor mean(const Tensor& input, int dim, bool keepdim);

// Shape operations
Tensor reshape(const Tensor& input, const std::vector<int64_t>& shape);
Tensor view(const Tensor& input, const std::vector<int64_t>& shape);
Tensor flatten(const Tensor& input);

} // namespace cpu
} // namespace ops
} // namespace neuronet
