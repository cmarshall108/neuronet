#pragma once

#include <neuronet/core/tensor.h>

namespace neuronet {
namespace ops {
namespace metal {

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

} // namespace metal
} // namespace ops
} // namespace neuronet
