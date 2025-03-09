#pragma once

#include <neuronet/core/tensor.h>

namespace neuronet {
namespace nn {

// Activation functions
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor softmax(const Tensor& input, int dim = -1);
Tensor gelu(const Tensor& input);

} // namespace nn
} // namespace neuronet
