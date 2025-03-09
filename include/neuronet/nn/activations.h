#pragma once

#include <neuronet/core/tensor.h>

namespace neuronet {
namespace nn {

// Basic activation functions
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor softmax(const Tensor& input, int dim = -1);
Tensor gelu(const Tensor& input);

// Additional activation functions
Tensor leaky_relu(const Tensor& input, float negative_slope = 0.01f);
Tensor elu(const Tensor& input, float alpha = 1.0f);
Tensor silu(const Tensor& input); // Also known as Swish

} // namespace nn
} // namespace neuronet
