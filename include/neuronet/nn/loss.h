#pragma once

#include <neuronet/core/tensor.h>

namespace neuronet {
namespace nn {

// Mean Squared Error Loss
Tensor mse_loss(const Tensor& pred, const Tensor& target);

// Cross Entropy Loss
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets, int dim = 1);

} // namespace nn
} // namespace neuronet
