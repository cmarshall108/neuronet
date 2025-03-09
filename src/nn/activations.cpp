#include <neuronet/nn/activations.h>
#include <neuronet/core/ops.h>

namespace neuronet {
namespace nn {

// ReLU activation
Tensor relu(const Tensor& input) {
    return ops::relu(input);
}

// Sigmoid activation
Tensor sigmoid(const Tensor& input) {
    return input.sigmoid();
}

// Tanh activation
Tensor tanh(const Tensor& input) {
    return input.tanh();
}

// Softmax activation
Tensor softmax(const Tensor& input, int dim) {
    return input.softmax(dim);
}

// GELU activation (Gaussian Error Linear Unit)
Tensor gelu(const Tensor& input) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    Tensor x_cubed = input * input * input;
    Tensor inner = sqrt_2_over_pi * (input + coeff * x_cubed);
    return 0.5 * input * (Tensor({1}, 1.0f, input.dtype(), input.device().type()) + inner.tanh());
}

} // namespace nn
} // namespace neuronet
