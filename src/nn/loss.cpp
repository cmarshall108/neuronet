#include <neuronet/nn/loss.h>
#include <neuronet/core/ops.h>
#include <neuronet/utils/logging.h>
#include <cmath>

namespace neuronet {
namespace nn {

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        log_error("MSE Loss requires tensors with same shape");
        return Tensor();
    }
    
    // Compute (pred - target)^2
    Tensor diff = pred - target;
    Tensor squared_diff = diff * diff;
    
    // Compute mean
    return squared_diff.mean();
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets, int dim) {
    // This is a simplified implementation for 2D tensors
    // logits shape: [batch_size, num_classes]
    // targets shape: [batch_size, num_classes] (one-hot) or [batch_size] (class indices)
    
    const auto& logits_shape = logits.shape();
    
    if (logits_shape.size() != 2) {
        log_error("Cross entropy loss expects 2D logits tensor [batch_size, num_classes]");
        return Tensor();
    }
    
    // Apply softmax to logits
    Tensor probs = logits.softmax(dim);
    
    // Create a small tensor for numerical stability
    float epsilon = 1e-7f;
    // Use the new scalar constructor
    Tensor eps_tensor({1}, epsilon, logits.dtype(), logits.device().type());
    
    // Clip probabilities to avoid log(0)
    // probs = max(probs, epsilon)
    Tensor clipped_probs = ops::maximum(probs, eps_tensor);
    
    // Compute negative log likelihood
    Tensor log_probs = ops::log(clipped_probs);
    
    // If targets are class indices (1D)
    if (targets.shape().size() == 1) {
        // One-hot encode targets
        // This part is simplified and would need full implementation
        log_error("Cross entropy with class indices not fully implemented");
        return Tensor();
    }
    
    // If targets are one-hot encoded (2D)
    if (targets.shape() == logits_shape) {
        // NLL: -sum(targets * log_probs, dim=1)
        Tensor mult = targets * log_probs;
        Tensor summed = mult.sum(dim);
        Tensor nll = ops::negate(summed);
        return nll.mean();
    }
    
    log_error("Cross entropy loss requires compatible target format");
    return Tensor();
}

} // namespace nn
} // namespace neuronet
