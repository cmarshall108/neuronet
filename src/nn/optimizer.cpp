#include <neuronet/nn/optimizer.h>
#include <neuronet/utils/logging.h>
#include <neuronet/core/ops.h>

namespace neuronet {
namespace nn {

// SGD Optimizer implementation
SGDOptimizer::SGDOptimizer(const std::vector<Tensor*>& parameters, float learning_rate, float momentum, float weight_decay)
    : learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {
    
    // Store parameters
    parameters_ = parameters;
    
    // Initialize velocity for momentum if needed
    if (momentum > 0) {
        for (auto* param : parameters_) {
            Tensor velocity(param->shape(), 0.0f, param->dtype(), param->device().type());
            velocities_.push_back(velocity);
        }
    }
    
    log_info("Created SGD optimizer with lr={}, momentum={}, weight_decay={}",
             std::to_string(learning_rate), 
             std::to_string(momentum),
             std::to_string(weight_decay));
}

void SGDOptimizer::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor* param = parameters_[i];
        
        // Get gradient - note: this is a placeholder implementation since we haven't
        // implemented automatic differentiation yet
        Tensor* grad = param->grad();
        
        if (grad == nullptr) {
            log_warn("Parameter has no gradient, skipping in optimizer step");
            continue;
        }
        
        // Apply weight decay if needed
        if (weight_decay_ > 0) {
            // grad = grad + weight_decay * param
            // Use scalar multiplication helper
            Tensor weight_decay_term = ops::mul_scalar(*param, weight_decay_);
            *grad = *grad + weight_decay_term;
        }
        
        // Apply momentum if needed
        if (momentum_ > 0) {
            Tensor& velocity = velocities_[i];
            // v = momentum * v + grad
            Tensor momentum_term = ops::mul_scalar(velocity, momentum_);
            velocity = momentum_term + (*grad);
            
            // param = param - lr * v
            Tensor lr_velocity = ops::mul_scalar(velocity, learning_rate_);
            *param = *param - lr_velocity;
        } else {
            // Simple SGD update: param = param - lr * grad
            Tensor lr_grad = ops::mul_scalar(*grad, learning_rate_);
            *param = *param - lr_grad;
        }
    }
}

void SGDOptimizer::zero_grad() {
    for (auto* param : parameters_) {
        Tensor* grad = param->grad();
        if (grad != nullptr) {
            // Zero out gradient
            // This would need proper tensor operations to zero the tensor
            log_info("Setting gradients to zero");
        }
    }
}

} // namespace nn
} // namespace neuronet
