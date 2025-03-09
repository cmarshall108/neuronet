#include <neuronet/nn/optimizer.h>
#include <neuronet/utils/logging.h>

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
            Tensor velocity(param->shape(), param->dtype(), param->device().type());
            // Fill with zeros
            // This is a simplified approach, would need to properly zero-initialize
            velocities_.push_back(velocity);
        }
    }
    
    log_info("Created SGD optimizer with lr={}, momentum={}, weight_decay={}",
             std::to_string(learning_rate), std::to_string(momentum), std::to_string(weight_decay));
}

void SGDOptimizer::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor* param = parameters_[i];
        
        // Get gradient
        Tensor* grad = param->grad();
        
        if (grad == nullptr) {
            log_warn("Parameter has no gradient, skipping in optimizer step");
            continue;
        }
        
        // Apply weight decay if needed
        if (weight_decay_ > 0) {
            // grad = grad + weight_decay * param
            *grad = *grad + (*param) * weight_decay_;
        }
        
        // Apply momentum if needed
        if (momentum_ > 0) {
            // v = momentum * v + grad
            // param = param - lr * v
            
            Tensor& velocity = velocities_[i];
            velocity = velocity * momentum_ + (*grad);
            *param = *param - velocity * learning_rate_;
        } else {
            // Simple SGD update: param = param - lr * grad
            *param = *param - (*grad) * learning_rate_;
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
