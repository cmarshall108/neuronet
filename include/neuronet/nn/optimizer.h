#pragma once

#include <neuronet/core/tensor.h>
#include <vector>

namespace neuronet {
namespace nn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    // Update parameters based on gradients
    virtual void step() = 0;
    
    // Zero out gradients
    virtual void zero_grad() = 0;
};

// SGD Optimizer
class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(const std::vector<Tensor*>& parameters, float learning_rate = 0.01f,
                float momentum = 0.0f, float weight_decay = 0.0f);
    
    void step() override;
    void zero_grad() override;
    
private:
    std::vector<Tensor*> parameters_;
    std::vector<Tensor> velocities_;
    float learning_rate_;
    float momentum_;
    float weight_decay_;
};

} // namespace nn
} // namespace neuronet
