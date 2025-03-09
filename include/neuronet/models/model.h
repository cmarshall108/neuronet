#pragma once

#include <neuronet/nn/layers.h>
#include <string>
#include <unordered_map>
#include <memory>

namespace neuronet {
namespace models {

class Model {
public:
    Model();
    virtual ~Model() = default;
    
    virtual Tensor forward(const Tensor& input) = 0;
    
    // Save and load model weights
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    // Get model parameters
    std::unordered_map<std::string, Tensor> state_dict() const;
    
    // Load model parameters
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict);
    
    // Move model to device
    void to(DeviceType device_type);
    
protected:
    std::shared_ptr<nn::Module> module_;
};

} // namespace models
} // namespace neuronet
