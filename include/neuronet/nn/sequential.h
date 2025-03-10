#pragma once

#include <neuronet/nn/layers.h>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace neuronet {
namespace nn {

// Sequential container for modules
class Sequential : public Module {
public:
    Sequential();
    ~Sequential() override = default;
    
    // Add a module to the sequence with a name
    void add_module(const std::string& name, std::shared_ptr<Module> module);
    
    // Forward pass: run each module in sequence
    Tensor forward(const Tensor& input) override;
    
    // Load and save state dictionaries
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, 
                        std::string prefix = "") override;
    std::unordered_map<std::string, Tensor> state_dict(std::string prefix = "") const override;
    
    // Move to device
    void to(DeviceType device_type) override;

    // Get device
    Device device() const override;
    
private:
    // Vector of (name, module) pairs to maintain order
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> modules_;
};

} // namespace nn
} // namespace neuronet
