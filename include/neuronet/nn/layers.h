#pragma once

#include <neuronet/core/tensor.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace neuronet {
namespace nn {

class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    
    virtual Tensor forward(const Tensor& input) = 0;
    
    virtual void load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, 
                                std::string prefix = "");
    virtual std::unordered_map<std::string, Tensor> state_dict(std::string prefix = "") const;
    
    virtual void to(DeviceType device_type);
    
    virtual Device device() const = 0;
};

class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool bias = true);
    ~Linear() override = default;
    
    Tensor forward(const Tensor& input) override;
    
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, 
                        std::string prefix = "") override;
    std::unordered_map<std::string, Tensor> state_dict(std::string prefix = "") const override;
    
    void to(DeviceType device_type) override;

    Device device() const;
    
private:
    int in_features_;
    int out_features_;
    bool has_bias_;
    Tensor weight_;
    Tensor bias_;
};

class Conv2d : public Module {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0, bool bias = true);
    ~Conv2d() override = default;
    
    Tensor forward(const Tensor& input) override;
    
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, 
                        std::string prefix = "") override;
    std::unordered_map<std::string, Tensor> state_dict(std::string prefix = "") const override;
    
    void to(DeviceType device_type) override;

    Device device() const;
    
private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool has_bias_;
    Tensor weight_;
    Tensor bias_;
};

class LayerNorm : public Module {
public:
    LayerNorm(const std::vector<int64_t>& normalized_shape, float eps = 1e-5);
    ~LayerNorm() override = default;
    
    Tensor forward(const Tensor& input) override;
    
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, 
                        std::string prefix = "") override;
    std::unordered_map<std::string, Tensor> state_dict(std::string prefix = "") const override;
    
    void to(DeviceType device_type) override;
    
private:
    std::vector<int64_t> normalized_shape_;
    float eps_;
    Tensor weight_;
    Tensor bias_;
};

class Dropout : public Module {
public:
    Dropout(float p = 0.5);
    ~Dropout() override = default;
    
    Tensor forward(const Tensor& input) override;
    
private:
    float p_;
    bool training_;
};

class Sequential : public Module {
public:
    Sequential();
    ~Sequential() override = default;
    
    template<typename... Modules>
    void add(Modules&&... modules);
    
    void add_module(const std::string& name, std::shared_ptr<Module> module);
    
    Tensor forward(const Tensor& input) override;
    
    void load_state_dict(const std::unordered_map<std::string, Tensor>& state_dict, 
                        std::string prefix = "") override;
    std::unordered_map<std::string, Tensor> state_dict(std::string prefix = "") const override;
    
    void to(DeviceType device_type) override;

    Device device() const;
    
private:
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> modules_;
};

} // namespace nn
} // namespace neuronet
