#pragma once

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <functional>
#include <neuronet/core/device.h>
#include <neuronet/core/memory.h>

namespace neuronet {

enum class DType {
    Float32,
    Float16,
    Int32,
    Int64,
    Bool
};

class TensorImpl;

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::Float32, 
           DeviceType device_type = DeviceType::CPU);
    Tensor(const std::vector<int64_t>& shape, const void* data, 
           DType dtype = DType::Float32, DeviceType device_type = DeviceType::CPU);
    // Add scalar constructor
    Tensor(const std::vector<int64_t>& shape, float scalar_value,
           DType dtype = DType::Float32, DeviceType device_type = DeviceType::CPU);
    
    // Copy and move constructors/assignment
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();
    
    // Basic info
    const std::vector<int64_t>& shape() const;
    int64_t dim() const;
    int64_t size() const;
    DType dtype() const;
    Device device() const;
    
    // Data access
    template<typename T>
    T* data();
    
    template<typename T>
    const T* data() const;

    // Device transfer
    Tensor to(DeviceType device_type) const;
    Tensor cpu() const;
    Tensor cuda() const;
    Tensor metal() const;
    
    // Reshape operations
    Tensor reshape(const std::vector<int64_t>& shape) const;
    Tensor flatten() const;
    Tensor view(const std::vector<int64_t>& shape) const;

    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    // Math operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose(int dim0, int dim1) const;
    
    // Reduction operations
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    Tensor max(int dim = -1, bool keepdim = false) const;
    
    // Element-wise operations
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor softmax(int dim = -1) const;

    // Print tensor
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
    // Save/load
    void save(const std::string& filename) const;
    static Tensor load(const std::string& filename, DeviceType device_type = DeviceType::CPU);

    // Add a grad() method for optimizer support
    Tensor* grad() const { return nullptr; } // Placeholder, to be implemented properly

    bool empty() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace neuronet
