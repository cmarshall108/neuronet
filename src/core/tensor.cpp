#include <neuronet/core/tensor.h>
#include <neuronet/core/ops.h>
#include <neuronet/core/memory.h>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <fstream>

namespace neuronet {

class TensorImpl {
public:
    TensorImpl(const std::vector<int64_t>& shape, DType dtype, Device device)
        : shape_(shape), dtype_(dtype), device_(device) {
        
        size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        allocate_memory();
    }
    
    TensorImpl(const std::vector<int64_t>& shape, const void* data, DType dtype, Device device)
        : shape_(shape), dtype_(dtype), device_(device) {
        
        size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        allocate_memory();
        
        // Copy data to device
        size_t bytes = size_ * element_size();
        if (device.type() == DeviceType::CPU) {
            std::memcpy(data_, data, bytes);
        } else {
            // First copy to CPU memory
            void* cpu_data = malloc(bytes);
            std::memcpy(cpu_data, data, bytes);
            
            // Then copy to device
            memory::copy_to_device(device_, data_, cpu_data, bytes);
            free(cpu_data);
        }
    }
    
    ~TensorImpl() {
        free_memory();
    }
    
    // Copy constructor
    TensorImpl(const TensorImpl& other)
        : shape_(other.shape_), size_(other.size_), 
          dtype_(other.dtype_), device_(other.device_) {
        
        allocate_memory();
        
        // Copy data
        size_t bytes = size_ * element_size();
        if (device_.type() == DeviceType::CPU && other.device_.type() == DeviceType::CPU) {
            std::memcpy(data_, other.data_, bytes);
        } else {
            memory::copy_between_devices(other.device_, device_, 
                                        other.data_, data_, bytes);
        }
    }
    
    void allocate_memory() {
        size_t bytes = size_ * element_size();
        data_ = memory::allocate(device_, bytes);
    }
    
    void free_memory() {
        if (data_) {
            memory::free(device_, data_);
            data_ = nullptr;
        }
    }
    
    size_t element_size() const {
        switch (dtype_) {
            case DType::Float32: return 4;
            case DType::Float16: return 2;
            case DType::Int32: return 4;
            case DType::Int64: return 8;
            case DType::Bool: return 1;
            default: return 0;
        }
    }
    
    std::vector<int64_t> shape_;
    int64_t size_;
    DType dtype_;
    Device device_;
    void* data_ = nullptr;
};

// Tensor implementation

Tensor::Tensor() 
    : impl_(std::make_shared<TensorImpl>(std::vector<int64_t>{0}, DType::Float32, Device::cpu())) {
}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, DeviceType device_type)
    : impl_(std::make_shared<TensorImpl>(shape, dtype, Device(device_type))) {
}

Tensor::Tensor(const std::vector<int64_t>& shape, const void* data, DType dtype, DeviceType device_type)
    : impl_(std::make_shared<TensorImpl>(shape, data, dtype, Device(device_type))) {
}

Tensor::Tensor(const Tensor& other) 
    : impl_(std::make_shared<TensorImpl>(*other.impl_)) {
}

Tensor::Tensor(Tensor&& other) noexcept
    : impl_(std::move(other.impl_)) {
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        impl_ = std::make_shared<TensorImpl>(*other.impl_);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

Tensor::~Tensor() = default;

const std::vector<int64_t>& Tensor::shape() const {
    return impl_->shape_;
}

int64_t Tensor::dim() const {
    return impl_->shape_.size();
}

int64_t Tensor::size() const {
    return impl_->size_;
}

DType Tensor::dtype() const {
    return impl_->dtype_;
}

Device Tensor::device() const {
    return impl_->device_;
}

template<typename T>
T* Tensor::data() {
    return static_cast<T*>(impl_->data_);
}

template<typename T>
const T* Tensor::data() const {
    return static_cast<const T*>(impl_->data_);
}

Tensor Tensor::to(DeviceType device_type) const {
    if (impl_->device_.type() == device_type) {
        return *this;
    }
    
    // Create a new tensor on the target device
    Tensor result(impl_->shape_, impl_->dtype_, device_type);
    
    // Copy data
    size_t bytes = impl_->size_ * impl_->element_size();
    memory::copy_between_devices(impl_->device_, result.impl_->device_,
                               impl_->data_, result.impl_->data_, bytes);
    
    return result;
}

Tensor Tensor::cpu() const {
    return to(DeviceType::CPU);
}

Tensor Tensor::cuda() const {
    return to(DeviceType::CUDA);
}

Tensor Tensor::metal() const {
    return to(DeviceType::Metal);
}

// Explicit instantiations for common types
template float* Tensor::data<float>();
template const float* Tensor::data<float>() const;
template int* Tensor::data<int>();
template const int* Tensor::data<int>() const;
template int64_t* Tensor::data<int64_t>();
template const int64_t* Tensor::data<int64_t>() const;

// Basic tensor operations - these will dispatch to the appropriate backend
Tensor Tensor::operator+(const Tensor& other) const {
    // Dispatch to the appropriate backend
    if (impl_->device_.type() == DeviceType::CPU) {
        return ops::cpu::add(*this, other);
    } else if (impl_->device_.type() == DeviceType::CUDA) {
        return ops::cuda::add(*this, other);
    } else if (impl_->device_.type() == DeviceType::Metal) {
        return ops::metal::add(*this, other);
    }
    
    // Fall back to CPU if device not supported
    return ops::cpu::add(this->cpu(), other.cpu()).to(impl_->device_.type());
}

// Add implementations for other operations similarly...
// For brevity, only a few operations are implemented here.

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    // Move tensor to CPU for printing
    Tensor cpu_tensor = tensor.cpu();
    
    // Print shape
    os << "Tensor(shape=[";
    const auto& shape = cpu_tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i < shape.size() - 1) {
            os << ", ";
        }
    }
    os << "], dtype=";
    
    // Print dtype
    switch (cpu_tensor.dtype()) {
        case DType::Float32: os << "float32"; break;
        case DType::Float16: os << "float16"; break;
        case DType::Int32: os << "int32"; break;
        case DType::Int64: os << "int64"; break;
        case DType::Bool: os << "bool"; break;
    }
    
    os << ", device=" << cpu_tensor.device().toString() << ")\n";
    
    // Print data (for small tensors only)
    if (cpu_tensor.size() <= 100) {
        // Print tensor data depending on its shape and dtype
        // Implementation omitted for brevity
        os << "[data...]";
    } else {
        os << "[...]";  // For large tensors
    }
    
    return os;
}

void Tensor::save(const std::string& filename) const {
    // Move tensor to CPU for saving
    Tensor cpu_tensor = this->cpu();
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    // Write header information
    int32_t magic = 0x4E544E53;  // "NTNS" (NeuroNet Tensor)
    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    // Write dtype
    int32_t dtype = static_cast<int32_t>(cpu_tensor.dtype());
    file.write(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    
    // Write shape
    int32_t ndim = static_cast<int32_t>(cpu_tensor.dim());
    file.write(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    
    const auto& shape = cpu_tensor.shape();
    for (int64_t dim : shape) {
        int64_t dim_val = dim;
        file.write(reinterpret_cast<char*>(&dim_val), sizeof(dim_val));
    }
    
    // Write data
    size_t bytes = cpu_tensor.size() * cpu_tensor.impl_->element_size();
    file.write(static_cast<char*>(cpu_tensor.impl_->data_), bytes);
}

Tensor Tensor::load(const std::string& filename, DeviceType device_type) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    
    // Read header information
    int32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x4E544E53) {
        throw std::runtime_error("Invalid tensor file format");
    }
    
    // Read dtype
    int32_t dtype;
    file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    
    // Read shape
    int32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    
    std::vector<int64_t> shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        int64_t dim_val;
        file.read(reinterpret_cast<char*>(&dim_val), sizeof(dim_val));
        shape[i] = dim_val;
    }
    
    // Create empty tensor
    Tensor tensor(shape, static_cast<DType>(dtype), DeviceType::CPU);
    
    // Read data
    size_t bytes = tensor.size() * tensor.impl_->element_size();
    file.read(static_cast<char*>(tensor.impl_->data_), bytes);
    
    // Move to requested device if needed
    if (device_type != DeviceType::CPU) {
        tensor = tensor.to(device_type);
    }
    
    return tensor;
}

} // namespace neuronet
