#pragma once

#include <string>

namespace neuronet {

enum class DeviceType {
    CPU,
    CUDA,
    Metal
};

class Device {
public:
    Device(DeviceType type, int index = 0);
    
    DeviceType type() const;
    int index() const;
    
    // Get a string representation (e.g., "cpu", "cuda:0", "metal:0")
    std::string toString() const;
    
    // Static helpers
    static Device cpu();
    static Device cuda(int index = 0);
    static Device metal(int index = 0);
    
    // Check if device type is available
    static bool isCudaAvailable();
    static bool isMetalAvailable();
    
private:
    DeviceType type_;
    int index_;
};

} // namespace neuronet
