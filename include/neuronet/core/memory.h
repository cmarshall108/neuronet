#pragma once

#include <neuronet/core/device.h>
#include <cstddef>

namespace neuronet {
namespace memory {

// Allocate memory on specified device
void* allocate(const Device& device, size_t size);

// Free memory on specified device
void free(const Device& device, void* ptr);

// Copy memory from host to device
void copy_to_device(const Device& device, void* dst, const void* src, size_t size);

// Copy memory from device to host
void copy_from_device(const Device& device, void* dst, const void* src, size_t size);

// Copy memory between devices
void copy_between_devices(const Device& src_device, const Device& dst_device,
                         const void* src, void* dst, size_t size);

} // namespace memory
} // namespace neuronet
