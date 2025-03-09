#pragma once

#include <neuronet/core/tensor.h>

namespace neuronet {
namespace metal {

// C-friendly wrapper functions to be called from C++ code
// These will be implemented in metal_wrapper.mm

// Allocate memory with Metal
void* metal_allocate_buffer(size_t size);

// Free memory allocated with Metal
void metal_free_buffer(void* buffer);

// Copy memory between host and Metal device
void metal_copy_to_device(void* dst, const void* src, size_t size);
void metal_copy_from_device(void* dst, const void* src, size_t size);

} // namespace metal
} // namespace neuronet
