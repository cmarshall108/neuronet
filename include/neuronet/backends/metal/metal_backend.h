#pragma once

namespace neuronet {
namespace metal {

// Initialize Metal backend
bool initialize();

// Cleanup Metal resources
void cleanup();

// Get Metal device, command queue, and library (defined in .mm file)
#if defined(__APPLE__) && defined(NEURONET_USE_METAL)
void* get_metal_device();
void* get_command_queue();
void* get_metal_library();
#endif

} // namespace metal
} // namespace neuronet
