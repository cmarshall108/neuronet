#pragma once

namespace neuronet {
namespace cpu {

// Initialize CPU backend
bool initialize();

// Cleanup CPU resources
void cleanup();

// Set number of threads for parallel processing
void set_num_threads(int threads);

// Get current number of threads
int get_num_threads();

} // namespace cpu
} // namespace neuronet
