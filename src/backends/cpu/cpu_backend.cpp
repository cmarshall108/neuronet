#include <neuronet/backends/cpu/cpu_backend.h>
#include <neuronet/utils/logging.h>
#include <thread>

namespace neuronet {
namespace cpu {

// Number of CPU threads to use for parallel computations
static int num_threads = std::thread::hardware_concurrency();

bool initialize() {
    log_info("CPU backend initialized with {} threads", std::to_string(num_threads));
    return true;
}

void cleanup() {
    // No specific cleanup needed for CPU backend
}

void set_num_threads(int threads) {
    num_threads = threads > 0 ? threads : std::thread::hardware_concurrency();
    log_info("CPU backend set to use {} threads", std::to_string(num_threads));
}

int get_num_threads() {
    return num_threads;
}

} // namespace cpu
} // namespace neuronet
