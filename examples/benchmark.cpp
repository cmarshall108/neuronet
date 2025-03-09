#include <neuronet/neuronet.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace neuronet;

// Helper function to measure execution time
template<typename Func>
double measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Run matrix multiplication benchmark for a specific device
void run_matmul_benchmark(DeviceType device_type, const std::vector<int>& sizes, int iterations) {
    std::cout << "\nRunning matrix multiplication benchmark on " << Device(device_type).toString() << std::endl;
    std::cout << std::setw(10) << "Size" << std::setw(15) << "Time (ms)" << std::setw(15) << "GFLOPS" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (int size : sizes) {
        // Create matrices of shape [size, size]
        std::vector<float> a_data(size * size, 1.0f);
        std::vector<float> b_data(size * size, 1.0f);
        
        Tensor a({size, size}, a_data.data(), DType::Float32, device_type);
        Tensor b({size, size}, b_data.data(), DType::Float32, device_type);
        
        // Warm-up iteration
        Tensor c = ops::matmul(a, b);
        
        // Measure execution time over multiple iterations
        std::vector<double> times;
        for (int i = 0; i < iterations; i++) {
            double time = measureTime([&]() {
                Tensor c = ops::matmul(a, b);
            });
            times.push_back(time);
        }
        
        // Calculate median time
        std::sort(times.begin(), times.end());
        double median_time = times[times.size() / 2];
        
        // Calculate GFLOPS (2 * n³ operations for matrix multiplication)
        double ops = 2.0 * std::pow(size, 3);
        double gflops = (ops / 1e9) / (median_time / 1000.0);
        
        std::cout << std::setw(10) << size << std::setw(15) << std::fixed << std::setprecision(3) 
                  << median_time << std::setw(15) << std::fixed << std::setprecision(2) << gflops << std::endl;
    }
}

// Run element-wise operation benchmark
void run_elementwise_benchmark(DeviceType device_type, const std::vector<int>& sizes, int iterations) {
    std::cout << "\nRunning element-wise operation benchmark on " << Device(device_type).toString() << std::endl;
    std::cout << std::setw(10) << "Size" << std::setw(15) << "Add (ms)" << std::setw(15) << "ReLU (ms)" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (int size : sizes) {
        int total_elements = size * size;
        std::vector<float> data(total_elements, 1.0f);
        
        Tensor a({size, size}, data.data(), DType::Float32, device_type);
        Tensor b({size, size}, data.data(), DType::Float32, device_type);
        
        // Warm-up
        Tensor c = a + b;
        Tensor d = ops::relu(a);
        
        // Benchmark addition
        std::vector<double> add_times;
        for (int i = 0; i < iterations; i++) {
            double time = measureTime([&]() {
                Tensor c = a + b;
            });
            add_times.push_back(time);
        }
        
        // Benchmark ReLU
        std::vector<double> relu_times;
        for (int i = 0; i < iterations; i++) {
            double time = measureTime([&]() {
                Tensor d = ops::relu(a);
            });
            relu_times.push_back(time);
        }
        
        // Calculate median times
        std::sort(add_times.begin(), add_times.end());
        std::sort(relu_times.begin(), relu_times.end());
        double median_add_time = add_times[add_times.size() / 2];
        double median_relu_time = relu_times[relu_times.size() / 2];
        
        std::cout << std::setw(10) << size << std::setw(15) << std::fixed << std::setprecision(3) 
                  << median_add_time << std::setw(15) << std::fixed << std::setprecision(3) 
                  << median_relu_time << std::endl;
    }
}

int main(int argc, char** argv) {
    // Initialize NeuroNet
    neuronet::initialize();
    
    // Enable colored logging (it's enabled by default, but showing for demonstration)
    neuronet::set_log_color_enabled(true);
    
    std::cout << "NeuroNet Benchmark" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << neuronet::libraryInfo() << std::endl;
    
    // Test all log levels
    log_debug("This is a DEBUG message (cyan)");
    log_info("This is an INFO message (green)");
    log_warn("This is a WARNING message (yellow)");
    log_error("This is an ERROR message (red)");
    // Uncomment to test fatal (will terminate program): log_fatal("This is a FATAL message (bold red)");
    
    // Benchmark parameters
    std::vector<int> matmul_sizes = {128, 256, 512, 1024, 2048, 4096};
    std::vector<int> elementwise_sizes = {1024, 2048, 4096, 8192};
    int iterations = 10;
    
    // Test CPU performance
    run_matmul_benchmark(DeviceType::CPU, matmul_sizes, iterations);
    run_elementwise_benchmark(DeviceType::CPU, elementwise_sizes, iterations);
    
    // Test CUDA if available
    if (Device::isCudaAvailable()) {
        run_matmul_benchmark(DeviceType::CUDA, matmul_sizes, iterations);
        run_elementwise_benchmark(DeviceType::CUDA, elementwise_sizes, iterations);
    }
    
    // Test Metal if available
    if (Device::isMetalAvailable()) {
        run_matmul_benchmark(DeviceType::Metal, matmul_sizes, iterations);
        run_elementwise_benchmark(DeviceType::Metal, elementwise_sizes, iterations);
    }
    
    // Clean up
    neuronet::cleanup();
    
    return 0;
}
