#include <gtest/gtest.h>
#include <neuronet/neuronet.h>

using namespace neuronet;

TEST(OpsTest, MatMul) {
    // Create input matrices
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
    
    Tensor a({2, 2}, a_data.data(), DType::Float32);
    Tensor b({2, 2}, b_data.data(), DType::Float32);
    
    // Perform matrix multiplication
    Tensor c = ops::matmul(a, b);
    
    // Expected result:
    // [1 2] × [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    const float* c_data = c.data<float>();
    
    ASSERT_FLOAT_EQ(c_data[0], 19.0f);  // 1*5 + 2*7
    ASSERT_FLOAT_EQ(c_data[1], 22.0f);  // 1*6 + 2*8
    ASSERT_FLOAT_EQ(c_data[2], 43.0f);  // 3*5 + 4*7
    ASSERT_FLOAT_EQ(c_data[3], 50.0f);  // 3*6 + 4*8
}

TEST(OpsTest, Relu) {
    // Create input tensor with positive and negative values
    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor input({5}, input_data.data(), DType::Float32);
    
    // Apply ReLU
    Tensor output = ops::relu(input);
    
    // Expected: [0, 0, 0, 1, 2]
    const float* output_data = output.data<float>();
    ASSERT_FLOAT_EQ(output_data[0], 0.0f);
    ASSERT_FLOAT_EQ(output_data[1], 0.0f);
    ASSERT_FLOAT_EQ(output_data[2], 0.0f);
    ASSERT_FLOAT_EQ(output_data[3], 1.0f);
    ASSERT_FLOAT_EQ(output_data[4], 2.0f);
}

TEST(OpsTest, DeviceConsistency) {
    // Create tensors on different available devices and check for consistent results
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
    
    Tensor a_cpu({2, 2}, a_data.data(), DType::Float32, DeviceType::CPU);
    Tensor b_cpu({2, 2}, b_data.data(), DType::Float32, DeviceType::CPU);
    
    // CPU computation
    Tensor result_cpu = ops::matmul(a_cpu, b_cpu);
    const float* cpu_data = result_cpu.data<float>();
    
    // Reference values
    const float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    
    // Test on CUDA if available
    if (Device::isCudaAvailable()) {
        Tensor a_cuda = a_cpu.cuda();
        Tensor b_cuda = b_cpu.cuda();
        
        Tensor result_cuda = ops::matmul(a_cuda, b_cuda);
        Tensor result_cuda_on_cpu = result_cuda.cpu();
        
        const float* cuda_data = result_cuda_on_cpu.data<float>();
        
        // Compare results
        for (int i = 0; i < 4; i++) {
            ASSERT_NEAR(cuda_data[i], expected[i], 1e-5);
        }
    }
    
    // Test on Metal if available
    if (Device::isMetalAvailable()) {
        Tensor a_metal = a_cpu.metal();
        Tensor b_metal = b_cpu.metal();
        
        Tensor result_metal = ops::matmul(a_metal, b_metal);
        Tensor result_metal_on_cpu = result_metal.cpu();
        
        const float* metal_data = result_metal_on_cpu.data<float>();
        
        // Compare results
        for (int i = 0; i < 4; i++) {
            ASSERT_NEAR(metal_data[i], expected[i], 1e-5);
        }
    }
}
