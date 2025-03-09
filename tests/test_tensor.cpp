#include <gtest/gtest.h>
#include <neuronet/neuronet.h>

using namespace neuronet;

TEST(TensorTest, Construction) {
    // Create a tensor with dimensions 2x3
    Tensor t({2, 3}, DType::Float32);
    
    // Check dimensions and size
    ASSERT_EQ(t.dim(), 2);
    ASSERT_EQ(t.shape()[0], 2);
    ASSERT_EQ(t.shape()[1], 3);
    ASSERT_EQ(t.size(), 6);
}

TEST(TensorTest, DeviceTransfer) {
    // Create a CPU tensor
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor cpu_tensor({2, 3}, data.data(), DType::Float32, DeviceType::CPU);
    
    // Move to CUDA if available
    if (Device::isCudaAvailable()) {
        Tensor cuda_tensor = cpu_tensor.cuda();
        ASSERT_EQ(cuda_tensor.device().type(), DeviceType::CUDA);
        
        // Move back to CPU
        Tensor back_to_cpu = cuda_tensor.cpu();
        
        // Check data is preserved
        const float* cpu_data = back_to_cpu.data<float>();
        for (size_t i = 0; i < data.size(); ++i) {
            ASSERT_FLOAT_EQ(cpu_data[i], data[i]);
        }
    }
    
    // Move to Metal if available
    if (Device::isMetalAvailable()) {
        Tensor metal_tensor = cpu_tensor.metal();
        ASSERT_EQ(metal_tensor.device().type(), DeviceType::Metal);
        
        // Move back to CPU
        Tensor back_to_cpu = metal_tensor.cpu();
        
        // Check data is preserved
        const float* cpu_data = back_to_cpu.data<float>();
        for (size_t i = 0; i < data.size(); ++i) {
            ASSERT_FLOAT_EQ(cpu_data[i], data[i]);
        }
    }
}

TEST(TensorTest, BasicOperations) {
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
    
    Tensor a({2, 2}, a_data.data(), DType::Float32);
    Tensor b({2, 2}, b_data.data(), DType::Float32);
    
    // Test addition
    Tensor c = a + b;
    const float* c_data = c.data<float>();
    
    ASSERT_FLOAT_EQ(c_data[0], 6.0f);   // 1 + 5
    ASSERT_FLOAT_EQ(c_data[1], 8.0f);   // 2 + 6
    ASSERT_FLOAT_EQ(c_data[2], 10.0f);  // 3 + 7
    ASSERT_FLOAT_EQ(c_data[3], 12.0f);  // 4 + 8
    
    // Test multiplication
    Tensor d = a * b;
    const float* d_data = d.data<float>();
    
    ASSERT_FLOAT_EQ(d_data[0], 5.0f);   // 1 * 5
    ASSERT_FLOAT_EQ(d_data[1], 12.0f);  // 2 * 6
    ASSERT_FLOAT_EQ(d_data[2], 21.0f);  // 3 * 7
    ASSERT_FLOAT_EQ(d_data[3], 32.0f);  // 4 * 8
}
