#include <gtest/gtest.h>
#include <neuronet/neuronet.h>

using namespace neuronet;
using namespace neuronet::nn;

TEST(NNTest, LinearLayer) {
    // Create a Linear layer with known weights and biases
    Linear layer(3, 2, true);
    
    // Set weights and biases manually for predictable output
    std::vector<float> weight_data = {
        0.1f, 0.2f, 0.3f,  // First output neuron weights
        0.4f, 0.5f, 0.6f   // Second output neuron weights
    };
    std::vector<float> bias_data = {0.1f, 0.2f};
    
    Tensor weight({2, 3}, weight_data.data(), DType::Float32);
    Tensor bias({2}, bias_data.data(), DType::Float32);
    
    // Set state dictionary
    std::unordered_map<std::string, Tensor> state_dict;
    state_dict["weight"] = weight;
    state_dict["bias"] = bias;
    
    layer.load_state_dict(state_dict);
    
    // Create input tensor
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f};  // Single sample
    Tensor input({1, 3}, input_data.data(), DType::Float32);
    
    // Forward pass
    Tensor output = layer.forward(input);
    
    // Expected output: [0.1*1 + 0.2*2 + 0.3*3 + 0.1, 0.4*1 + 0.5*2 + 0.6*3 + 0.2]
    //                = [1.4, 3.4]
    ASSERT_EQ(output.shape().size(), 2);
    ASSERT_EQ(output.shape()[0], 1);
    ASSERT_EQ(output.shape()[1], 2);
    
    const float* output_data = output.data<float>();
    ASSERT_NEAR(output_data[0], 1.4f, 1e-5);
    ASSERT_NEAR(output_data[1], 3.4f, 1e-5);
}

TEST(NNTest, Sequential) {
    // Create a simple sequential model
    auto seq = Sequential();
    
    // Add layers
    seq.add_module("fc1", std::make_shared<Linear>(2, 3));
    seq.add_module("fc2", std::make_shared<Linear>(3, 1));
    
    // Set weights and biases manually
    std::unordered_map<std::string, Tensor> state_dict;
    
    // First layer weights and bias
    std::vector<float> fc1_weight_data = {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f
    };
    std::vector<float> fc1_bias_data = {0.1f, 0.2f, 0.3f};
    
    // Second layer weights and bias
    std::vector<float> fc2_weight_data = {0.7f, 0.8f, 0.9f};
    std::vector<float> fc2_bias_data = {0.4f};
    
    state_dict["fc1.weight"] = Tensor({3, 2}, fc1_weight_data.data(), DType::Float32);
    state_dict["fc1.bias"] = Tensor({3}, fc1_bias_data.data(), DType::Float32);
    state_dict["fc2.weight"] = Tensor({1, 3}, fc2_weight_data.data(), DType::Float32);
    state_dict["fc2.bias"] = Tensor({1}, fc2_bias_data.data(), DType::Float32);
    
    seq.load_state_dict(state_dict);
    
    // Create input tensor
    std::vector<float> input_data = {1.0f, 2.0f};
    Tensor input({1, 2}, input_data.data(), DType::Float32);
    
    // Forward pass
    Tensor output = seq.forward(input);
    
    // Expected output calculation:
    // fc1 output = [0.1*1 + 0.2*2 + 0.1, 0.3*1 + 0.4*2 + 0.2, 0.5*1 + 0.6*2 + 0.3]
    //            = [0.5, 1.1, 1.7]
    // fc2 output = [0.7*0.5 + 0.8*1.1 + 0.9*1.7 + 0.4]
    //            = [2.68]
    
    ASSERT_EQ(output.shape().size(), 2);
    ASSERT_EQ(output.shape()[0], 1);
    ASSERT_EQ(output.shape()[1], 1);
    
    const float* output_data = output.data<float>();
    ASSERT_NEAR(output_data[0], 2.68f, 1e-5);
}

TEST(NNTest, Activations) {
    // Test various activation functions
    std::vector<float> input_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor input({5}, input_data.data(), DType::Float32);
    
    // Test ReLU
    Tensor relu_output = relu(input);
    const float* relu_data = relu_output.data<float>();
    ASSERT_FLOAT_EQ(relu_data[0], 0.0f);
    ASSERT_FLOAT_EQ(relu_data[1], 0.0f);
    ASSERT_FLOAT_EQ(relu_data[2], 0.0f);
    ASSERT_FLOAT_EQ(relu_data[3], 1.0f);
    ASSERT_FLOAT_EQ(relu_data[4], 2.0f);
    
    // Test sigmoid
    Tensor sigmoid_output = sigmoid(input);
    const float* sigmoid_data = sigmoid_output.data<float>();
    ASSERT_NEAR(sigmoid_data[0], 0.11920292f, 1e-6);  // sigmoid(-2)
    ASSERT_NEAR(sigmoid_data[1], 0.26894143f, 1e-6);  // sigmoid(-1)
    ASSERT_NEAR(sigmoid_data[2], 0.5f, 1e-6);         // sigmoid(0)
    ASSERT_NEAR(sigmoid_data[3], 0.73105858f, 1e-6);  // sigmoid(1)
    ASSERT_NEAR(sigmoid_data[4], 0.88079708f, 1e-6);  // sigmoid(2)
    
    // Test GELU
    Tensor gelu_output = gelu(input);
    const float* gelu_data = gelu_output.data<float>();
    ASSERT_NEAR(gelu_data[0], -0.04540229f, 1e-6);  // approximate gelu(-2)
    ASSERT_NEAR(gelu_data[1], -0.15880796f, 1e-6);  // approximate gelu(-1)
    ASSERT_NEAR(gelu_data[2], 0.0f, 1e-6);          // gelu(0)
    ASSERT_NEAR(gelu_data[3], 0.8411922f, 1e-6);    // approximate gelu(1)
    ASSERT_NEAR(gelu_data[4], 1.95459771f, 1e-6);   // approximate gelu(2)
}
