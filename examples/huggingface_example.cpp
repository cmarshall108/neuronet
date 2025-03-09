#include <neuronet/core/tensor.h>
#include <neuronet/core/ops.h>
#include <neuronet/models/huggingface.h>
#include <neuronet/utils/logging.h>
#include <iostream>
#include <vector>

using namespace neuronet;
using namespace neuronet::ops;

int main(int argc, char** argv) {
    // Initialize the operations backends
    ops::initialize_backends();
    
    // Set log level
    set_log_level(LogLevel::Info);
    
    log_info("NeuroNet HuggingFace Model Example");
    
    // Choose the available device with highest priority: CUDA > Metal > CPU
    DeviceType device_type = DeviceType::CPU;
    
    if (Device::isCudaAvailable()) {
        device_type = DeviceType::CUDA;
        log_info("Using CUDA device");
    } else if (Device::isMetalAvailable()) {
        device_type = DeviceType::Metal;
        log_info("Using Metal device");
    } else {
        log_info("Using CPU device");
    }
    
    // Default model to load
    std::string model_id = "bert-base-uncased";
    
    // Allow custom model ID from command line
    if (argc > 1) {
        model_id = argv[1];
    }
    
    log_info("Loading model: {}", model_id);
    
    try {
        // Load model from HuggingFace
        auto model = models::HuggingFaceModel::from_pretrained(model_id, "", device_type);
        
        if (!model) {
            log_error("Failed to load model");
            return 1;
        }
        
        log_info("Model loaded successfully");
        
        // Create some input tensors (this is a simplified example)
        // In a real application, you'd need tokenization and proper input preparation
        std::vector<int64_t> input_shape = {1, 128}; // Batch size 1, sequence length 128
        Tensor input(input_shape, DType::Int64, device_type);
        
        // Fill with dummy token IDs (all 101, which is [CLS] token in BERT)
        std::vector<int64_t> dummy_tokens(128, 101);
        for (int i = 0; i < 128; i++) {
            *((int64_t*)input.data<int64_t>() + i) = dummy_tokens[i];
        }
        
        // Forward pass
        log_info("Running model inference...");
        Tensor output = model->forward(input);
        
        log_info("Model inference completed");
        log_info("Output tensor shape: [");
        for (size_t i = 0; i < output.shape().size(); ++i) {
            std::cout << output.shape()[i];
            if (i < output.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
    } catch (const std::exception& e) {
        log_error("Error: {}", e.what());
        return 1;
    }
    
    // Clean up
    ops::cleanup_backends();
    
    return 0;
}
