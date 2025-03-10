#include <neuronet/neuronet.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>

using namespace neuronet;
using namespace neuronet::ops;

// Simple tokenizer for demonstration purposes
std::vector<int64_t> simple_tokenize(const std::string& text, int max_length = 128) {
    // This is a very simplified tokenizer for demonstration
    // In a real application, you would use a proper tokenizer
    std::vector<int64_t> tokens;
    
    // Start with [CLS] token (ID 101 in BERT vocabulary)
    tokens.push_back(101);
    
    // Split by space and assign dummy token IDs
    std::string word;
    for (char c : text) {
        if (std::isspace(c)) {
            if (!word.empty()) {
                // Just use a hash function to get a token ID between 1000-30000
                // This is just for demonstration, not a real tokenization
                int64_t token_id = 1000 + (std::hash<std::string>{}(word) % 29000);
                tokens.push_back(token_id);
                word.clear();
            }
        } else {
            word += std::tolower(c);
        }
    }
    
    // Add the last word if any
    if (!word.empty()) {
        int64_t token_id = 1000 + (std::hash<std::string>{}(word) % 29000);
        tokens.push_back(token_id);
    }
    
    // Add [SEP] token (ID 102 in BERT vocabulary)
    tokens.push_back(102);
    
    // Pad or truncate to max_length
    if (tokens.size() > max_length) {
        // Truncate, but keep [CLS] and [SEP]
        tokens.resize(max_length - 1);
        tokens.push_back(102); // Add [SEP] at the end
    } else {
        // Pad with 0s (padding token ID)
        while (tokens.size() < max_length) {
            tokens.push_back(0);
        }
    }
    
    return tokens;
}

// Simple detokenizer for demonstration purposes
std::string simple_detokenize(const std::vector<int>& token_ids) {
    // This is a very simplified detokenizer for demonstration
    // In a real application, you would use a proper detokenizer
    
    std::string result;
    
    // Skip special tokens ([CLS]=101, [SEP]=102, [PAD]=0, etc.)
    for (int token_id : token_ids) {
        if (token_id > 999) {  // Skip special tokens
            // In a real implementation, we would look up the token in a vocabulary
            // Here we just use the token ID as a character code for demonstration
            char c = 'a' + (token_id % 26);
            result += c;
            result += ' ';
        }
    }
    
    return result;
}

// Function to print the model's response in a nicely formatted way
void print_response(const std::string& response) {
    std::cout << "\n+" << std::string(78, '-') << "+" << std::endl;
    
    // Split response into lines that fit within 76 characters
    std::string line;
    std::istringstream iss(response);
    std::string word;
    
    while (iss >> word) {
        if (line.length() + word.length() + 1 > 76) {
            // Print current line and start a new one
            std::cout << "| " << std::left << std::setw(76) << line << " |" << std::endl;
            line = word;
        } else {
            if (!line.empty()) line += ' ';
            line += word;
        }
    }
    
    // Print the last line
    if (!line.empty()) {
        std::cout << "| " << std::left << std::setw(76) << line << " |" << std::endl;
    }
    
    std::cout << "+" << std::string(78, '-') << "+" << std::endl;
}

int main(int argc, char** argv) {
    // Initialize NeuroNet
    neuronet::initialize();
    
    // Enable colored output
    set_log_color_enabled(true);
    
    std::cout << "=========================================" << std::endl;
    std::cout << "NeuroNet Interactive Question Answering" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Select the device with best performance
    DeviceType device_type = DeviceType::CPU;
    
    if (Device::isCudaAvailable()) {
        device_type = DeviceType::CUDA;
        log_info("Using CUDA device for inference");
    } else if (Device::isMetalAvailable()) {
        device_type = DeviceType::Metal;
        log_info("Using Metal device for inference");
    } else {
        log_info("Using CPU device for inference");
    }
    
    // Default model to load - use a smaller model for interactivity
    std::string model_id = "distilbert-base-uncased-distilled-squad";
    
    // Allow custom model ID from command line
    if (argc > 1) {
        model_id = argv[1];
    }
    
    log_info("Loading model: {}...", model_id);
    
    try {
        // Load the question answering model
        auto model = models::HuggingFaceModel::from_pretrained(model_id, "", device_type);
        
        if (!model) {
            log_error("Failed to load model");
            return 1;
        }
        
        log_info("Model loaded successfully!");
        
        // Main interaction loop
        std::string context;
        std::string question;
        bool has_context = false;
        
        std::cout << "\nWelcome to the interactive QA system!" << std::endl;
        std::cout << "You can provide a context paragraph and then ask questions about it." << std::endl;
        
        while (true) {
            // Ask for context if we don't have one yet
            if (!has_context) {
                std::cout << "\nPlease provide a paragraph of context (or type 'exit' to quit):" << std::endl;
                std::cout << "> ";
                std::getline(std::cin, context);
                
                if (context == "exit" || context == "quit") {
                    break;
                }
                
                has_context = true;
                std::cout << "\nContext set! You can now ask questions about it." << std::endl;
            }
            
            // Ask for a question
            std::cout << "\nYour question (or type 'new' for new context, 'exit' to quit):" << std::endl;
            std::cout << "> ";
            std::getline(std::cin, question);
            
            if (question == "exit" || question == "quit") {
                break;
            }
            
            if (question == "new") {
                has_context = false;
                continue;
            }
            
            // Prepare input for the model
            log_info("Processing question...");
            
            // Combine context and question with [SEP] token between them
            std::string combined = context + " [SEP] " + question;
            
            // Tokenize the input
            std::vector<int64_t> token_ids = simple_tokenize(combined);
            
            // Create input tensor
            Tensor input_ids({1, static_cast<int64_t>(token_ids.size())}, token_ids.data(), DType::Int64, device_type);
            
            // Create attention mask (1 for real tokens, 0 for padding)
            std::vector<int64_t> attention_mask(token_ids.size(), 1);
            for (size_t i = 0; i < token_ids.size(); i++) {
                if (token_ids[i] == 0) attention_mask[i] = 0;
            }
            
            Tensor attn_mask({1, static_cast<int64_t>(attention_mask.size())}, attention_mask.data(), DType::Int64, device_type);
            
            // Forward pass - in a real implementation, we would properly pass both tensors
            // For now, we'll just demonstrate with input_ids
            log_info("Running model inference...");
            Tensor output = model->forward(input_ids);
            
            // In a real implementation, we would process the output to extract the answer span
            // For this demo, we'll just generate a simulated response
            
            // Move output to CPU for processing
            Tensor output_cpu = output.cpu();
            
            // Simulate extracting an answer from the output
            // In reality, this would involve finding start/end logits and extracting text
            std::string answer = "This is a simulated answer since we're not fully implementing the QA model output processing. "
                               "In a real implementation, we would extract the answer from the model output.";
            
            // Print the response
            std::cout << "\nAnswer:" << std::endl;
            print_response(answer);
        }
        
    } catch (const std::exception& e) {
        log_error("Error: {}", e.what());
        return 1;
    }
    
    // Clean up
    neuronet::cleanup();
    std::cout << "Goodbye!" << std::endl;
    
    return 0;
}
