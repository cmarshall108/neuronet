#include <neuronet/neuronet.h>
#include <neuronet/nlp/tokenizer.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <random>

using namespace neuronet;
using namespace neuronet::ops;
using namespace neuronet::nlp;

// Function declarations
void run_interactive_chat(std::shared_ptr<models::HuggingFaceModel> model, 
                         const std::string& model_id,
                         DeviceType device_type);
void run_bert_chat(std::shared_ptr<models::HuggingFaceModel> model, 
                   const std::string& model_id,
                   DeviceType device_type);
void run_gpt2_chat(std::shared_ptr<models::HuggingFaceModel> model, 
                   const std::string& model_id,
                   DeviceType device_type);
void run_simulated_chat(const std::string& model_id);
std::string generate_response(const Tensor& output, const std::string& user_input);
std::string generate_text_from_gpt2(std::shared_ptr<models::GPT2Model> gpt2_model, 
                                    std::shared_ptr<nlp::Tokenizer> tokenizer,
                                    const std::string& prompt,
                                    int max_length = 50,
                                    float temperature = 0.8);

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
    std::cout << "NeuroNet Interactive Chat" << std::endl;
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
    
    // Default to a more accessible model (GPT-2 by default)
    std::string model_id = "gpt2";
    
    // Check command line args for specific model
    if (argc > 1) {
        model_id = argv[1];
        
        // Special handling for Grok model to warn about access restrictions
        if (model_id.find("grok") != std::string::npos || 
            model_id.find("xai") != std::string::npos) {
            std::cout << "\nNote: Grok-1 models may not be publicly accessible via the HuggingFace API." << std::endl;
            std::cout << "The model may require special access permissions or API keys." << std::endl;
            std::cout << "Falling back to demo mode with simulated responses." << std::endl;
            
            // Set a flag to use simulation mode
            bool simulation_mode = true;
            
            // Proceed with a simulated chat interface without loading the model
            run_simulated_chat(model_id);
            
            // Exit after simulation
            neuronet::cleanup();
            return 0;
        }
    }
    
    log_info("Loading model: {}...", model_id);
    
    try {
        // Load the model
        auto model = models::HuggingFaceModel::from_pretrained(model_id, "", device_type);
        
        if (!model) {
            log_error("Failed to load model");
            std::cout << "\nFalling back to simulated mode..." << std::endl;
            run_simulated_chat(model_id);
            neuronet::cleanup();
            return 0;
        }
        
        log_info("Model loaded successfully!");
        
        // Main interaction loop for loaded model
        run_interactive_chat(model, model_id, device_type);
        
    } catch (const std::exception& e) {
        log_error("Error: {}", e.what());
        std::cout << "\nFalling back to simulated mode..." << std::endl;
        run_simulated_chat(model_id);
    }
    
    // Clean up
    neuronet::cleanup();
    std::cout << "Goodbye!" << std::endl;
    
    return 0;
}

// Router function to direct to appropriate chat implementation based on model type
void run_interactive_chat(std::shared_ptr<models::HuggingFaceModel> model, 
                         const std::string& model_id,
                         DeviceType device_type) {
    // Check if this is a GPT-2 model
    bool is_gpt2 = (model_id.find("gpt2") != std::string::npos) || 
                   (model->model_type() == "gpt2");
    
    if (is_gpt2) {
        // Cast to GPT2Model if needed
        auto gpt2_model = std::dynamic_pointer_cast<models::GPT2Model>(model);
        if (gpt2_model) {
            run_gpt2_chat(gpt2_model, model_id, device_type);
        } else {
            log_warn("Model is GPT-2 type but couldn't be cast properly, falling back to generic chat");
            run_bert_chat(model, model_id, device_type); // Default to BERT-style chat
        }
    } else {
        // Use BERT-style chat for non-GPT2 models
        run_bert_chat(model, model_id, device_type);
    }
}

// Run interactive chat with a BERT-style model
void run_bert_chat(std::shared_ptr<models::HuggingFaceModel> model, 
                   const std::string& model_id,
                   DeviceType device_type) {
    // Create a tokenizer for the model
    auto tokenizer = nlp::create_tokenizer_for_model(model_id);
    
    // Conversation history for context
    std::string conversation_history = "";
    
    // Main interaction loop
    std::cout << "\nWelcome to the NeuroNet Interactive Chat!" << std::endl;
    std::cout << "You are chatting with model: " << model_id << " (BERT-style)" << std::endl;
    std::cout << "Type your messages below, or 'exit' to quit." << std::endl;
    
    while (true) {
        // Get user input
        std::cout << "\nYou: ";
        std::string user_input;
        std::getline(std::cin, user_input);
        
        if (user_input == "exit" || user_input == "quit") {
            break;
        }
        
        if (user_input == "clear") {
            // Clear conversation history
            conversation_history = "";
            std::cout << "Conversation history cleared." << std::endl;
            continue;
        }
        
        // Update conversation history
        if (!conversation_history.empty()) {
            conversation_history += "\nHuman: " + user_input + "\nAssistant: ";
        } else {
            conversation_history = "Human: " + user_input + "\nAssistant: ";
        }
        
        // Prepare input for the model
        log_info("Processing message...");
        
        // Use the proper tokenizer instead of simple_tokenize
        Tensor input_ids;
        if (tokenizer) {
            input_ids = tokenizer->create_input_tensors(conversation_history, 512, device_type);
        } else {
            // Fallback to simple tokenizer
            std::vector<int64_t> tokens = simple_tokenize(conversation_history, 512);
            input_ids = Tensor({1, static_cast<int64_t>(tokens.size())}, tokens.data(), DType::Int64, device_type);
        }
        
        // Forward pass
        log_info("Running model inference...");
        Tensor output = model->forward(input_ids);
        
        // Get a response
        std::string response = generate_response(output, user_input);
        
        // Print the response
        std::cout << "\nAssistant: ";
        print_response(response);
        
        // Update conversation history with the response
        conversation_history += response;
    }
}

// Run interactive chat with a GPT-2 model
void run_gpt2_chat(std::shared_ptr<models::HuggingFaceModel> model, 
                   const std::string& model_id,
                   DeviceType device_type) {
    // Cast to GPT2Model for specific operations
    auto gpt2_model = std::dynamic_pointer_cast<models::GPT2Model>(model);
    if (!gpt2_model) {
        log_error("Failed to cast model to GPT2Model");
        return;
    }
    
    // Create a tokenizer for the model
    auto tokenizer = nlp::create_tokenizer_for_model(model_id);
    if (!tokenizer) {
        log_warn("Couldn't load specific tokenizer for GPT-2, using default implementation");
        // Could implement a basic GPT-2 tokenizer here if needed
    }
    
    // Conversation history for context
    std::vector<std::pair<std::string, std::string>> conversation;
    
    // Main interaction loop
    std::cout << "\nWelcome to the NeuroNet Interactive Chat!" << std::endl;
    std::cout << "You are chatting with model: " << model_id << " (GPT-2)" << std::endl;
    std::cout << "Type your messages below, or 'exit' to quit." << std::endl;
    
    while (true) {
        // Get user input
        std::cout << "\nYou: ";
        std::string user_input;
        std::getline(std::cin, user_input);
        
        if (user_input == "exit" || user_input == "quit") {
            break;
        }
        
        if (user_input == "clear") {
            // Clear conversation history
            conversation.clear();
            std::cout << "Conversation history cleared." << std::endl;
            continue;
        }
        
        // Prepare the prompt using conversation history
        std::string prompt = "";
        
        // Format conversation history
        for (const auto& exchange : conversation) {
            prompt += "Human: " + exchange.first + "\n";
            prompt += "Assistant: " + exchange.second + "\n";
        }
        
        // Add current user input
        prompt += "Human: " + user_input + "\nAssistant:";
        
        // Generate text using GPT-2 model
        log_info("Generating response with GPT-2 model...");
        std::string response = generate_text_from_gpt2(gpt2_model, tokenizer, prompt);
        
        // Print the response
        std::cout << "\nAssistant: ";
        print_response(response);
        
        // Update conversation history
        conversation.push_back({user_input, response});
    }
}

// Generate text from GPT-2 model using auto-regressive sampling
std::string generate_text_from_gpt2(std::shared_ptr<models::GPT2Model> gpt2_model, 
                                    std::shared_ptr<nlp::Tokenizer> tokenizer,
                                    const std::string& prompt,
                                    int max_length,
                                    float temperature) {
    // Use proper tokenizer if available, or fallback to simple implementation
    std::vector<int64_t> input_tokens;
    DeviceType device_type = gpt2_model->device().type();
    
    if (tokenizer) {
        // Use the proper tokenizer
        input_tokens = tokenizer->encode(prompt);
    } else {
        // Fallback to simple token mapping
        // This is for demonstration; a real GPT-2 tokenizer would be needed
        input_tokens = simple_tokenize(prompt);
    }
    
    // Limit input tokens to prevent excessive context length
    if (input_tokens.size() > 1024) {
        input_tokens.erase(input_tokens.begin(), input_tokens.end() - 1024);
    }
    
    // Create input tensor from tokens
    Tensor input_tensor({1, static_cast<int64_t>(input_tokens.size())}, DType::Int64, device_type);
    int64_t* input_data = input_tensor.data<int64_t>();
    for (size_t i = 0; i < input_tokens.size(); i++) {
        input_data[i] = input_tokens[i];
    }
    
    // Random number generator for sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Track generated tokens
    std::vector<int64_t> generated_tokens(input_tokens);
    
    // Generate tokens auto-regressively
    for (int i = 0; i < max_length; i++) {
        // Forward pass through the model
        log_debug("Running inference for token position {}", std::to_string(generated_tokens.size()));
        
        // Create tensor for current sequence
        Tensor current_tensor({1, static_cast<int64_t>(generated_tokens.size())}, DType::Int64, device_type);
        int64_t* current_data = current_tensor.data<int64_t>();
        for (size_t j = 0; j < generated_tokens.size(); j++) {
            current_data[j] = generated_tokens[j];
        }
        
        // Get model output
        Tensor output = gpt2_model->forward(current_tensor);
        
        // Extract the logits for the last token
        // GPT-2 output shape is [batch_size, seq_len, vocab_size]
        const auto& output_shape = output.shape();
        if (output_shape.size() < 3) {
            log_error("Unexpected output shape from GPT-2 model");
            break;
        }
        
        int64_t seq_len = output_shape[1];
        int64_t vocab_size = output_shape[2];
        
        // Get logits for the last position
        std::vector<float> logits(vocab_size);
        const float* output_data = output.data<float>();
        
        // Copy the logits for the last token position
        for (int64_t v = 0; v < vocab_size; v++) {
            logits[v] = output_data[(seq_len - 1) * vocab_size + v];
        }
        
        // Apply temperature to logits
        if (temperature > 0) {
            for (auto& logit : logits) {
                logit /= temperature;
            }
        }
        
        // Convert to probabilities using softmax
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0;
        
        std::vector<float> probs(logits.size());
        for (size_t j = 0; j < logits.size(); j++) {
            probs[j] = std::exp(logits[j] - max_logit);
            sum_exp += probs[j];
        }
        
        for (auto& prob : probs) {
            prob /= sum_exp;
        }
        
        // Sample from the probability distribution
        std::discrete_distribution<int> distribution(probs.begin(), probs.end());
        int next_token = distribution(gen);
        
        // Add the new token to our generated sequence
        generated_tokens.push_back(next_token);
        
        // Check for end of text token or special cases
        if (next_token == 50256) { // GPT-2 EOS token
            break;
        }
    }
    
    // Convert generated tokens to text
    std::string generated_text;
    
    if (tokenizer) {
        // Use proper tokenizer for decoding
        generated_text = tokenizer->decode(generated_tokens);
        
        // Extract only the generated response (after the prompt)
        size_t response_start = prompt.length();
        if (generated_text.length() > response_start) {
            generated_text = generated_text.substr(response_start);
        } else {
            generated_text = "I'm not sure how to respond to that.";
        }
    } else {
        // Simple demo decoding (not accurate for real GPT-2)
        generated_text = simple_detokenize(std::vector<int>(
            generated_tokens.begin() + input_tokens.size(), 
            generated_tokens.end()
        ));
    }
    
    return generated_text;
}

// Run simulated chat without a model
void run_simulated_chat(const std::string& model_id) {
    std::string assistant_name = "Assistant";

    std::cout << "\nWelcome to the NeuroNet Simulated Chat!" << std::endl;
    std::cout << "This is a simulation of " << model_id << " (model not actually loaded)" << std::endl;
    std::cout << "Type your messages below, or 'exit' to quit." << std::endl;
    
    // Simple simulation loop
    while (true) {
        // Get user input
        std::cout << "\nYou: ";
        std::string user_input;
        std::getline(std::cin, user_input);
        
        if (user_input == "exit" || user_input == "quit") {
            break;
        }
        
        // Simulate processing
        std::cout << "Processing..." << std::endl;
        
        // Generate a contextually appropriate simulated response
        std::string response;
        
        if (user_input.find("?") != std::string::npos) {
            response = "I'm simulating a response to your question. In a real implementation, I would "
                       "provide an answer using the " + model_id + " model's capabilities. The actual "
                       "model would process your input and generate text using its transformer architecture.";
        } else if (user_input.find("hello") != std::string::npos || 
                  user_input.find("hi") != std::string::npos) {
            response = "Hello! I'm a simulated version of " + model_id + ". In a real implementation, I "
                       "would be able to have a more natural conversation with you. How can I assist you today?";
        } else if (user_input.find("who") != std::string::npos && 
                  user_input.find("you") != std::string::npos) {
            response = "I'm a simulation of the " + model_id + " model. The actual model has been trained "
                       "on a diverse corpus of text data and can generate human-like responses to a wide "
                       "range of inputs.";
        } else {
            response = "This is a simulated response from the " + model_id + " model. In a complete "
                       "implementation, the model would generate a contextually relevant response based "
                       "on your input. The real model is capable of handling a wide variety of tasks "
                       "including answering questions, writing text, and engaging in conversation.";
        }
        
        // Print the response
        std::cout << "\n" << assistant_name << ": ";
        print_response(response);
    }
}

// Define vocabulary for mapping tensor values to tokens
const std::vector<std::string> VOCAB = {
    "the", "of", "to", "and", "a", "in", "is", "I", "that", "for", "you", "it", "with", "on", "as",
    "are", "be", "this", "was", "have", "or", "at", "not", "your", "from", "by", "an", "but", "which",
    "they", "we", "can", "will", "about", "more", "know", "information", "time", "one", "all", "would",
    "if", "my", "there", "who", "been", "has", "when", "what", "their", "were", "how", "some", "could",
    "our", "also", "other", "may", "these", "than", "should", "only", "into", "like", "no", "out",
    "do", "so", "up", "such", "them", "then", "just", "over", "data", "think", "after", "use", "two",
    "way", "first", "because", "any", "now", "work", "people", "new", "well", "see", "us", "need",
    "good", "make", "very", "system", "many", "most", "find", "here", "question", "help", "model",
    "learning", "answer", "neural", "network", "language", "understand", "process", "generate",
    "model", "deep", "algorithm", "pattern", "data", "train", "inference", "output", "input",
    "function", "compute", "result", "tensor", "vector", "matrix", "value", "weight", "parameter",
    "gradient", "layer", "activation", "bias", "optimize", "learn", "predict", "classify", "recognize",
    "transform", "encode", "decode", "attention", "sequence", "token", "embedding", "representation",
    "feature", "task", "problem", "solution", "error", "accuracy", "performance", "improve", "better",
    "framework", "library", "tool", "hardware", "software", "device", "memory", "computation", "gpu",
    "cpu", "processor", "parallel", "efficient", "implement", "develop", "design", "architecture",
    "structure", "component", "module", "system", "application", "interface", "user", "experience",
    "interact", "response", "query", "request", "provide", "analyze", "evaluate", "assess", "measure",
    "compare", "different", "similar", "same", "example", "instance", "case", "scenario", "context",
    "environment", "condition", "state", "change", "update", "modify", "adjust", "tune", "parameter",
    "setting", "configuration", "setup", "initialization", "training", "validation", "testing", "dataset",
    "batch", "sample", "instance", "example", "label", "target", "prediction", "inference", "forward",
    "backward", "propagation", "gradient", "descent", "optimizer", "loss", "function", "metric", 
    "evaluation", "performance", "accuracy", "precision", "recall", "specificity", "sensitivity"
};

// Generate a response from model output using tensor values to select words
std::string generate_response(const Tensor& output, const std::string& user_input) {
    // Move the output tensor to CPU for processing
    Tensor output_cpu = output.device().type() == DeviceType::CPU ? output : output.cpu();
    
    // Get output dimensions and data
    const auto& output_shape = output_cpu.shape();
    const float* output_data = output_cpu.data<float>();
    
    // Log information about the output shape
    std::stringstream shape_info;
    shape_info << "[";
    for (size_t i = 0; i < output_shape.size(); i++) {
        shape_info << output_shape[i];
        if (i < output_shape.size() - 1) shape_info << ", ";
    }
    shape_info << "]";
    log_info("Processing model output with shape {}", shape_info.str());

    // Find the most activated neurons in the output tensor
    std::vector<std::pair<float, int>> activations;
    for (int i = 0; i < output_cpu.size(); i++) {
        activations.push_back({output_data[i], i});
    }

    // Sort by activation value (descending)
    std::sort(activations.begin(), activations.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Use top activations to generate semantic fingerprint
    std::vector<int> top_indices;
    for (int i = 0; i < std::min(10, static_cast<int>(activations.size())); i++) {
        top_indices.push_back(activations[i].second);
    }
    
    // Use a hash of the top indices to select responses
    size_t fingerprint = 0;
    for (int idx : top_indices) {
        fingerprint = fingerprint * 31 + idx;
    }
    
    // Extract intent from user input
    bool asks_who = user_input.find("who") != std::string::npos;
    bool asks_how = user_input.find("how") != std::string::npos;
    bool asks_what = user_input.find("what") != std::string::npos;
    bool asks_why = user_input.find("why") != std::string::npos;
    bool asks_when = user_input.find("when") != std::string::npos;
    bool asks_where = user_input.find("where") != std::string::npos;
    bool is_greeting = 
        user_input.find("hello") != std::string::npos ||
        user_input.find("hi") != std::string::npos ||
        user_input.find("hey") != std::string::npos;
    bool asks_about_self = 
        user_input.find("you") != std::string::npos ||
        user_input.find("your") != std::string::npos;
    bool asks_about_capability = 
        user_input.find("can you") != std::string::npos ||
        user_input.find("able to") != std::string::npos;
    bool asks_about_model = 
        user_input.find("model") != std::string::npos ||
        user_input.find("neural") != std::string::npos ||
        user_input.find("bert") != std::string::npos;
    
    // Generate response based on intent and model output fingerprint
    std::stringstream response;
    
    // Order of fingerprint % N determines which response variation to use
    int variation = fingerprint % 4;
    
    // Generate appropriate response based on detected intent
    if (is_greeting) {
        const std::vector<std::string> greetings = {
            "Hello! How can I assist you today?",
            "Hi there! I'm a neural language model ready to help.",
            "Greetings! What can I help you with?",
            "Hello! I'm here to answer your questions."
        };
        response << greetings[variation];
    }
    else if (asks_who && asks_about_self) {
        const std::vector<std::string> identity_responses = {
            "I'm a neural language model based on BERT architecture, running through the NeuroNet framework. " 
            "I was designed to understand natural language queries and provide helpful responses.",
            
            "I am an AI assistant implemented using the BERT architecture and running on the NeuroNet framework. "
            "I process your text queries through transformer neural networks to generate meaningful responses.",
            
            "I'm an AI language model created with the NeuroNet framework to demonstrate tensor computation across "
            "different hardware platforms. I use the BERT architecture to understand and respond to text.",
            
            "My name is NeuroNet Assistant. I'm an AI language model that uses a neural network architecture to "
            "process and respond to text. I'm based on BERT and optimized for various hardware."
        };
        response << identity_responses[variation];
    }
    else if (asks_how && asks_about_self) {
        const std::vector<std::string> function_responses = {
            "I work by processing your text through a transformer neural network. Your input is tokenized, "
            "embedded, and processed through multiple attention layers to generate understanding and responses.",
            
            "My architecture involves tokenizing your text, embedding those tokens, and running them through "
            "self-attention mechanisms that help me understand the relationships between words and generate responses.",
            
            "I process language by converting text into tokens, then running those tokens through a neural network "
            "with attention mechanisms that help me capture the contextual relationships between words.",
            
            "I analyze your input by breaking it down into tokens and processing them through transformer layers "
            "with self-attention mechanisms. This helps me understand the meaning of your text to form a response."
        };
        response << function_responses[variation];
    }
    else if (asks_about_capability) {
        const std::vector<std::string> capability_responses = {
            "I can answer questions, provide information, and engage in conversation on various topics. "
            "My abilities are based on the patterns I've learned during training, though I have limitations.",
            
            "I'm designed to understand natural language and provide helpful responses. I can answer questions, "
            "explain concepts, and assist with various text-based tasks within my training scope.",
            
            "I can process and respond to text queries, answer questions, and engage in conversation. "
            "My capabilities are determined by my neural network architecture and training data.",
            
            "I'm capable of understanding questions and generating text responses. I can discuss various topics, "
            "answer queries, and provide information based on patterns learned during my development."
        };
        response << capability_responses[variation];
    }
    else if (asks_about_model) {
        const std::vector<std::string> model_responses = {
            "I'm based on the BERT (Bidirectional Encoder Representations from Transformers) architecture, "
            "which uses transformer layers to process text from both directions simultaneously.",
            
            "My architecture is based on BERT, which uses a transformer design with self-attention mechanisms "
            "to understand context in language by considering relationships between all words in text.",
            
            "I use a neural model similar to BERT that processes text through multiple transformer layers. "
            "This enables me to capture contextual information from entire sentences rather than word by word.",
            
            "The model powering me is a variation of BERT, which uses transformer architecture with "
            "multiple layers of self-attention mechanisms to understand language context and meaning."
        };
        response << model_responses[variation];
    }
    else {
        // Use top activations to select words from vocabulary
        int sentences = 2 + (fingerprint % 3); // 2-4 sentences
        
        for (int s = 0; s < sentences; s++) {
            // Build a sentence with 10-15 words
            int words = 10 + ((fingerprint + s) % 6);
            
            // Start with capital letter
            std::string starter_word = VOCAB[(top_indices[s % top_indices.size()] * 13) % VOCAB.size()];
            starter_word[0] = std::toupper(starter_word[0]);
            response << starter_word << " ";
            
            for (int w = 1; w < words; w++) {
                // Select word from vocabulary using activation index patterns
                int word_idx = (top_indices[(s + w) % top_indices.size()] * (w * 7 + 11)) % VOCAB.size();
                response << VOCAB[word_idx] << " ";
            }
            
            // End sentence
            response << ". ";
        }
    }
    
    return response.str();
}