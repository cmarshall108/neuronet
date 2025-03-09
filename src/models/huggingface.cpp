#include <neuronet/models/huggingface.h>
#include <neuronet/utils/logging.h>
#include <neuronet/utils/json.h>
#include <curl/curl.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace fs = std::filesystem;

namespace neuronet {
namespace models {

// Helper function for CURL data writing
size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    return fwrite(ptr, size, nmemb, stream);
}

HuggingFaceModelLoader::HuggingFaceModelLoader() {
    // Initialize CURL globally
    curl_global_init(CURL_GLOBAL_ALL);
}

bool HuggingFaceModelLoader::download(const std::string& model_id, const std::string& local_dir) {
    log_info("Downloading model {} to {}", model_id, local_dir);
    
    // Create local directory if it doesn't exist
    fs::create_directories(local_dir);
    
    // File URLs to download
    std::vector<std::pair<std::string, std::string>> files = {
        {"config.json", "https://huggingface.co/" + model_id + "/raw/main/config.json"},
        {"model.safetensors", "https://huggingface.co/" + model_id + "/resolve/main/model.safetensors"},
        {"pytorch_model.bin", "https://huggingface.co/" + model_id + "/resolve/main/pytorch_model.bin"}
    };
    
    bool success = true;
    
    for (const auto& file : files) {
        std::string local_path = local_dir + "/" + file.first;
        
        // Skip if file already exists
        if (fs::exists(local_path)) {
            log_info("File {} already exists, skipping download", file.first);
            continue;
        }
        
        // Download file
        log_info("Downloading {}", file.second);
        success = download_file(file.second, local_path);
        
        // If download fails, try the next file (could be different model format)
        if (!success) {
            log_warn("Failed to download {}, will try alternative formats if available", file.first);
            continue;
        }
    }
    
    return success;
}

bool HuggingFaceModelLoader::download_file(const std::string& url, const std::string& local_path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        log_error("Failed to initialize CURL");
        return false;
    }
    
    FILE* fp = fopen(local_path.c_str(), "wb");
    if (!fp) {
        log_error("Failed to open file {} for writing", local_path);
        curl_easy_cleanup(curl);
        return false;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "neuronet/0.1.0");
    
    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    
    // Check for errors
    bool success = (res == CURLE_OK);
    if (!success) {
        log_error("Failed to download {}: {}", url, curl_easy_strerror(res));
        // Remove the potentially partially downloaded file
        fclose(fp);
        fs::remove(local_path);
    } else {
        fclose(fp);
    }
    
    curl_easy_cleanup(curl);
    return success;
}

std::unordered_map<std::string, std::string> HuggingFaceModelLoader::load_config(const std::string& config_path) {
    std::unordered_map<std::string, std::string> config;
    
    try {
        // Read config file
        std::ifstream file(config_path);
        if (!file.is_open()) {
            log_error("Failed to open config file: {}", config_path);
            return config;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        
        // Parse JSON
        auto json = utils::parse_json(buffer.str());
        
        // Convert JSON to config map
        for (auto& [key, value] : json.items()) {
            if (value.is_string()) {
                config[key] = value.get<std::string>();
            } else if (value.is_number_integer()) {
                config[key] = std::to_string(value.get<int>());
            } else if (value.is_number_float()) {
                config[key] = std::to_string(value.get<float>());
            } else if (value.is_boolean()) {
                config[key] = value.get<bool>() ? "true" : "false";
            }
        }
        
        log_info("Loaded model config with {} parameters", config.size());
    } catch (const std::exception& e) {
        log_error("Error loading config: {}", e.what());
    }
    
    return config;
}

std::unordered_map<std::string, Tensor> HuggingFaceModelLoader::convert_weights(const std::string& weights_path) {
    std::unordered_map<std::string, Tensor> weights;
    
    // Determine weight format based on file extension
    std::string ext = fs::path(weights_path).extension();
    if (ext == ".bin") {
        weights = read_pytorch_weights(weights_path);
    } else if (ext == ".safetensors") {
        weights = read_safetensors(weights_path);
    } else {
        log_error("Unsupported weight format: {}", weights_path);
    }
    
    return weights;
}

std::unordered_map<std::string, Tensor> HuggingFaceModelLoader::read_pytorch_weights(const std::string& path) {
    log_info("Loading PyTorch weights not fully implemented: {}", path);
    // This is a placeholder - full implementation would require linking with PyTorch 
    // or implementing the torch serialization format
    
    std::unordered_map<std::string, Tensor> weights;
    return weights;
}

std::unordered_map<std::string, Tensor> HuggingFaceModelLoader::read_safetensors(const std::string& path) {
    log_info("Loading safetensors weights not fully implemented: {}", path);
    // This is a placeholder - full implementation would require parsing safetensors format
    
    std::unordered_map<std::string, Tensor> weights;
    return weights;
}

// HuggingFaceModel implementation
HuggingFaceModel::HuggingFaceModel() : Model() {}

std::shared_ptr<HuggingFaceModel> HuggingFaceModel::from_pretrained(
    const std::string& model_id,
    const std::string& cache_dir,
    DeviceType device_type) {
    
    // Determine cache directory
    std::string model_cache_dir = cache_dir;
    if (model_cache_dir.empty()) {
        const char* home_dir = getenv("HOME");
        if (!home_dir) {
            home_dir = getenv("USERPROFILE"); // Windows
        }
        
        if (home_dir) {
            model_cache_dir = std::string(home_dir) + "/.cache/neuronet/models/" + model_id;
        } else {
            model_cache_dir = "/tmp/neuronet/models/" + model_id;
        }
    }
    
    // Download model if needed
    HuggingFaceModelLoader loader;
    bool download_success = loader.download(model_id, model_cache_dir);
    
    if (!download_success) {
        log_error("Failed to download all model files");
    }
    
    // Load model config
    std::string config_path = model_cache_dir + "/config.json";
    auto config = loader.load_config(config_path);
    
    // Create appropriate model based on configuration
    auto model = create_model_from_config(config);
    if (!model) {
        log_error("Failed to create model from config");
        return nullptr;
    }
    
    // Load weights
    std::string weights_path;
    if (fs::exists(model_cache_dir + "/model.safetensors")) {
        weights_path = model_cache_dir + "/model.safetensors";
    } else if (fs::exists(model_cache_dir + "/pytorch_model.bin")) {
        weights_path = model_cache_dir + "/pytorch_model.bin";
    } else {
        log_error("No model weights found in {}", model_cache_dir);
        return model;
    }
    
    // Convert weights to neuronet format
    auto weights = loader.convert_weights(weights_path);
    
    // Load weights into model
    model->load_state_dict(weights);
    
    // Move model to desired device
    model->to(device_type);
    
    return model;
}

std::shared_ptr<HuggingFaceModel> HuggingFaceModel::create_model_from_config(
    const std::unordered_map<std::string, std::string>& config) {
    
    // Determine the model type from config
    auto it = config.find("model_type");
    if (it == config.end()) {
        log_error("Config doesn't contain 'model_type'");
        return nullptr;
    }
    
    std::string model_type = it->second;
    
    if (model_type == "bert") {
        return std::make_shared<BertModel>(config);
    } else if (model_type == "gpt2") {
        return std::make_shared<GPT2Model>(config);
    } else {
        log_error("Unsupported model type: {}", model_type);
        return nullptr;
    }
}

// BertModel implementation
BertModel::BertModel(const std::unordered_map<std::string, std::string>& config) 
    : HuggingFaceModel() {
    
    model_type_ = "bert";
    config_ = config;
    
    try {
        hidden_size_ = std::stoi(config.at("hidden_size"));
        num_hidden_layers_ = std::stoi(config.at("num_hidden_layers"));
        num_attention_heads_ = std::stoi(config.at("num_attention_heads"));
        intermediate_size_ = std::stoi(config.at("intermediate_size"));
    } catch (const std::exception& e) {
        log_error("Error parsing BERT config: {}", e.what());
        return;
    }
    
    // Initialize BERT model architecture - simplified implementation
    // A full implementation would create a proper neural network matching the BERT architecture
    // with embedding layers, attention layers, etc.
    log_info("Initialized BERT model with {} hidden layers, {} attention heads", 
             num_hidden_layers_, num_attention_heads_);
}

Tensor BertModel::forward(const Tensor& input) {
    // Placeholder for actual BERT forward processing logic
    log_info("BERT model forward pass with input shape: {}", 
             input.shape().empty() ? "[]" : std::to_string(input.shape()[0]));
    
    // For now, just return a dummy tensor
    std::vector<int64_t> output_shape = {input.shape()[0], hidden_size_};
    return Tensor(output_shape, DType::Float32, input.device().type());
}

// GPT2Model implementation
GPT2Model::GPT2Model(const std::unordered_map<std::string, std::string>& config) 
    : HuggingFaceModel() {
    
    model_type_ = "gpt2";
    config_ = config;
    
    try {
        hidden_size_ = std::stoi(config.at("n_embd"));
        num_hidden_layers_ = std::stoi(config.at("n_layer"));
        num_attention_heads_ = std::stoi(config.at("n_head"));
        // GPT2 uses a different parameter name for this
        intermediate_size_ = std::stoi(config.at("n_inner")); 
    } catch (const std::exception& e) {
        log_error("Error parsing GPT2 config: {}", e.what());
        return;
    }
    
    // Initialize GPT2 model architecture - simplified implementation
    log_info("Initialized GPT2 model with {} hidden layers, {} attention heads", 
             num_hidden_layers_, num_attention_heads_);
}

Tensor GPT2Model::forward(const Tensor& input) {
    // Placeholder for actual GPT2 forward processing logic
    log_info("GPT2 model forward pass with input shape: {}", 
             input.shape().empty() ? "[]" : std::to_string(input.shape()[0]));
    
    // For now, just return a dummy tensor
    std::vector<int64_t> output_shape = {input.shape()[0], hidden_size_};
    return Tensor(output_shape, DType::Float32, input.device().type());
}

} // namespace models
} // namespace neuronet
