#pragma once

#include <neuronet/models/model.h>
#include <string>
#include <vector>
#include <memory>
#include <curl/curl.h> // Add this include for curl_off_t type definition

namespace neuronet {
namespace models {

// Progress callback for CURL downloads
int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow);

// Class to download and load models from HuggingFace
class HuggingFaceModelLoader {
public:
    HuggingFaceModelLoader();
    
    // Download model from HuggingFace
    bool download(const std::string& model_id, const std::string& local_dir);
    
    // Load model configuration
    std::unordered_map<std::string, std::string> load_config(const std::string& config_path);
    
    // Convert model weights from HuggingFace format to neuronet format
    std::unordered_map<std::string, Tensor> convert_weights(const std::string& weights_path);
    
private:
    // Helper methods for downloading and extracting files
    bool download_file(const std::string& url, const std::string& local_path);
    bool extract_archive(const std::string& archive_path, const std::string& extract_dir);
    
    // Read specific file formats
    std::unordered_map<std::string, Tensor> read_pytorch_weights(const std::string& path);
    std::unordered_map<std::string, Tensor> read_safetensors(const std::string& path);
};

// Base class for HuggingFace models
class HuggingFaceModel : public Model {
public:
    HuggingFaceModel();
    virtual ~HuggingFaceModel() = default;
    
    // Load model from HuggingFace repository
    static std::shared_ptr<HuggingFaceModel> from_pretrained(
        const std::string& model_id,
        const std::string& cache_dir = "",
        DeviceType device_type = DeviceType::CPU);
        
    // Factory method to create specific model type
    static std::shared_ptr<HuggingFaceModel> create_model_from_config(
        const std::unordered_map<std::string, std::string>& config);

protected:
    std::string model_id_;
    std::string model_type_;
    std::unordered_map<std::string, std::string> config_;
};

// Specific HuggingFace model implementations
class BertModel : public HuggingFaceModel {
public:
    BertModel(const std::unordered_map<std::string, std::string>& config);
    ~BertModel() override = default;
    
    Tensor forward(const Tensor& input) override;
    
private:
    int hidden_size_;
    int num_hidden_layers_;
    int num_attention_heads_;
    int intermediate_size_;
};

class GPT2Model : public HuggingFaceModel {
public:
    GPT2Model(const std::unordered_map<std::string, std::string>& config);
    ~GPT2Model() override = default;
    
    Tensor forward(const Tensor& input) override;
    
private:
    int hidden_size_;
    int num_hidden_layers_;
    int num_attention_heads_;
    int intermediate_size_;
};

} // namespace models
} // namespace neuronet
