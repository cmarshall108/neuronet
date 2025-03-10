#include <neuronet/nlp/tokenizer.h>
#include <neuronet/utils/logging.h>
#include <neuronet/utils/json.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cctype>
#include <algorithm>

namespace fs = std::filesystem;

namespace neuronet {
namespace nlp {

Tensor Tokenizer::create_input_tensors(const std::string& text, int max_length, DeviceType device_type) const {
    // Convert text to token IDs
    std::vector<int64_t> token_ids = encode(text, max_length);
    
    // Create input tensor
    Tensor input_ids({1, static_cast<int64_t>(token_ids.size())}, token_ids.data(), DType::Int64, device_type);
    
    return input_ids;
}

BertTokenizer::BertTokenizer(const std::string& vocab_path) {
    load_vocab(vocab_path);
}

void BertTokenizer::load_vocab(const std::string& vocab_path) {
    log_info("Loading BERT vocabulary from: {}", vocab_path);
    
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        log_error("Failed to open vocabulary file: {}", vocab_path);
        return;
    }
    
    // Read vocab line by line
    std::string line;
    int64_t index = 0;
    while (std::getline(file, line)) {
        // Remove trailing whitespace
        line.erase(std::find_if(line.rbegin(), line.rend(), 
                                [](unsigned char ch) { return !std::isspace(ch); }).base(), 
                    line.end());
        
        if (!line.empty()) {
            vocab_[line] = index;
            id_to_token_.push_back(line);
            index++;
        }
    }
    
    log_info("Loaded vocabulary with {} tokens", std::to_string(vocab_.size()));
    
    // Make sure special tokens exist
    if (vocab_.find("[CLS]") != vocab_.end()) cls_token_id_ = vocab_["[CLS]"];
    if (vocab_.find("[SEP]") != vocab_.end()) sep_token_id_ = vocab_["[SEP]"];
    if (vocab_.find("[PAD]") != vocab_.end()) pad_token_id_ = vocab_["[PAD]"];
    if (vocab_.find("[UNK]") != vocab_.end()) unk_token_id_ = vocab_["[UNK]"];
}

std::vector<int64_t> BertTokenizer::encode(const std::string& text, int max_length) const {
    std::vector<int64_t> token_ids;
    
    // Start with [CLS] token
    token_ids.push_back(cls_token_id_);
    
    // Tokenize text
    std::vector<std::string> tokens = tokenize(text);
    
    // Convert tokens to IDs
    for (const auto& token : tokens) {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            token_ids.push_back(it->second);
        } else {
            // Handle unknown tokens - try to split into subwords
            bool added = false;
            if (token.length() > 1) {
                // Try character by character fallback
                for (char c : token) {
                    std::string char_token(1, c);
                    auto char_it = vocab_.find(char_token);
                    if (char_it != vocab_.end()) {
                        token_ids.push_back(char_it->second);
                        added = true;
                    } else {
                        token_ids.push_back(unk_token_id_);
                        added = true;
                    }
                }
            }
            
            if (!added) {
                token_ids.push_back(unk_token_id_);
            }
        }
    }
    
    // Add [SEP] token
    token_ids.push_back(sep_token_id_);
    
    // Handle max_length constraint
    if (max_length > 0) {
        if (token_ids.size() > max_length) {
            // Truncate, but keep [CLS] and [SEP]
            token_ids.resize(max_length - 1);
            token_ids.push_back(sep_token_id_);
        } else if (token_ids.size() < max_length) {
            // Pad to max_length
            token_ids.resize(max_length, pad_token_id_);
        }
    }
    
    return token_ids;
}

std::string BertTokenizer::decode(const std::vector<int64_t>& token_ids) const {
    std::ostringstream result;
    
    for (size_t i = 0; i < token_ids.size(); i++) {
        // Skip special tokens
        if (token_ids[i] == pad_token_id_ || 
            token_ids[i] == cls_token_id_ || 
            token_ids[i] == sep_token_id_) {
            continue;
        }
        
        // Get token string
        if (token_ids[i] >= 0 && token_ids[i] < id_to_token_.size()) {
            std::string token = id_to_token_[token_ids[i]];
            
            // Handle WordPiece tokens (remove ## prefix)
            if (token.substr(0, 2) == "##") {
                result << token.substr(2);
            } else if (i > 0 && token_ids[i-1] != cls_token_id_ && 
                       token_ids[i-1] != sep_token_id_ && !is_punctuation(token[0])) {
                // Add space before token unless it's the first token or punctuation
                result << " " << token;
            } else {
                result << token;
            }
        }
    }
    
    return result.str();
}

std::vector<std::string> BertTokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current_token;
    
    // Basic tokenization: split by whitespace and handle punctuation
    for (size_t i = 0; i < text.length(); i++) {
        char c = text[i];
        
        if (is_whitespace(c)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else if (is_punctuation(c)) {
            // Handle punctuation as separate tokens
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(std::string(1, c));
        } else {
            current_token += c;
        }
    }
    
    // Handle last token
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    // Apply WordPiece algorithm
    std::vector<std::string> wordpiece_tokens;
    
    for (const auto& token : tokens) {
        // Convert to lowercase
        std::string lower_token = token;
        std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), 
                       [](unsigned char c){ return std::tolower(c); });
        
        // Check if the whole token is in vocabulary
        if (vocab_.find(lower_token) != vocab_.end()) {
            wordpiece_tokens.push_back(lower_token);
            continue;
        }
        
        // Try to split into subwords
        bool is_bad = false;
        int start = 0;
        std::vector<std::string> sub_tokens;
        
        while (start < lower_token.length()) {
            int end = lower_token.length();
            std::string cur_substr;
            bool found = false;
            
            while (start < end) {
                std::string substr = lower_token.substr(start, end-start);
                if (start > 0) {
                    substr = "##" + substr;
                }
                
                if (vocab_.find(substr) != vocab_.end()) {
                    cur_substr = substr;
                    found = true;
                    break;
                }
                end--;
            }
            
            if (!found) {
                is_bad = true;
                break;
            }
            
            sub_tokens.push_back(cur_substr);
            start = end;
        }
        
        if (is_bad) {
            // If we couldn't break the word into subwords, use [UNK]
            wordpiece_tokens.push_back("[UNK]");
        } else {
            // Add all subword tokens
            wordpiece_tokens.insert(wordpiece_tokens.end(), sub_tokens.begin(), sub_tokens.end());
        }
    }
    
    return wordpiece_tokens;
}

bool BertTokenizer::is_punctuation(char c) const {
    return std::ispunct(c) || c == '`';
}

bool BertTokenizer::is_whitespace(char c) const {
    return std::isspace(c) || c == '\t' || c == '\n' || c == '\r';
}

std::shared_ptr<Tokenizer> create_tokenizer_for_model(const std::string& model_id, const std::string& cache_dir) {
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
    
    // Check what type of model and load appropriate tokenizer
    std::string vocab_path = model_cache_dir + "/vocab.txt";
    
    if (model_id.find("bert") != std::string::npos) {
        return std::make_shared<BertTokenizer>(vocab_path);
    }
    
    // Default to BERT tokenizer for now
    return std::make_shared<BertTokenizer>(vocab_path);
}

} // namespace nlp
} // namespace neuronet
