#pragma once

#include <neuronet/core/tensor.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace neuronet {
namespace nlp {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    
    // Convert text to token IDs
    virtual std::vector<int64_t> encode(const std::string& text, int max_length = 0) const = 0;
    
    // Convert token IDs back to text
    virtual std::string decode(const std::vector<int64_t>& token_ids) const = 0;
    
    // Get the vocabulary size
    virtual size_t vocab_size() const = 0;
    
    // Get special token IDs
    virtual int64_t cls_token_id() const = 0;
    virtual int64_t sep_token_id() const = 0;
    virtual int64_t pad_token_id() const = 0;
    virtual int64_t unk_token_id() const = 0;
    
    // Create tensors for model input
    virtual Tensor create_input_tensors(const std::string& text, int max_length, DeviceType device_type) const;
};

// BERT Tokenizer implementation specifically for BERT models
class BertTokenizer : public Tokenizer {
public:
    BertTokenizer(const std::string& vocab_path);
    
    std::vector<int64_t> encode(const std::string& text, int max_length = 0) const override;
    std::string decode(const std::vector<int64_t>& token_ids) const override;
    
    size_t vocab_size() const override { return vocab_.size(); }
    
    int64_t cls_token_id() const override { return cls_token_id_; }
    int64_t sep_token_id() const override { return sep_token_id_; }
    int64_t pad_token_id() const override { return pad_token_id_; }
    int64_t unk_token_id() const override { return unk_token_id_; }

private:
    // Tokenizes text into subword units
    std::vector<std::string> tokenize(const std::string& text) const;
    
    // Maps from word piece to token ID
    std::unordered_map<std::string, int64_t> vocab_;
    
    // Reverse mapping from token ID to word piece
    std::vector<std::string> id_to_token_;
    
    // Special token IDs
    int64_t cls_token_id_ = 101; // [CLS]
    int64_t sep_token_id_ = 102; // [SEP]
    int64_t pad_token_id_ = 0;   // [PAD]
    int64_t unk_token_id_ = 100; // [UNK]
    
    // Helper for wordpiece tokenization
    bool is_punctuation(char c) const;
    bool is_whitespace(char c) const;
    
    // Load vocabulary from file
    void load_vocab(const std::string& vocab_path);
};

// GPT-2 Tokenizer implementation specifically for GPT-2 models
class GPT2Tokenizer : public Tokenizer {
public:
    GPT2Tokenizer(const std::string& vocab_path);
    
    std::vector<int64_t> encode(const std::string& text, int max_length = 0) const override;
    std::string decode(const std::vector<int64_t>& token_ids) const override;
    
    size_t vocab_size() const override { return vocab_.size(); }
    
    int64_t cls_token_id() const override { return -1; } // Not used in GPT-2
    int64_t sep_token_id() const override { return -1; } // Not used in GPT-2
    int64_t pad_token_id() const override { return pad_token_id_; }
    int64_t unk_token_id() const override { return unk_token_id_; }

private:
    // Tokenizes text into subword units
    std::vector<std::string> tokenize(const std::string& text) const;
    
    // Maps from word piece to token ID
    std::unordered_map<std::string, int64_t> vocab_;
    
    // Reverse mapping from token ID to word piece
    std::vector<std::string> id_to_token_;
    
    // Special token IDs
    int64_t pad_token_id_ = 50256;   // [PAD]
    int64_t unk_token_id_ = 50257;   // [UNK]
    
    // Load vocabulary from file
    void load_vocab(const std::string& vocab_path);
};

// Create a tokenizer based on model type
std::shared_ptr<Tokenizer> create_tokenizer_for_model(const std::string& model_id, const std::string& cache_dir = "");

} // namespace nlp
} // namespace neuronet
