#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace neuronet {
namespace utils {

// Use nlohmann::json as our JSON implementation
using json = nlohmann::json;

// Parse JSON string
json parse_json(const std::string& content);

// Load JSON from file
json load_json_file(const std::string& filepath);

// Save JSON to file
bool save_json_file(const std::string& filepath, const json& data, int indent = 4);

} // namespace utils
} // namespace neuronet
