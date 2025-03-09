#include <neuronet/utils/json.h>
#include <neuronet/utils/logging.h>
#include <fstream>

namespace neuronet {
namespace utils {

// Simple JSON parser (based on nlohmann::json)
json parse_json(const std::string& content) {
    try {
        return json::parse(content);
    } catch (const std::exception& e) {
        log_error("JSON parse error: {}", e.what());
        return json::object();
    }
}

json load_json_file(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            log_error("Failed to open JSON file: {}", filepath);
            return json::object();
        }
        
        return json::parse(file);
    } catch (const std::exception& e) {
        log_error("Error loading JSON file {}: {}", 
                 filepath, std::string(e.what()));
        return json::object();
    }
}

bool save_json_file(const std::string& filepath, const json& data, int indent) {
    try {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            log_error("Failed to open file for writing: {}", filepath);
            return false;
        }
        
        file << data.dump(indent);
        return true;
    } catch (const std::exception& e) {
        log_error("Error saving JSON file {}: {}", 
                 filepath, std::string(e.what()));
        return false;
    }
}

} // namespace utils
} // namespace neuronet
