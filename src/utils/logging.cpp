#include <neuronet/utils/logging.h>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <mutex>

namespace neuronet {

// Global log level
static LogLevel g_log_level = LogLevel::Info;
static std::mutex g_log_mutex;

void set_log_level(LogLevel level) {
    g_log_level = level;
}

LogLevel get_log_level() {
    return g_log_level;
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

void log_message(LogLevel level, const std::string& message) {
    if (level < g_log_level) {
        return;
    }
    
    std::string level_str;
    switch (level) {
        case LogLevel::Debug: level_str = "DEBUG"; break;
        case LogLevel::Info: level_str = "INFO"; break;
        case LogLevel::Warning: level_str = "WARN"; break;
        case LogLevel::Error: level_str = "ERROR"; break;
        case LogLevel::Fatal: level_str = "FATAL"; break;
    }
    
    std::lock_guard<std::mutex> lock(g_log_mutex);
    
    std::cerr << "[" << get_timestamp() << "][" << level_str << "] " << message << std::endl;
    
    if (level == LogLevel::Fatal) {
        std::abort();
    }
}

// Log implementations for different severity levels
void log_debug(const std::string& format, const std::vector<std::string>& args) {
    log_message(LogLevel::Debug, format_string(format, args));
}

void log_info(const std::string& format, const std::vector<std::string>& args) {
    log_message(LogLevel::Info, format_string(format, args));
}

void log_warn(const std::string& format, const std::vector<std::string>& args) {
    log_message(LogLevel::Warning, format_string(format, args));
}

void log_error(const std::string& format, const std::vector<std::string>& args) {
    log_message(LogLevel::Error, format_string(format, args));
}

void log_fatal(const std::string& format, const std::vector<std::string>& args) {
    log_message(LogLevel::Fatal, format_string(format, args));
}

std::string format_string(const std::string& format, const std::vector<std::string>& args) {
    std::string result = format;
    size_t pos = 0;
    size_t arg_idx = 0;
    
    while ((pos = result.find("{}", pos)) != std::string::npos && arg_idx < args.size()) {
        result.replace(pos, 2, args[arg_idx++]);
        pos += args[arg_idx - 1].length();
    }
    
    return result;
}

} // namespace neuronet
