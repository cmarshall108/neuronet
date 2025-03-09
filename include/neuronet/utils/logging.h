#pragma once

#include <string>
#include <vector>

namespace neuronet {

// Log levels
enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Fatal
};

// Enable or disable colored output
void set_log_color_enabled(bool enabled);
bool is_log_color_enabled();

// Set/get global log level
void set_log_level(LogLevel level);
LogLevel get_log_level();

// Get current timestamp as string
std::string get_timestamp();

// Get color code for log level
std::string get_level_color(LogLevel level);

// Log message at specific level
void log_message(LogLevel level, const std::string& message);

// Format string with arguments
std::string format_string(const std::string& format, const std::vector<std::string>& args);

// Helper functions for different log levels - simplified versions
void log_debug(const std::string& format);
void log_debug(const std::string& format, const std::string& arg);
void log_debug(const std::string& format, const std::string& arg1, const std::string& arg2);

void log_info(const std::string& format);
void log_info(const std::string& format, const std::string& arg);
void log_info(const std::string& format, const std::string& arg1, const std::string& arg2);
// Add the missing 3-argument version
void log_info(const std::string& format, const std::string& arg1, const std::string& arg2, const std::string& arg3);
// Special version for Conv2d layer creation that has 7 parameters
void log_info(const std::string& format, const std::string& arg1, const std::string& arg2,
              const std::string& arg3, const std::string& arg4, const std::string& arg5, 
              const std::string& arg6, const std::string& arg7);

void log_warn(const std::string& format);
void log_warn(const std::string& format, const std::string& arg);

void log_error(const std::string& format);
void log_error(const std::string& format, const std::string& arg);
void log_error(const std::string& format, const std::string& arg1, const std::string& arg2);
void log_error(const std::string& format, const std::string& arg1, const std::string& arg2, const std::string& arg3);
void log_error(const std::string& format, const std::string& arg1, const std::string& arg2, 
               const std::string& arg3, const std::string& arg4);

void log_fatal(const std::string& format);
void log_fatal(const std::string& format, const std::string& arg);

// Legacy version with vector args
void log_debug(const std::string& format, const std::vector<std::string>& args);
void log_info(const std::string& format, const std::vector<std::string>& args);
void log_warn(const std::string& format, const std::vector<std::string>& args);
void log_error(const std::string& format, const std::vector<std::string>& args);
void log_fatal(const std::string& format, const std::vector<std::string>& args);

} // namespace neuronet
