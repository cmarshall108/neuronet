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

// Set/get global log level
void set_log_level(LogLevel level);
LogLevel get_log_level();

// Get current timestamp as string
std::string get_timestamp();

// Log message at specific level
void log_message(LogLevel level, const std::string& message);

// Format string with arguments
std::string format_string(const std::string& format, const std::vector<std::string>& args);

// Helper functions for different log levels
void log_debug(const std::string& format, const std::vector<std::string>& args = {});
void log_info(const std::string& format, const std::vector<std::string>& args = {});
void log_warn(const std::string& format, const std::vector<std::string>& args = {});
void log_error(const std::string& format, const std::vector<std::string>& args = {});
void log_fatal(const std::string& format, const std::vector<std::string>& args = {});

// Convenience macros for variadic logging
#define LOG_DEBUG(fmt, ...) log_debug(fmt, {__VA_ARGS__})
#define LOG_INFO(fmt, ...) log_info(fmt, {__VA_ARGS__})
#define LOG_WARN(fmt, ...) log_warn(fmt, {__VA_ARGS__})
#define LOG_ERROR(fmt, ...) log_error(fmt, {__VA_ARGS__})
#define LOG_FATAL(fmt, ...) log_fatal(fmt, {__VA_ARGS__})

} // namespace neuronet
