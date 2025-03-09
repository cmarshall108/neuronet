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
static bool g_color_enabled = true;

// ANSI color codes
const std::string RESET = "\033[0m";
const std::string BLACK = "\033[30m";
const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN = "\033[36m";
const std::string WHITE = "\033[37m";
const std::string BOLD = "\033[1m";

void set_log_color_enabled(bool enabled) {
    g_color_enabled = enabled;
}

bool is_log_color_enabled() {
    return g_color_enabled;
}

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

std::string get_level_color(LogLevel level) {
    if (!g_color_enabled) {
        return "";
    }
    
    switch (level) {
        case LogLevel::Debug:   return CYAN;     // Cyan for debug messages
        case LogLevel::Info:    return GREEN;    // Green for info
        case LogLevel::Warning: return YELLOW;   // Yellow for warnings
        case LogLevel::Error:   return RED;      // Red for errors
        case LogLevel::Fatal:   return BOLD + RED; // Bold red for fatal errors
        default:                return "";
    }
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
    
    std::string color = get_level_color(level);
    std::string reset = g_color_enabled ? RESET : "";
    
    // Format: [timestamp][level_colored] message
    std::cerr << "[" << get_timestamp() << "]" << color << "[" << level_str << "]" << reset << " " << message << std::endl;
    
    if (level == LogLevel::Fatal) {
        std::abort();
    }
}

// Modified log functions to accept variadic arguments directly
void log_debug(const std::string& format) {
    log_message(LogLevel::Debug, format);
}

void log_debug(const std::string& format, const std::string& arg) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg);
    }
    log_message(LogLevel::Debug, message);
}

void log_debug(const std::string& format, const std::string& arg1, const std::string& arg2) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg1);
        pos = message.find("{}", pos + arg1.length());
        if (pos != std::string::npos) {
            message.replace(pos, 2, arg2);
        }
    }
    log_message(LogLevel::Debug, message);
}

void log_info(const std::string& format) {
    log_message(LogLevel::Info, format);
}

void log_info(const std::string& format, const std::string& arg) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg);
    }
    log_message(LogLevel::Info, message);
}

void log_info(const std::string& format, const std::string& arg1, const std::string& arg2) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg1);
        pos = message.find("{}", pos + arg1.length());
        if (pos != std::string::npos) {
            message.replace(pos, 2, arg2);
        }
    }
    log_message(LogLevel::Info, message);
}

// Add the new 3-argument version for log_info
void log_info(const std::string& format, const std::string& arg1, const std::string& arg2, const std::string& arg3) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg1);
        pos = message.find("{}", pos + arg1.length());
        if (pos != std::string::npos) {
            message.replace(pos, 2, arg2);
            pos = message.find("{}", pos + arg2.length());
            if (pos != std::string::npos) {
                message.replace(pos, 2, arg3);
            }
        }
    }
    log_message(LogLevel::Info, message);
}

// For longer argument lists (like Conv2d creation)
void log_info(const std::string& format, const std::string& arg1, const std::string& arg2,
              const std::string& arg3, const std::string& arg4, const std::string& arg5, 
              const std::string& arg6, const std::string& arg7) {
    std::string message = format;
    size_t pos = 0;
    
    if ((pos = message.find("{}")) != std::string::npos) {
        message.replace(pos, 2, arg1);
        if ((pos = message.find("{}")) != std::string::npos) {
            message.replace(pos, 2, arg2);
            if ((pos = message.find("{}")) != std::string::npos) {
                message.replace(pos, 2, arg3);
                if ((pos = message.find("{}")) != std::string::npos) {
                    message.replace(pos, 2, arg4);
                    if ((pos = message.find("{}")) != std::string::npos) {
                        message.replace(pos, 2, arg5);
                        if ((pos = message.find("{}")) != std::string::npos) {
                            message.replace(pos, 2, arg6);
                            if ((pos = message.find("{}")) != std::string::npos) {
                                message.replace(pos, 2, arg7);
                            }
                        }
                    }
                }
            }
        }
    }
    
    log_message(LogLevel::Info, message);
}

void log_warn(const std::string& format) {
    log_message(LogLevel::Warning, format);
}

void log_warn(const std::string& format, const std::string& arg) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg);
    }
    log_message(LogLevel::Warning, message);
}

void log_error(const std::string& format) {
    log_message(LogLevel::Error, format);
}

void log_error(const std::string& format, const std::string& arg) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg);
    }
    log_message(LogLevel::Error, message);
}

void log_error(const std::string& format, const std::string& arg1, const std::string& arg2) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg1);
        pos = message.find("{}", pos + arg1.length());
        if (pos != std::string::npos) {
            message.replace(pos, 2, arg2);
        }
    }
    log_message(LogLevel::Error, message);
}

void log_error(const std::string& format, const std::string& arg1, const std::string& arg2, const std::string& arg3) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg1);
        pos = message.find("{}", pos + arg1.length());
        if (pos != std::string::npos) {
            message.replace(pos, 2, arg2);
            pos = message.find("{}", pos + arg2.length());
            if (pos != std::string::npos) {
                message.replace(pos, 2, arg3);
            }
        }
    }
    log_message(LogLevel::Error, message);
}

void log_error(const std::string& format, const std::string& arg1, const std::string& arg2, 
               const std::string& arg3, const std::string& arg4) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg1);
        pos = message.find("{}", pos + arg1.length());
        if (pos != std::string::npos) {
            message.replace(pos, 2, arg2);
            pos = message.find("{}", pos + arg2.length());
            if (pos != std::string::npos) {
                message.replace(pos, 2, arg3);
                pos = message.find("{}", pos + arg3.length());
                if (pos != std::string::npos) {
                    message.replace(pos, 2, arg4);
                }
            }
        }
    }
    log_message(LogLevel::Error, message);
}

void log_fatal(const std::string& format) {
    log_message(LogLevel::Fatal, format);
}

void log_fatal(const std::string& format, const std::string& arg) {
    std::string message = format;
    size_t pos = message.find("{}");
    if (pos != std::string::npos) {
        message.replace(pos, 2, arg);
    }
    log_message(LogLevel::Fatal, message);
}

// Keeping this for backward compatibility, but we'll use direct string versions above
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
