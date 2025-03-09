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
