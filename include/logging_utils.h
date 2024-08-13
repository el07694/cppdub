#ifndef CPPDUB_LOGGING_UTILS_H
#define CPPDUB_LOGGING_UTILS_H

#include <string>
#include <iostream>

namespace cppdub {

// Enum for log levels
enum class LogLevel {
    INFO,
    WARNING,
    ERROR
};

// Set the global log level
void set_log_level(LogLevel level);

// Log messages with various severity levels
void log_message(LogLevel level, const std::string& message);

} // namespace cppdub

#endif // CPPDUB_LOGGING_UTILS_H
