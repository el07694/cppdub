#include "logging_utils.h"
#include <fstream>
#include <mutex>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <iostream> // Added include for std::cerr

namespace cppdub {

class Logger {
public:
    // Constructor
    Logger(const std::string& logFile) : logFile(logFile), logLevel(LogLevel::INFO) {
        if (!logFile.empty()) {
            logStream.open(logFile, std::ios::out | std::ios::app);
            if (!logStream.is_open()) {
                std::cerr << "Error opening log file: " << logFile << std::endl;
            }
        }
    }

    // Destructor
    ~Logger() {
        if (logStream.is_open()) {
            logStream.close();
        }
    }

    // Set the logging level
    void set_log_level(LogLevel level) {
        std::lock_guard<std::mutex> guard(logMutex);
        logLevel = level;
    }

    // Log messages with various levels
    void log_message(LogLevel level, const std::string& message) {
        std::lock_guard<std::mutex> guard(logMutex);
        if (should_log(level)) {
            std::string formattedMessage = format_message(level, message);
            if (logStream.is_open()) {
                logStream << formattedMessage << std::endl;
            } else {
                std::cerr << formattedMessage << std::endl;
            }
        }
    }

private:
    std::string logFile;
    std::ofstream logStream;
    std::mutex logMutex;
    LogLevel logLevel;

    // Determine if the message should be logged based on level
    bool should_log(LogLevel level) const {
        return level >= logLevel;
    }

    // Format the message with timestamp and log level
    std::string format_message(LogLevel level, const std::string& message) {
        std::ostringstream oss;
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << " - ";
        switch (level) {
            case LogLevel::INFO:    oss << "INFO: "; break;
            case LogLevel::WARNING: oss << "WARNING: "; break;
            case LogLevel::ERROR:   oss << "ERROR: "; break;
        }
        oss << message;
        return oss.str();
    }
};

// Global Logger instance
Logger globalLogger("cppdub.log");

// Set the global log level
void set_log_level(LogLevel level) {
    globalLogger.set_log_level(level);
}

// Log messages with various severity levels
void log_message(LogLevel level, const std::string& message) {
    globalLogger.log_message(level, message);
}

} // namespace cppdub
