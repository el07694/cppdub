#ifndef CPPDUB_EXCEPTIONS_H
#define CPPDUB_EXCEPTIONS_H

#include <exception>
#include <string>

namespace cppdub {

// Base class for all exceptions in the cppdub library
class PyDubException : public std::exception {
public:
    explicit PyDubException(const std::string& message)
        : message_(message) {}

    virtual const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

// Derived exception classes
class FileNotFoundException : public PyDubException {
public:
    explicit FileNotFoundException(const std::string& message)
        : PyDubException("File Not Found: " + message) {}
};

class InvalidParameterException : public PyDubException {
public:
    explicit InvalidParameterException(const std::string& message)
        : PyDubException("Invalid Parameter: " + message) {}
};

class OperationFailedException : public PyDubException {
public:
    explicit OperationFailedException(const std::string& message)
        : PyDubException("Operation Failed: " + message) {}
};


class TooManyMissingFrames : public PyDubException {
public:
    explicit TooManyMissingFrames(const std::string& message)
        : PyDubException("File Not Found: " + message) {}
};

class InvalidDuration : public PyDubException {
public:
    explicit InvalidDuration(const std::string& message)
        : PyDubException("File Not Found: " + message) {}
};

class InvalidTag : public PyDubException {
public:
    explicit InvalidTag(const std::string& message)
        : PyDubException("File Not Found: " + message) {}
};

class InvalidID3TagVersion : public PyDubException {
public:
    explicit InvalidID3TagVersion(const std::string& message)
        : PyDubException("File Not Found: " + message) {}
};

class CouldntDecodeError : public PyDubException {
public:
    explicit CouldntDecodeError(const std::string& message)
        : PyDubException("File Not Found: " + message) {}
};

class CouldntEncodeError : public PyDubException {
public:
    explicit CouldntEncodeError(const std::string& message)
        : PyDubException("File Not Found: " + message) {}
};

class MissingAudioParameter : public PyDubException {
public:
    explicit MissingAudioParameter(const std::string& message)
        : PyDubException("File Not Found: " + message) {}
};

} // namespace cppdub

#endif // CPPDUB_EXCEPTIONS_H
