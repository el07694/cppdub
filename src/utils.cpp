#ifndef UTILS_CPP
#define UTILS_CPP
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>
#include <cmath>
#include <stdexcept>
#include <map>
#include <array>
#include <vector>
#include <string>
#include <memory>
#include <cstdio>
#include <utility> // for std::pair
#include <cerrno>
#include <functional>
#include <nlohmann/json.hpp> // Include the json library
#include <set>

#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define access _access
#define F_OK 0
#define X_OK 1
#else
#include <unistd.h>
#endif

#include <cstring>
#include <cstdlib>
#include <unordered_map>

namespace cppdub{

// Constants
const std::map<int, int> FRAME_WIDTHS = {
    {8, 1},
    {16, 2},
    {32, 4}
};

const std::map<int, std::string> ARRAY_TYPES = {
    {8, "b"},
    {16, "h"},
    {32, "i"}
};

const std::map<int, std::pair<int, int>> ARRAY_RANGES = {
    {8, {-128, 127}},                      // 8-bit
    {16, {-32768, 32767}},                 // 16-bit
    {32, {-2147483648, 2147483647}}       // 32-bit
};

// Functions
int get_frame_width(int bit_depth) {
    auto it = cppdub::FRAME_WIDTHS.find(bit_depth);
    if (it != cppdub::FRAME_WIDTHS.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unsupported bit depth");
}

std::string get_array_type(int bit_depth, bool signed_type) {
    auto it = cppdub::ARRAY_TYPES.find(bit_depth);
    if (it != cppdub::ARRAY_TYPES.end()) {
        std::string type = it->second;
        if (!signed_type) {
            type = static_cast<char>(std::toupper(type[0])) + type.substr(1);  // Convert first letter to uppercase
        }
        return type;
    }
    throw std::invalid_argument("Unsupported bit depth");
}

std::pair<int, int> get_min_max_value(int bit_depth) {
    auto it = cppdub::ARRAY_RANGES.find(bit_depth);
    if (it != cppdub::ARRAY_RANGES.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unsupported bit depth");
}

// Custom function to handle file-like operations similar to Python
std::pair<std::unique_ptr<std::FILE, decltype(&std::fclose)>, bool> _fd_or_path_or_tempfile(int* fd, const std::string& path, bool tempfile, const std::string& mode) {
    
    bool close_fd = false;

    // Handle the case where fd is nullptr and tempfile is true (create a temporary file)
    if (fd == nullptr && tempfile) {
        std::unique_ptr<std::FILE, decltype(&std::fclose)> temp_file(std::tmpfile(), std::fclose);
        if (!temp_file) {
            throw std::runtime_error("Unable to create temporary file");
        }
        close_fd = true;
        return {std::move(temp_file), close_fd};
    }

    // If fd is a valid file descriptor, use it
    if (fd != nullptr && *fd >= 0) {
#ifdef _WIN32
        // Convert Windows file descriptor to FILE* using _fdopen
        std::unique_ptr<std::FILE, decltype(&std::fclose)> file(_fdopen(*fd, mode.c_str()), std::fclose);
#else
        // Convert POSIX file descriptor to FILE* using fdopen
        std::unique_ptr<std::FILE, decltype(&std::fclose)> file(fdopen(*fd, mode.c_str()), std::fclose);
#endif
        if (!file) {
            throw std::runtime_error("Unable to open file from file descriptor");
        }
        return {std::move(file), close_fd};
    }

    // If a path is provided, open the file at the given path
    if (!path.empty()) {
        std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::fopen(path.c_str(), mode.c_str()), std::fclose);
        if (!file) {
            throw std::runtime_error("Unable to open file from path: " + path);
        }
        close_fd = true;
        return {std::move(file), close_fd};
    }

    // If neither valid fd nor path is provided, throw an error
    throw std::invalid_argument("Invalid arguments provided: fd is null, path is empty, and tempfile is false");
}

/* test previous function:
int main() {
    try {
        // Test cases

        // Case 1: Create a temporary file
        auto result = _fd_or_path_or_tempfile(nullptr, "", true);
        std::cout << "Temporary file created, should close: " << result.second << std::endl;

        // Case 2: Use a valid file descriptor
        int fd;
#ifdef _WIN32
        fd = _open("example.txt", _O_RDWR | _O_CREAT, _S_IREAD | _S_IWRITE);
#else
        fd = open("example.txt", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
#endif
        if (fd < 0) {
            throw std::runtime_error("Failed to open file descriptor for 'example.txt'");
        }
        auto result_fd = _fd_or_path_or_tempfile(&fd);
        std::cout << "File descriptor used, should close: " << result_fd.second << std::endl;
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif

        // Case 3: Open a file by path
        auto result_path = _fd_or_path_or_tempfile(nullptr, "example.txt", false);
        std::cout << "File opened from path, should close: " << result_path.second << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
*/

float db_to_float(float db, bool using_amplitude) {
    if (using_amplitude) {
        return std::pow(10.0, db / 20.0);
    } else { // using power
        return std::pow(10.0, db / 10.0);
    }
}

float ratio_to_db(float ratio, float val2, bool using_amplitude) {
    // Handle ratio of zero
    if (ratio == 0) {
        return -std::numeric_limits<float>::infinity();
    }

    // Handle case where two values are given
    if (val2 != 0.0f) {
        ratio /= val2;
    }

    // Use appropriate formula based on amplitude or power
    if (using_amplitude) {
        return 20.0f * std::log10(ratio);
    } else { // using power
        return 10.0f * std::log10(ratio);
    }
}


std::vector<cppdub::AudioSegment> make_chunks(const cppdub::AudioSegment& audio_segment, size_t chunk_length_ms) {
    std::vector<cppdub::AudioSegment> chunks;
    size_t total_length = audio_segment.length_in_milliseconds();

    if (chunk_length_ms == 0) {
        throw std::invalid_argument("Chunk length must be greater than zero.");
    }

    size_t number_of_chunks = std::ceil(static_cast<double>(total_length) / chunk_length_ms);

    for (size_t i = 0; i < number_of_chunks; ++i) {
        size_t start = i * chunk_length_ms;
        size_t end = std::min(start + chunk_length_ms, total_length);
        
        // Get the slice using the get_sample_slice method
        uint32_t start_sample = static_cast<uint32_t>((static_cast<int>(start) * audio_segment.get_frame_rate()) / 1000);
        uint32_t end_sample = static_cast<uint32_t>((static_cast<int>(end) * audio_segment.get_frame_rate()) / 1000);
        std::vector<char> slice_123 = audio_segment.get_sample_slice(start_sample, end_sample);
        chunks.push_back(audio_segment._spawn(slice_123));
    }

    return chunks;
}

std::string which(const std::string& program) {
    // Add .exe extension for Windows if not already present
    std::string cmd = program;
#ifdef _WIN32
    if (cmd.find(".exe") == std::string::npos) {
        cmd += ".exe";
    }
#endif

    // Get the PATH environment variable
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return "not found";
    }

    // Split the PATH into directories
    std::vector<std::string> directories;
    std::string path_env_str(path_env);
    std::string delimiter;
    
#ifdef _WIN32
    delimiter = ";"; // Windows uses semicolon as the PATH separator
#else
    delimiter = ":"; // Unix-like systems use colon as the PATH separator
#endif

    size_t pos = 0;
    while ((pos = path_env_str.find(delimiter)) != std::string::npos) {
        directories.push_back(path_env_str.substr(0, pos));
        path_env_str.erase(0, pos + delimiter.length());
    }
    directories.push_back(path_env_str); // Add the last directory

    // Check each directory for the program
    for (const auto& dir : directories) {
        std::string full_path = dir + "/" + cmd;

#ifdef _WIN32
        // Windows uses backslashes in paths
        std::replace(full_path.begin(), full_path.end(), '/', '\\');
#endif

        // Check if the file exists and is executable
        if (access(full_path.c_str(), F_OK) == 0 && access(full_path.c_str(), X_OK) == 0) {
            return full_path;
        }
    }

    return "not found";
}

std::string get_encoder_name() {
    // Check for the presence of avconv
    if (which("avconv") != "not found") {
        return "avconv";
    }
    // Check for the presence of ffmpeg
    else if (which("ffmpeg") != "not found") {
        return "ffmpeg";
    }
    else {
        // Warn and default to ffmpeg
        std::cerr << "Warning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work" << std::endl;
        return "ffmpeg";
    }
}

std::string get_player_name() {
    // Check for the presence of avplay
    if (which("avplay") != "not found") {
        return "avplay";
    }
    // Check for the presence of ffplay
    else if (which("ffplay") != "not found") {
        return "ffplay";
    }
    else {
        // Warn and default to ffplay
        std::cerr << "Couldn't find ffplay or avplay - defaulting to ffplay, but may not work" << std::endl;
        return "ffplay";
    }
}

std::string get_prober_name() {
    // Check for the presence of avprobe
    if (which("avprobe") != "not found") {
        return "avprobe";
    }
    // Check for the presence of ffprobe
    else if (which("ffprobe") != "not found") {
        return "ffprobe";
    }
    else {
        // Warn and default to ffprobe
        std::cerr << "Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work" << std::endl;
        return "ffprobe";
    }
}

// Assuming you have C++17 or later for std::filesystem

std::string fsdecode(const std::filesystem::path& path) {
    // Convert std::filesystem::path to std::string
    // std::filesystem::path automatically handles different encodings
    return path.string();
}

std::string fsdecode(const std::string& path) {
    // In C++17, we assume the path is already in the correct format
    // if the input is std::string, we return it directly
    return path;
}

nlohmann::json get_extra_info(const std::string& stderr_) {
    nlohmann::json extra_info;

    // Adjusted regex pattern to match Python regex more closely
    std::regex re_stream(R"( *(Stream #0[:\.](\d+)):(.*?)(?:\n(?: *)*(Stream #0[:\.](\d+)):(.*?))?)", std::regex::extended);

    // Create an iterator for all matches
    auto begin = std::sregex_iterator(stderr_.begin(), stderr_.end(), re_stream);
    auto end = std::sregex_iterator();

    for (std::sregex_iterator i = begin; i != end; ++i) {
        std::smatch match = *i;

        // Extract stream ID and content
        std::string stream_id = match[2].str(); // Correctly extract numeric stream ID
        std::string content_0 = match[3].str();
        std::string content_1 = match[5].str();

        // Combine the content lines if needed
        std::string content_line = content_0;
        if (!content_1.empty()) {
            content_line += "," + content_1;
        }

        // Split the content line into tokens
        std::regex token_regex("[,:]");
        std::sregex_token_iterator token_begin(content_line.begin(), content_line.end(), token_regex, -1);
        std::sregex_token_iterator token_end;
        std::vector<std::string> tokens;

        for (std::sregex_token_iterator iter = token_begin; iter != token_end; ++iter) {
            std::string token = iter->str();
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }

        // Convert stream_id to integer and add to JSON object
        int id = std::stoi(stream_id);
        extra_info[std::to_string(id)] = tokens;
    }

    return extra_info;
}

// Overload for single string command

std::string exec_command_(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;
    
    #ifdef _WIN32
        std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
    #else
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    #endif

    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

nlohmann::json mediainfo_json(const std::string& file_path, int read_ahead_limit) {
    std::string prober = get_prober_name();

    std::vector<std::string> command_args = { "-v", "info", "-show_format", "-show_streams" };
    std::string stdin_data;
    bool use_stdin = false;

    try {
        command_args.push_back(fsdecode(file_path));
    }
    catch (const std::exception&) {
        if (prober == "ffprobe") {
            command_args.push_back("-read_ahead_limit");
            command_args.push_back(std::to_string(read_ahead_limit));
            command_args.push_back("cache:pipe:0");
        }
        else {
            command_args.push_back("-");
        }
        use_stdin = true;
        std::ifstream file(file_path, std::ios::binary);
        if (!file) throw std::runtime_error("Unable to open file: " + file_path);
        std::ostringstream oss;
        oss << file.rdbuf();
        stdin_data = oss.str();
    }

    command_args.insert(command_args.begin(), prober);
    command_args.insert(command_args.begin() + 1, "-of");
    command_args.insert(command_args.begin() + 2, "json");

    std::string output = exec_command(command_args, use_stdin ? stdin_data : "");
    std::string stderr_output; // Capture stderr if needed

    nlohmann::json info;
    try {
        info = nlohmann::json::parse(output);
    }
    catch (const nlohmann::json::parse_error&) {
        return nullptr; // Or handle error appropriately
    }

    nlohmann::json extra_info = get_extra_info(stderr_output);

    auto audio_streams = info["streams"].get<std::vector<nlohmann::json>>();
    auto it = std::find_if(audio_streams.begin(), audio_streams.end(), [](const nlohmann::json& stream) {
        return stream["codec_type"] == "audio";
        });

    if (it == audio_streams.end()) {
        return info; // No audio streams found
    }

    auto& stream = *it;

    auto set_property = [&](const std::string& prop, const nlohmann::json& value) {
        if (!stream.contains(prop) || stream[prop] == 0) {
            stream[prop] = value;
        }
        };

    for (const auto& token : extra_info[stream["index"].get<int>()]) {
        std::string token_str = token.get<std::string>();  // Ensure token is a string

        std::regex re_sample_fmt(R"(([su][0-9]{1,2}p?) \(([0-9]{1,2}) bit\)$)");
        std::regex re_sample_fmt_default(R"(([su][0-9]{1,2}p?)( \(default\))?$)");
        std::regex re_float(R"(flt(p)? \(default\)?)");
        std::regex re_double(R"(dbl(p)? \(default\)?)");

        std::smatch match;
        if (std::regex_match(token_str, match, re_sample_fmt)) {
            set_property("sample_fmt", match[1].str());
            set_property("bits_per_sample", std::stoi(match[2].str()));
            set_property("bits_per_raw_sample", std::stoi(match[2].str()));  // Corrected to match[2] as the third capture group was removed
        }
        else if (std::regex_match(token_str, match, re_sample_fmt_default)) {
            set_property("sample_fmt", match[1].str());
            set_property("bits_per_sample", 16);  // Defaulting to 16 bits for this case if match[2] does not exist
            set_property("bits_per_raw_sample", 16);
        }
        else if (std::regex_match(token_str, match, re_float)) {
            set_property("sample_fmt", token_str);
            set_property("bits_per_sample", 32);
            set_property("bits_per_raw_sample", 32);
        }
        else if (std::regex_match(token_str, match, re_double)) {
            set_property("sample_fmt", token_str);
            set_property("bits_per_sample", 64);
            set_property("bits_per_raw_sample", 64);
        }
    }

    return info;
}


nlohmann::json mediainfo(const std::string& file_path) {
    std::string prober = get_prober_name();
    std::string command = prober + " -v quiet -show_format -show_streams " + file_path;

    std::string output = exec_command_(command);

    // Retry command without 'quiet' if output is empty
    if (output.empty()) {
        command = prober + " -show_format -show_streams " + file_path;
        output = exec_command_(command);
    }

    // Regex to match key-value pairs
    std::regex rgx(R"((?:(.*?):)?([^=]+)=(.*))");
    std::smatch match;

    nlohmann::json info;

    std::istringstream stream(output);
    std::string line;

    while (std::getline(stream, line)) {
        // Remove carriage return for Windows compatibility
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

        if (std::regex_search(line, match, rgx)) {
            std::string inner_dict = match[1];
            std::string key = match[2];
            std::string value = match[3];

            if (!inner_dict.empty()) {
                info[inner_dict][key] = value;
            } else {
                info[key] = value;
            }
        }
    }

    return info;
}

// Function to get supported codecs with caching
std::pair<std::set<std::string>, std::set<std::string>> get_supported_codecs() {
    static std::pair<std::set<std::string>, std::set<std::string>> cached_codecs;
    static bool is_cached = false;

    if (!is_cached) {
        std::string encoder = "ffmpeg"; // Set dynamically based on your environment if needed
        std::string command = encoder + " -codecs";
        std::string output = exec_command_(command);

        std::regex rgx(R"(^([D.][E.][AVS.][I.][L.][S.]) (\w*) +(.*))");
        std::smatch match;
        std::set<std::string> decoders;
        std::set<std::string> encoders;

        std::istringstream stream(output);
        std::string line;
        while (std::getline(stream, line)) {
            // Remove carriage return characters (Windows compatibility)
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

            if (std::regex_match(line, match, rgx)) {
                std::string flags = match[1];
                std::string codec = match[2];

                if (flags[0] == 'D') {
                    decoders.insert(codec);
                }

                if (flags[1] == 'E') {
                    encoders.insert(codec);
                }
            }
        }

        cached_codecs = std::make_pair(decoders, encoders);
        is_cached = true;
    }

    return cached_codecs;
}

// Function to get supported decoders
std::set<std::string> get_supported_decoders() {
    return get_supported_codecs().first;
}

// Function to get supported encoders
std::set<std::string> get_supported_encoders() {
    return get_supported_codecs().second;
}

// Convert stereo to mid-side
AudioSegment stereo_to_ms(const AudioSegment& stereo_segment) {
    // Split the stereo audio into mono channels
    auto channels = stereo_segment.split_to_mono();

    if (channels.size() != 2) {
        throw std::invalid_argument("Stereo segment must contain exactly two channels.");
    }

    // Get the left and right channels
    auto left_channel = channels[0];
    auto right_channel = channels[1];

    // Create mid and side channels
    int position = 0;
    bool loop = false;
    int times = 1;
    int gain_during_overlay = 0;
    
    AudioSegment mid_channel = left_channel.overlay(right_channel, position, loop, times, gain_during_overlay);
    AudioSegment side_channel = left_channel.overlay(invert_phase(right_channel), position, loop, times, gain_during_overlay);
    const std::vector<AudioSegment>& mono_segments = { mid_channel ,side_channel };
    // Combine mid and side into a new AudioSegment
    return AudioSegment::from_mono_audiosegments(mono_segments);
}

// Convert mid-side to stereo
AudioSegment ms_to_stereo(const AudioSegment& ms_segment) {
    // Split the MS audio into mono channels
    auto channels = ms_segment.split_to_mono();

    if (channels.size() != 2) {
        throw std::invalid_argument("MS segment must contain exactly two channels.");
    }

    // Get the mid and side channels
    auto mid_channel = channels[0];
    auto side_channel = channels[1];

    // Create left and right channels
    int position = 0;
    bool loop = false;
    int times = 1;
    int gain_during_overlay = 0;
    auto left_channel = mid_channel.overlay(side_channel,position,loop,times,gain_during_overlay);
    auto right_channel = mid_channel.overlay(invert_phase(side_channel), position, loop, times, gain_during_overlay);

    // Combine left and right into a new AudioSegment
    const std::vector<AudioSegment>& mono_segments = { mid_channel ,side_channel };
    // Combine mid and side into a new AudioSegment
    return AudioSegment::from_mono_audiosegments(mono_segments);
}
}

#endif // UTILS_CPP
