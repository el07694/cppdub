#include "utils.h"
#include "audio_segment.h"
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
#include <nlohmann/json.hpp> // Include the nlohmann/json library

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cstring>
#include <cstdlib>
#include <unordered_map>

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
    auto it = FRAME_WIDTHS.find(bit_depth);
    if (it != FRAME_WIDTHS.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unsupported bit depth");
}

std::string get_array_type(int bit_depth, bool signed_type = true) {
    auto it = ARRAY_TYPES.find(bit_depth);
    if (it != ARRAY_TYPES.end()) {
        std::string type = it->second;
        if (!signed_type) {
            type = std::toupper(type[0]) + type.substr(1); // Convert first letter to uppercase
        }
        return type;
    }
    throw std::invalid_argument("Unsupported bit depth");
}

std::pair<int, int> get_min_max_value(int bit_depth) {
    auto it = ARRAY_RANGES.find(bit_depth);
    if (it != ARRAY_RANGES.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unsupported bit depth");
}

std::pair<std::string, bool> _fd_or_path_or_tempfile(int fd, const std::string& path, bool tempfile) {
    if (tempfile) {
        // Generate a unique temporary file name
        std::string temp_file_name = "tempfile_XXXXXX.tmp";
        std::unique_ptr<std::FILE, decltype(&std::pclose)> temp_file(std::tmpfile(), std::pclose);
        if (!temp_file) {
            throw std::runtime_error("Unable to create temporary file");
        }
        return {temp_file_name, true};
    }

    if (fd >= 0) {
        return {"File Descriptor: " + std::to_string(fd), false};
    }

    // Handle the case for file path
    if (!path.empty()) {
        return {path, false};
    }

    throw std::invalid_argument("Invalid arguments provided");
}

float db_to_float(float db, bool using_amplitude = true) {
    if (using_amplitude) {
        return std::pow(10.0, db / 20.0);
    } else { // using power
        return std::pow(10.0, db / 10.0);
    }
}


float ratio_to_db(float ratio, float val2 = 0.0f, bool using_amplitude = true) {
    // Handle ratio of zero
    if (ratio == 0) {
        return -std::numeric_limits<float>::infinity();
    }

    // Handle case where two values are given
    if (val2 != 0.0f) {
        ratio /= val2;
    }

    if (using_amplitude) {
        return 20.0 * std::log10(ratio);
    } else { // using power
        return 10.0 * std::log10(ratio);
    }
}


void register_pydub_effect(const std::string& effect_name, std::function<void(AudioSegment&)> effect_function) {
    AudioSegment::register_effect(effect_name, effect_function);
    std::cout << "Registered effect: " << effect_name << std::endl;
}

std::vector<AudioSegment> make_chunks(const AudioSegment& audio_segment, size_t chunk_length_ms) {
    std::vector<AudioSegment> chunks;
    size_t total_length = audio_segment.length();

    if (chunk_length_ms == 0) {
        throw std::invalid_argument("Chunk length must be greater than zero.");
    }

    size_t number_of_chunks = std::ceil(static_cast<double>(total_length) / chunk_length_ms);

    for (size_t i = 0; i < number_of_chunks; ++i) {
        size_t start = i * chunk_length_ms;
        size_t end = std::min(start + chunk_length_ms, total_length);
        chunks.push_back(audio_segment(start, end));
    }

    return chunks;
}

std::string which(const std::string& cmd) {
    std::string command;
    
    #ifdef _WIN32
        command = "where " + cmd;
    #else
        command = "which " + cmd;
    #endif

    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(_popen(command.c_str(), "r"), pclose);

    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    // Trim any trailing newlines or carriage returns
    if (!result.empty()) {
        result.erase(result.find_last_not_of("\r\n") + 1);
    }

    return result.empty() ? "not found" : result;
}

std::string get_encoder_name() {
    // Check for the presence of avconv
    if (!which("avconv").empty()) {
        return "avconv";
    }
    // Check for the presence of ffmpeg
    else if (!which("ffmpeg").empty()) {
        return "ffmpeg";
    }
    else {
        // Warn and default to ffmpeg
        std::cerr << "Warning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work" << std::endl;
        return "ffmpeg";
    }
}

std::string get_player_name() {
    if (!which("avplay").empty()) {
        return "avplay";
    } else if (!which("ffplay").empty()) {
        return "ffplay";
    } else {
        std::cerr << "Couldn't find ffplay or avplay - defaulting to ffplay, but may not work" << std::endl;
        return "ffplay";
    }
}

std::string get_prober_name() {
    if (!which("avprobe").empty()) {
        return "avprobe";
    } else if (!which("ffprobe").empty()) {
        return "ffprobe";
    } else {
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

nlohmann::json get_extra_info(const std::string& stderr) {
    nlohmann::json extra_info;

    // Define the regex pattern to match the stream information
    std::regex re_stream(R"( *(Stream #0:(?:[0-9]+)):(.*?)(?:\n(?: *)*(Stream #0:([0-9]+)):(.*?))?)", std::regex::extended);

    // Create an iterator for all matches
    auto begin = std::sregex_iterator(stderr.begin(), stderr.end(), re_stream);
    auto end = std::sregex_iterator();

    for (std::sregex_iterator i = begin; i != end; ++i) {
        std::smatch match = *i;

        std::string stream_id = match[1].str();
        std::string content_0 = match[2].str();
        std::string content_1 = match[4].str();

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
        int id = std::stoi(stream_id.substr(stream_id.find_last_of(':') + 1));
        extra_info[std::to_string(id)] = tokens;
    }

    return extra_info;
}

// Function to execute a command and capture its output
std::string exec_command(const std::vector<std::string>& command_args, const std::string& stdin_data = "") {
    std::string command;
    for (const auto& arg : command_args) {
        command += arg + " ";
    }

    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(_popen(command.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

std::string exec_command(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

nlohmann::json mediainfo_json(const std::string& file_path, int read_ahead_limit = -1) {
    std::string prober = which("avprobe").empty() ? "ffprobe" : "avprobe";
    
    std::vector<std::string> command_args = {"-v", "info", "-show_format", "-show_streams"};
    std::string stdin_data;
    bool use_stdin = false;
    
    try {
        command_args.push_back(fsdecode(file_path));
    } catch (const std::exception&) {
        if (prober == "ffprobe") {
            command_args.push_back("-read_ahead_limit");
            command_args.push_back(std::to_string(read_ahead_limit));
            command_args.push_back("cache:pipe:0");
        } else {
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
    } catch (const nlohmann::json::parse_error&) {
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
        std::regex re_sample_fmt(R"(([su][0-9]{1,2}p?) \(([0-9]{1,2}) bit\)$)");
        std::regex re_sample_fmt_default(R"(([su][0-9]{1,2}p?)( \(default\))?$)");
        std::regex re_float(R"(flt(p)? \(default\)?)");
        std::regex re_double(R"(dbl(p)? \(default\)?)");

        std::smatch match;
        if (std::regex_match(token, match, re_sample_fmt)) {
            set_property("sample_fmt", match[1].str());
            set_property("bits_per_sample", std::stoi(match[2].str()));
            set_property("bits_per_raw_sample", std::stoi(match[3].str()));
        } else if (std::regex_match(token, match, re_sample_fmt_default)) {
            set_property("sample_fmt", match[1].str());
            set_property("bits_per_sample", std::stoi(match[2].str()));
            set_property("bits_per_raw_sample", std::stoi(match[2].str()));
        } else if (std::regex_match(token, match, re_float)) {
            set_property("sample_fmt", token);
            set_property("bits_per_sample", 32);
            set_property("bits_per_raw_sample", 32);
        } else if (std::regex_match(token, match, re_double)) {
            set_property("sample_fmt", token);
            set_property("bits_per_sample", 64);
            set_property("bits_per_raw_sample", 64);
        }
    }

    return info;
}

nlohmann::json mediainfo(const std::string& file_path) {
    std::string prober = get_prober_name();
    std::string command = prober + " -v quiet -show_format -show_streams " + file_path;

    std::string output = exec_command(command);

    // In case the initial command fails, retry without quiet
    if (output.empty()) {
        command = prober + " -show_format -show_streams " + file_path;
        output = exec_command(command);
    }

    // Regex to match key-value pairs
    std::regex rgx(R"((?:(?P<inner_dict>.*?):)?(?P<key>.*?)\=(?P<value>.*?))");
    std::smatch match;

    nlohmann::json info;

    std::istringstream stream(output);
    std::string line;

    while (std::getline(stream, line)) {
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
        std::string encoder = "ffmpeg"; // You can set this dynamically based on your environment
        std::string command = encoder + " -codecs";
        std::string output = exec_command(command);

        std::regex rgx("^([D.][E.][AVS.][I.][L.][S.]) (\\w*) +(.*)");
        std::smatch match;
        std::set<std::string> decoders;
        std::set<std::string> encoders;

        std::istringstream stream(output);
        std::string line;
        while (std::getline(stream, line)) {
            line = std::regex_replace(line, std::regex("\\r"), ""); // Handle Windows CR

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
    auto mid_channel = left_channel.overlay(right_channel);
    auto side_channel = left_channel.overlay(right_channel.invert_phase());

    // Combine mid and side into a new AudioSegment
    return AudioSegment::from_mono_audiosegments(mid_channel, side_channel);
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
    auto left_channel = mid_channel.overlay(side_channel);
    auto right_channel = mid_channel.overlay(side_channel.invert_phase());

    // Combine left and right into a new AudioSegment
    return AudioSegment::from_mono_audiosegments(left_channel, right_channel);
}
