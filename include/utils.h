#ifndef UTILS_H
#define UTILS_H

#ifndef AUDIO_SEGMENT_H
#include "audio_segment.cpp"
#endif

#ifndef EFFECTS_H
#include "effects.cpp"
#endif

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <set>
#include <functional>
#include <optional>
#include <cstdio>      // For std::FILE
#include <memory>      // For std::unique_ptr
#include <filesystem>

namespace cppdub{
    using namespace nlohmann;
// Constants
extern const std::map<int, int> FRAME_WIDTHS; // Updated declaration
extern const std::map<int, std::string> ARRAY_TYPES;
extern const std::map<int, std::pair<int, int>> ARRAY_RANGES;

// Functions
int get_frame_width(int bit_depth);
std::string get_array_type(int bit_depth, bool signed_type = true);
std::pair<int, int> get_min_max_value(int bit_depth);
std::pair<std::unique_ptr<std::FILE, decltype(&std::fclose)>, bool> _fd_or_path_or_tempfile(int* fd, const std::string& path = "", bool tempfile = true, const std::string& mode = "w+b");
float db_to_float(float db, bool using_amplitude = true);
float ratio_to_db(float ratio, float val2 = 0.0f, bool using_amplitude = true);
std::string which(const std::string& program);
std::string get_encoder_name();
std::string get_player_name();
std::string get_prober_name();
std::string fsdecode(const std::filesystem::path& path);
std::string fsdecode(const std::string& path);
nlohmann::json get_extra_info(const std::string& stderr_);
// Function to execute a command and capture its output
std::string exec_command_(const std::string& command);
// Function to get media information as JSON
nlohmann::json mediainfo_json(const std::string& file_path, int read_ahead_limit = -1);
nlohmann::json mediainfo(const std::string& file_path);

// Template function to cache the result of a callable
template<typename Func>
auto cache_codecs(Func&& func) {
    // Cache container
    std::optional<decltype(func())> cache;

    // Lambda to wrap the function with caching
    auto wrapper = [func, &cache]() -> decltype(func()) {
        if (cache) {
            return *cache; // Return cached result
        } else {
            cache = func(); // Compute and cache result
            return *cache;
        }
    };

    return wrapper;
}

// Function to get supported codecs with caching
std::pair<std::set<std::string>, std::set<std::string>> get_supported_codecs();
// Function to get supported decoders
std::set<std::string> get_supported_decoders();
// Function to get supported encoders
std::set<std::string> get_supported_encoders();
// Function to convert stereo to mid-side format
AudioSegment stereo_to_ms(const AudioSegment& stereo_segment);

// Function to convert mid-side format to stereo
AudioSegment ms_to_stereo(const AudioSegment& ms_segment);

// New function declarations for handling audio segments
std::vector<AudioSegment> make_chunks(const AudioSegment& audio_segment, size_t chunk_length_ms);

}
#endif // UTILS_H
