#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include "audio_segment.h" // Include AudioSegment class
#include <set>
#include <functional>

// Constants
extern const std::map<int, int> FRAME_WIDTHS; // Updated declaration
extern const std::map<int, std::string> ARRAY_TYPES;
extern const std::map<int, std::pair<int, int>> ARRAY_RANGES;

// Functions
int get_frame_width(int bit_depth);
std::string get_array_type(int bit_depth, bool signed_type = true);
std::pair<int, int> get_min_max_value(int bit_depth);
std::pair<std::string, bool> _fd_or_path_or_tempfile(int fd, const std::string& path, bool tempfile);
float db_to_float(float db, bool using_amplitude = true);
float ratio_to_db(float ratio, float val2 = 0.0f, bool using_amplitude = true);
void register_pydub_effect(const std::string& effect_name, std::function<void(AudioSegment&)> effect_function);
std::string which(const std::string& cmd);
std::string get_encoder_name();
std::string get_player_name();
std::string get_prober_name();
std::string fsdecode(const std::filesystem::path& path);
std::string fsdecode(const std::string& path);
nlohmann::json get_extra_info(const std::string& stderr);
// Function to execute a command and capture its output
std::string exec_command(const std::vector<std::string>& command_args, const std::string& stdin_data = "");
std::string exec_command(const std::string& command);
// Function to get media information as JSON
nlohmann::json mediainfo_json(const std::string& file_path, int read_ahead_limit = -1);
nlohmann::json mediainfo(const std::string& file_path);
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


#endif // UTILS_H
