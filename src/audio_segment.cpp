#include "utils.h"
#include "audio_segment.h"
#include "cppaudioop.h"  // Make sure this header is included for the rms function
#include "utils.h"       // Include for the ratio_to_db function
#include "effects.h"

#include <unordered_map>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sys/stat.h>
#include <fstream>
#include <cstring> // For std::memcpy
#include <array>
#include <type_traits>
#include <functional>
#include <algorithm> // For std::min and std::max
#include <vector>
#include <stdexcept>
#include <iterator> // For std::back_inserter
#include <limits> // For std::numeric_limits
#include <sstream>
#include <memory>
#include <cstdlib>  // For std::getenv
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>

#include <iomanip>
//#include <openssl/evp.h> // For base64 encoding


// Add other necessary includes
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/samplefmt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/avutil.h>
#include <libavutil/log.h>

namespace cppdub {
	
// Helper function to convert dB to linear float
double db_to_float(double db) {
    return std::pow(10.0, db / 20.0);
}

// Helper function to convert milliseconds to frame count
int milliseconds_to_frames(int milliseconds, int frame_rate) {
    return static_cast<int>((milliseconds / 1000.0) * frame_rate);
}

// Static member initialization
std::string AudioSegment::ffmpeg_converter_ = "ffmpeg";

const std::unordered_map<std::string, std::string> AudioSegment::DEFAULT_CODECS = {
    {"ogg", "libvorbis"}
};

// Convert time (in milliseconds) to sample index
uint32_t AudioSegment::time_to_sample_index(int time_in_ms) const {
	return (time_in_ms * frame_rate_) / 1000;
}

// Overload the subscript operator to accept Range
AudioSegment operator[](Range range) const {
	// Convert times to sample indices
	uint32_t start_sample = time_to_sample_index(range.start_time_ms);
	uint32_t end_sample = time_to_sample_index(range.end_time_ms);

	// Get the slice using the get_sample_slice method
	std::vector<char> sliced_data = get_sample_slice(start_sample, end_sample);

	// Return a new AudioSegment with the sliced data
	return _spawn(sliced_data);
}

// Constructor and Destructor
AudioSegment::AudioSegment()
    : sample_width_(2), frame_rate_(44100), channels_(2), frame_width_(0) {
    initialize_ffmpeg();
}

AudioSegment::AudioSegment(const std::string& file_path)
    : file_path_(file_path), sample_width_(2), frame_rate_(44100), channels_(2), frame_width_(0) {
    initialize_ffmpeg();
    load_data_from_file(file_path);
}

// Constructor from raw data
AudioSegment::AudioSegment(const char* data, size_t size, const std::map<std::string, int>& metadata)
    : sample_width_(metadata.at("sample_width")), frame_rate_(metadata.at("frame_rate")),
      frame_width_(metadata.at("frame_width")), channels_(metadata.at("channels")), data_size_(size) {
    data_.assign(data, data + size);
}

// Private constructor implementation
AudioSegment::AudioSegment(std::vector<char> data, uint16_t sample_width, uint32_t frame_rate, 
                           uint16_t channels, uint32_t frame_width)
    : data_(std::move(data)), sample_width_(sample_width), frame_rate_(frame_rate), 
      channels_(channels), frame_width_(frame_width) {}
	  
AudioSegment::AudioSegment(const std::vector<uint8_t>& data) {
    data_.assign(data.begin(), data.end());  // Convert from uint8_t to char
}

AudioSegment::~AudioSegment() {
    // Clean up FFmpeg libraries
    avformat_network_deinit();
    av_log_set_callback(nullptr); // Optional: Reset log callback if it was set
}

// Static method to create an AudioSegment from a file
AudioSegment AudioSegment::from_file(const std::string& file_path, const std::string& format,
                                     const std::string& codec, const std::map<std::string, int>& parameters,
                                     int start_second, int duration) {
    std::string command = "ffmpeg -y -i " + file_path;
    if (!format.empty()) {
        command += " -f " + format;
    }
    if (!codec.empty()) {
        command += " -acodec " + codec;
    }
    if (start_second > 0) {
        command += " -ss " + std::to_string(start_second);
    }
    if (duration > 0) {
        command += " -t " + std::to_string(duration);
    }
    command += " -f wav -";

    std::vector<char> output = execute_conversion(command, {});
    fix_wav_headers(output);

    // Dummy metadata for illustration; replace with actual metadata extraction
    std::map<std::string, int> metadata = {
        {"sample_width", 2},
        {"frame_rate", 44100},
        {"channels", 2},
        {"frame_width", 4}
    };

    return AudioSegment(static_cast<const char*>(output.data()), output.size(), metadata);
}

AudioSegment AudioSegment::from_file(const std::string& file_path, const std::string& format,const std::map<std::string, int>& parameters) {
    AudioSegment audio_segment = AudioSegment::from_file(file_path, format,"",parameters,0,0);	
	return audio_segment    
}

// Implementation of the fade method
// Implementation of the fade method
AudioSegment AudioSegment::fade(double to_gain, double from_gain, 
                                int start, int end, int duration) const {
    if ((start != -1 && end != -1 && duration != -1) ||
        (start == -1 && end == -1 && duration == -1)) {
        throw std::invalid_argument("Only two of the three arguments, 'start', 'end', and 'duration' may be specified");
    }

    // No fade == the same audio
    if (to_gain == 0 && from_gain == 0) {
        return *this;
    }

    // Adjust start and end
    int length = this->length_in_milliseconds(); // Assuming you have a method to get duration in milliseconds
    start = (start != -1) ? std::min(length, start) : 0;
    end = (end != -1) ? std::min(length, end) : length;

    if (start < 0) start += length;
    if (end < 0) end += length;

    if (duration < 0) {
        duration = end - start;
    } else {
        if (start != -1) end = start + duration;
        else if (end != -1) start = end - duration;
    }

    if (duration <= 0) duration = end - start;

    double from_power = db_to_float(from_gain);
    double gain_delta = db_to_float(to_gain) - from_power;

    std::vector<char> output;

    // Convert start and end times to sample indices
    uint32_t start_sample = time_to_sample_index(start);
    uint32_t end_sample = time_to_sample_index(end);

    // Original data before fade
    std::vector<char> before_fade = this->get_sample_slice(0, start_sample);
    if (from_gain != 0) {
        before_fade = audioop::mul(before_fade, static_cast<int>(this->sample_width()), static_cast<int>(from_power)); // Assuming `audioop::mul` exists
    }
    output.insert(output.end(), before_fade.begin(), before_fade.end());

    // Fade logic
    if (duration > 100) {
        double scale_step = gain_delta / duration;
        for (int i = 0; i < duration; ++i) {
            double volume_change = from_power + (scale_step * i);
            std::vector<char> chunk = this->get_sample_slice(start_sample + i, start_sample + i + 1);
            chunk = audioop::mul(chunk, static_cast<int>(this->sample_width()), static_cast<int>(volume_change));
            output.insert(output.end(), chunk.begin(), chunk.end());
        }
    } else {
        int start_frame = static_cast<int>(this->frame_count(start)); // Assumed method exists
        int end_frame = static_cast<int>(this->frame_count(end)); // Assumed method exists
        int fade_frames = end_frame - start_frame;
        double scale_step = gain_delta / fade_frames;

        for (int i = 0; i < fade_frames; ++i) {
            double volume_change = from_power + (scale_step * i);
            std::vector<char> sample = static_cast<std::vector<char>>(this->get_frame(start_frame + i)); // Assuming `get_frame` method exists
            sample = audioop::mul(sample, static_cast<int>(this->sample_width()), static_cast<int>(volume_change));
            output.insert(output.end(), sample.begin(), sample.end());
        }
    }

    // Original data after fade
    std::vector<char> after_fade = this->get_sample_slice(end_sample, static_cast<uint32_t>(data_.size() / frame_width_));
    if (to_gain != 0) {
        after_fade = audioop::mul(after_fade, static_cast<int>(this->sample_width()), static_cast<int>(db_to_float(to_gain)));
    }
    output.insert(output.end(), after_fade.begin(), after_fade.end());

    return this->_spawn(output); // Assuming `spawn` method exists and works with raw data
}

// Implementation of fade_out method
AudioSegment AudioSegment::fade_out(int duration) const {
    // Call fade method with to_gain set to -120 dB and end set to infinity
    return fade(-120.0, 0.0, nullptr, duration);
}

// Implementation of fade_in method
AudioSegment AudioSegment::fade_in(int duration) const {
    // Call fade method with from_gain set to -120 dB and start set to 0
    return fade(0.0, -120.0, 0, nullptr, duration);
}

// Implementation of reverse method
AudioSegment AudioSegment::reverse() const {
    // Call the reverse function to reverse the audio data
    std::vector<char> reversed_data = audioop::reverse(data_,static_cast<int>(sample_width_));

    // Create a new AudioSegment with the reversed data
    return _spawn(reversed_data);
}

/*
// Function to base64 encode data
std::string base64_encode(const std::vector<char>& data) {
    // Initialize OpenSSL base64 encoder
    EVP_ENCODE_CTX* ctx = EVP_ENCODE_CTX_new();
    int output_len = 4 * ((data.size() + 2) / 3);
    std::string output(output_len, '\0');
    int final_len = 0;

    EVP_EncodeInit(ctx);
    EVP_EncodeUpdate(ctx, reinterpret_cast<unsigned char*>(&output[0]), &output_len, reinterpret_cast<const unsigned char*>(data.data()), data.size());
    EVP_EncodeFinal(ctx, reinterpret_cast<unsigned char*>(&output[0]) + output_len, &final_len);
    output.resize(output_len + final_len);
    EVP_ENCODE_CTX_free(ctx);

    return output;
}
*/

// Implementation of remove_dc_offset
AudioSegment AudioSegment::remove_dc_offset(int channel, int offset) const {
    if (channel && (channel < 1 || channel > 2)) {
        throw std::invalid_argument("channel value must be None, 1 (left) or 2 (right)");
    }
    if (offset && (offset < -1.0 || offset > 1.0)) {
        throw std::invalid_argument("offset value must be in range -1.0 to 1.0");
    }

    if (offset) {
        offset = static_cast<int>(round(offset * max_possible_amplitude()));
    }

    auto remove_data_dc = [this, offset](const std::vector<char>& data) {
        if (!offset) {
            offset = static_cast<int>(round(avg(data, static_cast<int>(sample_width_))));
        }
        return bias(data, static_cast<int>(sample_width_), -offset);
    };

    if (channels_ == 1) {
        return _spawn(remove_data_dc(data_));
    }

    auto left_channel = tomono(data_, static_cast<int>(sample_width_), 1, 0);
    auto right_channel = tomono(data_, static_cast<int>(sample_width_), 0, 1);

    if (!channel || channel == 1) {
        left_channel = remove_data_dc(left_channel);
    }

    if (!channel || channel == 2) {
        right_channel = remove_data_dc(right_channel);
    }

    left_channel = tostereo(left_channel, static_cast<int>(sample_width_), 1, 0);
    right_channel = tostereo(right_channel, static_cast<int>(sample_width_), 0, 1);

    return _spawn(add(left_channel, right_channel, static_cast<int>(sample_width_)));
}

// Implementation of the rms method
double AudioSegment::rms() const {
    // Ensure data_ is in the correct format for the rms function
    return rms(data_,static_cast<int>(sample_width_));
}

// Implementation of the dBFS method
float AudioSegment::dBFS() const {
    double rms_value = rms();
    if (rms_value == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    double max_amplitude = max_possible_amplitude();
    return ratio_to_db(static_cast<float>(rms_value) / static_cast<float>(max_amplitude));
}

// Implementation of the max method
int AudioSegment::max() const {
    return max(data_, static_cast<int>(sample_width_));
}

// Implementation of the max_possible_amplitude method
double AudioSegment::max_possible_amplitude() const {
    int bits = static_cast<int>(sample_width_) * 8;
    double max_possible_val = std::pow(2, bits);
    return max_possible_val / 2;
}

// Implementation of the get_dc_offset method
double AudioSegment::get_dc_offset(int channel) const {
    if (channel < 1 || channel > 2) {
        throw std::out_of_range("channel value must be 1 (left) or 2 (right)");
    }

    std::vector<char> data;

    if (channels_ == 1) {
        data = data_;
    } else if (channel == 1) {
        data = tomono(data_, static_cast<int>(sample_width_), 1, 0);
    } else {
        data = tomono(data_, static_cast<int>(sample_width_), 0, 1);
    }

    double avg_dc_offset = avg(data, static_cast<int>(sample_width_));
    return avg_dc_offset / max_possible_amplitude();
}

// Implementation of the max_dBFS method
float AudioSegment::max_dBFS() const {
    double max_value = static_cast<double>(max());  // Use the existing max() method
    double max_amplitude = max_possible_amplitude();  // Use the existing max_possible_amplitude() method

    return ratio_to_db(max_value / max_amplitude);
}

// Implementation of the duration_seconds method
double AudioSegment::duration_seconds() const {
    if (frame_rate_ > 0) {
        return static_cast<double>(frame_count()) / frame_rate_;
    }
    return 0.0;
}

AudioSegment AudioSegment::from_mp3(const std::string& file, const std::map<std::string, std::string>& parameters) {
    AudioSegment audio_segment = AudioSegment::from_file(file, "mp3", parameters);	
	return audio_segment
}

AudioSegment AudioSegment::from_flv(const std::string& file, const std::map<std::string, std::string>& parameters) {
    AudioSegment audio_segment = AudioSegment::from_file(file, "flv", parameters);	
	return audio_segment
}

AudioSegment AudioSegment::from_ogg(const std::string& file, const std::map<std::string, std::string>& parameters) {
    AudioSegment audio_segment = AudioSegment::from_file(file, "ogg", parameters);	
	return audio_segment
}

AudioSegment AudioSegment::from_wav(const std::string& file, const std::map<std::string, std::string>& parameters) {
    AudioSegment audio_segment = AudioSegment::from_file(file, "wav", parameters);	
	return audio_segment
}

AudioSegment AudioSegment::from_raw(const std::string& file, int sample_width, int frame_rate, int channels) {
    std::map<std::string, int> parameters;
    parameters["sample_width"] = sample_width;
    parameters["frame_rate"] = frame_rate;
    parameters["channels"] = channels;

    // Default codec and duration
    std::string codec = ""; // Empty or default codec; adjust as needed
    int start_second = 0;
    int duration = 0;

    return AudioSegment::from_file(file, "raw", codec, parameters, start_second, duration);
}

// Static method to create an AudioSegment from a WAV file
AudioSegment AudioSegment::_from_safe_wav(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file.");
    }
    
    // Read file contents into a std::vector<uint8_t>
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());

    file.close();

    // Create and return AudioSegment using the vector
    return AudioSegment(data);
}

std::ifstream AudioSegment::export(const std::string& out_f, const std::string& format,
                                   const std::string& codec, const std::string& bitrate,
                                   const std::vector<std::string>& parameters, const std::map<std::string, std::string>& tags,
                                   const std::string& id3v2_version, const std::string& cover) {
    if (format == "raw" && (!codec.empty() || !parameters.empty())) {
        throw std::invalid_argument("Cannot invoke ffmpeg when export format is 'raw'; specify an ffmpeg raw format or call export(format='raw') with no codec or parameters");
    }

    std::ofstream out_file(out_f, std::ios::binary | std::ios::trunc);
    if (!out_file) {
        throw std::runtime_error("Failed to open output file");
    }

    bool easy_wav = (format == "wav" && codec.empty() && parameters.empty());

    std::string temp_data_filename = "temp_data.wav";
    std::string temp_output_filename = "temp_output." + format;

    if (!easy_wav) {
        // Write to temp data file
        std::ofstream data(temp_data_filename, std::ios::binary | std::ios::trunc);
        if (!data) {
            throw std::runtime_error("Failed to create temp data file");
        }

        // Write WAV headers and data to the temp file
        // Assume appropriate WAV header writing here
        data.write(reinterpret_cast<const char*>(this->_data.data()), this->_data.size());
        data.close();
    } else {
        // Write WAV data directly
        out_file.write(reinterpret_cast<const char*>(this->_data.data()), this->_data.size());
        out_file.flush();
        return out_file;
    }

    // Build ffmpeg command
    std::string command = "ffmpeg -y -f wav -i " + temp_data_filename;

    if (!codec.empty()) {
        command += " -acodec " + codec;
    }
    if (!bitrate.empty()) {
        command += " -b:a " + bitrate;
    }
    for (const auto& param : parameters) {
        command += " " + param;
    }
    if (!tags.empty()) {
        for (const auto& tag : tags) {
            command += " -metadata " + tag.first + "=" + tag.second;
        }
        if (format == "mp3" && (id3v2_version == "3" || id3v2_version == "4")) {
            command += " -id3v2_version " + id3v2_version;
        }
    }
    if (!cover.empty()) {
        command += " -i " + cover + " -map 0 -map 1 -c:v mjpeg";
    }
    command += " " + temp_output_filename;

    // Execute the command
    int ret_code = system(command.c_str());
    if (ret_code != 0) {
        throw std::runtime_error("Encoding failed");
    }

    // Read from temp output file
    std::ifstream temp_output(temp_output_filename, std::ios::binary);
    if (!temp_output) {
        throw std::runtime_error("Failed to open temp output file");
    }
    out_file << temp_output.rdbuf();

    // Cleanup
    temp_output.close();
    std::remove(temp_data_filename.c_str());
    std::remove(temp_output_filename.c_str());

    out_file.flush();
    return out_file;
}

std::vector<uint8_t> AudioSegment::get_frame(int index) const {
    size_t frame_start = static_cast<size_t>(index * this->frame_width_);
    size_t frame_end = static_cast<size_t>(frame_start + this->frame_width_);
    
    if (frame_start >= static_cast<size_t>(this->_data.size())) {
        throw std::out_of_range("Frame index out of range");
    }
    
    if (frame_end > static_cast<size_t>(this->_data.size())) {
        frame_end = static_cast<size_t>(this->_data.size()); // Adjust frame_end if it exceeds data size
    }
    
    return std::vector<uint8_t>(this->_data.begin() + frame_start, this->_data.begin() + frame_end);
}

double AudioSegment::frame_count(int ms) const {
    if (ms >= 0) {
        return static_cast<double>(ms * (static_cast<double>(this->frame_rate_) / 1000.0));
    } else {
        return static_cast<double>(this->_data.size()) / this->frame_width_;
    }
}

// Helper function to execute a command and capture output
std::vector<char> AudioSegment::execute_command(const std::string& command, const std::vector<char>& input_data) {
    std::array<char, 128> buffer;
    std::vector<char> result;
    
    // Use popen with "r+" to allow both reading and writing to the process
    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r+"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    // Write input data to the pipe
    size_t written = fwrite(input_data.data(), sizeof(char), input_data.size(), pipe.get());
    if (written != input_data.size()) {
        throw std::runtime_error("fwrite() failed!");
    }

    // Close the writing end of the pipe (to signal EOF to the command)
    if (fflush(pipe.get()) != 0) {
        throw std::runtime_error("fflush() failed!");
    }

    // Read the output from the pipe
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result.insert(result.end(), buffer.data(), buffer.data() + strlen(buffer.data()));
    }

    // Check if there's any error in reading
    if (ferror(pipe.get())) {
        throw std::runtime_error("fgets() failed!");
    }

    return result; // Return the result as a vector of chars
}

// Implementation of execute_conversion
std::vector<char> AudioSegment::execute_conversion(const std::string& conversion_command, const std::vector<char>& stdin_data) const {
    // Ensure the command ends with " -" to signify output to stdout
    std::string command = conversion_command;
    if (command.back() != ' ') {
        command += " -";
    }

    // Execute the command and capture the output
    std::vector<char> output_data;
    try {
        output_data = execute_command(command, stdin_data);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to execute command: " + std::string(e.what()));
    }

    // Validate output data
    if (output_data.empty()) {
        throw std::runtime_error("Conversion failed: No output data returned.");
    }

    // Optionally, further checks on output_data could be added here based on your requirements

    return output_data;
}

void AudioSegment::initialize_ffmpeg() {
    // Register all codecs, formats, and protocols
    av_register_all();
    avformat_network_init();
    avcodec_register_all();

    // Optional: Set FFmpeg logging level
    av_log_set_level(AV_LOG_INFO); // Adjust logging level as needed
}

std::string AudioSegment::ffmpeg() {
    return ffmpeg_converter_;
}

void AudioSegment::ffmpeg(const std::string& value) {
    ffmpeg_converter_ = value;
}

const std::unordered_map<std::string, std::string>& AudioSegment::default_codecs() {
    return DEFAULT_CODECS;
}

void AudioSegment::load_data_from_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw CouldntDecodeError("Failed to open file: " + file_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    data_.resize(size);
    if (!file.read(data_.data(), size)) {
        throw CouldntDecodeError("Failed to read file: " + file_path);
    }
}

void AudioSegment::load_data_from_memory(const uint8_t* data, size_t size) {
    data_.assign(data, data + size);
}

// Getter for raw audio data
std::vector<char> AudioSegment::raw_data() const {
    return data_;
}

// Method to get the length of the audio segment in milliseconds
size_t AudioSegment::length_in_milliseconds() const {
    // Check if the sample rate and frame width are properly initialized
    if (frame_rate_ == 0 || frame_width_ == 0) {
        throw CouldntDecodeError("Sample rate or frame width not set.");
    }

    // Number of samples = total data size / frame width
    size_t num_samples = data_.size() / frame_width_;

    // Duration in seconds = number of samples / sample rate
    double duration_in_seconds = static_cast<double>(num_samples) / frame_rate_;

    // Convert to milliseconds
    return static_cast<size_t>(duration_in_seconds * 1000);
}

// Convert time (in milliseconds) to sample index
uint32_t time_to_sample_index(int time_in_ms) const {
    return (time_in_ms * frame_rate_) / 1000;
}

// New method to get a slice of samples
std::vector<char> AudioSegment::get_sample_slice(uint32_t start_sample, uint32_t end_sample) const {
    // Calculate maximum valid index
    uint32_t max_val = static_cast<uint32_t>(data_.size() / frame_width_);

    // Helper lambda function to bound the value within range
    auto bounded = [max_val](uint32_t val, uint32_t default_val) {
        if (val == std::numeric_limits<uint32_t>::max()) {
            return default_val;
        }
        // No negative values allowed
        if (val > max_val) {
            return max_val;
        }
        return val;
    };

    // Bound the start and end sample indices
    start_sample = bounded(start_sample, 0);
    end_sample = bounded(end_sample, max_val);

    // Calculate start and end indices for slicing
    size_t start_i = start_sample * frame_width_;
    size_t end_i = end_sample * frame_width_;

    // Ensure end_i does not exceed data size
    end_i = std::min(end_i, data_.size());

    // Slice the data
    std::vector<char> result(data_.begin() + start_i, data_.begin() + end_i);

    // Calculate the expected length of the result
    size_t expected_length = (end_sample - start_sample) * frame_width_;
    size_t actual_length = result.size();

    // If the result size is less than expected, pad with silence
    if (expected_length > actual_length) {
        // Define silence (all zeroes)
        std::vector<char> silence(frame_width_, 0);

        // Calculate number of missing frames
        size_t missing_frames = (expected_length - actual_length) / frame_width_;

        // Check if the missing frames exceed the 2 ms threshold
        if (missing_frames > 2 * frame_rate_ / 1000) {
            throw CouldntDecodeError("Too many missing frames; exceeding 2 ms of silence");
        }

        // Append silence to the result
        result.insert(result.end(), missing_frames * frame_width_, 0);
    }

    return result;
}

// Implementation of the split_to_mono method
std::vector<AudioSegment> AudioSegment::split_to_mono() const {
    std::vector<AudioSegment> mono_channels;

    if (channels_ == 1) {
        mono_channels.push_back(*this);
        return mono_channels;
    }

    std::vector<char> samples = raw_data();
    size_t total_samples = samples.size() / sample_width_;
    size_t frame_count = total_samples / channels_;
    
    for (int i = 0; i < channels_; ++i) {
        std::vector<char> samples_for_current_channel;

        for (size_t j = i * sample_width_; j < samples.size(); j += channels_ * sample_width_) {
            samples_for_current_channel.insert(samples_for_current_channel.end(),
                                               samples.begin() + j,
                                               samples.begin() + j + sample_width_);
        }

        std::unordered_map<std::string, int> overrides = {{"channels", 1}, {"frame_width", sample_width_}};
        mono_channels.push_back(_spawn(samples_for_current_channel, overrides));
    }

    return mono_channels;
}

// Equality operator
bool AudioSegment::operator==(const AudioSegment& other) const {
    return data_ == other.data_ &&
           sample_width_ == other.sample_width_ &&
           frame_rate_ == other.frame_rate_ &&
           channels_ == other.channels_;
}

// Inequality operator
bool AudioSegment::operator!=(const AudioSegment& other) const {
    return !(*this == other);
}

std::size_t AudioSegmentHash::operator()(const AudioSegment& segment) const {
    std::size_t h1 = std::hash<uint16_t>{}(segment.sample_width_);
    std::size_t h2 = std::hash<uint32_t>{}(segment.frame_rate_);
    std::size_t h3 = std::hash<uint16_t>{}(segment.channels_);
    std::size_t h4 = std::hash<std::string>{}(std::string(segment.data_.begin(), segment.data_.end()));
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
}

// Helper function to convert milliseconds to frame index
size_t millisecond_to_frame(size_t milliseconds, uint32_t frame_rate) {
    return static_cast<size_t>((milliseconds * frame_rate) / 1000);
}

// Retrieve data based on millisecond
std::vector<char> AudioSegment::get_data_by_millisecond(size_t start_ms, size_t end_ms) const {
    // Convert milliseconds to frame indices
    size_t start_frame = millisecond_to_frame(start_ms, frame_rate_);
    size_t end_frame = millisecond_to_frame(end_ms, frame_rate_);

    // Ensure indices are within bounds
    start_frame = std::min(start_frame, data_.size() / frame_width_);
    end_frame = std::min(end_frame, data_.size() / frame_width_);

    // Calculate start and end positions in the data vector
    size_t start_pos = start_frame * frame_width_;
    size_t end_pos = end_frame * frame_width_;
    end_pos = std::min(end_pos, data_.size()); // Ensure we do not exceed bounds

    std::vector<char> result(data_.begin() + start_pos, data_.begin() + end_pos);

    // Handle missing frames by padding with silence if needed
    size_t expected_length = (end_frame - start_frame) * frame_width_;
    size_t actual_length = result.size();
    if (expected_length > actual_length) {
        // Define silence (all zeroes)
        size_t missing_length = expected_length - actual_length;
        result.insert(result.end(), missing_length, 0); // Efficiently append silence
    }

    return result;
}

// Implement the '+' operator for combining two AudioSegments
AudioSegment AudioSegment::operator+(const AudioSegment& other) const {
    return append(other, 0);
}

// Implement the '+' operator for applying gain
AudioSegment AudioSegment::operator+(int db) const {
    return apply_gain(db);
}

// Global operator to support adding gain to AudioSegment
AudioSegment operator+(int db, const AudioSegment& segment) {
    return segment + db;  // Leverage the existing operator+ defined for AudioSegment
}

// Implement the '-' operator to subtract gain from AudioSegment
AudioSegment operator-(const AudioSegment& segment, int db) {
    return segment + (-db);  // Apply negative gain using existing operator+
}

// Implement the '-' operator for subtracting one AudioSegment from another
AudioSegment operator-(const AudioSegment& lhs, const AudioSegment& rhs) {
    throw std::invalid_argument("AudioSegment objects can't be subtracted from each other");
}

// Implement the '*' operator to repeat the audio segment
AudioSegment operator*(const AudioSegment& segment, int times) {
    if (times <= 0) {
        throw std::invalid_argument("Repeat count must be positive.");
    }

    AudioSegment result = segment;

    std::vector<char> repeated_data;
    repeated_data.reserve(segment.data_.size() * times);

    for (int i = 0; i < times; ++i) {
        repeated_data.insert(repeated_data.end(), segment.data_.begin(), segment.data_.end());
    }

    result.data_ = std::move(repeated_data);
    return result;
}

// Implement the '*' operator to overlay two AudioSegments
AudioSegment operator*(const AudioSegment& lhs, const AudioSegment& rhs) {
    // Check if both segments have the same format
    if (lhs.frame_rate_ != rhs.frame_rate_ || lhs.channels_ != rhs.channels_) {
        throw std::invalid_argument("Audio segments must have the same sample rate and channels.");
    }

    AudioSegment result;
    result.frame_rate_ = lhs.frame_rate_;
    result.channels_ = lhs.channels_;
    result.sample_width_ = lhs.sample_width_; // Assumes both segments have the same sample width
    result.frame_width_ = lhs.frame_width_; // Assumes both segments have the same frame width

    size_t max_size = std::max(lhs.data_.size(), rhs.data_.size());
    result.data_.resize(max_size, 0);

    // Overlay audio data
    for (size_t i = 0; i < max_size; ++i) {
        if (i < lhs.data_.size()) {
            result.data_[i] += lhs.data_[i];
        }
        if (i < rhs.data_.size()) {
            result.data_[i] += rhs.data_[i];
        }
    }

    // Ensure data values are within valid range, e.g., 0 to 255 for 8-bit audio
    std::transform(result.data_.begin(), result.data_.end(), result.data_.begin(),
                   [](char sample) { return std::min(std::max(sample, 0), 255); });

    return result;
}

// Helper method to apply gain
AudioSegment AudioSegment::apply_gain(int db) const {
    AudioSegment result = *this;
    if (sample_width_ != 2) {
        throw std::runtime_error("Unsupported sample width for gain adjustment");
    }

    int16_t gain_factor = static_cast<int16_t>(db);
    size_t num_samples = result.data_.size() / sizeof(int16_t);

    for (size_t i = 0; i < num_samples; ++i) {
        int16_t* sample = reinterpret_cast<int16_t*>(result.data_.data() + i * sizeof(int16_t));
        *sample = std::clamp(*sample + gain_factor, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max());
    }

    return result;
}

AudioSegment AudioSegment::overlay(const AudioSegment& seg, int position, bool loop, int times, int gain_during_overlay) const {
    if (loop) {
        times = -1; // Loop indefinitely
    } else if (times == 0) {
        return *this; // No-op, return a copy of the current segment
    } else if (times < 0) {
        times = 1; // Default to looping once if times is negative
    }

    // Sync segments (assume _sync method returns a pair of AudioSegments)
    auto [seg1, seg2] = _sync(*this, seg);

    int sample_width = seg1.sample_width_;
    std::vector<char> data1 = seg1.raw_data();
    std::vector<char> data2 = seg2.raw_data();

    std::vector<char> output_data;
    output_data.insert(output_data.end(), data1.begin(), data1.begin() + position);

    int seg1_len = data1.size();
    int seg2_len = data2.size();
    int pos = position;

    while (times != 0) {
        int remaining = std::max(0, seg1_len - pos);
        if (seg2_len >= remaining) {
            data2.resize(remaining);
            seg2_len = remaining;
            times = 1; // Last iteration if we reach the end
        }

        std::vector<char> seg1_overlaid(data1.begin() + pos, data1.begin() + pos + seg2_len);

        if (gain_during_overlay != 0) {
            // Apply gain (convert dB to float factor)
            float gain_factor = db_to_float(gain_during_overlay);
            std::vector<char> seg1_adjusted_gain = mul(seg1_overlaid, sample_width, gain_factor);
            std::vector<char> overlay_result = add(seg1_adjusted_gain, data2, sample_width);
            output_data.insert(output_data.end(), overlay_result.begin(), overlay_result.end());
        } else {
            std::vector<char> overlay_result = add(seg1_overlaid, data2, sample_width);
            output_data.insert(output_data.end(), overlay_result.begin(), overlay_result.end());
        }

        pos += seg2_len;
        times -= 1;
    }

    output_data.insert(output_data.end(), data1.begin() + pos, data1.end());

    // Create a new AudioSegment with the output data
    return _spawn(output_data);
}

// Implementation of the append method
AudioSegment AudioSegment::append(const AudioSegment& seg, int crossfade) const {
    // Synchronize the audio segments to ensure they have the same sample width and frame rate
    AudioSegment seg1, seg2;
    std::tie(seg1, seg2) = AudioSegment::_sync(*this, seg);

    // Check crossfade validity
    int crossfade_frames = milliseconds_to_frames(crossfade, frame_rate_);
    if (crossfade == 0) {
        // No crossfade, just concatenate
        std::vector<char> combined_data = seg1.raw_data();
        combined_data.insert(combined_data.end(), seg2.raw_data().begin(), seg2.raw_data().end());
        return seg1._spawn(combined_data);
    }
    else if (crossfade_frames > seg1.frame_count()) {
        throw std::invalid_argument("Crossfade is longer than the original AudioSegment.");
    }
    else if (crossfade_frames > seg2.frame_count()) {
        throw std::invalid_argument("Crossfade is longer than the appended AudioSegment.");
    }

    // Create crossfade segments
    auto fade_out_data = seg1.slice(-crossfade_frames, crossfade_frames).fade(-120.0);
    auto fade_in_data = seg2.slice(0, crossfade_frames).fade(-120.0);

    // Concatenate segments
    std::vector<char> combined_data;
    combined_data.reserve(seg1.raw_data().size() + fade_out_data.raw_data().size() + seg2.raw_data().size());

    // Append the first segment excluding the crossfade portion
    combined_data.insert(combined_data.end(), seg1.slice(0, -crossfade_frames).raw_data().begin(), seg1.slice(0, -crossfade_frames).raw_data().end());
    // Append the crossfade portion
    combined_data.insert(combined_data.end(), fade_out_data.raw_data().begin(), fade_out_data.raw_data().end());
    // Append the second segment excluding the crossfade portion
    combined_data.insert(combined_data.end(), seg2.slice(crossfade_frames).raw_data().begin(), seg2.slice(crossfade_frames).raw_data().end());

    return seg1._spawn(combined_data);
}

// Spawn a new AudioSegment with the given data and optional metadata overrides
AudioSegment AudioSegment::_spawn(const std::vector<char>& data) const {
    AudioSegment new_segment;

    // Copy metadata from the current segment
    new_segment.sample_width_ = sample_width_;
    new_segment.frame_rate_ = frame_rate_;
    new_segment.frame_width_ = frame_width_;
    new_segment.channels_ = channels_;

    // Set the new data
    new_segment.data_ = data;

    return new_segment;
}

// Implementation of convert_audio_data using libavcodec or other audio library
std::vector<char> AudioSegment::convert_audio_data(const std::vector<char>& data, int src_rate, int dest_rate) const {
    // Set up libavformat structures
    AVCodecContext *codec_ctx = avcodec_alloc_context3(nullptr);
    AVFrame *frame = av_frame_alloc();
    AVPacket packet;
    av_init_packet(&packet);

    // Configure codec context (e.g., sample rate, channels)
    codec_ctx->sample_rate = src_rate;
    codec_ctx->channels = channels_;
    codec_ctx->sample_fmt = AV_SAMPLE_FMT_S16; // Example format
    codec_ctx->channel_layout = av_get_default_channel_layout(channels_);

    // Open codec context
    AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE);
    avcodec_open2(codec_ctx, codec, nullptr);

    // Decode input data
    packet.data = reinterpret_cast<uint8_t*>(const_cast<char*>(data.data()));
    packet.size = data.size();
    avcodec_send_packet(codec_ctx, &packet);
    avcodec_receive_frame(codec_ctx, frame);

    // Set up conversion context
    SwrContext *swr_ctx = swr_alloc();
    av_opt_set_int(swr_ctx, "in_channel_layout", codec_ctx->channel_layout, 0);
    av_opt_set_int(swr_ctx, "out_channel_layout", codec_ctx->channel_layout, 0);
    av_opt_set_int(swr_ctx, "in_sample_rate", src_rate, 0);
    av_opt_set_int(swr_ctx, "out_sample_rate", dest_rate, 0);
    av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);
    av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", codec_ctx->sample_fmt, 0);
    swr_init(swr_ctx);

    // Convert audio
    std::vector<char> converted_data;
    int64_t in_samples = frame->nb_samples;
    int out_samples = av_rescale_rnd(swr_get_delay(swr_ctx, src_rate) + in_samples, dest_rate, src_rate, AV_ROUND_UP);
    int out_buffer_size = av_samples_get_buffer_size(nullptr, codec_ctx->channels, out_samples, codec_ctx->sample_fmt, 1);
    std::vector<uint8_t> out_buffer(out_buffer_size);

    swr_convert(swr_ctx, &out_buffer.data(), out_samples, (const uint8_t **)frame->data, in_samples);

    converted_data.assign(out_buffer.begin(), out_buffer.end());

    // Cleanup
    av_frame_free(&frame);
    avcodec_free_context(&codec_ctx);
    swr_free(&swr_ctx);

    return converted_data;
}

// Implementation of set_frame_rate
AudioSegment AudioSegment::set_frame_rate(int frame_rate) const {
    if (frame_rate == frame_rate_) {
        return *this;
    }

    std::vector<char> converted_data;
    if (!data_.empty()) {
        converted_data = convert_audio_data(data_, frame_rate_, frame_rate);
    } else {
        converted_data = data_;
    }
	
    return _spawn(converted_data);
}

// Convert data to stereo format
std::vector<char> AudioSegment::convert_to_stereo(const std::vector<char>& data) const {
    // Assuming 16-bit PCM audio and using a simple conversion
    // for illustration. Adjust as needed.
    size_t frame_size = sample_width_ * channels_;
    std::vector<char> converted_data(data.size() * 2); // Double the size for stereo

    for (size_t i = 0; i < data.size(); i += frame_size) {
        std::memcpy(&converted_data[i * 2], &data[i], frame_size);
        std::memcpy(&converted_data[i * 2 + frame_size], &data[i], frame_size);
    }

    return converted_data;
}

// Convert data to mono format
std::vector<char> AudioSegment::convert_to_mono(const std::vector<char>& data) const {
    // Assuming 16-bit PCM audio and using a simple conversion
    // for illustration. Adjust as needed.
    size_t frame_size = sample_width_ * channels_;
    std::vector<char> converted_data(data.size() / 2); // Half the size for mono

    for (size_t i = 0; i < data.size(); i += frame_size * 2) {
        int16_t left = *reinterpret_cast<const int16_t*>(&data[i]);
        int16_t right = *reinterpret_cast<const int16_t*>(&data[i + sample_width_]);
        int16_t mono = (left + right) / 2;
        std::memcpy(&converted_data[i / 2], &mono, sample_width_);
    }

    return converted_data;
}

// Merge multiple channels into a single data buffer
std::vector<char> AudioSegment::merge_channels(const std::vector<std::vector<char>>& channel_data) const {
    size_t frame_size = sample_width_ * channels_;
    size_t frame_count = channel_data[0].size() / (sample_width_ * channels_);
    std::vector<char> merged_data(frame_size * frame_count);

    for (size_t i = 0; i < frame_count; ++i) {
        int16_t sample_sum = 0;
        for (const auto& data : channel_data) {
            sample_sum += *reinterpret_cast<const int16_t*>(&data[i * sample_width_]);
        }
        int16_t average = sample_sum / channels_;
        std::memcpy(&merged_data[i * sample_width_], &average, sample_width_);
    }

    return merged_data;
}

// Split the audio into individual channel data
std::vector<std::vector<char>> AudioSegment::split_to_channels() const {
    std::vector<std::vector<char>> channels_data(channels_);
    size_t frame_size = sample_width_ * channels_;

    // Resize each channel's data vector to fit the correct amount of data per channel
    size_t num_frames = data_.size() / frame_size;  // Total number of frames in the audio segment
    for (int i = 0; i < channels_; ++i) {
        channels_data[i].resize(num_frames * sample_width_);
    }

    // Split the data into channels
    for (size_t j = 0; j < num_frames; ++j) {
        for (int i = 0; i < channels_; ++i) {
            std::memcpy(&channels_data[i][j * sample_width_], &data_[j * frame_size + i * sample_width_], sample_width_);
        }
    }

    return channels_data;
}

// Implementation of set_channels
AudioSegment AudioSegment::set_channels(int channels) const {
    if (channels == channels_) {
        return *this;
    }

    std::vector<char> converted_data;
    int frame_width;

    if (channels == 2 && channels_ == 1) {
        converted_data = convert_to_stereo(data_);
        frame_width = frame_width_ * 2;
    } else if (channels == 1 && channels_ == 2) {
        converted_data = convert_to_mono(data_);
        frame_width = frame_width_ / 2;
    } else if (channels == 1) {
        auto channels_data = split_to_channels();
        converted_data = merge_channels(channels_data);
        frame_width = frame_width_ / channels_;
    } else if (channels_ == 1) {
        std::vector<AudioSegment> duplicated_channels(channels, *this);
        return AudioSegment::from_mono_audiosegments(duplicated_channels);
    } else {
        throw std::invalid_argument("AudioSegment.set_channels only supports mono-to-multi channel and multi-to-mono channel conversion");
    }

    std::unordered_map<std::string, int> overrides;
    overrides["channels"] = channels;
    overrides["frame_width"] = frame_width;
	converted_segment = _spawn(converted_data);
	
	// Update with overrides
    if (overrides.find("frame_width") != overrides.end()) {
        converted_segment.frame_width_ = overrides.at("frame_width");
    }
    if (overrides.find("channels") != overrides.end()) {
        converted_segment.channels_ = overrides.at("channels");
    }
	
    return converted_segment;
}

AudioSegment AudioSegment::set_sample_width(int sample_width) const {
    if (sample_width == sample_width_) {
        return *this;
    }

    int new_frame_width = channels_ * sample_width;

    std::vector<char> new_data;
    // Convert data to the new sample width (using an equivalent conversion function)
    // For example, if you have a function for this conversion:
    // new_data = audioop::lin2lin(data_, sample_width_, sample_width);

    std::unordered_map<std::string, int> overrides = {
        {"sample_width", sample_width},
        {"frame_width", new_frame_width}
    };	
   	converted_segment = _spawn(new_data);
	
	// Update with overrides
    if (overrides.find("frame_width") != overrides.end()) {
        converted_segment.frame_width_ = overrides.at("frame_width");
    }
    if (overrides.find("channels") != overrides.end()) {
        converted_segment.channels_ = overrides.at("channels");
    }
	
    return converted_segment;
}

// Synchronize metadata (channels, frame rate, sample width) across multiple AudioSegment instances
std::vector<AudioSegment> AudioSegment::sync(const std::vector<AudioSegment>& segments) {
    if (segments.empty()) {
        return {}; // Return an empty vector if no segments are provided
    }

    // Find the maximum values for channels, frame rate, and sample width
    uint16_t max_channels = 0;
    uint32_t max_frame_rate = 0;
    uint16_t max_sample_width = 0;

    for (const auto& seg : segments) {
        max_channels = std::max(max_channels, seg.channels_);
        max_frame_rate = std::max(max_frame_rate, seg.frame_rate_);
        max_sample_width = std::max(max_sample_width, seg.sample_width_);
    }

    // Create synchronized segments
    std::vector<AudioSegment> synced_segments;
    synced_segments.reserve(segments.size());

    for (const auto& seg : segments) {
        AudioSegment synced_seg = seg;
        synced_seg.sample_width_ = max_sample_width;
        synced_seg.frame_rate_ = max_frame_rate;
        synced_seg.channels_ = max_channels;
        synced_segments.push_back(synced_seg);
    }

    return synced_segments;
}

// Convert milliseconds to frame count
uint32_t AudioSegment::frame_count(uint32_t ms) const {
    return static_cast<uint32_t>((ms * frame_rate_) / 1000);
}

// Parse position considering negative values and infinity
uint32_t AudioSegment::parse_position(double val) const {
    if (val < 0) {
        val = static_cast<double>(data_size_) - std::abs(val);
    }
    if (val == std::numeric_limits<double>::infinity()) {
        val = static_cast<double>(data_size_);
    }
    // Assuming frame_count() needs milliseconds, so we convert the value to milliseconds
    uint32_t ms = static_cast<uint32_t>(val); // Assuming val is in milliseconds for simplicity
    return frame_count(ms);
}

// Static method to create an empty audio segment with default metadata
AudioSegment AudioSegment::empty() {
    // Default values
    return AudioSegment(
        std::vector<char>{}, // Empty data
        1,  // sample_width
        1,  // frame_rate
        1,  // channels
        1   // frame_width
    );
}

// Static method to generate a silent audio segment
AudioSegment AudioSegment::silent(size_t duration_ms, uint32_t frame_rate) {
    // Calculate the number of frames needed
    size_t frames = static_cast<size_t>(frame_rate * (duration_ms / 1000.0));
    
    // Calculate the size of the data buffer: sample_width * number of frames * number of channels
    size_t sample_width = 2;  // Default sample width (2 bytes for 16-bit PCM)
    size_t frame_width = sample_width * 1;  // 1 channel
    std::vector<char> data(frames * frame_width, 0);  // Silent data

    return AudioSegment(data, sample_width, frame_rate, 1, frame_width);
}

// Static method to combine mono audio segments into a multi-channel segment
AudioSegment AudioSegment::from_mono_audiosegments(const std::vector<AudioSegment>& mono_segments) {
    if (mono_segments.empty()) {
        throw std::invalid_argument("At least one AudioSegment instance is required");
    }

    // Validate that all segments are mono
    for (const auto& seg : mono_segments) {
        if (seg.channels_ != 1) {
            throw std::invalid_argument("All segments must be mono (1 channel)");
        }
    }

    size_t num_segments = mono_segments.size();
    uint16_t sample_width = mono_segments[0].sample_width_;
    uint32_t frame_rate = mono_segments[0].frame_rate_;

    // Determine the frame count
    size_t frame_count = 0;
    for (const auto& seg : mono_segments) {
        frame_count = std::max(frame_count, static_cast<size_t>(seg.data_.size() / sample_width));
    }

    // Prepare the combined data buffer
    std::vector<char> combined_data(frame_count * sample_width * num_segments, 0);

    // Interleave the samples
    for (size_t i = 0; i < num_segments; ++i) {
        const auto& seg = mono_segments[i];
        const auto& seg_data = seg.data_;
        size_t segment_frame_count = seg_data.size() / sample_width;

        for (size_t j = 0; j < segment_frame_count; ++j) {
            std::copy(seg_data.begin() + j * sample_width, seg_data.begin() + (j + 1) * sample_width,
                      combined_data.begin() + j * sample_width * num_segments + i * sample_width);
        }
    }

    return AudioSegment(combined_data, sample_width, frame_rate, num_segments, sample_width * num_segments);
}

// Global functions
std::vector<WavSubChunk> extract_wav_headers(const std::vector<char>& data) {
    std::vector<WavSubChunk> subchunks;
    size_t pos = 12;  // The size of the RIFF chunk descriptor

    while (pos + 8 <= data.size() && subchunks.size() < 10) {
        if (pos + 8 + 4 > data.size()) break;  // Ensure we have enough data for the next chunk
        
        // Read subchunk ID
        std::string subchunk_id(data.begin() + pos, data.begin() + pos + 4);
        
        // Read subchunk size
        uint32_t subchunk_size = 0;
        std::memcpy(&subchunk_size, data.data() + pos + 4, sizeof(subchunk_size));
        subchunk_size = __builtin_bswap32(subchunk_size);  // Convert from little-endian to host byte order

        subchunks.emplace_back(subchunk_id, pos, subchunk_size);

        if (subchunk_id == "data") {
            // 'data' is the last subchunk
            break;
        }

        pos += subchunk_size + 8;
    }

    return subchunks;
}

WavData read_wav_audio(const std::vector<char>& data, const std::vector<WavSubChunk>* headers) {
    std::vector<WavSubChunk> hdrs;
    if (headers) {
        hdrs = *headers;
    } else {
        hdrs = extract_wav_headers(data);
    }

    // Find 'fmt ' subchunk
    auto fmt_iter = std::find_if(hdrs.begin(), hdrs.end(), [](const WavSubChunk& subchunk) {
        return subchunk.id == "fmt ";
    });

    if (fmt_iter == hdrs.end() || fmt_iter->size < 16) {
        throw CouldntDecodeError("Couldn't find fmt header in wav data");
    }

    const WavSubChunk& fmt = *fmt_iter;
    size_t pos = fmt.position + 8;

    // Read audio format
    uint16_t audio_format = 0;
    std::memcpy(&audio_format, data.data() + pos, sizeof(audio_format));
    audio_format = __builtin_bswap16(audio_format);  // Convert from little-endian to host byte order

    if (audio_format != 1 && audio_format != 0xFFFE) {
        throw CouldntDecodeError("Unknown audio format 0x" + std::to_string(audio_format) + " in wav data");
    }

    // Read channels
    uint16_t channels = 0;
    std::memcpy(&channels, data.data() + pos + 2, sizeof(channels));
    channels = __builtin_bswap16(channels);  // Convert from little-endian to host byte order

    // Read sample rate
    uint32_t sample_rate = 0;
    std::memcpy(&sample_rate, data.data() + pos + 4, sizeof(sample_rate));
    sample_rate = __builtin_bswap32(sample_rate);  // Convert from little-endian to host byte order

    // Read bits per sample
    uint16_t bits_per_sample = 0;
    std::memcpy(&bits_per_sample, data.data() + pos + 14, sizeof(bits_per_sample));
    bits_per_sample = __builtin_bswap16(bits_per_sample);  // Convert from little-endian to host byte order

    // Find 'data' subchunk
    auto data_hdr_iter = std::find_if(hdrs.rbegin(), hdrs.rend(), [](const WavSubChunk& subchunk) {
        return subchunk.id == "data";
    });

    if (data_hdr_iter == hdrs.rend()) {
        throw CouldntDecodeError("Couldn't find data header in wav data");
    }

    const WavSubChunk& data_hdr = *data_hdr_iter;
    pos = data_hdr.position + 8;

    return WavData{
        audio_format,
        channels,
        sample_rate,
        bits_per_sample,
        std::vector<char>(data.begin() + pos, data.begin() + pos + data_hdr.size)
    };
}

void fix_wav_headers(std::vector<char>& data) {
    std::vector<WavSubChunk> headers = extract_wav_headers(data);

    if (headers.empty() || headers.back().id != "data") {
        return;
    }

    if (data.size() > 0xFFFFFFFF) {  // Check if file size exceeds 4GB
        throw CouldntDecodeError("Unable to process >4GB files");
    }

    // Update the RIFF chunk size
    size_t riff_chunk_size = data.size() - 8;
    std::memcpy(data.data() + 4, &riff_chunk_size, sizeof(riff_chunk_size));

    // Update the data subchunk size
    const WavSubChunk& data_chunk = headers.back();
    size_t data_size = data.size() - data_chunk.position - 8;
    std::memcpy(data.data() + data_chunk.position + 4, &data_size, sizeof(data_size));
}

// Helper function to execute a command and capture output
std::string exec_command(const std::vector<std::string>& command_args, const std::string& stdin_data = "") {
    std::string command;
    for (const auto& arg : command_args) {
        command += arg + " ";
    }

    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);

    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

// Function to handle temporary file creation and cleanup
std::filesystem::path create_temp_file(const std::string& data) {
    // Create a temporary file path with a unique name
    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "tempfile_XXXXXX.tmp";
    
    // Convert to std::string for use with mkstemp
    std::string temp_path_string = temp_path.string();

    // Generate a unique file path
    int fd = mkstemp(temp_path_string.data());
    if (fd == -1) {
        throw std::runtime_error("Unable to create temporary file");
    }

    // Update the path with the actual unique file name
    temp_path = std::filesystem::path(temp_path_string);

    // Use RAII to open the file and ensure it closes properly
    std::ofstream temp_file(temp_path, std::ios::binary);
    if (!temp_file) {
        throw std::runtime_error("Unable to open the temporary file");
    }

    // Write data
    temp_file.write(data.data(), data.size());
    if (!temp_file) {
        throw std::runtime_error("Error writing to the temporary file");
    }

    // No need to explicitly close; RAII will handle it
    return temp_path;
}

AudioSegment from_file_using_temporary_files(const std::string& file_path, const std::string& format = "", const std::string& codec = "", 
                                             const std::vector<std::string>& parameters = {}, double start_second = 0, double duration = 0) {
    // Read input file data
    std::ifstream input_file(file_path, std::ios::binary | std::ios::ate);
    if (!input_file) throw std::runtime_error("Unable to open file: " + file_path);
    std::streamsize size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    std::string file_data(size, '\0');
    if (!input_file.read(file_data.data(), size)) throw std::runtime_error("Unable to read file data");

    // Create temporary input file
    std::filesystem::path temp_input_path = create_temp_file(file_data);

    // Temporary output file
    std::filesystem::path temp_output_path = std::filesystem::temp_directory_path() / "tempfile_output.wav";

    // Build ffmpeg command
    std::vector<std::string> conversion_command = {
        "ffmpeg", "-y", "-i", temp_input_path.string(), "-vn", "-f", "wav", temp_output_path.string()
    };

    if (!format.empty()) {
        conversion_command.insert(conversion_command.begin() + 3, "-f");
        conversion_command.insert(conversion_command.begin() + 4, format);
    }

    if (!codec.empty()) {
        conversion_command.push_back("-acodec");
        conversion_command.push_back(codec);
    }

    if (start_second > 0) {
        conversion_command.push_back("-ss");
        conversion_command.push_back(std::to_string(start_second));
    }

    if (duration > 0) {
        conversion_command.push_back("-t");
        conversion_command.push_back(std::to_string(duration));
    }

    conversion_command.insert(conversion_command.end(), parameters.begin(), parameters.end());

    // Execute command
    std::string command_output = exec_command(conversion_command);

    // Load the output audio segment
    AudioSegment audio_segment = AudioSegment::from_wav(temp_output_path.string());

    // Clean up temporary files
    std::filesystem::remove(temp_input_path);
    std::filesystem::remove(temp_output_path);

    return audio_segment;
}

} // namespace cppdub
