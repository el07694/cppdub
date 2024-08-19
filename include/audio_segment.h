#ifndef AUDIO_SEGMENT_H
#define AUDIO_SEGMENT_H

#pragma once


#include <algorithm>
#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include <fstream>
#include <sstream>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/samplefmt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/avutil.h>
#include <libavutil/log.h>
#include <filesystem>
#include <limits>
#include "exceptions.h"


#include "cppaudioop.h"



// Forward declaration of custom hash function
struct AudioSegmentHash;

namespace cppdub {
/*
// Exception class
class CouldntDecodeError : public std::runtime_error {
public:
    explicit CouldntDecodeError(const std::string& message)
        : std::runtime_error(message) {}
};
*/

// Struct to represent a range
struct Range {
	uint32_t start_time_ms;
	uint32_t end_time_ms;
};

std::unordered_map<std::string, std::string> AUDIO_FILE_EXT_ALIASES = {
        {"m4a", "mp4"},
        {"wave", "wav"}
};

// Struct to hold WAV file sub-chunk information
struct WavSubChunk {
    std::string id;
    size_t position;
    size_t size;

    WavSubChunk(const std::string& id, size_t pos, size_t sz)
        : id(id), position(pos), size(sz) {}
};

// Struct to hold WAV file data
struct WavData {
    uint16_t audio_format;
    uint16_t channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    std::vector<char> raw_data;
};

// AudioSegment class
class AudioSegment {
public:
    // Constructors and Destructor
    AudioSegment();
    explicit AudioSegment(const std::string& file_path);
	//AudioSegment(const char* data, size_t size);
    AudioSegment(const char* data, size_t size, const std::map<std::string, int>& metadata);
    AudioSegment(const std::vector<uint8_t>& data);
    ~AudioSegment();

    uint16_t get_sample_width() const { return sample_width_; }
    uint32_t get_frame_rate() const { return frame_rate_; }
    uint16_t get_channels() const { return channels_; }
	

    
    // Static method to create an AudioSegment from file
    static AudioSegment from_file(const std::string& file_path, const std::string& format, const std::map<std::string, int>& parameters);

    static AudioSegment from_file(const std::string& file_path, const std::string& format,
                                     const std::string& codec, const std::map<std::string, int>& parameters,
                                     int start_second, int duration);
							


    static AudioSegment from_mp3(const std::string& file, const std::map<std::string, int>& parameters);
    static AudioSegment from_flv(const std::string& file, const std::map<std::string, int>& parameters);
    static AudioSegment from_ogg(const std::string& file, const std::map<std::string, int>& parameters);
    static AudioSegment from_wav(const std::string& file, const std::map<std::string, int>& parameters);
    static AudioSegment from_raw(const std::string& file, int sample_width, int frame_rate, int channels);

    // Static method to create an AudioSegment from a WAV file
    static AudioSegment _from_safe_wav(const std::string& file_path);

    // Method to export an AudioSegment to a file with given options
    std::ofstream export_segment(const std::string& out_f, const std::string& format, const std::string& codec, const std::string& bitrate, const std::vector<std::string>& parameters, const std::map<std::string, std::string>& tags, const std::string& id3v2_version, const std::string& cover);

    // Method to get a specific frame from the AudioSegment
    std::vector<uint8_t> get_frame(int index) const;
	
	// Method to get the number of frames for the given number of milliseconds,
    // or the number of frames in the whole AudioSegment if no argument is provided
    double frame_count(int ms = -1) const;

    // Sets the sample width and returns a new AudioSegment
    AudioSegment set_sample_width(int sample_width) const;
	
    // Iterator support
    class Iterator {
    public:
        Iterator(const std::vector<char>::const_iterator& it) : it_(it) {}

        bool operator!=(const Iterator& other) const {
            return it_ != other.it_;
        }

        const char& operator*() const {
            return *it_;
        }

        Iterator& operator++() {
            ++it_;
            return *this;
        }

    private:
        std::vector<char>::const_iterator it_;
    };

    Iterator begin() const {
        return Iterator(data_.cbegin());
    }

    Iterator end() const {
        return Iterator(data_.cend());
    }

    // Static methods
    static std::string ffmpeg();
    static void ffmpeg(const std::string& value);
    static const std::unordered_map<std::string, std::string>& default_codecs();
	


    // Getter for raw audio data
    std::vector<char> raw_data() const;

    // Methods
    void load_data_from_file(const std::string& file_path);
    void load_data_from_memory(const uint8_t* data, size_t size);

    // Template method to get array of samples
	/*
    template<typename T>
    std::vector<T> get_array_of_samples() const {
        static_assert(std::is_arithmetic<T>::value, "Template parameter must be an arithmetic type");

        size_t num_samples = data_.size() / sizeof(T);
        std::vector<T> samples(num_samples);
        std::memcpy(samples.data(), data_.data(), data_.size());
        return samples;
    }
	*/

    size_t length_in_milliseconds() const;

    // New method to get data by millisecond range
    size_t millisecond_to_frame(size_t milliseconds, uint32_t frame_rate);
    std::vector<char> get_data_by_millisecond(size_t start_ms, size_t end_ms) const;

    // New method to get a slice of samples
    std::vector<char> get_sample_slice(uint32_t start_sample, uint32_t end_sample) const;
    AudioSegment slice(int64_t start_ms, int64_t end_ms) const;

    // Operator overloads
    bool operator==(const AudioSegment& other) const;
    bool operator!=(const AudioSegment& other) const;
    AudioSegment operator+(const AudioSegment& other) const;
    AudioSegment operator+(int db) const;
    friend AudioSegment operator+(int db, const AudioSegment& segment);
    friend AudioSegment operator-(const AudioSegment& segment, int db); // Corrected operator overload
    friend AudioSegment operator-(const AudioSegment& lhs, const AudioSegment& rhs); // Corrected operator overload
    friend AudioSegment operator*(const AudioSegment& segment, int times); // Corrected operator overload
    friend AudioSegment operator*(const AudioSegment& lhs, const AudioSegment& rhs); // Corrected operator overload
    AudioSegment operator[](Range range) const;
	
	/**
     * Overlay the provided segment on to this segment starting at the specified position 
     * and using the specified looping behavior.
     *
     * @param seg The audio segment to overlay on top of this one.
     * @param position The position to start overlaying the provided segment into this one.
     * @param loop Loop seg as many times as necessary to match this segment's length.
     * @param times Loop seg the specified number of times or until it matches this segment's length.
     * @param gain_during_overlay Changes this segment's volume by the specified amount during the overlay.
     * @return A new AudioSegment with the overlay applied.
     */
    AudioSegment overlay(const AudioSegment& seg, int position, bool loop, int times, int gain_during_overlay) const;


	/**
     * Append the provided segment to this segment with an optional crossfade.
     *
     * @param seg The audio segment to append to this one.
     * @param crossfade Duration of the crossfade in milliseconds.
     * @return A new AudioSegment with the appended content.
     */
    AudioSegment append(const AudioSegment& seg, int crossfade) const;


	/**
     * Fade the volume of this audio segment.
     *
     * @param to_gain Resulting volume change in dB.
     * @param from_gain Starting volume change in dB.
     * @param start When to start fading in milliseconds. Default is the beginning of the segment.
     * @param end When to end fading in milliseconds. Default is the end of the segment.
     * @param duration Duration of the fade in milliseconds. Default is until the end of the segment.
     * @return A new AudioSegment with the faded volume.
     */
    AudioSegment fade(double to_gain, double from_gain, 
                                int start, int end, int duration) const;
	
	// Declaration of fade_out method
	AudioSegment fade_out(int duration) const;

	// Declaration of fade_in method
	AudioSegment fade_in(int duration) const;
	
	// Declaration of reverse method
	AudioSegment reverse() const;
	
	// Declaration of _repr_html_ method
	//std::string repr_html() const;

	// Declaration of the split_to_mono method
	std::vector<AudioSegment> split_to_mono() const;

	// Declaration of the rms method
	double rms() const;

	// Declaration of the dBFS method
	float dBFS() const;

	// Declaration of the max method
	int max() const;
	
	// Declaration of the max_dBFS method
	float max_dBFS() const;
	
	// Declaration of the duration_seconds method
	double duration_seconds() const;
	
	// Declaration of the get_dc_offset method
    double get_dc_offset(int channel) const;
	
	// Declaration of the remove_dc_offset method
    AudioSegment AudioSegment::remove_dc_offset(int channel, double offset) const;
    double max_possible_amplitude() const;
	
	uint32_t time_to_sample_index(int time_in_ms);

    // Synchronize metadata (channels, frame rate, sample width) across multiple AudioSegment instances
    static std::vector<AudioSegment> sync(const std::vector<AudioSegment>& segments);

    // Method to parse a position, considering negative values and infinity
    uint32_t parse_position(double val) const;

    // Static method to create an empty audio segment with default metadata
    static AudioSegment empty();

    // Static method to generate a silent audio segment
    static AudioSegment silent(size_t duration_ms, uint32_t frame_rate);

    // Static method to combine mono audio segments into a multi-channel segment
    static AudioSegment from_mono_audiosegments(const std::vector<AudioSegment>& mono_segments);

    // Static method to create an AudioSegment from a file with conversion
    static AudioSegment from_file_using_temporary_files(const std::string& file_path, const std::string& format = "", const std::string& codec = "", const std::vector<std::string>& parameters = {}, double start_second = -1, double duration = -1);

    std::vector<char> execute_conversion(const std::string& conversion_command, const std::vector<char>& stdin_data) const;

    // Helper function to execute a command and capture output
    static std::vector<char> execute_command(const std::string& command, const std::vector<char>& input_data);
    
	
	AudioSegment _spawn(const std::vector<char>& data) const;
	
	std::vector<char> convert_audio_data(const std::vector<char>& data, int src_rate, int dest_rate) const;
	
	AudioSegment set_frame_rate(int frame_rate) const;
	
	std::vector<char> convert_to_stereo(const std::vector<char>& data) const;
	std::vector<char> convert_to_mono(const std::vector<char>& data) const;
	std::vector<char> merge_channels(const std::vector<std::vector<char>>& channel_data) const;
	std::vector<std::vector<char>> split_to_channels() const;
	AudioSegment set_channels(int channels) const;

private:
    // Helper method to convert milliseconds to frame count
    uint32_t frame_count(uint32_t ms) const;

    // Initialization function for FFmpeg
    void initialize_ffmpeg();

    // Helper method for applying gain
    AudioSegment apply_gain(int db) const;

    // Member variables
    std::vector<char> data_;
    std::string file_path_;
    uint16_t sample_width_;
    uint32_t frame_rate_;
    uint16_t channels_;
    uint32_t frame_width_;
    std::string format_;
    uint32_t data_size_;  // Assuming data_size_ holds the size of audio data


    static std::string ffmpeg_converter_;
    static const std::unordered_map<std::string, std::string> DEFAULT_CODECS;

    //std::vector<uint8_t> execute_conversion(const std::string& conversion_command, const std::vector<uint8_t>& stdin_data) const;


    // Private constructor for creating with default values
	//Warning check declaration
    AudioSegment(std::vector<char> data, uint16_t sample_width, uint32_t frame_rate, uint16_t channels, uint32_t frame_width);
};

// Global functions
std::vector<WavSubChunk> extract_wav_headers(const std::vector<char>& data);
WavData read_wav_audio(const std::vector<char>& data, const std::vector<WavSubChunk>* headers);
void fix_wav_headers(std::vector<char>& data);
double db_to_float(double db);
int milliseconds_to_frames(int milliseconds, int frame_rate);
std::vector<AudioSegment> _sync(const std::vector<AudioSegment>& segs);

// Function to execute a command and capture output
std::string exec_command(const std::vector<std::string>& command_args, const std::string& stdin_data);

// Function to handle temporary file creation and cleanup
std::filesystem::path create_temp_file(const std::string& data);



// Function to convert an audio file to an AudioSegment using temporary files
AudioSegment from_file_using_temporary_files(const std::string& file_path, const std::string& format = "", const std::string& codec = "", 
                                             const std::vector<std::string>& parameters = {}, float start_second = 0, float duration = 0);

// Custom hash function
struct AudioSegmentHash {
    std::size_t operator()(const AudioSegment& segment) const;
};

const std::map<std::string, int>& get_empty_map();

} // namespace cppdub

#endif // AUDIO_SEGMENT_H
