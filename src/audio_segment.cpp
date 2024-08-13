#include "audio_segment.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/samplefmt.h>
#include <libavutil/channel_layout.h>
#include <iomanip>
#include <iostream>

#include <string>
#include <map>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/samplefmt.h>
#include <libavutil/channel_layout.h>

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

// Helper function to convert various data types to raw audio data
std::vector<uint8_t> convert_to_raw_data(const void* data, size_t size) {
    const uint8_t* raw_data = static_cast<const uint8_t*>(data);
    return std::vector<uint8_t>(raw_data, raw_data + size);
}

// Constructor and Destructor
AudioSegment::AudioSegment()
    : sample_width_(2), frame_rate_(44100), channels_(2), frame_width_(0) {
    initialize_ffmpeg();
}

AudioSegment::AudioSegment(const std::string& file_path)
    : file_path_(file_path), sample_width_(2), frame_rate_(44100), channels_(2), frame_width_(0) {
    initialize_ffmpeg();
    // Add file loading logic here
}

AudioSegment::AudioSegment(const uint8_t* data, size_t size, const std::map<std::string, int>& metadata)
    : sample_width_(metadata.at("sample_width")), frame_rate_(metadata.at("frame_rate")),
      frame_width_(metadata.at("frame_width")), channels_(metadata.at("channels")) {
    data_.assign(data, data + size);
}

AudioSegment::~AudioSegment() {
    avformat_network_deinit();
}

// Static methods
AudioSegment AudioSegment::empty() {
    return AudioSegment();
}

AudioSegment AudioSegment::silent(double duration) {
    int num_samples = static_cast<int>(duration * 44100); // Assuming 44100 Hz sample rate
    std::vector<uint8_t> silence(num_samples * 2, 0); // 2 bytes per sample
    return AudioSegment(silence.data(), silence.size(), {
        {"sample_width", 2},
        {"frame_rate", 44100},
        {"channels", 1},
        {"frame_width", 2}
    });
}

AudioSegment AudioSegment::from_mono_audiosegments(const std::vector<AudioSegment>& segments) {
    std::vector<uint8_t> combined_data;
    for (const auto& segment : segments) {
        combined_data.insert(combined_data.end(), segment.raw_data().begin(), segment.raw_data().end());
    }
    return AudioSegment(combined_data.data(), combined_data.size(), {
        {"sample_width", 2},
        {"frame_rate", 44100},
        {"channels", 1},
        {"frame_width", 2}
    });
}

AudioSegment AudioSegment::from_file_using_temporary_files(const std::string& file_path) {
    // Placeholder: Implement logic using temporary files
    return from_file(file_path);
}

AudioSegment AudioSegment::from_file(const std::string& file_path) {
    // Implement file loading based on file extension
    std::string extension = file_path.substr(file_path.find_last_of('.') + 1);
    if (AUDIO_FILE_EXT_ALIASES.find(extension) != AUDIO_FILE_EXT_ALIASES.end()) {
        // Call appropriate loading function based on extension
    }
    return AudioSegment();
}

AudioSegment AudioSegment::from_mp3(const std::string& file_path) {
    AudioSegment segment;
    segment.file_path_ = file_path;
    // Implement MP3 loading logic here
    return segment;
}

AudioSegment AudioSegment::from_flv(const std::string& file_path) {
    AudioSegment segment;
    segment.file_path_ = file_path;
    // Implement FLV loading logic here
    return segment;
}

AudioSegment AudioSegment::from_ogg(const std::string& file_path) {
    AudioSegment segment;
    segment.file_path_ = file_path;
    // Implement OGG loading logic here
    return segment;
}

AudioSegment AudioSegment::from_wav(const std::string& file_path) {
    AudioSegment segment;
    segment.file_path_ = file_path;
    // Implement WAV loading logic here
    return segment;
}

AudioSegment AudioSegment::from_raw(const std::string& file_path) {
    AudioSegment segment;
    segment.file_path_ = file_path;
    // Implement RAW loading logic here
    return segment;
}

AudioSegment AudioSegment::_from_safe_wav(const std::string& file_path) {
    // Implement safe WAV loading
    return from_wav(file_path);
}

std::vector<uint8_t> AudioSegment::raw_data() const {
    return data_;
}

std::vector<uint8_t> AudioSegment::get_array_of_samples() const {
    return data_;
}

std::string AudioSegment::array_type() const {
    return "uint8_t";
}

size_t AudioSegment::__len__() const {
    return data_.size();
}

bool AudioSegment::__eq__(const AudioSegment& other) const {
    return data_ == other.data_;
}

size_t AudioSegment::__hash__() const {
    // Implement a hash function
    return std::hash<std::string>{}(std::string(data_.begin(), data_.end()));
}

bool AudioSegment::__ne__(const AudioSegment& other) const {
    return !(__eq__(other));
}

std::vector<uint8_t>::const_iterator AudioSegment::__iter__() const {
    return data_.begin();
}

int16_t AudioSegment::__getitem__(size_t index) const {
    if (index >= data_.size() / (sample_width_ / 8)) {
        throw std::out_of_range("Index out of range.");
    }
    return *reinterpret_cast<const int16_t*>(data_.data() + index * (sample_width_ / 8));
}

std::vector<uint8_t> AudioSegment::get_sample_slice(size_t start, size_t end) const {
    if (start >= data_.size() || end > data_.size() || start > end) {
        throw std::out_of_range("Slice indices are out of range.");
    }
    return std::vector<uint8_t>(data_.begin() + start, data_.begin() + end);
}

void AudioSegment::bounded(size_t size) {
    if (size > data_.size()) {
        throw std::out_of_range("Size is larger than data size.");
    }
    data_.resize(size);
}

AudioSegment AudioSegment::__add__(const AudioSegment& other) const {
    if (frame_rate_ != other.frame_rate_ || sample_width_ != other.sample_width_ || channels_ != other.channels_) {
        throw std::invalid_argument("Audio segments must have the same parameters.");
    }
    std::vector<uint8_t> combined_data = data_;
    combined_data.insert(combined_data.end(), other.data_.begin(), other.data_.end());
    return AudioSegment(combined_data.data(), combined_data.size(), {
        {"sample_width", sample_width_},
        {"frame_rate", frame_rate_},
        {"channels", channels_},
        {"frame_width", frame_width_}
    });
}

AudioSegment AudioSegment::__radd__(const AudioSegment& other) const {
    return __add__(other);
}

AudioSegment AudioSegment::__sub__(const AudioSegment& other) const {
    // Subtracting audio segments is not straightforward, so handle it as an exception for now.
    throw std::runtime_error("Subtraction of audio segments is not supported.");
}

AudioSegment AudioSegment::__mul__(double factor) const {
    AudioSegment result = *this;
    // Multiplying audio data with a factor would typically involve adjusting amplitude levels.
    // Implement appropriate logic if needed.
    return result;
}

AudioSegment AudioSegment::_spawn() {
    // Clone the current AudioSegment object.
    return AudioSegment(data_.data(), data_.size(), {
        {"sample_width", sample_width_},
        {"frame_rate", frame_rate_},
        {"channels", channels_},
        {"frame_width", frame_width_}
    });
}

std::tuple<AudioSegment, AudioSegment> AudioSegment::_sync(const AudioSegment& seg1, const AudioSegment& seg2) {
    // Syncing audio segments
    // Assuming same length and format
    return std::make_tuple(seg1, seg2);
}

size_t AudioSegment::_parse_position(const std::string& position) const {
    return std::stoul(position);
}

std::vector<uint8_t> AudioSegment::get_frame(size_t index) const {
    size_t frame_size = frame_width_;
    size_t start = index * frame_size;
    if (start >= data_.size()) {
        throw std::out_of_range("Frame index out of range.");
    }
    size_t end = std::min(start + frame_size, data_.size());
    return std::vector<uint8_t>(data_.begin() + start, data_.begin() + end);
}

size_t AudioSegment::frame_count() const {
    return data_.size() / frame_width_;
}

void AudioSegment::set_sample_width(int width) {
    sample_width_ = width;
}

void AudioSegment::set_frame_rate(int rate) {
    frame_rate_ = rate;
}

void AudioSegment::set_channels(int channels) {
    channels_ = channels;
}

std::vector<AudioSegment> AudioSegment::split_to_mono() const {
    std::vector<AudioSegment> mono_segments;
    size_t sample_size = sample_width_ / 8;
    size_t frame_size = frame_width_;

    size_t num_samples = data_.size() / (sample_size * channels_);
    size_t mono_sample_size = sample_size;

    std::vector<uint8_t> left_channel(num_samples * mono_sample_size);
    std::vector<uint8_t> right_channel(num_samples * mono_sample_size);

    for (size_t i = 0; i < num_samples; ++i) {
        size_t offset = i * frame_size;
        std::memcpy(left_channel.data() + i * mono_sample_size, data_.data() + offset, mono_sample_size);
        std::memcpy(right_channel.data() + i * mono_sample_size, data_.data() + offset + mono_sample_size, mono_sample_size);
    }

    mono_segments.push_back(AudioSegment(left_channel.data(), left_channel.size(), {
        {"sample_width", sample_width_},
        {"frame_rate", frame_rate_},
        {"channels", 1},
        {"frame_width", mono_sample_size}
    }));

    mono_segments.push_back(AudioSegment(right_channel.data(), right_channel.size(), {
        {"sample_width", sample_width_},
        {"frame_rate", frame_rate_},
        {"channels", 1},
        {"frame_width", mono_sample_size}
    }));

    return mono_segments;
}

double AudioSegment::rms() const {
    double sum = 0.0;
    size_t sample_count = data_.size() / (sample_width_ / 8);
    for (size_t i = 0; i < sample_count; ++i) {
        int16_t sample = __getitem__(i);
        sum += sample * sample;
    }
    return std::sqrt(sum / sample_count);
}

double AudioSegment::dBFS() const {
    return 20 * std::log10(rms() / max_possible_amplitude());
}

int16_t AudioSegment::max() const {
    int16_t max_value = std::numeric_limits<int16_t>::min();
    size_t sample_count = data_.size() / (sample_width_ / 8);
    for (size_t i = 0; i < sample_count; ++i) {
        int16_t sample = __getitem__(i);
        if (sample > max_value) {
            max_value = sample;
        }
    }
    return max_value;
}

double AudioSegment::max_possible_amplitude() const {
    return std::pow(2, (sample_width_ * 8 - 1));
}

double AudioSegment::max_dBFS() const {
    return dBFS();
}

double AudioSegment::duration_seconds() const {
    return static_cast<double>(data_.size()) / (frame_rate_ * (sample_width_ / 8) * channels_);
}

int16_t AudioSegment::get_dc_offset() const {
    int16_t sum = 0;
    size_t sample_count = data_.size() / (sample_width_ / 8);
    for (size_t i = 0; i < sample_count; ++i) {
        sum += __getitem__(i);
    }
    return sum / sample_count;
}

void AudioSegment::remove_dc_offset() {
    int16_t offset = get_dc_offset();
    size_t sample_count = data_.size() / (sample_width_ / 8);
    for (size_t i = 0; i < sample_count; ++i) {
        int16_t* sample = reinterpret_cast<int16_t*>(data_.data() + i * (sample_width_ / 8));
        *sample -= offset;
    }
}

void AudioSegment::remove_data_dc() {
    // Implement if needed
}

void AudioSegment::apply_gain(double gain) {
    size_t sample_count = data_.size() / (sample_width_ / 8);
    for (size_t i = 0; i < sample_count; ++i) {
        int16_t* sample = reinterpret_cast<int16_t*>(data_.data() + i * (sample_width_ / 8));
        *sample = std::clamp(static_cast<int16_t>(*sample * gain), std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max());
    }
}

AudioSegment AudioSegment::overlay(const AudioSegment& other, double position) const {
    if (frame_rate_ != other.frame_rate_ || sample_width_ != other.sample_width_ || channels_ != other.channels_) {
        throw std::invalid_argument("Audio segments must have the same parameters.");
    }

    size_t offset = static_cast<size_t>(position * frame_rate_);
    std::vector<uint8_t> combined_data(data_);

    if (offset + other.data_.size() > combined_data.size()) {
        combined_data.resize(offset + other.data_.size());
    }

    std::memcpy(combined_data.data() + offset, other.data_.data(), other.data_.size());
    return AudioSegment(combined_data.data(), combined_data.size(), {
        {"sample_width", sample_width_},
        {"frame_rate", frame_rate_},
        {"channels", channels_},
        {"frame_width", frame_width_}
    });
}

void AudioSegment::append(const AudioSegment& other) {
    if (frame_rate_ != other.frame_rate_ || sample_width_ != other.sample_width_ || channels_ != other.channels_) {
        throw std::invalid_argument("Audio segments must have the same parameters.");
    }
    data_.insert(data_.end(), other.data_.begin(), other.data_.end());
}

AudioSegment AudioSegment::fade(double start_time, double end_time) const {
    // Implement fading effect
    return *this;
}

AudioSegment AudioSegment::fade_out(double duration) const {
    // Implement fade-out effect
    return *this;
}

AudioSegment AudioSegment::fade_in(double duration) const {
    // Implement fade-in effect
    return *this;
}

AudioSegment AudioSegment::reverse() const {
    std::vector<uint8_t> reversed_data(data_.rbegin(), data_.rend());
    return AudioSegment(reversed_data.data(), reversed_data.size(), {
        {"sample_width", sample_width_},
        {"frame_rate", frame_rate_},
        {"channels", channels_},
        {"frame_width", frame_width_}
    });
}

std::string AudioSegment::_repr_html_() const {
    std::ostringstream oss;
    oss << "<audio controls><source src=\"" << file_path_ << "\" type=\"" << AUDIO_FILE_EXT_ALIASES.at(file_path_.substr(file_path_.find_last_of('.') + 1)) << "\">Your browser does not support the audio tag.</audio>";
    return oss.str();
}

void AudioSegment::save_to_file(const std::string& file_path) const {
    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing.");
    }
    ofs.write(reinterpret_cast<const char*>(data_.data()), data_.size());
}

void AudioSegment::apply_ffmpeg_filter(const std::string& filter) {
    // Apply FFmpeg filter
}

void AudioSegment::initialize_ffmpeg() {
    av_register_all();
}

void AudioSegment::set_ffmpeg_options() {
    // Set FFmpeg options
}

WavData AudioSegment::read_wav_audio(const std::string& file_path) {
    WavData wav_data;
    // Implement WAV audio reading
    return wav_data;
}

void AudioSegment::fix_wav_headers(const std::string& file_path) {
    // Implement fixing WAV headers
}

std::unordered_map<std::string, AudioSegment::EffectFunction> AudioSegment::effects;

void AudioSegment::register_effect(const std::string& effect_name, EffectFunction effect_function) {
    effects[effect_name] = effect_function;
}

AudioSegment::EffectFunction AudioSegment::get_effect(const std::string& effect_name) {
    auto it = effects.find(effect_name);
    if (it != effects.end()) {
        return it->second;
    }
    throw std::runtime_error("Effect not found: " + effect_name);
}