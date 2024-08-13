#ifndef AUDIO_SEGMENT_H
#define AUDIO_SEGMENT_H

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <cstring>  // For std::memcpy


extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavutil/opt.h>
#include <libavutil/frame.h>
#include <libavutil/samplefmt.h>
#include <libavutil/channel_layout.h>
}


// Descriptor class to handle getter and setter
template <typename T>
class ClassPropertyDescriptor {
public:
    using Getter = std::function<T()>;
    using Setter = std::function<void(T)>;

    ClassPropertyDescriptor(Getter fget, Setter fset = nullptr)
        : fget_(fget), fset_(fset) {}

    T get() const {
        if (!fget_) {
            throw std::runtime_error("Getter function not set");
        }
        return fget_();
    }

    void set(T value) {
        if (!fset_) {
            throw std::runtime_error("Setter function not set");
        }
        fset_(value);
    }

    void set_setter(Setter fset) {
        fset_ = fset;
    }

private:
    Getter fget_;
    Setter fset_;
};

// Helper function to create ClassPropertyDescriptor
template <typename T>
ClassPropertyDescriptor<T> classproperty(std::function<T()> getter, std::function<void(T)> setter = nullptr) {
    return ClassPropertyDescriptor<T>(getter, setter);
}


// Define a map to hold audio file extension aliases
const std::unordered_map<std::string, std::string> AUDIO_FILE_EXT_ALIASES = {
    {"m4a", "mp4"},
    {"wave", "wav"},
};

// Define structs to hold WAV file information
struct WavSubChunk {
    std::string id;
    size_t position;
    size_t size;
};

// Function to extract WAV headers
std::vector<WavSubChunk> extract_wav_headers(const std::vector<char>& data);

struct WavData {
    uint16_t audio_format;
    uint16_t channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    std::vector<char> raw_data;
};

struct CouldntDecodeError : public std::runtime_error {
    explicit CouldntDecodeError(const std::string& message)
        : std::runtime_error(message) {}
};

// Function to read WAV audio data
WavData read_wav_audio(const std::vector<char>& data, const std::vector<WavSubChunk>* headers = nullptr);

// Function to fix WAV headers
void fix_wav_headers(std::vector<char>& data);

// Custom exception class
class MissingAudioParameter : public std::runtime_error {
public:
    explicit MissingAudioParameter(const std::string& msg) : std::runtime_error(msg) {}
};

class CouldntDecodeError : public std::runtime_error {
public:
    explicit CouldntDecodeError(const std::string& msg) : std::runtime_error(msg) {}
};

class AudioSegment {
public:
    // Constructors and Destructor
    AudioSegment();
    AudioSegment(const std::string& file_path);
    AudioSegment(const uint8_t* data, size_t size, const std::map<std::string, int>& metadata);
    ~AudioSegment();

    // Static methods
    static AudioSegment empty();
    static AudioSegment silent(double duration);
    static AudioSegment from_mono_audiosegments(const std::vector<AudioSegment>& segments);
    static AudioSegment from_file_using_temporary_files(const std::string& file_path);
    static AudioSegment from_file(const std::string& file_path);
    static AudioSegment from_mp3(const std::string& file_path);
    static AudioSegment from_flv(const std::string& file_path);
    static AudioSegment from_ogg(const std::string& file_path);
    static AudioSegment from_wav(const std::string& file_path);
    static AudioSegment from_raw(const std::string& file_path);
    static AudioSegment _from_safe_wav(const std::string& file_path);

    AudioSegment set_frame_rate(int frame_rate) const;
    AudioSegment set_channels(int channels) const;

	
	using EffectFunction = std::function<void(AudioSegment&)>;
    
    static void register_effect(const std::string& effect_name, EffectFunction effect_function);
    static EffectFunction get_effect(const std::string& effect_name);


    // Instance methods
    std::vector<uint8_t> raw_data() const;
    std::vector<uint8_t> get_array_of_samples() const;
    std::string array_type() const;
    size_t __len__() const;
    bool __eq__(const AudioSegment& other) const;
    size_t __hash__() const;
    bool __ne__(const AudioSegment& other) const;
    std::vector<uint8_t>::const_iterator __iter__() const;
    int16_t __getitem__(size_t index) const;
    std::vector<uint8_t> get_sample_slice(size_t start, size_t end) const;
    void bounded(size_t size);
    AudioSegment __add__(const AudioSegment& other) const;
    AudioSegment __radd__(const AudioSegment& other) const;
    AudioSegment __sub__(const AudioSegment& other) const;
    AudioSegment __mul__(double factor) const;
    AudioSegment _spawn();
    static std::tuple<AudioSegment, AudioSegment> _sync(const AudioSegment& seg1, const AudioSegment& seg2);
    size_t _parse_position(const std::string& position) const;
    std::vector<uint8_t> get_frame(size_t index) const;
    size_t frame_count() const;
    void set_sample_width(int width);
    void set_frame_rate(int rate);
    void set_channels(int channels);
    std::vector<AudioSegment> split_to_mono() const;
    double rms() const;
    double dBFS() const;
    int16_t max() const;
    double max_possible_amplitude() const;
    double max_dBFS() const;
    double duration_seconds() const;
    int16_t get_dc_offset() const;
    void remove_dc_offset();
    void remove_data_dc();
    void apply_gain(double gain);
    AudioSegment overlay(const AudioSegment& other, double position) const;
    void append(const AudioSegment& other);
    AudioSegment fade(double start_time, double end_time) const;
    AudioSegment fade_out(double duration) const;
    AudioSegment fade_in(double duration) const;
    AudioSegment reverse() const;
    std::string _repr_html_() const;
    void save_to_file(const std::string& file_path) const;
    void apply_ffmpeg_filter(const std::string& filter);

private:
    void set_ffmpeg_options();
    void initialize_ffmpeg();
    static WavData read_wav_audio(const std::string& file_path);
    static void fix_wav_headers(const std::string& file_path);

    std::vector<char> convert_audio_data(const std::vector<char>& data, int src_rate, int dest_rate) const;
    std::vector<char> convert_to_stereo(const std::vector<char>& data) const;
    std::vector<char> convert_to_mono(const std::vector<char>& data) const;
    std::vector<char> merge_channels(const std::vector<std::vector<char>>& channel_data) const;
    std::vector<std::vector<char>> split_to_channels() const;

    // Member variables
    std::string file_path_;
    std::vector<uint8_t> data_;
    int sample_width_;
    int frame_rate_;
    int channels_;
    int frame_width_;
	
	// Map to store effects by name
    static std::unordered_map<std::string, EffectFunction> effects;
};

#endif // AUDIO_SEGMENT_H
