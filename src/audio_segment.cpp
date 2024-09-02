#define NOMINMAX

#include "audio_segment.h"
#include "utils.h"
#include <sys/stat.h>
#ifdef _WIN32
	#include <windows.h>
#else
	#include <unistd.h>
#endif
extern "C" {
	#include <libavformat/avformat.h>
	#include <libavcodec/avcodec.h>
	#include <libavutil/samplefmt.h>
	#include <libavutil/channel_layout.h>
	#include <libavutil/avutil.h>
	#include <libavutil/log.h>
	#include <libavutil/opt.h>
	#include <libavutil/channel_layout.h>
	#include <libswresample/swresample.h>
	#include <libavutil/imgutils.h>
	#include <libavutil/frame.h>
}

#include <unordered_map>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <array>
#include <type_traits>
#include <functional>
#include <vector>
#include <stdexcept>
#include <iterator>
#include <limits>
#include <sstream>
#include <memory>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <iomanip>
#include <algorithm>


namespace cppdub {

void AudioSegment::initialize_ffmpeg() {
	avformat_network_init();
	av_log_set_level(AV_LOG_INFO);
}

AudioSegment::AudioSegment()
	: sample_width_(2), frame_rate_(44100), channels_(2), frame_width_(4) {
	initialize_ffmpeg();
}

AudioSegment::AudioSegment(const char* data, size_t size, const std::map<std::string, int>& metadata)
	: sample_width_(metadata.at("sample_width")), frame_rate_(metadata.at("frame_rate")),
	  frame_width_(metadata.at("frame_width")), channels_(metadata.at("channels")), data_size_(size) {
	initialize_ffmpeg();
	data_.assign(data, data + size);
}

AudioSegment::AudioSegment(std::vector<char> data, uint16_t sample_width, uint32_t frame_rate, uint16_t channels, uint32_t frame_width): data_(std::move(data)), sample_width_(sample_width), frame_rate_(frame_rate), channels_(channels), frame_width_(frame_width), data_size_(data.size()) { 
	initialize_ffmpeg();
}
	  
AudioSegment::AudioSegment(const std::vector<uint8_t>& data) {
	initialize_ffmpeg();
	data_.assign(data.begin(), data.end());  // Convert from uint8_t to char
}

AudioSegment::~AudioSegment() {
	avformat_network_deinit();
	av_log_set_callback(nullptr);
}

AudioSegment AudioSegment::empty() {
	return AudioSegment(std::vector<char>{},2,44100,2,4);
}

AudioSegment AudioSegment::silent(size_t duration_ms, uint32_t frame_rate) {
	size_t frames = static_cast<size_t>(frame_rate * (duration_ms / 1000.0));

	size_t sample_width = 2;
	size_t frame_width = sample_width * 1;
	std::vector<char> data(frames * frame_width, 0);

	return AudioSegment(data, sample_width, frame_rate, 1, frame_width);
}

AudioSegment AudioSegment::from_file(const std::string& file_path, const std::string& format, const std::string& codec,
	const std::map<std::string, int>& parameters, int start_second, int duration) {

	avformat_network_init();
	av_log_set_level(AV_LOG_ERROR); // Adjust logging level as needed

	AVFormatContext* format_ctx = nullptr;
	if (avformat_open_input(&format_ctx, file_path.c_str(), nullptr, nullptr) != 0) {
		std::cerr << "Error: Could not open audio file." << std::endl;
		return AudioSegment();  // Return an empty AudioSegment on failure
	}

	if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
		std::cerr << "Error: Could not find stream information." << std::endl;
		avformat_close_input(&format_ctx);
		return AudioSegment();
	}

	int audio_stream_index = -1;
	for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
		if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
			audio_stream_index = i;
			break;
		}
	}

	if (audio_stream_index == -1) {
		std::cerr << "Error: Could not find audio stream." << std::endl;
		avformat_close_input(&format_ctx);
		return AudioSegment();
	}

	AVCodecParameters* codec_par = format_ctx->streams[audio_stream_index]->codecpar;
	const AVCodec* my_codec = avcodec_find_decoder(codec_par->codec_id);
	AVCodecContext* codec_ctx = avcodec_alloc_context3(my_codec);

	if (avcodec_parameters_to_context(codec_ctx, codec_par) < 0) {
		std::cerr << "Error: Could not initialize codec context." << std::endl;
		avformat_close_input(&format_ctx);
		return AudioSegment();
	}

	if (avcodec_open2(codec_ctx, my_codec, nullptr) < 0) {
		std::cerr << "Error: Could not open codec." << std::endl;
		avcodec_free_context(&codec_ctx);
		avformat_close_input(&format_ctx);
		return AudioSegment();
	}

	SwrContext* swr_ctx = swr_alloc();
	if (!swr_ctx) {
		std::cerr << "Error: Could not allocate SwrContext." << std::endl;
		avcodec_free_context(&codec_ctx);
		avformat_close_input(&format_ctx);
		return AudioSegment();
	}

	av_opt_set_chlayout(swr_ctx, "in_chlayout", &codec_ctx->ch_layout, 0);
	av_opt_set_int(swr_ctx, "in_sample_rate", codec_ctx->sample_rate, 0);
	av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);

	AVChannelLayout dst_ch_layout;
	av_channel_layout_copy(&dst_ch_layout, &codec_ctx->ch_layout);
	av_channel_layout_uninit(&dst_ch_layout);
	av_channel_layout_default(&dst_ch_layout, 2);

	av_opt_set_chlayout(swr_ctx, "out_chlayout", &dst_ch_layout, 0);
	av_opt_set_int(swr_ctx, "out_sample_rate", 48000, 0);
	av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);

	if (swr_init(swr_ctx) < 0) {
		std::cerr << "Error: Failed to initialize the resampling context" << std::endl;
		swr_free(&swr_ctx);
		avcodec_free_context(&codec_ctx);
		avformat_close_input(&format_ctx);
		return AudioSegment();
	}

	AVPacket packet;
	AVFrame* frame = av_frame_alloc();
	if (!frame) {
		std::cerr << "Error: Could not allocate frame." << std::endl;
		swr_free(&swr_ctx);
		avcodec_free_context(&codec_ctx);
		avformat_close_input(&format_ctx);
		return AudioSegment();
	}

	std::vector<char> output;
	while (av_read_frame(format_ctx, &packet) >= 0) {
		if (packet.stream_index == audio_stream_index) {
			if (avcodec_send_packet(codec_ctx, &packet) == 0) {
				while (avcodec_receive_frame(codec_ctx, frame) == 0) {
					if (frame->pts != AV_NOPTS_VALUE) {
						frame->pts = av_rescale_q(frame->pts, codec_ctx->time_base, format_ctx->streams[audio_stream_index]->time_base);
					}

					uint8_t* output_buffer;
					int output_samples = av_rescale_rnd(
						swr_get_delay(swr_ctx, codec_ctx->sample_rate) + frame->nb_samples,
						48000, codec_ctx->sample_rate, AV_ROUND_UP);

					int output_buffer_size = av_samples_get_buffer_size(
						nullptr, 2, output_samples, AV_SAMPLE_FMT_S16, 1);

					output_buffer = (uint8_t*)av_malloc(output_buffer_size);

					if (output_buffer) {
						memset(output_buffer, 0, output_buffer_size); // Zero padding to avoid random noise
						int converted_samples = swr_convert(swr_ctx, &output_buffer, output_samples,
							(const uint8_t**)frame->extended_data, frame->nb_samples);

						if (converted_samples >= 0) {
							output.insert(output.end(), output_buffer, output_buffer + output_buffer_size);
						}
						else {
							std::cerr << "Error: Failed to convert audio samples." << std::endl;
						}

						av_free(output_buffer);
					}
					else {
						std::cerr << "Error: Could not allocate output buffer." << std::endl;
					}
				}
			}
			else {
				std::cerr << "Error: Failed to send packet to codec context." << std::endl;
			}
		}
		av_packet_unref(&packet);
	}

	av_frame_free(&frame);
	swr_free(&swr_ctx);
	avcodec_free_context(&codec_ctx);
	avformat_close_input(&format_ctx);

	std::map<std::string, int> metadata = {
		{"sample_width", 2},
		{"frame_rate", 48000},
		{"channels", 2},
		{"frame_width", 4}
	};

	return AudioSegment(static_cast<const char*>(output.data()), output.size(), metadata);
}





/*
AudioSegment AudioSegment::from_file(const std::string& file_path,const std::string& format,const std::string& codec,const std::map<std::string, int>& parameters ,int start_second,int duration) {
	std::string command = "ffmpeg -y -i \"" + file_path + "\"";

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

	command += " -f wav -";  // Output as WAV format to stdout

	AudioSegment audioSegmentInstance;
	std::vector<char> output = audioSegmentInstance.execute_conversion(command, {});

	if (output.size() < 44) {
		std::cerr << "Error: FFmpeg output is too small to be a valid WAV file." << std::endl;
	}

	fix_wav_headers(output);

	std::map<std::string, int> metadata = {
		{"sample_width", 2},
		{"frame_rate", 44100},
		{"channels", 2},
		{"frame_width", 4}
	};

	return AudioSegment(static_cast<const char*>(output.data()), output.size(), metadata);
}

AudioSegment AudioSegment::from_mp3(const std::string& file, const std::map<std::string, int>& parameters) {
	return AudioSegment::from_file(file,"mp3","", parameters,0,0);
}

AudioSegment AudioSegment::from_flv(const std::string& file, const std::map<std::string, int>& parameters) {
	return AudioSegment::from_file(file, "flv", "", parameters, 0, 0);
}

AudioSegment AudioSegment::from_ogg(const std::string& file, const std::map<std::string, int>& parameters) {
	return AudioSegment::from_file(file, "ogg", "", parameters, 0, 0);
}

AudioSegment AudioSegment::from_wav(const std::string& file, const std::map<std::string, int>& parameters) {
	return AudioSegment::from_file(file, "wav", "", parameters, 0, 0);
}

AudioSegment AudioSegment::from_raw(const std::string& file, int sample_width, int frame_rate, int channels) {
	std::map<std::string, int> parameters;
	parameters["sample_width"] = sample_width;
	parameters["frame_rate"] = frame_rate;
	parameters["channels"] = channels;
	return AudioSegment::from_file(file, "raw", "", parameters, 0, 0);
}
*/

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

bool AudioSegment::operator==(const AudioSegment& other) const {
	return data_ == other.data_ &&
		sample_width_ == other.sample_width_ &&
		frame_rate_ == other.frame_rate_ &&
		channels_ == other.channels_;
}

bool AudioSegment::operator!=(const AudioSegment& other) const {
	return !(*this == other);
}

AudioSegment AudioSegment::operator+(const AudioSegment& other) const {
	return append(other, 0);
}

AudioSegment AudioSegment::operator+(int db) const {
	return apply_gain(db);
}

AudioSegment operator+(int db, const AudioSegment& segment) {
	return segment + db;  // Leverage the existing operator+ defined for AudioSegment
}

AudioSegment operator-(const AudioSegment& segment, int db) {
	return segment + (-db);  // Apply negative gain using existing operator+
}

AudioSegment operator-(const AudioSegment& lhs, const AudioSegment& rhs) {
	throw std::invalid_argument("AudioSegment objects can't be subtracted from each other");
}

std::string AudioSegment::ffmpeg() {
	return ffmpeg_converter_;
}

void AudioSegment::ffmpeg(const std::string& value) {
	ffmpeg_converter_ = value;
}

const std::unordered_map<std::string, std::string> AudioSegment::DEFAULT_CODECS = {
	{"ogg", "libvorbis"}
};

const std::unordered_map<std::string, std::string>& AudioSegment::default_codecs() {
	return DEFAULT_CODECS;
}

std::vector<char> AudioSegment::raw_data() const {
	return data_;
}

double AudioSegment::rms() const {
	return _rms(data_, static_cast<int>(sample_width_));
}

float AudioSegment::dBFS() const {
	double rms_value = rms();
	if (rms_value == 0) {
		return -std::numeric_limits<double>::infinity();
	}
	double max_amplitude = max_possible_amplitude();
	return ratio_to_db(static_cast<float>(rms_value) / static_cast<float>(max_amplitude));
}

int AudioSegment::max() const {
	return _max(data_, static_cast<int>(sample_width_));
}

double AudioSegment::max_possible_amplitude() const {
	int bits = static_cast<int>(sample_width_) * 8;
	double max_possible_val = std::pow(2, bits);
	return max_possible_val / 2;
}

float AudioSegment::max_dBFS() const {
	double max_value = static_cast<double>(max());  // Use the existing max() method
	double max_amplitude = max_possible_amplitude();  // Use the existing max_possible_amplitude() method

	return ratio_to_db(max_value / max_amplitude);
}

double AudioSegment::duration_seconds() const {
	if (frame_rate_ > 0) {
		return static_cast<double>(frame_count()) / frame_rate_;
	}
	return 0.0;
}

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

std::vector<uint8_t> AudioSegment::get_frame(int index) const {
	size_t frame_start = static_cast<size_t>(index * this->frame_width_);
	size_t frame_end = static_cast<size_t>(frame_start + this->frame_width_);

	if (frame_start >= static_cast<size_t>(this->raw_data().size())) {
		throw std::out_of_range("Frame index out of range");
	}

	if (frame_end > static_cast<size_t>(this->raw_data().size())) {
		frame_end = static_cast<size_t>(this->raw_data().size()); // Adjust frame_end if it exceeds data size
	}

	return std::vector<uint8_t>(this->raw_data().begin() + frame_start, this->raw_data().begin() + frame_end);
}

double AudioSegment::frame_count(int ms) const {
	if (ms >= 0) {
		return static_cast<double>(ms * (static_cast<double>(this->frame_rate_) / 1000.0));
	}
	else {
		return static_cast<double>(this->raw_data().size()) / this->frame_width_;
	}
}

std::ofstream AudioSegment::export_segment(std::string& out_f,
	const std::string& format,
	const std::string& codec,
	const std::string& bitrate,
	const std::vector<std::string>& parameters,
	const std::map<std::string, std::string>& tags,
	const std::string& id3v2_version,
	const std::string& cover) {
	av_log_set_level(AV_LOG_DEBUG);
	AVCodecContext* codec_ctx = nullptr;
	AVFormatContext* format_ctx = nullptr;
	AVStream* stream = nullptr;
	AVFrame* frame = nullptr;
	AVPacket* pkt = nullptr;
	int ret;

	// Open output file
	std::ofstream out_file(out_f, std::ios::binary);
	if (!out_file) {
		throw std::runtime_error("Failed to open output file.");
	}

	// Initialize format context
	avformat_alloc_output_context2(&format_ctx, nullptr, format.c_str(), out_f.c_str());
	if (!format_ctx) {
		throw std::runtime_error("Could not allocate format context.");
	}

	// Find encoder
	const AVCodec* codec_ptr = avcodec_find_encoder_by_name(codec.c_str());
	if (!codec_ptr) {
		throw std::runtime_error("Codec not found.");
	}

	// Add stream
	stream = avformat_new_stream(format_ctx, codec_ptr);
	if (!stream) {
		throw std::runtime_error("Failed to create new stream.");
	}

	// Allocate codec context
	codec_ctx = avcodec_alloc_context3(codec_ptr);
	if (!codec_ctx) {
		throw std::runtime_error("Could not allocate audio codec context.");
	}

	// Set codec parameters
	codec_ctx->bit_rate = std::stoi(bitrate);
	codec_ctx->sample_fmt = codec_ptr->sample_fmts ? codec_ptr->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
	codec_ctx->sample_rate = 48000;
	codec_ctx->ch_layout.nb_channels = this->get_channels();
	AVChannelLayout ch_layout_1;
	av_channel_layout_uninit(&ch_layout_1);
	av_channel_layout_default(&ch_layout_1, this->get_channels());
	codec_ctx->ch_layout = ch_layout_1;

	// Open codec
	ret = avcodec_open2(codec_ctx, codec_ptr, nullptr);
	if (ret < 0) {
		throw std::runtime_error("Could not open codec.");
	}

	// Initialize packet
	pkt = av_packet_alloc();
	if (!pkt) {
		throw std::runtime_error("Could not allocate AVPacket.");
	}

	// Initialize frame
	frame = av_frame_alloc();
	if (!frame) {
		throw std::runtime_error("Could not allocate AVFrame.");
	}

	frame->nb_samples = codec_ctx->frame_size;
	frame->format = codec_ctx->sample_fmt;
	frame->ch_layout = codec_ctx->ch_layout;
	frame->sample_rate = 48000;

	// Allocate data buffer
	ret = av_frame_get_buffer(frame, 0);
	if (ret < 0) {
		throw std::runtime_error("Could not allocate audio data buffers.");
	}

	// Encode frames
	int samples_read = 0;
	while (samples_read < data_.size()) {
		ret = av_frame_make_writable(frame);
		if (ret < 0) {
			throw std::runtime_error("Frame not writable.");
		}

		// Determine the number of samples to copy into the frame
		int frame_size = std::min<int>(codec_ctx->frame_size, (data_.size() - samples_read) / frame_width_);
		int buffer_size = frame_size * frame_width_;

		// Clear the frame data to avoid artifacts from previous data
		std::memset(frame->data[0], 0, codec_ctx->frame_size * frame_width_);

		// Copy the actual audio data into the frame
		std::memcpy(frame->data[0], data_.data() + samples_read, buffer_size);
		samples_read += buffer_size;

		// If the frame is partially filled, pad the remaining part with zeros
		if (frame_size < codec_ctx->frame_size) {
			std::memset(frame->data[0] + buffer_size, 0, (codec_ctx->frame_size - frame_size) * frame_width_);
		}

		// Send the frame for encoding
		ret = avcodec_send_frame(codec_ctx, frame);
		if (ret < 0) {
			throw std::runtime_error("Error sending frame for encoding.");
		}

		// Receive and write packets
		while (ret >= 0) {
			ret = avcodec_receive_packet(codec_ctx, pkt);
			if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
				break;
			}
			else if (ret < 0) {
				throw std::runtime_error("Error encoding frame.");
			}

			out_file.write(reinterpret_cast<char*>(pkt->data), pkt->size);
			av_packet_unref(pkt);
		}
	}

	// **Explicitly flush the encoder**
	ret = avcodec_send_frame(codec_ctx, nullptr);
	if (ret < 0) {
		throw std::runtime_error("Error flushing the encoder.");
	}

	// Receive and write remaining packets after flushing
	while (ret >= 0) {
		ret = avcodec_receive_packet(codec_ctx, pkt);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			break;
		}
		else if (ret < 0) {
			throw std::runtime_error("Error encoding frame during flush.");
		}

		out_file.write(reinterpret_cast<char*>(pkt->data), pkt->size);
		av_packet_unref(pkt);
	}

	// Cleanup
	av_frame_free(&frame);
	av_packet_free(&pkt);
	avcodec_free_context(&codec_ctx);
	avformat_free_context(format_ctx);

	return out_file;
}











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

AudioSegment AudioSegment::slice(int64_t start_ms, int64_t end_ms) const {
	// Convert start and end times from milliseconds to samples
	int64_t start_sample = milliseconds_to_frames(start_ms, frame_rate_);
	int64_t end_sample = milliseconds_to_frames(end_ms, frame_rate_);

	// Handle negative values (from the end of the segment)
	if (start_ms < 0) {
		start_sample = frame_count() + start_sample;
	}

	if (end_ms < 0) {
		end_sample = frame_count() + end_sample;
	}

	// Retrieve the sample slice
	std::vector<char> slice_data = get_sample_slice(static_cast<uint32_t>(start_sample), static_cast<uint32_t>(end_sample));

	// Return a new AudioSegment created from the sliced data
	return _spawn(slice_data);
}

std::vector<AudioSegment> _sync(const std::vector<AudioSegment>& segs) {
	// Find the maximum channels, frame_rate, and sample_width
	int channels = 0;
	int frame_rate = 0;
	int sample_width = 0;

	for (const auto& seg : segs) {
		if (seg.get_channels() > channels) channels = seg.get_channels();
		if (seg.get_frame_rate() > frame_rate) frame_rate = seg.get_frame_rate();
		if (seg.get_sample_width() > sample_width) sample_width = seg.get_sample_width();
	}

	// Return the modified segments
	std::vector<AudioSegment> result;
	for (const auto& seg : segs) {
		result.push_back(
			seg.set_channels(channels)
			   .set_frame_rate(frame_rate)
			   .set_sample_width(sample_width)
		);
	}

	return result;
}

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

AudioSegment AudioSegment::apply_gain(int db) const {
	AudioSegment result = *this;
	if (sample_width_ != 2) {
		throw std::runtime_error("Unsupported sample width for gain adjustment");
	}

	int16_t gain_factor = static_cast<int16_t>(db);
	size_t num_samples = result.data_.size() / sizeof(int16_t);

	for (size_t i = 0; i < num_samples; ++i) {
		int16_t* sample = reinterpret_cast<int16_t*>(result.data_.data() + i * sizeof(int16_t));
		// Ensure that all the arguments to std::clamp are of the same type (int16_t)
		*sample = std::clamp<int16_t>(*sample + gain_factor, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max());
	}

	return result;
}

AudioSegment AudioSegment::overlay(const AudioSegment& seg, int position, bool loop, int times, int gain_during_overlay) const {
	if (loop) {
		times = -1; // Loop indefinitely
	}
	else if (times == 0) {
		return *this; // No-op, return a copy of the current segment
	}
	else if (times < 0) {
		times = 1; // Default to looping once if times is negative
	}

	// Sync segments
	std::vector<AudioSegment> segs = { *this, seg };
	std::vector<AudioSegment> segs_synced = _sync(segs);
	AudioSegment seg1 = segs_synced[0];
	AudioSegment seg2 = segs_synced[1];

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
		int overlay_len = std::min(seg2_len, remaining);

		// Prepare segments for overlay
		std::vector<char> seg1_overlaid(data1.begin() + pos, data1.begin() + pos + overlay_len);
		std::vector<char> seg2_resized(data2.begin(), data2.begin() + overlay_len);

		// Overlay segments
		std::vector<char> overlay_result;
		if (gain_during_overlay != 0) {
			float gain_factor = static_cast<float>(db_to_float(static_cast<double>(gain_during_overlay)));
			std::vector<char> seg1_adjusted_gain = mul(seg1_overlaid, sample_width, gain_factor);
			overlay_result = add(seg1_adjusted_gain, seg2_resized, sample_width);
		}
		else {
			overlay_result = add(seg1_overlaid, seg2_resized, sample_width);
		}

		// Append result to output
		output_data.insert(output_data.end(), overlay_result.begin(), overlay_result.end());

		// Move position for the next iteration
		pos += overlay_len;

		// Decrement times or set to loop indefinitely
		times = (times > 0) ? times - 1 : times;
	}

	// Append the remaining part of the first segment
	output_data.insert(output_data.end(), data1.begin() + pos, data1.end());

	// Create and return a new AudioSegment with the output data
	return _spawn(output_data);
}

AudioSegment AudioSegment::append(const AudioSegment& seg, int crossfade) const {
	// Synchronize the audio segments to ensure they have the same sample width and frame rate
	const std::vector<AudioSegment> seg_1_2 = { *this, seg };
	const std::vector<AudioSegment> segs_synced = _sync(seg_1_2);
	AudioSegment seg1 = segs_synced[0];
	AudioSegment seg2 = segs_synced[1];

	// Create a copy of seg2 to avoid self-referencing issues
	AudioSegment seg2_copy = seg2;

	// Convert crossfade from milliseconds to frames
	int crossfade_frames = milliseconds_to_frames(crossfade, frame_rate_);
	int crossfade_ms = static_cast<int>(crossfade_frames / frame_rate_ * 1000.0);

	if (crossfade == 0) {
		// No crossfade, just concatenate
		std::vector<char> combined_data = seg1.raw_data();
		const std::vector<char>& seg2_data = seg2_copy.raw_data();
		if (seg2_data.empty()) {
			throw std::runtime_error("seg2_copy.raw_data() is empty.");
		}
		combined_data.insert(combined_data.end(), seg2_data.begin(), seg2_data.end());
		return seg1._spawn(combined_data);
	}

	// Check if crossfade is valid
	if (crossfade_ms > static_cast<int>(seg1.frame_count() / frame_rate_ * 1000.0)) {
		throw std::invalid_argument("Crossfade is longer than the original AudioSegment.");
	}
	if (crossfade_ms > static_cast<int>(seg2_copy.frame_count() / frame_rate_ * 1000.0)) {
		throw std::invalid_argument("Crossfade is longer than the appended AudioSegment.");
	}

	// Create crossfade segments
	auto fade_out_data = seg1.slice(seg1.frame_count() - crossfade_ms, seg1.frame_count()).fade_out(crossfade_frames);
	auto fade_in_data = seg2_copy.slice(0, crossfade_ms).fade_in(crossfade_frames);

	// Concatenate segments
	std::vector<char> combined_data;
	combined_data.reserve(seg1.raw_data().size() + fade_out_data.raw_data().size() + seg2_copy.raw_data().size());

	// Append the first segment excluding the crossfade portion
	const auto& seg1_data = seg1.slice(0, seg1.frame_count() - crossfade_ms).raw_data();
	combined_data.insert(combined_data.end(), seg1_data.begin(), seg1_data.end());

	// Append the crossfade portion
	const auto& fade_out_data_raw = fade_out_data.raw_data();
	combined_data.insert(combined_data.end(), fade_out_data_raw.begin(), fade_out_data_raw.end());

	// Append the second segment excluding the crossfade portion
	const auto& seg2_data = seg2_copy.slice(crossfade_ms, seg2_copy.frame_count()).raw_data();
	combined_data.insert(combined_data.end(), seg2_data.begin(), seg2_data.end());

	return seg1._spawn(combined_data);
}

std::vector<char> AudioSegment::convert_audio_data(const std::vector<char>& data, int src_rate, int dest_rate) const {
	// Set up libavformat structures
	AVCodecContext* codec_ctx = avcodec_alloc_context3(nullptr);
	AVFrame* frame = av_frame_alloc();
	AVPacket packet;
	av_init_packet(&packet);

	// Configure codec context (e.g., sample rate, channels)

	codec_ctx->ch_layout.nb_channels = this->get_channels();  // Ensure 'channels_' is correctly defined and used
	codec_ctx->sample_fmt = AV_SAMPLE_FMT_S16; // Example format
	//codec_ctx->ch_layout = av_channel_layout_default(codec_ctx->ch_layout.nb_channels);

	// Open codec context
	const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE);
	avcodec_open2(codec_ctx, codec, nullptr);

	// Decode input data
	packet.data = reinterpret_cast<uint8_t*>(const_cast<char*>(data.data()));
	packet.size = data.size();
	avcodec_send_packet(codec_ctx, &packet);
	avcodec_receive_frame(codec_ctx, frame);

	// Set up conversion context
	SwrContext* swr_ctx = swr_alloc();
	if (!swr_ctx) {
		// Handle error
	}

	av_opt_set_int(swr_ctx, "in_channels", codec_ctx->ch_layout.nb_channels, 0);
	av_opt_set_int(swr_ctx, "out_channels", codec_ctx->ch_layout.nb_channels, 0);
	av_opt_set_int(swr_ctx, "in_sample_rate", src_rate, 0);
	av_opt_set_int(swr_ctx, "out_sample_rate", dest_rate, 0);
	av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);
	av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", codec_ctx->sample_fmt, 0);
	swr_init(swr_ctx);

	// Convert audio
	int64_t in_samples = frame->nb_samples;
	int out_samples = av_rescale_rnd(swr_get_delay(swr_ctx, src_rate) + in_samples, dest_rate, src_rate, AV_ROUND_UP);
	int out_buffer_size = av_samples_get_buffer_size(nullptr, codec_ctx->ch_layout.nb_channels, out_samples, codec_ctx->sample_fmt, 1);
	std::vector<uint8_t> out_buffer(out_buffer_size);

	// Note: Use a pointer to the buffer for swr_convert
	uint8_t* out_buffer_ptr = out_buffer.data();
	swr_convert(swr_ctx, &out_buffer_ptr, out_samples, (const uint8_t**)frame->data, in_samples);

	// Convert output to std::vector<char>
	std::vector<char> converted_data(out_buffer.begin(), out_buffer.end());

	// Cleanup
	av_frame_free(&frame);
	avcodec_free_context(&codec_ctx);
	swr_free(&swr_ctx);

	return converted_data;
}

AudioSegment AudioSegment::set_frame_rate(int frame_rate) const {
	if (frame_rate == frame_rate_) {
		return *this;
	}

	std::vector<char> converted_data;
	if (!data_.empty()) {
		converted_data = convert_audio_data(data_, frame_rate_, frame_rate);
	}
	else {
		converted_data = data_;
	}

	return _spawn(converted_data);
}

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
	AudioSegment converted_segment = _spawn(converted_data);
	
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
		return *this;  // No conversion needed
	}

	// Calculate the new frame width based on the new sample width
	int new_frame_width = channels_ * sample_width;

	// Convert the raw audio data to the new sample width
	std::vector<char> new_data = lin2lin(this->raw_data(), this->get_sample_width(), sample_width);

	// Create a new AudioSegment with the converted data
	AudioSegment converted_segment = _spawn(new_data);

	// Update the metadata to reflect the new sample width and frame width
	converted_segment.sample_width_ = sample_width;  // Set to the new sample width
	converted_segment.frame_width_ = new_frame_width;

	// Ensure the frame rate remains the same (to prevent tempo issues)
	converted_segment.frame_rate_ = this->get_frame_rate();  // This can be frame_rate or sample_rate, depending on your system

	return converted_segment;
}

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

std::vector<AudioSegment> AudioSegment::split_to_mono() const {
	std::vector<AudioSegment> mono_channels;

	if (channels_ == 1) {
		mono_channels.push_back(*this);
		return mono_channels;
	}

	std::vector<char> samples = raw_data();
	size_t total_samples = samples.size() / sample_width_;
	size_t frame_count = total_samples / channels_;

	// Calculate the expected size for each channel
	size_t expected_size_per_channel = frame_count * sample_width_;

	for (int i = 0; i < channels_; ++i) {
		std::vector<char> samples_for_current_channel;
		samples_for_current_channel.reserve(expected_size_per_channel);

		// Extract samples for the current channel
		for (size_t j = 0; j < frame_count; ++j) {
			size_t start_index = j * channels_ * sample_width_ + i * sample_width_;
			size_t end_index = start_index + sample_width_;

			// Ensure we don't go out of bounds
			if (end_index > samples.size()) {
				end_index = samples.size();
			}

			samples_for_current_channel.insert(samples_for_current_channel.end(),
				samples.begin() + start_index,
				samples.begin() + end_index);
		}

		// Ensure the size matches the expected size
		if (samples_for_current_channel.size() != expected_size_per_channel) {
			std::cerr << "Size mismatch for channel " << i << ": "
				<< samples_for_current_channel.size() << " vs. "
				<< expected_size_per_channel << std::endl;
		}

		// Create a new AudioSegment for the current channel
		AudioSegment slice_x = AudioSegment::_spawn(samples_for_current_channel);
		slice_x.frame_width_ = sample_width_;
		slice_x.channels_ = 1;
		mono_channels.push_back(slice_x);
	}

	return mono_channels;
}

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

std::vector<char> AudioSegment::convert_to_stereo(const std::vector<char>& data) const {
	// Calculate the frame size based on the sample width and the number of input channels.
	size_t frame_size = sample_width_ * channels_;

	// Output vector size is doubled as we're converting from mono to stereo
	// i.e., two channels from one, so we need twice the number of frames.
	std::vector<char> converted_data(data.size() * 2 / channels_); // Double the size for stereo

	// Loop through the input data in steps of `frame_size`
	for (size_t i = 0; i < data.size(); i += frame_size) {
		// Copy the original mono frame to the left channel in stereo
		std::memcpy(&converted_data[i * 2], &data[i], frame_size);

		// Copy the same mono frame to the right channel in stereo
		std::memcpy(&converted_data[i * 2 + frame_size], &data[i], frame_size);
	}

	return converted_data;
}

std::vector<char> AudioSegment::convert_to_mono(const std::vector<char>& data) const {
	// Calculate the frame size for one channel based on the sample width
	size_t frame_size = sample_width_ * channels_;

	// Calculate the size of the mono data (half the size for stereo to mono conversion)
	std::vector<char> converted_data(data.size() / channels_);

	for (size_t i = 0; i < data.size(); i += frame_size) {
		// Accumulator for averaging the samples
		std::vector<int> channel_sum(sample_width_, 0);

		// Iterate over all channels, summing their data
		for (size_t ch = 0; ch < channels_; ++ch) {
			for (size_t byte = 0; byte < sample_width_; ++byte) {
				// Add the sample value byte by byte (treating them as unsigned)
				channel_sum[byte] += static_cast<unsigned char>(data[i + ch * sample_width_ + byte]);
			}
		}

		// Compute the average per byte
		for (size_t byte = 0; byte < sample_width_; ++byte) {
			channel_sum[byte] /= channels_;
			converted_data[i / channels_ + byte] = static_cast<char>(channel_sum[byte]);
		}
	}

	return converted_data;
}

AudioSegment AudioSegment::fade(double to_gain, double from_gain, int start, int end, int duration) const {
	if ((start != -1 && end != -1 && duration != -1) ||
		(start == -1 && end == -1 && duration == -1)) {
		throw std::invalid_argument("Only two of the three arguments, 'start', 'end', and 'duration' may be specified");
	}

	if (to_gain == 0 && from_gain == 0) {
		return *this;
	}

	int length = this->length_in_milliseconds();
	start = (start != -1) ? std::min(length, start) : 0;
	end = (end != -1) ? std::min(length, end) : length;

	if (start < 0) start += length;
	if (end < 0) end += length;

	if (duration < 0) {
		duration = end - start;
	}
	else {
		if (start != -1) end = start + duration;
		else if (end != -1) start = end - duration;
	}

	if (duration <= 0) duration = end - start;

	double from_power = db_to_float(from_gain);
	double to_power = db_to_float(to_gain);
	double gain_delta = to_power - from_power;

	std::vector<char> output;

	uint32_t start_sample = static_cast<uint32_t>((start * frame_rate_) / 1000);
	uint32_t end_sample = static_cast<uint32_t>((end * frame_rate_) / 1000);

	std::vector<char> before_fade = this->get_sample_slice(0, start_sample);
	if (from_gain != 0.0) {
		before_fade = mul(before_fade, static_cast<int>(this->sample_width_), from_power);
	}
	output.insert(output.end(), before_fade.begin(), before_fade.end());

	int fade_samples = end_sample - start_sample;
	if (fade_samples > 0) {
		double scale_step = gain_delta / fade_samples;
		for (int i = 0; i < fade_samples; ++i) {
			double progress = static_cast<double>(i) / fade_samples;
			double volume_change = from_power + (scale_step * i);

			// Debugging: Print out the volume change
			//std::cout << "Volume change at step " << i << ": " << volume_change << std::endl;

			std::vector<char> chunk = this->get_sample_slice(start_sample + i, start_sample + i + 1);
			chunk = mul(chunk, static_cast<int>(this->sample_width_), volume_change);
			output.insert(output.end(), chunk.begin(), chunk.end());
		}
	}

	std::vector<char> after_fade = this->get_sample_slice(end_sample, static_cast<uint32_t>(data_.size() / frame_width_));
	if (to_gain != 0) {
		after_fade = mul(after_fade, static_cast<int>(this->sample_width_), to_power);
	}
	output.insert(output.end(), after_fade.begin(), after_fade.end());

	return this->_spawn(output);
}

AudioSegment AudioSegment::fade_out(int duration) const {
	// Call fade method with to_gain set to -120 dB and end set to infinity
	return fade(-120.0, 0.0, -1, static_cast<int>(this->length_in_milliseconds()), duration);
}

AudioSegment AudioSegment::fade_in(int duration) const {
	// Call fade method with from_gain set to -120 dB and start set to 0
	return fade(0.0, -120.0, 0, -1, duration);
}

double AudioSegment::get_dc_offset(int channel) const {
	if (channel < 1 || channel > 2) {
		throw std::out_of_range("channel value must be 1 (left) or 2 (right)");
	}

	std::vector<char> data;

	if (channels_ == 1) {
		data = data_;
	}
	else if (channel == 1) {
		data = tomono(data_, static_cast<int>(sample_width_), 1, 0);
	}
	else {
		data = tomono(data_, static_cast<int>(sample_width_), 0, 1);
	}

	double avg_dc_offset = _avg(data, static_cast<int>(sample_width_));
	return avg_dc_offset / max_possible_amplitude();
}

AudioSegment AudioSegment::remove_dc_offset(int channel, double offset) const {
	if (channel && (channel < 1 || channel > 2)) {
		throw std::invalid_argument("channel value must be None, 1 (left) or 2 (right)");
	}
	if (offset < -1.0 || offset > 1.0) {
		throw std::invalid_argument("offset value must be in range -1.0 to 1.0");
	}

	int offset_value = 0;
	if (offset) {
		offset_value = static_cast<int>(round(offset * this->max_possible_amplitude()));
	}

	auto remove_data_dc = [this, offset_value](const std::vector<char>& data) {
		int current_offset = offset_value;
		if (!offset_value) {
			current_offset = static_cast<int>(round(_avg(data, static_cast<int>(sample_width_))));
		}
		return bias(data, static_cast<int>(sample_width_), -current_offset);
		};

	if (channels_ == 1) {
		return _spawn(remove_data_dc(data_));
	}

	std::vector<char> left_channel = tomono(data_, static_cast<int>(sample_width_), 1, 0);
	std::vector<char> right_channel = tomono(data_, static_cast<int>(sample_width_), 0, 1);

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

void write_wav_header(std::ofstream& out_file, int sample_rate, int num_channels, int bits_per_sample, int data_size) {
	// Calculate ByteRate and BlockAlign based on the provided parameters
	int byte_rate = sample_rate * num_channels * (bits_per_sample / 8);  // Byte rate = SampleRate * NumChannels * BitsPerSample/8
	int block_align = num_channels * (bits_per_sample / 8);  // Block align = NumChannels * BitsPerSample/8

	// Start writing the RIFF header
	out_file.write("RIFF", 4);  // ChunkID

	// ChunkSize = 36 (WAV header size) + data_size (audio data size)
	int chunk_size = 36 + data_size;  // Size of the entire file minus 8 bytes for "RIFF" and the chunk size
	out_file.write(reinterpret_cast<const char*>(&chunk_size), 4);  // ChunkSize

	// Write the format, which is always "WAVE"
	out_file.write("WAVE", 4);  // Format

	// Write the "fmt " subchunk
	out_file.write("fmt ", 4);  // Subchunk1ID
	int subchunk1_size = 16;  // PCM format has a fixed subchunk size of 16 bytes
	out_file.write(reinterpret_cast<const char*>(&subchunk1_size), 4);  // Subchunk1Size

	// AudioFormat = 1 for PCM (no compression)
	short audio_format = 1;
	out_file.write(reinterpret_cast<const char*>(&audio_format), 2);  // AudioFormat

	// Write the number of channels (1 for mono, 2 for stereo, etc.)
	out_file.write(reinterpret_cast<const char*>(&num_channels), 2);  // NumChannels

	// Write the sample rate (e.g., 44100 for CD-quality audio)
	out_file.write(reinterpret_cast<const char*>(&sample_rate), 4);  // SampleRate

	// Write the byte rate (SampleRate * NumChannels * BitsPerSample/8)
	out_file.write(reinterpret_cast<const char*>(&byte_rate), 4);  // ByteRate

	// Write the block align (NumChannels * BitsPerSample/8)
	out_file.write(reinterpret_cast<const char*>(&block_align), 2);  // BlockAlign

	// Write the bits per sample (e.g., 16 for 16-bit audio)
	out_file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);  // BitsPerSample

	// Write the "data" subchunk
	out_file.write("data", 4);  // Subchunk2ID

	// Write the size of the audio data in bytes
	out_file.write(reinterpret_cast<const char*>(&data_size), 4);  // Subchunk2Size
}

WavData read_wav_audio(const std::vector<char>& data, const std::vector<WavSubChunk>* headers) {
	std::vector<WavSubChunk> hdrs;
	if (headers) {
		hdrs = *headers;
	}
	else {
		hdrs = extract_wav_headers(data);
	}

	// Find 'fmt ' subchunk
	auto fmt_iter = std::find_if(hdrs.begin(), hdrs.end(), [](const WavSubChunk& subchunk) {
		return subchunk.id == "fmt ";
		});

	if (fmt_iter == hdrs.end() || fmt_iter->size < 16) {
		throw std::runtime_error("Couldn't find fmt header in wav data");
	}

	const WavSubChunk& fmt = *fmt_iter;
	size_t pos = fmt.position + 8;

	// Read audio format
	uint16_t audio_format = 0;
	std::memcpy(&audio_format, data.data() + pos, sizeof(audio_format));
	audio_format = _byteswap_ushort(audio_format);  // Convert from little-endian to host byte order

	if (audio_format != 1 && audio_format != 0xFFFE) {
		throw std::runtime_error("Unknown audio format 0x" + std::to_string(audio_format) + " in wav data");
	}

	// Read channels
	uint16_t channels = 0;
	std::memcpy(&channels, data.data() + pos + 2, sizeof(channels));
	channels = _byteswap_ushort(channels);  // Convert from little-endian to host byte order

	// Read sample rate
	uint32_t sample_rate = 0;
	std::memcpy(&sample_rate, data.data() + pos + 4, sizeof(sample_rate));
	sample_rate = _byteswap_ulong(sample_rate);  // Convert from little-endian to host byte order

	// Read bits per sample
	uint16_t bits_per_sample = 0;
	std::memcpy(&bits_per_sample, data.data() + pos + 14, sizeof(bits_per_sample));
	bits_per_sample = _byteswap_ushort(bits_per_sample);  // Convert from little-endian to host byte order

	// Find 'data' subchunk
	auto data_hdr_iter = std::find_if(hdrs.rbegin(), hdrs.rend(), [](const WavSubChunk& subchunk) {
		return subchunk.id == "data";
		});

	if (data_hdr_iter == hdrs.rend()) {
		throw std::runtime_error("Couldn't find data header in wav data");
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

std::vector<WavSubChunk> extract_wav_headers(const std::vector<char>& data) {
	std::vector<WavSubChunk> subchunks;
	size_t pos = 12;  // The size of the RIFF chunk descriptor

	while (pos + 8 <= data.size()) {
		// Ensure we have enough data for the next chunk
		if (pos + 8 + 4 > data.size()) break;

		// Read subchunk ID
		std::string subchunk_id(data.begin() + pos, data.begin() + pos + 4);

		// Read subchunk size
		uint32_t subchunk_size = 0;
		std::memcpy(&subchunk_size, data.data() + pos + 4, sizeof(subchunk_size));
		subchunk_size = _byteswap_ulong(subchunk_size);  // Convert from little-endian to host byte order

		subchunks.emplace_back(subchunk_id, pos, subchunk_size);

		if (subchunk_id == "data") {
			// 'data' is the last subchunk
			break;
		}

		pos += 8 + subchunk_size;
	}

	return subchunks;
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

}
