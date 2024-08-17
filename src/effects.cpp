#include "effects.h"
#include "utils.h"
#include "audio_segment.h"
#include <vector>
#include "silence.h" // For split_on_silence
#include "exceptions.h" // For InvalidDuration
#include "cppaudioop.h"

#include <cmath>
#include <array>

void apply_mono_filter_to_each_channel(AudioSegment& seg, std::function<AudioSegment(const AudioSegment&)> filter_fn) {
    int n_channels = seg.channels_;

    // Split the segment into mono channels
    std::vector<AudioSegment> channel_segs = seg.split_to_mono();
    
    // Apply the filter function to each mono channel
    for (auto& channel_seg : channel_segs) {
        channel_seg = filter_fn(channel_seg);
    }

    // Get the array of samples
    std::vector<char> out_data = seg.raw_data(); // Assuming this returns a vector of samples

    // Reassemble channels
    int sample_count = channel_segs[0].get_sample_count(); // Assuming a method to get the number of samples
    for (int channel_i = 0; channel_i < n_channels; ++channel_i) {
        const auto& channel_seg = channel_segs[channel_i];
        const auto& channel_samples = channel_seg.get_array_of_samples();

        for (int sample_i = 0; sample_i < sample_count; ++sample_i) {
            int index = (sample_i * n_channels) + channel_i;
            out_data[index] = channel_samples[sample_i];
        }
    }

    // Spawn a new AudioSegment with the processed data
    seg._spawn(out_data);
}

void normalize(AudioSegment& seg, double headroom = 0.1) {
    // Get the peak sample value
    double peak_sample_val = seg.max(); // Assuming a method to get peak value
    double max_possible_amplitude = seg.get_max_possible_amplitude(); // Assuming a method to get the max amplitude
    
    // If the max is 0, this audio segment is silent and can't be normalized
    if (peak_sample_val == 0) {
        return;
    }

    // Calculate the target peak
    double target_peak = max_possible_amplitude * db_to_float(-headroom);

    // Calculate the needed boost
    double needed_boost = static_cast<double>(ratio_to_db(target_peak / peak_sample_val));

    // Apply the gain
    seg.apply_gain(needed_boost); // Assuming a method to apply gain
}

AudioSegment speedup(const AudioSegment& seg, double playback_speed = 1.5, int chunk_size = 150, int crossfade = 25) {
    // Portion of audio to keep
    double atk = 1.0 / playback_speed;

    // Determine ms to remove per chunk
    int ms_to_remove_per_chunk;
    if (playback_speed < 2.0) {
        ms_to_remove_per_chunk = static_cast<int>(chunk_size * (1 - atk) / atk);
    } else {
        ms_to_remove_per_chunk = chunk_size;
        chunk_size = static_cast<int>(atk * chunk_size / (1 - atk));
    }

    // Adjust crossfade
    crossfade = std::min(crossfade, ms_to_remove_per_chunk - 1);

    // Chunk the audio
    auto chunks = make_chunks(seg, static_cast<size_t>(chunk_size + ms_to_remove_per_chunk));
    if (chunks.size() < 2) {
        throw std::runtime_error("Could not speed up AudioSegment, it was too short.");
    }

    // Adjust ms_to_remove_per_chunk for crossfade
    ms_to_remove_per_chunk -= crossfade;

    // Process chunks
    AudioSegment out = chunks[0].slice(0, chunks[0].length_in_milliseconds() - ms_to_remove_per_chunk);
    for (size_t i = 1; i < chunks.size() - 1; ++i) {
        out = out.append(chunks[i].slice(0, chunks[i].length_in_milliseconds() - ms_to_remove_per_chunk), crossfade);
    }

    // Add the last chunk
    out = out.append(chunks.back(), 0);

    return out;
}

AudioSegment strip_silence(const AudioSegment& seg, int silence_len = 1000, int silence_thresh = -16, int padding = 100) {
    // Check if padding is valid
    if (padding > silence_len) {
        throw InvalidDuration("padding cannot be longer than silence_len");
    }

    // Split the audio segment on silence
    auto chunks = split_on_silence(seg, silence_len, silence_thresh, padding);
    int crossfade = padding / 2;

    // If no chunks are found, return an empty segment
    if (chunks.empty()) {
        return seg.slice(0, 0); // Equivalent to seg[0:0] in Python
    }

    // Combine the chunks with crossfade
    AudioSegment result = chunks[0];
    for (size_t i = 1; i < chunks.size(); ++i) {
        result = result.append(chunks[i], crossfade);
    }

    return result;
}

AudioSegment compress_dynamic_range(const AudioSegment& seg, double threshold = -20.0, double ratio = 4.0, double attack = 5.0, double release = 50.0) {
    double thresh_rms = seg.max_possible_amplitude() * db_to_float(threshold);

    int look_frames = static_cast<int>(seg.frame_count(attack));
    auto rms_at = [&](int frame_i) {
        return seg.get_sample_slice(frame_i - look_frames, frame_i).rms();
    };

    auto db_over_threshold = [&](double rms) {
        if (rms == 0) return 0.0;
        double db = ratio_to_db(rms / thresh_rms);
        return std::max(db, 0.0);
    };

    std::vector<char> output;
    output.reserve(seg.raw_data().size());

    double attenuation = 0.0;
    int attack_frames = static_cast<int>(seg.frame_count(attack));
    int release_frames = static_cast<int>(seg.frame_count(release));

    for (int i = 0; i < seg.frame_count(); ++i) {
        double rms_now = rms_at(i);

        double max_attenuation = (1.0 - (1.0 / ratio)) * db_over_threshold(rms_now);
        double attenuation_inc = max_attenuation / attack_frames;
        double attenuation_dec = max_attenuation / release_frames;

        if (rms_now > thresh_rms && attenuation <= max_attenuation) {
            attenuation += attenuation_inc;
            attenuation = std::min(attenuation, max_attenuation);
        } else {
            attenuation -= attenuation_dec;
            attenuation = std::max(attenuation, 0.0);
        }

        auto frame = seg.get_frame(i);
        if (attenuation != 0.0) {
            frame = mul(frame, seg.sample_width(), db_to_float(-attenuation));
        }

        output.insert(output.end(), frame.data(), frame.data() + frame.size());
    }

    return seg._spawn(output);
}

// Function to invert the phase of the audio segment
AudioSegment invert_phase(const AudioSegment& seg, std::pair<int, int> channels = {1, 1}) {
    if (channels == std::make_pair(1, 1)) {
        // Invert phase for the entire mono or stereo segment
        std::vector<char> inverted = mul(seg.raw_data(), static_cast<int>(seg.sample_width_), -1.0);
        return seg._spawn(inverted);
    } else {
        if (seg.channels() == 2) {
            // Split stereo into mono channels
            auto [left, right] = seg.split_to_mono();
            
            // Invert phase for the specified channel(s)
            if (channels == std::make_pair(1, 0)) {
                left = invert_phase(left, {1, 1});  // Invert phase for the left channel
            } else {
                right = invert_phase(right, {1, 1});  // Invert phase for the right channel
            }

            // Combine mono channels back into a stereo segment
            return seg.from_mono_audiosegments({left, right});
        } else {
            throw std::runtime_error("Can't implicitly convert an AudioSegment with " + std::to_string(seg.channels()) + " channels to stereo.");
        }
    }
}

AudioSegment low_pass_filter(const AudioSegment& seg, double cutoff) {
    double RC = 1.0 / (cutoff * 2 * M_PI);
    double dt = 1.0 / seg.frame_rate();
    double alpha = dt / (RC + dt);
    
    // Get the raw data from the segment
    std::vector<char> original_data = seg.raw_data();
    std::vector<char> filtered_data = original_data;
    
    size_t frame_count = seg.frame_count();
    size_t num_channels = seg.channels();
    size_t sample_width = seg.sample_width();
    
    // Convert raw data to sample array
    std::vector<int> original_samples(frame_count * num_channels);
    std::vector<int> filtered_samples(frame_count * num_channels);
    
    // Populate original_samples with data
    for (size_t i = 0; i < original_samples.size(); ++i) {
        original_samples[i] = static_cast<int>(original_data[i]);
    }
    
    // Apply the low-pass filter
    std::vector<double> last_val(num_channels, 0.0);
    for (size_t i = 0; i < frame_count; ++i) {
        for (size_t j = 0; j < num_channels; ++j) {
            size_t offset = i * num_channels + j;
            last_val[j] = last_val[j] + (alpha * (original_samples[offset] - last_val[j]));
            filtered_samples[offset] = static_cast<int>(last_val[j]);
        }
    }
    
    // Convert filtered_samples back to raw data
    for (size_t i = 0; i < filtered_samples.size(); ++i) {
        filtered_data[i] = static_cast<char>(filtered_samples[i]);
    }
    
    return seg._spawn(filtered_data);
}

AudioSegment high_pass_filter(const AudioSegment& seg, double cutoff) {
    double RC = 1.0 / (cutoff * 2 * M_PI);
    double dt = 1.0 / seg.frame_rate();
    double alpha = RC / (RC + dt);

    // Get the raw data from the segment
    std::vector<char> original_data = seg.raw_data();
    std::vector<char> filtered_data = original_data;
    
    size_t frame_count = seg.frame_count();
    size_t num_channels = seg.channels();
    size_t sample_width = seg.sample_width();
    
    // Convert raw data to sample array
    std::vector<int> original_samples(frame_count * num_channels);
    std::vector<int> filtered_samples(frame_count * num_channels);
    
    // Populate original_samples with data
    for (size_t i = 0; i < original_samples.size(); ++i) {
        original_samples[i] = static_cast<int>(original_data[i]);
    }
    
    // Get min and max value for the sample width
    auto [minval, maxval] = get_min_max_value(seg.sample_width() * 8);

    // Apply the high-pass filter
    std::vector<double> last_val(num_channels, 0.0);
    for (size_t i = 0; i < frame_count; ++i) {
        for (size_t j = 0; j < num_channels; ++j) {
            size_t offset = i * num_channels + j;
            size_t offset_minus_1 = (i > 0) ? (i - 1) * num_channels + j : 0;
            
            if (i > 0) {
                last_val[j] = alpha * (last_val[j] + original_samples[offset] - original_samples[offset_minus_1]);
            } else {
                last_val[j] = original_samples[offset];
            }
            
            filtered_samples[offset] = static_cast<int>(std::min(std::max(last_val[j], minval), maxval));
        }
    }
    
    // Convert filtered_samples back to raw data
    for (size_t i = 0; i < filtered_samples.size(); ++i) {
        filtered_data[i] = static_cast<char>(filtered_samples[i]);
    }
    
    return seg._spawn(filtered_data);
}

AudioSegment pan(const AudioSegment& seg, double pan_amount) {
    if (pan_amount < -1.0 || pan_amount > 1.0) {
        throw std::invalid_argument("pan_amount should be between -1.0 (100% left) and +1.0 (100% right)");
    }

    double max_boost_db = ratio_to_db(2.0);
    double boost_db = std::abs(pan_amount) * max_boost_db;

    double boost_factor = db_to_float(boost_db);
    double reduce_factor = db_to_float(max_boost_db) - boost_factor;

    double reduce_db = ratio_to_db(reduce_factor);

    // Cut boost in half (max boost == 3dB)
    boost_db = boost_db / 2.0;

    if (pan_amount < 0) {
        return seg.apply_gain_stereo(boost_db, reduce_db);
    } else {
        return seg.apply_gain_stereo(reduce_db, boost_db);
    }
}

AudioSegment apply_gain_stereo(const AudioSegment& seg, double left_gain = 0.0, double right_gain = 0.0) {
    AudioSegment left, right;

    if (seg.channels() == 1) {
        // Mono to stereo conversion
        left = right = seg;
    } else if (seg.channels() == 2) {
        // Split stereo into left and right channels
        std::tie(left, right) = seg.split_to_mono();
    } else {
        throw std::invalid_argument("AudioSegment must have 1 or 2 channels.");
    }

    // Convert dB to float gain factor
    double l_mult_factor = db_to_float(left_gain);
    double r_mult_factor = db_to_float(right_gain);

    // Apply gain to left and right channels
    auto left_data = audioop::mul(left.data(), left.sample_width(), l_mult_factor);
    left_data = audioop::to_stereo(left_data, left.sample_width(), 1, 0);

    auto right_data = audioop::mul(right.data(), right.sample_width(), r_mult_factor);
    right_data = audioop::to_stereo(right_data, right.sample_width(), 0, 1);

    // Combine left and right channels
    auto output_data = audioop::add(left_data, right_data, seg.sample_width());

    // Create and return new AudioSegment
    return seg._spawn(output_data, /*overrides*/ {{"channels", 2}, {"frame_width", 2 * seg.sample_width()}});
}