#include "effects.h"

#include <vector>
#include <cmath>
#include <array>
using namespace cppdub;
namespace cppdub {


AudioSegment apply_mono_filter_to_each_channel(AudioSegment& seg, std::function<AudioSegment(const AudioSegment&)> filter_fn) {
    int n_channels = static_cast<int>(seg.get_channels());
    
    // Split the segment into mono channels
    std::vector<AudioSegment> channel_segs = seg.split_to_mono();
    
    if (channel_segs.size() != n_channels) {
        throw std::runtime_error("Mismatch between number of channels and split segments");
    }
    
    // Apply the filter function to each mono channel
    for (auto& channel_seg : channel_segs) {
        channel_seg = filter_fn(channel_seg);
    }
    
    // Assuming sample_width_ is the number of bytes per sample
    int sample_size = static_cast<int>(seg.get_sample_width());
    
    if (channel_segs.empty()) {
        throw std::runtime_error("No channels available for reassembly");
    }
    
    // Calculate the number of samples based on the raw data size of the first channel
    const auto& first_channel_data = channel_segs[0].raw_data();
    int sample_count = static_cast<int>(first_channel_data.size()) / (n_channels * sample_size);
    
    // Allocate output data buffer
    std::vector<char> out_data(n_channels * sample_count * sample_size);
    
    // Reassemble channels
    for (int channel_i = 0; channel_i < n_channels; ++channel_i) {
        const auto& channel_seg = channel_segs[channel_i];
        const auto& channel_samples = channel_seg.raw_data();
        
        if (channel_samples.size() / sample_size != sample_count) {
            throw std::runtime_error("Sample count mismatch in channel data");
        }
        
        for (int sample_i = 0; sample_i < sample_count; ++sample_i) {
            int index = (sample_i * n_channels + channel_i) * sample_size;
            std::copy(channel_samples.begin() + sample_i * sample_size, 
                      channel_samples.begin() + (sample_i + 1) * sample_size, 
                      out_data.begin() + index);
        }
    }
    
    // Spawn a new AudioSegment with the processed data
    return seg._spawn(out_data);
}

void normalize(AudioSegment& seg, double headroom) {
    // Get the peak sample value
    double peak_sample_val = seg.max();
    double max_possible_amplitude = seg.max_possible_amplitude();
    
    // If the max is 0, this audio segment is silent and can't be normalized
    if (peak_sample_val == 0) {
        return;
    }

    // Calculate the target peak (with headroom)
    double target_peak = max_possible_amplitude * db_to_float(-headroom);

    // Calculate the needed boost in decibels
    double needed_boost = static_cast<double>(   cppdub::ratio_to_db(   static_cast<float>(target_peak / peak_sample_val), 0.0f,true   )   );
    
    // Apply the gain
    seg = seg + needed_boost;
}

AudioSegment speedup(const AudioSegment& seg, double playback_speed, int chunk_size, int crossfade) {
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
    crossfade = static_cast<int>(std::min(crossfade, ms_to_remove_per_chunk - 1));

    // Chunk the audio
    auto chunks = make_chunks(seg, static_cast<size_t>(chunk_size + ms_to_remove_per_chunk));
    if (chunks.size() < 2) {
        throw std::runtime_error("Could not speed up AudioSegment, it was too short.");
    }

    // Adjust ms_to_remove_per_chunk for crossfade
    ms_to_remove_per_chunk -= crossfade;

    // Process chunks
    AudioSegment out = chunks[0].slice(0, static_cast<int64_t>(chunks[0].length_in_milliseconds() - ms_to_remove_per_chunk));
    for (size_t i = 1; i < chunks.size() - 1; ++i) {
        out = out.append(chunks[i].slice(0, static_cast<int64_t>(chunks[i].length_in_milliseconds() - ms_to_remove_per_chunk)), crossfade);
    }

    // Add the last chunk
    out = out.append(chunks.back(), 0);

    return out;
}

AudioSegment strip_silence(AudioSegment& seg, int silence_len, int silence_thresh, int padding) {
    // Check if padding is valid
    if (padding > silence_len) {
        throw std::invalid_argument("padding cannot be longer than silence_len");
    }

    // Split the audio segment on silence
    std::vector<cppdub::AudioSegment> chunks = split_on_silence(seg, silence_len, silence_thresh, padding);


    int crossfade = padding / 2;

    // If no chunks are found, return the original segment
    if (chunks.empty()) {
        return seg;
    }

    // Combine the chunks with crossfade
    AudioSegment result = chunks[0];
    for (size_t i = 1; i < chunks.size(); ++i) {
        result = result.append(chunks[i], crossfade);
    }

    return result;
}

AudioSegment compress_dynamic_range(const AudioSegment& seg, double threshold, double ratio, double attack, double release) {
    double thresh_rms = seg.max_possible_amplitude() * db_to_float(threshold);

    int look_frames = static_cast<int>(seg.frame_count(attack));
    auto rms_at = [&](int frame_i) {
        return seg._spawn(seg.get_sample_slice(static_cast<uint32_t>(frame_i - look_frames), static_cast<uint32_t>(frame_i))).rms();
    };

    auto db_over_threshold = [&](double rms) {
        if (rms == 0) return 0.0;
        double db = static_cast<double>(cppdub::ratio_to_db(static_cast<float>(rms / thresh_rms), 0.0f, true));
        return std::max(db, 0.0);
    };

    std::vector<char> output;
    output.reserve(seg.raw_data().size());

    double attenuation = 0.0;
    int attack_frames = static_cast<int>(seg.frame_count(attack));
    int release_frames = static_cast<int>(seg.frame_count(release));

    for (int i = 0; i < static_cast<int>(seg.frame_count()); ++i) {
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

        //auto frame = seg.get_frame(i); !!! I CHANGE THIS TO THE FOLLOWING !!!
        std::vector<char> frame = seg.get_sample_slice(static_cast<uint32_t>(i),static_cast<uint32_t>(i));
        if (attenuation != 0.0) {
            frame = mul(frame, static_cast<uint32_t>(seg.get_sample_width()), static_cast<uint32_t>(db_to_float(-attenuation)));
        }

        output.insert(output.end(), frame.data(), frame.data() + frame.size());
    }

    return seg._spawn(output);
}

// Function to invert the phase of the audio segment
AudioSegment invert_phase(const AudioSegment& seg, std::pair<int, int> channels) {
    if (channels == std::make_pair(1, 1)) {
        // Invert phase for the entire mono or stereo segment
        std::vector<char> inverted = mul(seg.raw_data(), static_cast<int>(seg.get_sample_width()), -1.0);
        return seg._spawn(inverted);
    } else {
        if (seg.get_channels() == 2) {
            // Split stereo into mono channels
            std::vector<AudioSegment> left_and_right = seg.split_to_mono();
			AudioSegment left = left_and_right[0];
			AudioSegment right = left_and_right[1];
            
            // Invert phase for the specified channel(s)
            if (channels == std::make_pair(1, 0)) {
                left = invert_phase(left, {1, 1});  // Invert phase for the left channel
            } else {
                right = invert_phase(right, {1, 1});  // Invert phase for the right channel
            }

            // Combine mono channels back into a stereo segment
            return seg.from_mono_audiosegments({left, right});
        } else {
            throw std::runtime_error("Can't implicitly convert an AudioSegment with " + std::to_string(seg.get_channels()) + " channels to stereo.");
        }
    }
}

AudioSegment low_pass_filter(const AudioSegment& seg, double cutoff) {
    double RC = 1.0 / (cutoff * 2 * M_PI);
    double dt = 1.0 / seg.get_frame_rate();
    double alpha = dt / (RC + dt);
    
    // Get the raw data from the segment
    std::vector<char> original_data = seg.raw_data();
    std::vector<char> filtered_data = original_data;
    
    size_t frame_count = static_cast<size_t>((seg.length_in_milliseconds() * seg.get_frame_rate()) / 1000);
    size_t num_channels = seg.get_channels();
    size_t sample_width = seg.get_sample_width();
    
    // Convert raw data to sample array
    std::vector<int16_t> original_samples(frame_count * num_channels);
    std::vector<int16_t> filtered_samples(frame_count * num_channels);
    
    // Populate original_samples with data
    for (size_t i = 0; i < original_samples.size(); ++i) {
        original_samples[i] = *reinterpret_cast<int16_t*>(&original_data[i * sample_width]);
    }
    
    // Apply the low-pass filter
    std::vector<double> last_val(num_channels, 0.0);
    for (size_t i = 0; i < frame_count; ++i) {
        for (size_t j = 0; j < num_channels; ++j) {
            size_t offset = i * num_channels + j;
            last_val[j] = last_val[j] + (alpha * (original_samples[offset] - last_val[j]));
            filtered_samples[offset] = static_cast<int16_t>(last_val[j]);
        }
    }
    
    // Convert filtered_samples back to raw data
    for (size_t i = 0; i < filtered_samples.size(); ++i) {
        *reinterpret_cast<int16_t*>(&filtered_data[i * sample_width]) = filtered_samples[i];
    }
    
    return seg._spawn(filtered_data);
}

AudioSegment high_pass_filter(const AudioSegment& seg, double cutoff) {
    double RC = 1.0 / (cutoff * 2 * M_PI);
    double dt = 1.0 / seg.get_frame_rate();
    double alpha = RC / (RC + dt);

    // Get the raw data from the segment
    std::vector<char> original_data = seg.raw_data();
    std::vector<char> filtered_data = original_data;
    
    size_t frame_count = static_cast<size_t>((seg.length_in_milliseconds() * seg.get_frame_rate()) / 1000);
    size_t num_channels = seg.get_channels();
    size_t sample_width = seg.get_sample_width();
    
    // Convert raw data to sample array
    std::vector<int16_t> original_samples(frame_count * num_channels);
    std::vector<int16_t> filtered_samples(frame_count * num_channels);
    
    // Populate original_samples with data
    for (size_t i = 0; i < original_samples.size(); ++i) {
        original_samples[i] = *reinterpret_cast<int16_t*>(&original_data[i * sample_width]);
    }
    
    // Get min and max value for the sample width
    int minval, maxval;
    std::tie(minval, maxval) = get_min_max_value(seg.get_sample_width() * 8);

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
            
            // Ensure minval and maxval are of type int16_t
            int16_t minval = static_cast<int16_t>(minval);
            int16_t maxval = static_cast<int16_t>(maxval);

            // Clamp the filtered value
            filtered_samples[offset] = static_cast<int16_t>(
                std::min<int16_t>(std::max<int16_t>(static_cast<int16_t>(last_val[j]), minval), maxval)
                );

        }
    }
    
    // Convert filtered_samples back to raw data
    for (size_t i = 0; i < filtered_samples.size(); ++i) {
        *reinterpret_cast<int16_t*>(&filtered_data[i * sample_width]) = filtered_samples[i];
    }
    
    return seg._spawn(filtered_data);
}

AudioSegment pan(const AudioSegment& seg, double pan_amount) {
    if (pan_amount < -1.0 || pan_amount > 1.0) {
        throw std::invalid_argument("pan_amount should be between -1.0 (100% left) and +1.0 (100% right)");
    }

    double max_boost_db = static_cast<double>(cppdub::ratio_to_db(2.0, 0.0f, true));
    double boost_db = std::abs(pan_amount) * max_boost_db;

    double boost_factor = db_to_float(boost_db);
    double reduce_factor = db_to_float(max_boost_db) - boost_factor;

    double reduce_db = static_cast<double>(cppdub::ratio_to_db(static_cast<float>(reduce_factor), 0.0f, true));

    // Cut boost in half (max boost == 3dB)
    boost_db = boost_db / 2.0;

    if (pan_amount < 0) {
        return apply_gain_stereo(seg,boost_db, reduce_db);
    } else {
        return apply_gain_stereo(seg,reduce_db, boost_db);
    }
}

AudioSegment apply_gain_stereo(const AudioSegment& seg, double left_gain, double right_gain) {
    AudioSegment left, right;

    if (seg.get_channels() == 1) {
        // Mono to stereo conversion
        left = right = seg;
    } else if (seg.get_channels() == 2) {
        // Split stereo into left and right channels
        std::vector<AudioSegment> mono_channels = seg.split_to_mono();
        left = mono_channels[0];
        right = mono_channels[1];
    } else {
        throw std::invalid_argument("AudioSegment must have 1 or 2 channels.");
    }

    // Convert dB to float gain factor
    double left_mult_factor = db_to_float(left_gain);
    double right_mult_factor = db_to_float(right_gain);

    // Apply gain to left and right channels
    std::vector<char> left_data = mul(left.raw_data(), static_cast<int>(left.get_sample_width()), static_cast<int>(left_mult_factor));
    left_data = tostereo(left_data, static_cast<int>(left.get_sample_width()), 1.0, 0.0);

    std::vector<char> right_data = mul(right.raw_data(), static_cast<int>(right.get_sample_width()), static_cast<int>(right_mult_factor));
    right_data = tostereo(right_data, static_cast<int>(right.get_sample_width()), 0.0, 1.0);

    // Combine left and right channels
    std::vector<char> output_data = add(left_data, right_data, static_cast<int>(left.get_sample_width()) * 2);

    // Create and return new AudioSegment
    return seg._spawn(output_data);
}


}