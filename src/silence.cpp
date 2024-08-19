#include "silence.h"
#include <cmath>
#include <iterator>
#include <stdexcept>

// Helper function to convert dB to float
double db_to_float(double db) {
    return std::pow(10, db / 20);
}

std::vector<std::pair<int, int>> detect_silence(const AudioSegment& audio_segment, int min_silence_len = 1000, double silence_thresh = -16, int seek_step = 1) {
    int seg_len = audio_segment.length_in_milliseconds(); // Get length of the audio segment in milliseconds

    // Check if the audio segment is shorter than the minimum silence length
    if (seg_len < min_silence_len) {
        return {};
    }

    // Convert silence threshold to a float value
    double max_possible_amplitude = audio_segment.max_possible_amplitude();
    double silence_threshold = db_to_float(silence_thresh) * max_possible_amplitude;

    std::vector<int> silence_starts;
    int last_slice_start = seg_len - min_silence_len;

    // Iterate over the audio segment in chunks
    for (int i = 0; i <= last_slice_start; i += seek_step) {
        // Slice the audio segment
        AudioSegment audio_slice = AudioSegment::_spawn(audio_segment.get_sample_slice(static_cast<uint32_t>(i),static_cast<uint32_t>(i+min_silence_len)));
        
        // Check if the RMS of the slice is below the threshold
        if (audio_slice.rms() <= silence_threshold) {
            silence_starts.push_back(i);
        }
    }

    // Ensure the last portion is included in the search
    if (last_slice_start % seek_step != 0) {
        AudioSegment audio_slice = AudioSegment::_spawn(audio_segment.get_sample_slice(static_cast<uint32_t>(last_slice_start),static_cast<uint32_t>(last_slice_start + min_silence_len)));
        if (audio_slice.rms() <= silence_threshold) {
            silence_starts.push_back(last_slice_start);
        }
    }

    // Combine consecutive silent sections into ranges
    std::vector<std::pair<int, int>> silent_ranges;
    if (silence_starts.empty()) {
        return silent_ranges;
    }

    int prev_i = silence_starts.front();
    int current_range_start = prev_i;

    for (size_t j = 1; j < silence_starts.size(); ++j) {
        int silence_start_i = silence_starts[j];
        bool continuous = (silence_start_i == prev_i + seek_step);
        bool silence_has_gap = (silence_start_i > (prev_i + min_silence_len));

        if (!continuous && silence_has_gap) {
            silent_ranges.push_back({current_range_start, prev_i + min_silence_len});
            current_range_start = silence_start_i;
        }
        prev_i = silence_start_i;
    }

    // Add the last range
    silent_ranges.push_back({current_range_start, prev_i + min_silence_len});

    return silent_ranges;
}

std::vector<std::pair<int, int>> detect_nonsilent(const AudioSegment& audio_segment, int min_silence_len = 1000, double silence_thresh = -16, int seek_step = 1) {
    std::vector<std::pair<int, int>> silent_ranges = detect_silence(audio_segment, min_silence_len, silence_thresh, seek_step);
    int seg_len = audio_segment.length_in_milliseconds(); // Get length of the audio segment in milliseconds

    // If there is no silence, the whole segment is nonsilent
    if (silent_ranges.empty()) {
        return {{0, seg_len}};
    }

    // Short circuit when the whole audio segment is silent
    if (silent_ranges.front().first == 0 && silent_ranges.front().second == seg_len) {
        return {};
    }

    // Identify nonsilent ranges by comparing with detected silent ranges
    std::vector<std::pair<int, int>> nonsilent_ranges;
    int prev_end_i = 0;

    for (const auto& range : silent_ranges) {
        int start_i = range.first;
        int end_i = range.second;

        if (prev_end_i < start_i) {
            nonsilent_ranges.push_back({prev_end_i, start_i});
        }
        prev_end_i = end_i;
    }

    // Add the final nonsilent range if necessary
    if (prev_end_i < seg_len) {
        nonsilent_ranges.push_back({prev_end_i, seg_len});
    }

    // Remove initial zero-length range if present
    if (!nonsilent_ranges.empty() && nonsilent_ranges.front() == std::make_pair(0, 0)) {
        nonsilent_ranges.erase(nonsilent_ranges.begin());
    }

    return nonsilent_ranges;
}

// Helper function to generate pairwise ranges
std::vector<std::pair<int, int>> pairwise(const std::vector<std::pair<int, int>>& ranges) {
    std::vector<std::pair<int, int>> result;
    for (size_t i = 0; i < ranges.size() - 1; ++i) {
        result.push_back({ranges[i].second, ranges[i + 1].first});
    }
    return result;
}

std::vector<AudioSegment> split_on_silence(const AudioSegment& audio_segment, int min_silence_len = 1000, double silence_thresh = -16, int keep_silence = 100, int seek_step = 1) {
    if (keep_silence < 0) {
        throw std::invalid_argument("keep_silence cannot be negative");
    }

    // Convert silence threshold to float
    double silence_thresh_float = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude();

    // Detect nonsilent ranges
    std::vector<std::pair<int, int>> output_ranges = detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step);

    // Adjust ranges to include silence
    for (auto& range : output_ranges) {
        range.first -= keep_silence;
        range.second += keep_silence;
    }

    // Adjust overlapping ranges
    auto pair_ranges = pairwise(output_ranges);
    for (auto& range_pair : pair_ranges) {
        int last_end = range_pair.first;
        int next_start = range_pair.second;
        if (next_start < last_end) {
            range_pair.first = (last_end + next_start) / 2;
            range_pair.second = range_pair.first;
        }
    }

    // Create segments from adjusted ranges
    std::vector<AudioSegment> segments;
    for (const auto& range : output_ranges) {
        int start = std::max(range.first, 0);
        int end = std::min(range.second, static_cast<int>(audio_segment.length_in_milliseconds()));
        if (start < end) {
            segments.push_back(AudioSegment::_spawn(audio_segment.get_sample_slice(static_cast<uint32_t>(start),static_cast<uint32_t>(end))));
        }
    }

    return segments;
}

int detect_leading_silence(const AudioSegment& sound, double silence_threshold = -50.0, int chunk_size = 10) {
    if (chunk_size <= 0) {
        throw std::invalid_argument("chunk_size must be greater than 0");
    }

    int trim_ms = 0; // milliseconds
    double silence_threshold_float = db_to_float(silence_threshold) * sound.max_possible_amplitude();

    while (trim_ms + chunk_size <= sound.length_in_milliseconds()) {
        // Slice the audio segment and calculate the dBFS value
        AudioSegment chunk = AudioSegment::_spawn(sound.get_sample_slice(static_cast<uint32_t>(trim_ms),static_cast<uint32_t>(trim_ms + chunk_size)));
        double chunk_dBFS = static_cast<double>(chunk.dBFS()); // Assume dBFS method is available in AudioSegment

        if (chunk_dBFS >= silence_threshold) {
            break;
        }

        trim_ms += chunk_size;
    }

    // Return the end of the silence or the length of the segment
    return std::min(trim_ms, static_cast<int>(sound.length_in_milliseconds()));
}