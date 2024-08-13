#ifndef SILENCE_H
#define SILENCE_H

#include <vector>
#include <algorithm>
#include "audio_segment.h"

// Function to detect silence in an AudioSegment
std::vector<std::pair<int, int>> detect_silence(const AudioSegment& audio_segment, int min_silence_len = 1000, double silence_thresh = -16, int seek_step = 1);

// Function to detect nonsilent sections in an AudioSegment
std::vector<std::pair<int, int>> detect_nonsilent(const AudioSegment& audio_segment, int min_silence_len = 1000, double silence_thresh = -16, int seek_step = 1);

// Function to split an AudioSegment on silence
std::vector<AudioSegment> split_on_silence(const AudioSegment& audio_segment, int min_silence_len = 1000, double silence_thresh = -16, int keep_silence = 100, int seek_step = 1);

// Function to detect leading silence
int detect_leading_silence(const AudioSegment& sound, double silence_threshold = -50.0, int chunk_size = 10);

#endif // SILENCE_H
