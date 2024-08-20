

#ifndef EFFECTS_H
#define EFFECTS_H

#include "audio_segment.h" // For AudioSegment class
#include <cmath>           // For math functionalities
#include <vector>          // For dynamic arrays (equivalent to Python's list)
#include "silence.h"       // For split_on_silence
#include "exceptions.h"    // For TooManyMissingFrames, InvalidDuration

#include <functional>



namespace cppdub {



AudioSegment apply_mono_filter_to_each_channel(AudioSegment& seg, std::function<AudioSegment(const AudioSegment&)> filter_fn);

// Function to normalize audio segment
void normalize(AudioSegment& seg, double headroom = 0.1);

// Function to speed up an audio segment
AudioSegment speedup(const AudioSegment& seg, double playback_speed = 1.5, int chunk_size = 150, int crossfade = 25);

// Function to strip silence from an audio segment
AudioSegment strip_silence(const AudioSegment& seg, int silence_len = 1000, 
                           int silence_thresh = -16, int padding = 100);
						   
// Function to compress the dynamic range of an audio segment
AudioSegment compress_dynamic_range(const AudioSegment& seg, 
                                     double threshold = -20.0, 
                                     double ratio = 4.0, 
                                     double attack = 5.0, 
                                     double release = 50.0);

// Function to invert the phase of an audio segment
AudioSegment invert_phase(const AudioSegment& seg, std::pair<int, int> channels = {1, 1});

// Function to apply a low-pass filter to the audio segment
AudioSegment low_pass_filter(const AudioSegment& seg, double cutoff);

// Function to apply a high-pass filter to the audio segment
AudioSegment high_pass_filter(const AudioSegment& seg, double cutoff);

// Function to apply panning effect to the audio segment
AudioSegment pan(const AudioSegment& seg, double pan_amount);

// Function to apply gain to left and right channels of a stereo audio segment
AudioSegment apply_gain_stereo(const AudioSegment& seg, double left_gain = 0.0, double right_gain = 0.0);

}
#endif // EFFECTS_H
