#include "generators.h"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <array>
#include <random>
#include <numeric>
#include <cstring>  // for std::memcpy

float db_to_float(float db, bool using_amplitude = true) {
    if (using_amplitude) {
        return std::pow(10.0, db / 20.0);
    } else { // using power
        return std::pow(10.0, db / 10.0);
    }
}

SignalGenerator::SignalGenerator(int sample_rate, int bit_depth)
    : sample_rate(sample_rate), bit_depth(bit_depth) {}

// Implementation of the to_audio_segment method
AudioSegment SignalGenerator::to_audio_segment(double duration, double volume) {
    // Get the min and max value based on bit depth
    int64_t minval, maxval;
    std::tie(minval, maxval) = static_cast<std::pair<int64_t, int64_t>>(get_min_max_value(bit_depth));

    // Get the sample width and array type
    int sample_width = get_frame_width(bit_depth);

    // Gain (volume) calculation
    float gain = db_to_float(static_cast<float>(volume));

    // Calculate the number of samples
    int sample_count = static_cast<int>(sample_rate * (duration / 1000.0));

    // Generate samples
    std::vector<double> generated_samples = generate();

    // Apply gain and convert to integers, limited by the max amplitude
    std::vector<int16_t> sample_data(sample_count);
    std::transform(generated_samples.begin(), generated_samples.begin() + sample_count, sample_data.begin(),
                   [maxval, gain](double val) {
                       return static_cast<int16_t>(val * maxval * gain);
                   });

    // Convert vector to byte data (equivalent to array.tobytes() in Python)
    std::vector<uint8_t> byte_data(sample_data.size() * sample_width);
    std::memcpy(byte_data.data(), sample_data.data(), sample_data.size() * sizeof(int16_t));

    // Create and return an AudioSegment object
    return AudioSegment::_spawn(byte_data);
}

// Pure virtual function to be implemented by subclasses
std::vector<double> SignalGenerator::generate() {
    throw std::runtime_error("SignalGenerator subclasses must implement the generate() method.");
}

// Constructor implementation for Sine
Sine::Sine(double freq, int sample_rate, int bit_depth)
    : SignalGenerator(sample_rate, bit_depth), freq(freq) {}

// Generate method implementation for Sine
std::vector<double> Sine::generate() {
    std::vector<double> samples;

    double sine_of = (freq * 2 * M_PI) / sample_rate;
    int sample_n = 0;

    for (int i = 0; i < sample_rate; ++i) {
        samples.push_back(std::sin(sine_of * sample_n));
        ++sample_n;
    }

    return samples;
}

// Constructor implementation for Pulse
Pulse::Pulse(double freq, double duty_cycle, int sample_rate, int bit_depth)
    : SignalGenerator(sample_rate, bit_depth), freq(freq), duty_cycle(duty_cycle) {}

// Generate method implementation for Pulse
std::vector<double> Pulse::generate() {
    std::vector<double> samples;

    double cycle_length = static_cast<double>(sample_rate) / freq;
    double pulse_length = cycle_length * duty_cycle;

    int sample_n = 0;

    for (int i = 0; i < sample_rate; ++i) {
        if ((sample_n % static_cast<int>(cycle_length)) < pulse_length) {
            samples.push_back(1.0);
        } else {
            samples.push_back(-1.0);
        }
        ++sample_n;
    }

    return samples;
}

// Constructor implementation for Square
Square::Square(double freq, int sample_rate, int bit_depth)
    : Pulse(freq, 0.5, sample_rate, bit_depth) {
    // This constructor calls the Pulse constructor with a fixed duty_cycle of 0.5
}

// Constructor implementation for Sawtooth
Sawtooth::Sawtooth(double freq, double duty_cycle, int sample_rate, int bit_depth)
    : SignalGenerator(sample_rate, bit_depth), freq(freq), duty_cycle(duty_cycle) {}

// Generate method implementation for Sawtooth
std::vector<double> Sawtooth::generate() {
    std::vector<double> samples;
    double cycle_length = static_cast<double>(sample_rate) / freq;
    double midpoint = cycle_length * duty_cycle;
    double ascend_length = midpoint;
    double descend_length = cycle_length - ascend_length;

    int sample_n = 0;

    for (int i = 0; i < sample_rate; ++i) {
        double cycle_position = static_cast<double>(sample_n % static_cast<int>(cycle_length));
        if (cycle_position < ascend_length) {
            samples.push_back((2 * cycle_position / ascend_length) - 1.0);
        } else {
            samples.push_back(1.0 - (2 * (cycle_position - ascend_length) / descend_length));
        }
        ++sample_n;
    }

    return samples;
}

// Constructor implementation for Triangle
Triangle::Triangle(double freq, int sample_rate, int bit_depth)
    : Sawtooth(freq, 0.5, sample_rate, bit_depth) {
    // This constructor calls the Sawtooth constructor with a fixed duty_cycle of 0.5
}

// Constructor implementation for WhiteNoise
WhiteNoise::WhiteNoise(int sample_rate, int bit_depth)
    : SignalGenerator(sample_rate, bit_depth), distribution(-1.0, 1.0) {
    std::random_device rd;
    generator.seed(rd());
}

// Generate method implementation for WhiteNoise
std::vector<double> WhiteNoise::generate() {
    std::vector<double> samples(sample_rate);

    for (int i = 0; i < sample_rate; ++i) {
        samples[i] = distribution(generator);
    }

    return samples;
}
