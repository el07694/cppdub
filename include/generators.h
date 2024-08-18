#ifndef SIGNAL_GENERATOR_H
#define SIGNAL_GENERATOR_H

#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <random>
#include "audio_segment.h"  // Include audio_segment.h

float db_to_float(float db, bool using_amplitude = true);

class SignalGenerator {
public:
    // Constructor
    SignalGenerator(int sample_rate = 44100, int bit_depth = 16);

    // Virtual destructor for abstract class
    virtual ~SignalGenerator() = default;

    // Converts the signal to an AudioSegment object
    AudioSegment to_audio_segment(double duration = 1000.0, double volume = 0.0);

    // Pure virtual function to be implemented by subclasses
    virtual std::vector<double> generate() = 0;

protected:
    int sample_rate;
    int bit_depth;
};

class Sine : public SignalGenerator {
public:
    // Constructor with frequency and passing additional arguments to SignalGenerator
    Sine(double freq, int sample_rate = 44100, int bit_depth = 16);

    // Override the generate method to produce sine wave samples
    std::vector<double> generate() override;

private:
    double freq;
};

class Pulse : public SignalGenerator {
public:
    // Constructor with frequency, duty cycle, and passing additional arguments to SignalGenerator
    Pulse(double freq, double duty_cycle = 0.5, int sample_rate = 44100, int bit_depth = 16);

    // Override the generate method to produce pulse wave samples
    std::vector<double> generate() override;

private:
    double freq;
    double duty_cycle;
};

// New Square class definition
class Square : public Pulse {
public:
    // Constructor with frequency, passing additional arguments to Pulse and setting duty_cycle to 0.5
    Square(double freq, int sample_rate = 44100, int bit_depth = 16);
};

// New Sawtooth class definition
class Sawtooth : public SignalGenerator {
public:
    // Constructor with frequency and duty cycle, passing additional arguments to SignalGenerator
    Sawtooth(double freq, double duty_cycle = 1.0, int sample_rate = 44100, int bit_depth = 16);

    // Override the generate method to produce sawtooth wave samples
    std::vector<double> generate() override;

private:
    double freq;
    double duty_cycle;
};

// New Triangle class definition
class Triangle : public Sawtooth {
public:
    // Constructor with frequency, passing additional arguments to Sawtooth and setting duty_cycle to 0.5
    Triangle(double freq, int sample_rate = 44100, int bit_depth = 16);
};

// New WhiteNoise class definition
class WhiteNoise : public SignalGenerator {
public:
    // Constructor with frequency and passing additional arguments to SignalGenerator
    WhiteNoise(int sample_rate = 44100, int bit_depth = 16);

    // Override the generate method to produce white noise samples
    std::vector<double> generate() override;

private:
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution{-1.0, 1.0};
};

#endif // SIGNAL_GENERATOR_H
