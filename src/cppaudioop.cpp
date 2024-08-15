#include "cppaudioop.h"
#include <bitset>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>
#include <functional>
#include <stdexcept>  // For std::runtime_error
#include <cstring> // For std::memcpy
#include <string>
#include <exception>

namespace cppdub {

// Error class implementation
error::error(const char* message) : msg_(message) {}
error::error(const std::string& message) : msg_(message) {}
error::~error() noexcept {}
const char* error::what() const noexcept { return msg_.c_str(); }

// Function implementations
void _check_size(int size) {
    if (size != 1 && size != 2 && size != 4) {
        throw error("Size should be 1, 2, or 4");
    }
}

void _check_params(int length, int size) {
    _check_size(size);  // Ensure size is valid
    if (length <= 0 || length % size != 0) {
        throw error("Invalid length or size");
    }
}

int _sample_count(const std::vector<char>& cp, int size) {
    return cp.size() / size;
}

std::vector<int> _get_samples(const std::vector<char>& cp, int size) {
    std::vector<int> samples;
    int sample_count = _sample_count(cp, size);
    for (int i = 0; i < sample_count; ++i) {
        samples.push_back(_get_sample(cp, size, i));
    }
    return samples;
}

std::string _struct_format(int size, bool signed_) {
    switch (size) {
        case 1:
            return signed_ ? "b" : "B";
        case 2:
            return signed_ ? "h" : "H";
        case 4:
            return signed_ ? "i" : "I";
        default:
            throw error("Unsupported size");
    }
}

int _get_sample(const std::vector<char>& cp, int size, int index, bool signed_=true) {
    // Determine the format and size
    std::string fmt = _struct_format(size, signed_);
    int start = index * size;
    int end = start + size;

    if (start < 0 || end > cp.size()) {
        throw std::out_of_range("Index out of range");
    }

    // Extract bytes from the vector
    std::vector<char> bytes(cp.begin() + start, cp.begin() + end);

    if (size == 1) {
        unsigned char value = static_cast<unsigned char>(bytes[0]);
        return signed_ ? static_cast<signed char>(value) : value;
    } else if (size == 2) {
        short value;
        std::memcpy(&value, bytes.data(), sizeof(value));
        if (signed_) {
            // Ensure correct signedness and endianness
            return value;
        } else {
            // Convert to unsigned short
            return static_cast<unsigned short>(value);
        }
    } else if (size == 4) {
        int value;
        std::memcpy(&value, bytes.data(), sizeof(value));
        if (signed_) {
            return value;
        } else {
            return static_cast<unsigned int>(value);
        }
    } else {
        throw std::runtime_error("Unsupported size");
    }
}

void _put_sample(std::vector<char>& result, int size, int index, int sample) {
    // Ensure we are writing into the correct range
    int start = index * size;
    if (start + size > result.size()) {
        throw error("Index out of bounds");
    }
    
    // Write bytes in little-endian order
    for (int i = 0; i < size; ++i) {
        result[start + i] = static_cast<char>((sample >> (i * 8)) & 0xFF);
    }
}

int _get_maxval(int size, bool signed_) {
    if (size == 1) {
        return signed_ ? 0x7F : 0xFF;
    } else if (size == 2) {
        return signed_ ? 0x7FFF : 0xFFFF;
    } else if (size == 4) {
        return signed_ ? 0x7FFFFFFF : 0xFFFFFFFF;
    }
    throw error("Unsupported size");
}

int _get_minval(int size, bool signed_) {
    if (!signed_) {
        return 0;
    } else if (size == 1) {
        return -0x80;
    } else if (size == 2) {
        return -0x8000;
    } else if (size == 4) {
        return -0x80000000;
    }
    throw error("Unsupported size");
}

std::function<int(int)> _get_clipfn(int size, bool signed_ = true) {
    int maxval = _get_maxval(size, signed_);
    int minval = _get_minval(size, signed_);
    return [maxval, minval](int sample) -> int {
        return std::max(minval, std::min(sample, maxval));
    };
}

int _overflow(int sample, int size, bool signed_ = true) {
    int maxval = _get_maxval(size, signed_);
    int minval = _get_minval(size, signed_);

    // Check if the sample is within the range
    if (sample >= minval && sample <= maxval) {
        return sample;
    }

    // Calculate the number of bits based on size
    int bits = size * 8;

    if (signed_) {
        int offset = 1 << (bits - 1);
        return ((sample + offset) % (1 << bits)) - offset;
    } else {
        return sample % (1 << bits);
    }
}

int getsample(const std::vector<char>& cp, int size, int index) {
    int sample_count = _sample_count(cp, size);
    if (index < 0 || index >= sample_count) {
        throw error("Index out of range");
    }
    return _get_sample(cp, size, index);
}

int max(const std::vector<char>& cp, int size) {
    _check_params(cp.size(), size);

    if (cp.empty()) {
        return 0;
    }

    int max_val = _get_minval(size,true);
    auto samples = _get_samples(cp, size);
    for (const auto& sample : samples) {
        max_val = std::max(max_val, std::abs(sample)); // Compare absolute values
    }
    return max_val;
}

int minmax(const std::vector<char>& cp, int size, int& minval, int& maxval) {
    _check_params(cp.size(), size);

    // Initialize minval and maxval to extreme values
    minval = _get_maxval(size);
    maxval = _get_minval(size);

    auto samples = _get_samples(cp, size);
    for (const auto& sample : samples) {
        minval = std::min(minval, sample);
        maxval = std::max(maxval, sample);
    }
    return 0;
}

double avg(const std::vector<char>& cp, int size) {
    _check_params(cp.size(), size);
    
    int count = _sample_count(cp, size);
    if (count == 0) {
        return 0.0;
    }
    
    int sum = 0;
    auto samples = _get_samples(cp, size);
    for (const auto& sample : samples) {
        sum += sample;
    }
    
    return static_cast<double>(sum) / count;
}

double rms(const std::vector<char>& cp, int size) {
    int count = _sample_count(cp, size);
    if (count == 0) {
        return 0.0;  // Handle division by zero
    }

    double sum_squares = 0.0;
    auto samples = _get_samples(cp, size);

    for (const auto& sample : samples) {
        sum_squares += static_cast<double>(sample * sample);
    }

    return std::sqrt(sum_squares / count); // Calculate RMS
}

int _sum2(const std::vector<char>& cp1, const std::vector<char>& cp2, int size) {
    int sum = 0;
    int sample_count = _sample_count(cp1, size);
    
    // Ensure both buffers have the same size
    if (_sample_count(cp2, size) != sample_count) {
        throw error("Buffers have different sizes");
    }

    for (int i = 0; i < sample_count; ++i) {
        int sample1 = _get_sample(cp1, size, i);
        int sample2 = _get_sample(cp2, size, i);
        sum += sample1 * sample2;
    }

    return sum;
}

// Function to calculate the fit of a sample
int findfit(const std::vector<char>& cp, int size, int value) {
    auto samples = _get_samples(cp, size);
    int fit = 0;
    int best_diff = std::numeric_limits<int>::max();
    for (const auto& sample : samples) {
        int diff = std::abs(sample - value);
        if (diff < best_diff) {
            best_diff = diff;
            fit = sample;
        }
    }
    return fit;
}

// Function to find the best alignment and scaling factor
std::pair<int, double> findfit(const std::vector<char>& cp1, const std::vector<char>& cp2) {
    int size = 2;

    if (cp1.size() % 2 != 0 || cp2.size() % 2 != 0) {
        throw error("Strings should be even-sized");
    }

    if (cp1.size() < cp2.size()) {
        throw error("First sample should be longer");
    }

    int len1 = _sample_count(cp1, size);
    int len2 = _sample_count(cp2, size);

    int sum_ri_2 = _sum2(cp2, cp2, len2);
    int sum_aij_2 = _sum2(cp1, cp1, len2);
    int sum_aij_ri = _sum2(cp1, cp2, len2);

    double result = static_cast<double>(sum_ri_2 * sum_aij_2 - sum_aij_ri * sum_aij_ri) / sum_aij_2;

    double best_result = result;
    int best_i = 0;

    for (int i = 1; i <= len1 - len2; ++i) {
        int aj_m1 = _get_sample(cp1, size, i - 1);
        int aj_lm1 = _get_sample(cp1, size, i + len2 - 1);

        sum_aij_2 += aj_lm1 * aj_lm1 - aj_m1 * aj_m1;
        sum_aij_ri = _sum2(std::vector<char>(cp1.begin() + i * size, cp1.begin() + i * size + len2 * size), cp2, len2);

        result = static_cast<double>(sum_ri_2 * sum_aij_2 - sum_aij_ri * sum_aij_ri) / sum_aij_2;

        if (result < best_result) {
            best_result = result;
            best_i = i;
        }
    }

    double factor = static_cast<double>(_sum2(std::vector<char>(cp1.begin() + best_i * size, cp1.begin() + best_i * size + len2 * size), cp2, len2)) / sum_ri_2;

    return std::make_pair(best_i, factor);
}

// Function to find the scaling factor between two vectors
double findfactor(const std::vector<char>& cp1, const std::vector<char>& cp2) {
    int size = 2;  // Size of each sample in bytes

    if (cp1.size() % size != 0) {
        throw error("Strings should be even-sized");
    }

    if (cp1.size() != cp2.size()) {
        throw error("Samples should be same size");
    }

    int sample_count = _sample_count(cp1, size);

    // Calculate sums needed for factor computation
    int sum_ri_2 = _sum2(cp2, cp2, sample_count);
    int sum_aij_ri = _sum2(cp1, cp2, sample_count);

    if (sum_ri_2 == 0) {
        throw error("Division by zero in findfactor calculation");
    }

    return static_cast<double>(sum_aij_ri) / sum_ri_2;
}

// Function to find the position with the maximum sum of squares in a sliding window
int findmax(const std::vector<char>& cp, int len2) {
    int size = 2;  // Size of each sample in bytes
    int sample_count = _sample_count(cp, size);

    if (cp.size() % 2 != 0) {
        throw error("Strings should be even-sized");
    }

    if (len2 < 0 || sample_count < len2) {
        throw error("Input sample should be longer");
    }

    if (sample_count == 0) {
        return 0;
    }

    int result = _sum2(cp, cp, len2);
    int best_result = result;
    int best_i = 0;

    for (int i = 1; i <= sample_count - len2; ++i) {
        int sample_leaving_window = getsample(cp, size, i - 1);
        int sample_entering_window = getsample(cp, size, i + len2 - 1);

        result = result - sample_leaving_window * sample_leaving_window + sample_entering_window * sample_entering_window;

        if (result > best_result) {
            best_result = result;
            best_i = i;
        }
    }

    return best_i;
}

double avgpp(const std::vector<char>& cp, int size) {
    _check_params(cp, size);
    int sample_count = _sample_count(cp, size);

    bool prevextremevalid = false;
    double prevextreme = 0.0;  // Use double for consistency
    double avg = 0.0;
    int nextreme = 0;

    int prevval = getsample(cp, size, 0);
    int val = getsample(cp, size, 1);

    int prevdiff = val - prevval;

    for (int i = 1; i < sample_count; ++i) {
        val = getsample(cp, size, i);
        int diff = val - prevval;

        if (diff * prevdiff < 0) {
            if (prevextremevalid) {
                avg += std::abs(prevval - prevextreme);  // Use std::abs for abs function
                nextreme += 1;
            }

            prevextremevalid = true;
            prevextreme = prevval;
        }

        prevval = val;
        if (diff != 0) {
            prevdiff = diff;
        }
    }

    if (nextreme == 0) {
        return 0.0;
    }

    return avg / nextreme;
}

int maxpp(const std::vector<char>& cp, int size) {
    _check_params(cp.size(), size);
    int sample_count = _sample_count(cp, size);

    bool prevextremevalid = false;
    int prevextreme = 0;  // Changed to int
    int max = 0;

    int prevval = getsample(cp, size, 0);
    int val = getsample(cp, size, 1);

    int prevdiff = val - prevval;

    for (int i = 1; i < sample_count; ++i) {
        val = getsample(cp, size, i);
        int diff = val - prevval;

        if (diff * prevdiff < 0) {
            if (prevextremevalid) {
                int extremediff = std::abs(prevval - prevextreme);
                if (extremediff > max) {
                    max = extremediff;
                }
            }
            prevextremevalid = true;
            prevextreme = prevval;
        }

        prevval = val;
        if (diff != 0) {
            prevdiff = diff;
        }
    }
    return max;  // Return int, not std::vector<char>
}

int cross(const std::vector<char>& cp, int size) {
    _check_params(cp.size(), size);

    int crossings = 0;
    int last_sample = 0;
    auto samples = _get_samples(cp, size);

    for (const auto& sample : samples) {
        // Properly decompose the conditional
        if ((sample <= 0 && last_sample > 0) || (sample >= 0 && last_sample < 0)) {
            crossings += 1;
        }
        last_sample = sample;
    }

    return crossings;
}

std::vector<char> mul(const std::vector<char>& cp, int size, int factor) {
    _check_params(cp.size(), size);
    auto clip = _get_clipfn(size);

    std::vector<char> result(cp.size());
    int sample_count = _sample_count(cp, size);

    for (int i = 0; i < sample_count; ++i) {
        int sample = _get_sample(cp, size, i);
        sample = _overflow(sample * factor, size); // Handle overflow if necessary
        sample = clip(sample); // Apply clipping
        _put_sample(result, size, i, sample);
    }

    return result;
}

// Function to combine stereo samples into mono using given factors
std::vector<char> tomono(const std::vector<char>& cp, int size, int fac1, int fac2) {
    // Validate parameters
    _check_params(cp.size(), size);
    
    // Get clipping function
    auto clip = _get_clipfn(size);

    int sample_count = _sample_count(cp, size);
    std::vector<char> result(sample_count / 2);  // Allocate space for mono result

    for (int i = 0; i < sample_count / 2; ++i) {
        // Get left and right channel samples
        int l_sample = _get_sample(cp, size, i * 2);
        int r_sample = _get_sample(cp, size, i * 2 + 1);

        // Combine samples using given factors
        int sample = (l_sample * fac1) + (r_sample * fac2);
        
        // Clip and store result
        sample = clip(sample);
        _put_sample(result, size / 2, i, sample);
    }

    return result;
}

// Function to convert mono to stereo using given factors
std::vector<char> tostereo(const std::vector<char>& cp, int size, int fac1, int fac2) {
    // Validate parameters
    _check_params(cp.size(), size);

    int sample_count = _sample_count(cp, size);
    std::vector<char> result(sample_count * 2 * size);  // Allocate space for stereo result

    // Get clipping function
    auto clip = _get_clipfn(size);

    for (int i = 0; i < sample_count; ++i) {
        // Get mono sample
        int sample = _get_sample(cp, size, i);

        // Apply factors and clip the result for left and right channels
        int l_sample = clip(sample * fac1);
        int r_sample = clip(sample * fac2);

        // Store left and right samples in the result vector
        _put_sample(result, size, i * 2, l_sample);
        _put_sample(result, size, i * 2 + 1, r_sample);
    }

    return result;
}


// Function to add two audio samples element-wise
std::vector<char> add(const std::vector<char>& cp1, const std::vector<char>& cp2, int size) {
    // Validate parameters
    _check_params(cp1.size(), size);
    if (cp1.size() != cp2.size()) {
        throw std::invalid_argument("Lengths of cp1 and cp2 should be the same");
    }

    int sample_count = _sample_count(cp1, size);
    std::vector<char> result(cp1.size());  // Allocate space for the result

    // Get clipping function
    auto clip = _get_clipfn(size);

    for (int i = 0; i < sample_count; ++i) {
        // Get samples from both input vectors
        int sample1 = _get_sample(cp1, size, i);
        int sample2 = _get_sample(cp2, size, i);

        // Compute the sum and apply overflow handling
        int sum = sample1 + sample2;
        sum = _overflow(sum, size);  // Ensure the result is within the valid range

        // Clip the result and store it in the result vector
        sum = clip(sum);
        _put_sample(result, size, i, sum);
    }

    return result;
}

// Function to apply a bias to each sample in the vector
std::vector<char> bias(const std::vector<char>& cp, int size, int amount) {
    // Validate parameters
    _check_params(cp.size(), size);

    int sample_count = _sample_count(cp, size);
    std::vector<char> result(cp.size());  // Allocate space for the result

    for (int i = 0; i < sample_count; ++i) {
        // Get the sample from the input vector
        int sample = _get_sample(cp, size, i);

        // Apply the bias and handle overflow
        sample = _overflow(sample + amount, size);

        // Store the result in the result vector
        _put_sample(result, size, i, sample);
    }

    return result;
}

// Function to reverse the order of samples in the vector
std::vector<char> reverse(const std::vector<char>& cp, int size) {
    // Validate parameters
    _check_params(cp.size(), size);

    int sample_count = _sample_count(cp, size);
    std::vector<char> result(cp.size());  // Allocate space for the result

    for (int i = 0; i < sample_count; ++i) {
        // Get the sample from the input vector
        int sample = _get_sample(cp, size, i);

        // Place the sample in the reversed position
        _put_sample(result, size, sample_count - i - 1, sample);
    }

    return result;
}

// Helper function to compute the scale factor for conversion
int compute_scale_factor(int from_size, int to_size) {
    return (from_size < to_size) ? (1 << (4 * (to_size - from_size) / from_size)) : (1 << (4 * (from_size - to_size) / to_size));
}

// Function to convert audio samples from one bit depth to another
std::vector<char> lin2lin(const std::vector<char>& cp, int size, int size2) {
     _check_params(cp.size(), size);
    _check_size(size2);
	if (size == size2) {
        return cp;  // No conversion needed
    }

    // Check if size2 is valid
    if (size2 <= 0) {
        throw std::invalid_argument("Invalid target size");
    }

    // Compute the length of the result
    int sample_count = _sample_count(cp, size);
    std::vector<char> result(sample_count * size2 / size);

    int scale_factor = compute_scale_factor(size, size2);

    for (int i = 0; i < sample_count; ++i) {
        int sample = _get_sample(cp, size, i);

        if (size < size2) {
            sample <<= scale_factor;
        } else if (size > size2) {
            sample >>= scale_factor;
        }

        sample = _overflow(sample, size2);
        _put_sample(result, size2, i, sample);
    }

    return result;
}


// Updated ratecv function
std::vector<char> ratecv(const std::vector<char>& cp, int size, int nchannels, int inrate, int outrate, std::pair<int, std::vector<std::pair<int, int>>> state, double weightA, double weightB) {

    _check_params(cp.size(), size);

    if (nchannels < 1) {
        throw std::invalid_argument("# of channels should be >= 1");
    }

    int bytes_per_frame = size * nchannels;
    int frame_count = cp.size() / bytes_per_frame;

    if (bytes_per_frame / nchannels != size) {
        throw std::overflow_error("width * nchannels too big for a C int");
    }

    if (weightA < 1 || weightB < 0) {
        throw std::invalid_argument("weightA should be >= 1, weightB should be >= 0");
    }

    if (cp.size() % bytes_per_frame != 0) {
        throw std::invalid_argument("not a whole number of frames");
    }

    if (inrate <= 0 || outrate <= 0) {
        throw std::invalid_argument("sampling rate not > 0");
    }

    int d = std::gcd(inrate, outrate);
    inrate /= d;
    outrate /= d;

    std::vector<int> prev_i(nchannels, 0);
    std::vector<int> cur_i(nchannels, 0);

    if (state.first == -1) {
        d = -outrate;
    } else {
        d = state.first;
        const std::vector<std::pair<int, int>>& samps = state.second;

        if (samps.size() != static_cast<size_t>(nchannels)) {
            throw std::invalid_argument("illegal state argument");
        }

        for (int i = 0; i < nchannels; ++i) {
            prev_i[i] = samps[i].first;
            cur_i[i] = samps[i].second;
        }
    }

    double q = static_cast<double>(frame_count) / inrate;
    double ceiling = (q + 1) * outrate;
    size_t nbytes = static_cast<size_t>(ceiling) * bytes_per_frame;

    std::vector<char> result(nbytes);
    auto samples = _get_samples(cp, size);
    int out_i = 0;

    while (true) {
        while (d < 0) {
            if (frame_count == 0) {
                std::vector<std::pair<int, int>> samps(nchannels);
                for (int chan = 0; chan < nchannels; ++chan) {
                    samps[chan] = std::make_pair(prev_i[chan], cur_i[chan]);
                }
                result.resize(out_i * bytes_per_frame);
                return result;
            }

            for (int chan = 0; chan < nchannels; ++chan) {
                prev_i[chan] = cur_i[chan];
                cur_i[chan] = samples[next()]; // Assuming samples[next()] gets the next sample

                cur_i[chan] = static_cast<int>(
                    (weightA * cur_i[chan] + weightB * prev_i[chan]) /
                    (weightA + weightB)
                );
            }

            --frame_count;
            d += outrate;
        }

        while (d >= 0) {
            for (int chan = 0; chan < nchannels; ++chan) {
                int cur_o = static_cast<int>(
                    (prev_i[chan] * d + cur_i[chan] * (outrate - d)) /
                    outrate
                );
                _put_sample(result, size, out_i, _overflow(cur_o, size));
                ++out_i;
            }
            d -= inrate;
        }
    }
}

// Function to convert linear PCM to u-law
std::vector<char> lin2ulaw(const std::vector<char>& cp, int size) {
    throw std::runtime_error("Function lin2ulaw is not implemented");
}

// Function to convert u-law to linear PCM
std::vector<char> ulaw2lin(const std::vector<char>& cp, int size) {
    throw std::runtime_error("Function ulaw2lin is not implemented");
}

// Function to convert linear PCM to a-law
std::vector<char> lin2alaw(const std::vector<char>& cp, int size) {
    throw std::runtime_error("Function lin2alaw is not implemented");
}

// Function to convert a-law to linear PCM
std::vector<char> alaw2lin(const std::vector<char>& cp, int size) {
    throw std::runtime_error("Function alaw2lin is not implemented");
}

// Function to convert linear PCM to ADPCM
std::vector<char> lin2adpcm(const std::vector<char>& cp, int size, std::pair<int, int> state) {
    throw std::runtime_error("Function lin2adpcm is not implemented");
}

// Function to convert ADPCM to linear PCM
std::vector<char> adpcm2lin(const std::vector<char>& cp, int size, std::pair<int, int> state) {
    throw std::runtime_error("Function adpcm2lin is not implemented");
}


} // namespace cppdub
