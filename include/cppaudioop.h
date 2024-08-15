#ifndef CPPAUDIOOP_H
#define CPPAUDIOOP_H

#include <vector>
#include <string>
#include <exception>
#include <algorithm>   // For std::max, std::min
#include <numeric>     // For std::gcd
#include <cmath>       // For math functions
#include <cstring>     // For C-style string handling
#include <utility>     // For std::pair

namespace cppdub {

class error : public std::exception {
public:
    explicit error(const char* message) : msg_(message) {}
    explicit error(const std::string& message) : msg_(message) {}
    virtual ~error() noexcept {}
    virtual const char* what() const noexcept { return msg_.c_str(); }

private:
    std::string msg_;
};

// Function declarations
void _check_size(int size);
void _check_params(int length, int size);
int _sample_count(const std::vector<char>& cp, int size);
std::vector<int> _get_samples(const std::vector<char>& cp, int size);
std::string _struct_format(int size, bool signed_);
int _get_sample(const std::vector<char>& cp, int size, int index, bool signed_=true);
void _put_sample(std::vector<char>& result, int size, int index, int sample);
int _get_maxval(int size, bool signed_);
int _get_minval(int size, bool signed_);
std::function<int(int)> _get_clipfn(int size, bool signed_ = true);
int _overflow(int sample, int size, bool signed_ = true);


int getsample(const std::vector<char>& cp, int size, int index);
int max(const std::vector<char>& cp, int size);
int minmax(const std::vector<char>& cp, int size, int& minval, int& maxval);
double avg(const std::vector<char>& cp, int size);
double rms(const std::vector<char>& cp, int size);
int _sum2(const std::vector<char>& cp1, const std::vector<char>& cp2, int size);  // Updated
int findfit(const std::vector<char>& cp, int size, int value);  // Added declaration
std::pair<int, double> findfit(const std::vector<char>& cp1, const std::vector<char>& cp2);  // Added declaration
double findfactor(const std::vector<char>& cp1, const std::vector<char>& cp2);  // Added declaration
int findmax(const std::vector<char>& cp, int len2);  // Added declaration
double avgpp(const std::vector<char>& cp, int size);  // Added declaration
std::vector<char> maxpp(const std::vector<char>& cp, int size);  // Added declaration
int cross(const std::vector<char>& cp, int size);  // Added declaration
std::vector<char> mul(const std::vector<char>& cp, int size, int factor);  // Added declaration
std::vector<char> tomono(const std::vector<char>& cp, int size, int fac1, int fac2);  // Added declaration
std::vector<char> tostereo(const std::vector<char>& cp, int size, int fac1, int fac2);  // Added declaration
std::vector<char> add(const std::vector<char>& cp1, const std::vector<char>& cp2, int size);  // Added declaration
std::vector<char> bias(const std::vector<char>& cp, int size, int amount);  // Added declaration
std::vector<char> reverse(const std::vector<char>& cp, int size);  // Added declaration
std::vector<char> lin2lin(const std::vector<char>& cp, int size, int size2);  // Added declaration
std::vector<char> ratecv(const std::vector<char>& cp, int size, int nchannels, int inrate, int outrate, std::pair<int, std::vector<std::pair<int, int>>> state, double weightA, double weightB);
std::vector<char> lin2ulaw(const std::vector<char>& cp, int size);
std::vector<char> ulaw2lin(const std::vector<char>& cp, int size);
std::vector<char> lin2alaw(const std::vector<char>& cp, int size);
std::vector<char> alaw2lin(const std::vector<char>& cp, int size);
std::vector<char> lin2adpcm(const std::vector<char>& cp, int size, std::pair<int, int> state);
std::vector<char> adpcm2lin(const std::vector<char>& cp, int size, std::pair<int, int> state);

} // namespace cppdub

#endif // CPPAUDIOOP_H
