#pragma once
#include <chrono>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace slimt {

class Tensor;

class Shape;

std::string checked_fpath();

class Verifier {
 public:
  static Verifier &instance() {
    static Verifier verifier;
    return verifier;
  }
  bool verify(Tensor &value, const std ::string &name);

 private:
  Verifier() : blob_path_(checked_fpath()) {}
  std::unordered_set<std::string> verified_;
  std::string blob_path_;
};

#define SLIMT_VERIFY_MATCH(value, name)            \
  do {                                             \
    const char *flag = std::getenv("SLIMT_TRACE"); \
    if (flag) {                                    \
      (Verifier::instance()).verify(value, name);  \
    }                                              \
  } while (0)

template <class Printable>
std::string fmt(Printable &printable) {
  std::stringstream stream;
  stream << printable;
  return stream.str();
}

template <class Scalar>
std::ostream &print_ndarray(std::ostream &out, const Scalar *data,
                            const std::vector<uint64_t> &dims);

template <class Scalar>
Tensor tensor_from_file(const std::string &fpath, const Shape &shape,
                        const std::string &name);

template <class Scalar, class Quant>
std::tuple<Tensor, float> quantized_tensor_from_file(const std::string &fpath,
                                                     const Shape &shape,
                                                     const std::string &name);

// This combinator is based on boost::hash_combine, but uses
// std::hash as the hash implementation. Used as a drop-in
// replacement for boost::hash_combine.
template <class T, class HashType = std::size_t>
inline void hash_combine(HashType &seed, const T &v) {
  std::hash<T> hasher;
  // Constants only appear here, and are taken from some StackOverflow (or
  // marian).
  seed ^= (                             //
      static_cast<HashType>(hasher(v))  //
      + 0x9e3779b9                      // NOLINT
      + (seed << 6) + (seed >> 2)       // NOLINT
  );
}

// Hash a whole chunk of memory, mostly used for diagnostics
template <class T, class HashType = std::size_t>
inline HashType hash_bytes(const T *data, size_t size) {
  HashType seed = 0;
  for (auto p = data; p < data + size; ++p) {
    hash_combine(seed, *p);
  }
  return seed;
}

class Timer {
 public:
  // Create and start the timer
  Timer() : start_(clock::now()) {}

  Timer(const Timer &timer) = delete;
  Timer &operator=(const Timer &timer) = delete;

  Timer(Timer &&timer) = default;
  Timer &operator=(Timer &&timer) = default;

  // Get the time elapsed without stopping the timer.  If the template type is
  // not specified, it returns the time counts as represented by
  // std::chrono::seconds
  template <class Duration = std::chrono::seconds>
  double elapsed() const {
    using duration_double =
        std::chrono::duration<double, typename Duration::period>;
    return std::chrono::duration_cast<duration_double>(clock::now() - start_)
        .count();
  }

 protected:
  using clock = std::chrono::steady_clock;
  using time_point = std::chrono::time_point<clock>;
  using duration = std::chrono::nanoseconds;

  time_point start_;     // Starting time point
  bool stopped_{false};  // Indicator if the timer has been stopped
  duration time_;        // Time duration from start() to stop()
};

template <class Scalar>
class AverageMeter {
 public:
  AverageMeter() = default;
  void reset();
  Scalar value();
  void record(Scalar point);

 private:
  Scalar running_avg_ = 0;
  size_t count_ = 0;
};

}  // namespace slimt
