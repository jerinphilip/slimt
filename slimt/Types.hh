#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace slimt {

/// ByteRange stores indices for half-interval [begin, end) in a string. Can be
/// used to represent a sentence, word.
struct ByteRange {
  size_t begin;
  size_t end;
  size_t size() const { return end - begin; }
};

inline bool operator==(ByteRange &a, ByteRange b) {
  return a.begin == b.begin && a.end == b.end;
}

using Word = uint32_t;
using Words = std::vector<Word>;
using Views = std::vector<std::string_view>;

using Segment = Words;
using Segments = std::vector<Segment>;

#define ABORT_IF(condition, ...) \
  do {                           \
    if (condition) {             \
      std::cerr << #condition;   \
      std::abort();              \
    }                            \
  } while (0)

#define ABORT(message)      \
  do {                      \
    std::cerr << (message); \
    std::abort();           \
  } while (0)

#define LOG(...) (void)0

}  // namespace slimt
