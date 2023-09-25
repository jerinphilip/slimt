#pragma once
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

#include "slimt/Cache.hh"

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
using Sentences = std::vector<Words>;

template <class T>
using Ptr = std::shared_ptr<T>;

using Alignment = std::vector<std::vector<float>>;
using Alignments = std::vector<Alignment>;

struct Hypothesis {
  Segment target;
  Alignment alignment;
};

using History = Ptr<Hypothesis>;
using Histories = std::vector<History>;
using TranslationCache = AtomicCache<size_t, History>;

}  // namespace slimt
