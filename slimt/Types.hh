#pragma once
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

#include "slimt/Cache.hh"

namespace slimt {

/// Range stores indices for half-interval [begin, end) in a string. Can be
/// used to represent a sentence, word.
struct Range {
  size_t begin;
  size_t end;
  size_t size() const { return end - begin; }
};

inline bool operator==(Range &a, Range b) {
  return a.begin == b.begin && a.end == b.end;
}

using Word = uint32_t;
using Words = std::vector<Word>;

struct View {
  void *data = nullptr;
  size_t size = 0;
};

using Views = std::vector<std::string_view>;

using Segment = Words;
using Segments = std::vector<Segment>;
using Sentences = std::vector<Words>;

template <class T>
using Ptr = std::shared_ptr<T>;

using Distribution = std::vector<float>;
using Alignment = std::vector<Distribution>;
using Alignments = std::vector<Alignment>;

struct Hypothesis {
  Segment target;
  Alignment alignment;
};

using History = Ptr<Hypothesis>;
using Histories = std::vector<History>;
using TranslationCache = AtomicCache<size_t, History>;

}  // namespace slimt
