#pragma once
#include <cstddef>

namespace slimt {

class BeamSearch {
 public:
  explicit BeamSearch(size_t max_length_factor);

 private:
  size_t max_length_factor_;
};
}  // namespace slimt
