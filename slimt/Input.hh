#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt {

class Input {
 public:
  Input(size_t batch_size, size_t sequence_length, uint32_t pad_id,
        size_t limit_factor);

  void add(const std::vector<uint32_t> &words);
  const Tensor &indices() { return batch_; }
  Tensor &mask() { return mask_; }
  std::vector<uint32_t> &words() { return words_; }
  std::vector<size_t> &lengths() { return lengths_; }
  size_t index() const { return index_; }
  float occupancy();
  float limit_factor() const;

 private:
  std::vector<uint32_t> words_;
  std::vector<size_t> lengths_;
  Tensor batch_;
  Tensor mask_;
  size_t index_ = 0;
  uint32_t pad_id_ = 0;
  size_t used_ = 0;
  size_t limit_factor_;
};
}  // namespace slimt
