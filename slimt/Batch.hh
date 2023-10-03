#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt {

class Batch {
 public:
  Batch(size_t batch_size, size_t sequence_length, uint32_t pad_id);

  void add(std::vector<uint32_t> &words);
  Tensor &indices() { return batch_; }
  Tensor &mask() { return mask_; }
  std::vector<uint32_t> &words() { return words_; }
  std::vector<size_t> &lengths() { return lengths_; }
  size_t index() const { return index_; }

 private:
  std::vector<uint32_t> words_;
  std::vector<size_t> lengths_;
  Tensor batch_;
  Tensor mask_;
  size_t index_ = 0;
  uint32_t pad_id_ = 0;
};
}  // namespace slimt
