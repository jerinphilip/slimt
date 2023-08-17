#include "slimt/Batch.hh"

#include <cassert>

#include "slimt/Tensor.hh"

namespace slimt {

// Tensor(Type type, Shape shape, std::string name);
Batch::Batch(size_t batch_size, size_t sequence_length, uint32_t pad_id)
    : batch_(Type::u32, Shape({batch_size, sequence_length}), "batch"),
      mask_(Type::f32, Shape({batch_size, sequence_length}), "mask"),
      pad_id_(pad_id) {}

void Batch::add(std::vector<uint32_t> &words) {
  size_t sequence_length = batch_.dim(-1);
  size_t batch_size = batch_.dim(-2);

  (void)batch_size;
  words_.insert(words_.end(), words.begin(), words.end());

  assert(words.size() <= sequence_length);
  assert(index_ < batch_size);

  auto *data = batch_.data<int>();
  auto *mask = mask_.data<float>();
  for (size_t j = 0; j < words.size(); j++) {
    data[index_ * sequence_length + j] = words[j];
    mask[index_ * sequence_length + j] = 1.0F;
  }

  int *pad_begin = data + index_ * sequence_length + words.size();
  int *pad_end = data + (index_ + 1) * sequence_length;
  std::fill(pad_begin, pad_end, pad_id_);

  float *mask_begin = mask + index_ * sequence_length + words.size();
  float *mask_end = mask + (index_ + 1) * sequence_length;
  std::fill(mask_begin, mask_end, 0.0F);
  ++index_;
  used_ += words.size();
}

float Batch::occupancy() {
  size_t sequence_length = batch_.dim(-1);
  size_t batch_size = batch_.dim(-2);
  return used_ / static_cast<float>(batch_size * sequence_length);
}

}  // namespace slimt
