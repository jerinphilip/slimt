#include "slimt/Aligned.hh"

#include <cassert>

namespace slimt {

Aligned::Aligned(size_t alignment, size_t size)
    : data_(allocate(alignment, size)), size_(size) {}

Aligned::~Aligned() { release(); }

void* Aligned::data() const { return data_; }

size_t Aligned::size() const { return size_; }

Aligned::Aligned(Aligned&& from) noexcept {
  if (this != &from) {
    release();
    consume(from);
  }
}

Aligned& Aligned::operator=(Aligned&& from) noexcept {
  if (this != &from) {
    release();
    consume(from);
  }
  return *this;
}

void Aligned::consume(Aligned& from) {
  data_ = from.data_;
  size_ = from.size_;

  from.data_ = nullptr;
  from.size_ = 0;
}

void* Aligned::allocate(size_t alignment, size_t size) {
  size_t aligned_size = (size / alignment) * alignment;
  if (size % alignment != 0) {
    aligned_size += alignment;
  }
  assert(aligned_size >= size);
  return aligned_alloc(alignment, aligned_size);
}

void Aligned::release() {
  if (data_ != nullptr) {
    free(data_);
  }
  size_ = 0;
}
}  // namespace slimt
