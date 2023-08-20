#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <string>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt {

constexpr static uint64_t kBinaryFileVersion = 1;

namespace io {

// Header is used to store metadata on the contents in a binary-storage format.
struct Header {
  uint64_t name_length;
  uint64_t type;
  uint64_t shape_length;
  uint64_t data_length;
};

struct Item {
  Aligned aligned;
  View view;

  std::string name;
  Shape shape;
  Type type{Type::f32};
};

void set_item(Item& item, Aligned&& aligned);

std::vector<io::Item> loadItems(void* current);
std::ostream& operator<<(std::ostream& out, const Item& item);

void unquantize_embedding_weights(const int8_t* quantized_weights,
                                  float quantization_multiplier, size_t size,
                                  float* weights);

class MmapFile {
 public:
  explicit MmapFile(const std::string& filepath);
  ~MmapFile();

  void* data() const { return data_; }
  size_t size() const { return size_; }

  // Disable copy and assignment
  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;

  MmapFile(MmapFile&& from) noexcept;

  MmapFile& operator=(MmapFile&& from) noexcept;

 private:
  void consume(MmapFile& from);
  void reset();

  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace io

}  // namespace slimt
