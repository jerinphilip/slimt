#include "slimt/Io.hh"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "slimt/QMM.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Utils.hh"

namespace slimt::io {

namespace {

template <typename Element, typename ReadHead = void*>
Element* emit(ReadHead& read_head, uint64_t size = 1) {
  auto* begin = reinterpret_cast<Element*>(read_head);
  Element* end = begin + size;
  read_head = reinterpret_cast<void*>(end);
  return begin;
}

// clang-format off
// NOLINTBEGIN
// Internal to types.h, don't use. Use test functions below.
enum class TypeClass : size_t {
  signed_type   = 0x0100,
  unsigned_type = 0x0200,
  float_type    = 0x0400,

  packed_type   = 0x0800, // special packed (CPU cache friendly) type class, used in FBGEMM, not meant to be used anywhere else
  avx2_type     = 0x1000, // processor-specific layout for avx2, currently used for FBGEMM only
  avx512_type   = 0x2000, // processor-specific layout for avx512, currently used for FBGEMM only

  intgemm_type = 0x4000, // intgemm quantized architecture agnostic models


  size_mask     = 0x00FF,
  class_mask    = 0xFF00
};

constexpr inline size_t operator+(TypeClass typeClass, size_t val) {
  return (size_t)typeClass + val;
}

constexpr inline size_t operator+(size_t val, TypeClass typeClass) {
  return val + (size_t)typeClass;
}



enum class OGType : size_t {
  int8     = TypeClass::signed_type + 1u,
  int16    = TypeClass::signed_type + 2u,
  int32    = TypeClass::signed_type + 4u,
  int64    = TypeClass::signed_type + 8u,

  uint8    = TypeClass::unsigned_type + 1u,
  uint16   = TypeClass::unsigned_type + 2u,
  uint32   = TypeClass::unsigned_type + 4u,
  uint64   = TypeClass::unsigned_type + 8u,

  float16  = TypeClass::float_type + 2u,
  float32  = TypeClass::float_type + 4u,
  float64  = TypeClass::float_type + 8u,

  packed16      = TypeClass::packed_type + 2u,                          // special type for FBGEMM, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint16) is meaningless.
  packed8avx2   = TypeClass::packed_type + 1u + TypeClass::avx2_type,   // special type for FBGEMM with AVX2, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint8) is meaningless.
  packed8avx512 = TypeClass::packed_type + 1u + TypeClass::avx512_type, // special type for FBGEMM with AVX512, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint8) is meaningless.

  intgemm8      = TypeClass::signed_type + 1u + TypeClass::intgemm_type, // Int8 quantized (not packed) matrices for intgemm
  intgemm16     = TypeClass::signed_type + 2u + TypeClass::intgemm_type // Int16 quantized (not packed) matrices for intgemm
};

// clang-format on
// NOLINTEND

Type intercept(uint64_t value) {
  auto type = static_cast<OGType>(value);
  switch (type) {
    case OGType::intgemm8:
      return Type::ig8;
    case OGType::int8:
      return Type::i8;
    case OGType::float32:
      return Type::f32;
    default:
      std::cerr << "Incompatible type.\n";
      std::abort();
  }
}

}  // namespace

void set_item(Item& item, Aligned&& aligned) {
  item.aligned = std::move(aligned);
  item.view = View{
      .data = reinterpret_cast<char*>(item.aligned.data()),  //
      .size = item.aligned.size()                            //
  };
}

std::vector<io::Item> loadItems(void* current) {
  uint64_t binary_file_version = *emit<uint64_t>(current);
  if (binary_file_version != kBinaryFileVersion) {
    std::cerr << "Binary file versions do not match: ";
    std::cerr << binary_file_version << "(file) != ";
    std::cerr << kBinaryFileVersion << " (expected)";

    std::abort();
  }

  // Read number of headers and based on the information, the headers.
  uint64_t num_headers = *emit<uint64_t>(current);
  Header* headers = emit<Header>(current, num_headers);  // NOLINT

  // prepopulate items with meta data from headers
  std::vector<io::Item> items;
  items.resize(num_headers);
  for (uint64_t i = 0; i < num_headers; ++i) {
    items[i].type = intercept(headers[i].type);

    // Can someone explain the -1? Remains a mystery to me.
    size_t length = headers[i].name_length;
    char* name = emit<char>(current, length);
    items[i].name = std::string(name, length - 1);
  }

  // read in actual shape and data
  for (uint64_t i = 0; i < num_headers; ++i) {
    Item& item = items[i];

    uint64_t size = headers[i].shape_length;
    int* shape = emit<int>(current, size);

    // This copy has to be incurred, because metadata.
    item.shape.set(shape, shape + size);
  }

  // move by offset bytes, aligned to 256-bytes boundary
  uint64_t offset = *emit<uint64_t>(current);
  emit<char>(current, offset);

  // Keep an extra item for embedding processed.
  Item embedding_processed;

  for (uint64_t i = 0; i < num_headers; ++i) {
    Item& item = items[i];
    uint64_t size = headers[i].data_length;
    char* ptr = emit<char>(current, size);
    // We're about to read-data.
    // We can either make it point to mmap, which is aligned,
    // or we can create a new aligned.
    if (item.type == Type::ig8) {
      // since Embedding layer quantized weights need to be dequantised, we
      // have a special case for items containing the name "Wemb"
      if (item.name == "Wemb_QuantMultA") {
        // Wemb_QuantMultA hints at this being the quantization multiplier for
        // when we have to process at linear multiply on embedding.  However,
        // this value does not hold anything useful.

        // It is `none_QuantMultA` of type `float32` that holds the useful
        // quantization multiplier.

        // Pointing to this, that's all, mostly a no-op and prevents falling
        // into the other branch.
        item.view = View{
            .data = ptr,  //
            .size = size  //
        };
      } else if (item.name == "Wemb") {  // NOLINT
        size_t num_elements = item.shape.elements();
        // At the end of items is the quantization multiplier.So we do some
        // pointer arithmetic to move ahead of the elements to extract the
        // quantization multiplier.
        char* end = ptr + num_elements;
        auto* quantization_multiplier_addr = reinterpret_cast<float*>(end);
        float quantization_multiplier = *(quantization_multiplier_addr);

        // Allocate aligned storage to write out unquantized embeddings.
        size_t size_as_float = num_elements * sizeof(float);
        Aligned aligned(/*alignment=*/64, size_as_float);

        auto* quantized_weights = reinterpret_cast<int8_t*>(ptr);
        auto* weights = reinterpret_cast<float*>(aligned.data());
        unquantize_embedding_weights(quantized_weights, quantization_multiplier,
                                     num_elements, weights);
        set_item(item, std::move(aligned));
        item.type = Type::f32;

        size_t rows = item.shape.dim(-2);
        size_t cols = item.shape.dim(-1);
        assert((rows * cols) % 8 == 0);

        // PrepareB and write.
        embedding_processed.name = "Wemb_intgemm8";
        embedding_processed.shape = Shape({cols, rows});
        embedding_processed.type = Type::i8;
        size_t prepared_size =
            embedding_processed.shape.elements() * sizeof(int8_t) +
            sizeof(float);
        Aligned embedding_aligned(/*alignment=*/64, prepared_size);
        auto* prepared = reinterpret_cast<int8_t*>(embedding_aligned.data());
        qmm::prepare_weight_transposed(weights, prepared, quantization_multiplier,
                                  cols, rows);

        // Save quantization multiplier.
        auto* embedding_quantization_multiplier_addr =
            reinterpret_cast<float*>(prepared + (rows * cols));
        *embedding_quantization_multiplier_addr = quantization_multiplier;

        // SLIMT_TRACE(embedding_processed.shape);
        set_item(embedding_processed, std::move(embedding_aligned));
      } else {
        // The matrix has to be processed to the format expected by intgemm.
        size_t rows = item.shape.dim(-2);
        size_t cols = item.shape.dim(-1);
        auto* input = reinterpret_cast<int8_t*>(ptr);

        Aligned aligned(/*alignment=*/64, rows * cols + sizeof(float));

        auto* output = reinterpret_cast<int8_t*>(aligned.data());
        qmm::prepare_weight_quantized_transposed(input, output, rows, cols);

        // Set b_quant at end.
        auto* output_end = reinterpret_cast<float*>(output + rows * cols);
        auto* input_end = reinterpret_cast<float*>(input + rows * cols);
        *output_end = *input_end;

        set_item(item, std::move(aligned));
        item.type = Type::i8;

        // This debug function exists here to inspect if need be.
        auto debug = [&]() {
          Tensor input_view;
          View original = {
              .data = ptr,                    //
              .size = headers[i].data_length  //
          };

          input_view.load(original, item.type, item.shape, item.name);
          std::cerr << "input" << input_view << "\n";

          Tensor output_view;
          output_view.load(item.view, item.type, item.shape, item.name);
          std::cerr << "output" << output_view << "\n";
          std::abort();
        };

        (void)debug;
      }
    } else {
      item.view = View{
          .data = ptr,  //
          .size = size  //
      };
    }
  }

  items.push_back(std::move(embedding_processed));
  return items;
}

void unquantize_embedding_weights(const int8_t* quantized_weights,
                                  float quantization_multiplier, size_t size,
                                  float* weights) {
  // Now proceed to unquantize the int8_ts into floats.
  for (size_t i = 0; i < size; i++) {
    weights[i] = static_cast<float>(quantized_weights[i]) *
                 (1 / quantization_multiplier);
  }
}

std::ostream& operator<<(std::ostream& out, const Item& item) {
  out << "Item(" << item.name << ", ";
  out << to_string(item.type) << ", ";
  out << item.shape << ")";
  return out;
}

MmapFile::MmapFile(const std::string& filepath) {
  fd_ = open(filepath.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw std::runtime_error("Failed to open file: " + filepath);
  }

  struct stat st;
  if (fstat(fd_, &st) == -1) {
    close(fd_);
    throw std::runtime_error("Failed to get file size: " + filepath);
  }
  size_ = st.st_size;

  data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (data_ == MAP_FAILED) {  // NOLINT
    close(fd_);
    throw std::runtime_error("Failed to mmap file: " + filepath);
  }
}

MmapFile::~MmapFile() {
  if (data_ != nullptr) {
    munmap(data_, size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

MmapFile::MmapFile(MmapFile&& from) noexcept
    : fd_(from.fd_), data_(from.data_), size_(from.size_) {
  reset();
};

MmapFile& MmapFile::operator=(MmapFile&& from) noexcept {
  if (this == &from) {
    return *this;
  }
  reset();
  consume(from);
  return *this;
}

void MmapFile::consume(MmapFile& from) {
  fd_ = (from.fd_);
  data_ = (from.data_);
  size_ = (from.size_);
  from.reset();
}

void MmapFile::reset() {
  fd_ = -1;
  data_ = nullptr;
  size_ = 0;
}

}  // namespace slimt::io
