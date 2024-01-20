#include "slimt/Utils.hh"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "slimt/Io.hh"
#include "slimt/Tensor.hh"

namespace slimt {

// clang-format off
template <class Scalar> struct Emit { using Type = Scalar; };
template <>             struct Emit <float> { using Type = float; };
template <>             struct Emit <int8_t> { using Type = int; };

// clang-format on

template <class Scalar>
std::ostream &print_ndarray(std::ostream &out, const Scalar *data,
                            const std::vector<uint64_t> &dims) {
  // Enable printing by recursing on dimensions.
  using EmitT = typename Emit<Scalar>::Type;
  constexpr size_t kTruncate = 4;

  std::function<uint64_t(uint64_t, uint64_t)> recurse;
  recurse = [&out, &dims, &recurse, data](uint64_t d, uint64_t offset) {
    // Base case. Print the vector.
    if (d + 1 == dims.size()) {
      out << "[";
      // we truncate only if we have 2*kTruncate
      bool truncate = dims[d] > 2 * kTruncate;
      if (truncate) {
        out << static_cast<EmitT>(data[offset]);
        for (uint64_t j = offset + 1; j < offset + kTruncate; j++) {
          out << ", " << static_cast<EmitT>(data[j]);
        }
        out << ", ... ";
        for (uint64_t j = offset + dims[d] - kTruncate; j < offset + dims[d];
             j++) {
          out << ", " << static_cast<EmitT>(data[j]);
        }
      } else {
        for (uint64_t j = offset; j < offset + dims[d]; j++) {
          if (j != offset) {
            out << ", ";
          }
          out << static_cast<EmitT>(data[j]);
        }
        // We've printed dims[d] more elements.
      }
      out << "]";
      return offset + dims[d];
    }

    out << "[";
    for (uint64_t j = 0; j < dims[d]; j++) {
      if (j != 0) {
        out << ",";
        if (d + 2 == dims.size()) {
          out << "\n";
        }
      }
      offset = recurse(d + 1, offset);
    }
    out << "]";

    return offset;
  };

  out << "\n";
  recurse(0, 0);
  return out;
}

// Explicit unrolls.
#define SLIMT_PRINT_NDARRAY_EXPLICIT(ScalarType)               \
  template std::ostream &print_ndarray(std::ostream &out,      \
                                       const ScalarType *data, \
                                       const std::vector<uint64_t> &dims)

SLIMT_PRINT_NDARRAY_EXPLICIT(float);
SLIMT_PRINT_NDARRAY_EXPLICIT(int);
SLIMT_PRINT_NDARRAY_EXPLICIT(int8_t);
SLIMT_PRINT_NDARRAY_EXPLICIT(uint32_t);

#undef SLIMT_PRINT_NDARRAY_EXPLICIT

namespace {
Tensor dispatch_by_type(Type type, const std::string &fpath, const Shape &shape,
                        const std::string &name) {
  switch (type) {
    case Type::f32:
      return tensor_from_file<float>(fpath, shape, name);
    case Type::ig8:
    case Type::i8:
      return tensor_from_file<int8_t>(fpath, shape, name);
    case Type::i32:
      return tensor_from_file<int>(fpath, shape, name);
    case Type::u32:
      return tensor_from_file<uint32_t>(fpath, shape, name);
  }
  return Tensor{};
}
}  // namespace

template <class Scalar, class Quant>
std::tuple<Tensor, float> quantized_tensor_from_file(const std::string &fpath,
                                                     const Shape &shape,
                                                     const std::string &name) {
  auto file_exists = [](const std::string &fpath) {
    return (access(fpath.c_str(), F_OK) == 0);
  };

  if (!file_exists(fpath)) {
    std::cerr << "File " << fpath << " not found on disk." << '\n';
  }

  io::MmapFile file(fpath);
  Type type = DeduceEnumType<Scalar>::value;
  Tensor tensor(type, shape, name);

  const auto *data = reinterpret_cast<const Scalar *>(file.data());
  size_t size_on_disk = file.size();
  size_t size_elements = tensor.size();
  size_t size_expected = sizeof(Scalar) * size_elements;

  if (size_on_disk != size_expected) {
    fprintf(stderr,
            "Mismatch in load size (%zu) vs (%zu x %zu = %zu) expected size "
            "detected\n",
            size_on_disk, sizeof(Scalar), size_elements, size_expected);
  }

  std::copy(data, data + size_elements, tensor.data<Scalar>());
  const auto *end = reinterpret_cast<const Quant *>(data + size_elements);
  Quant quantization_multiplier = *end;
  return std::make_tuple(std::move(tensor), quantization_multiplier);
}

template <class Scalar>
Tensor tensor_from_file(const std::string &fpath, const Shape &shape,
                        const std::string &name) {
  auto file_exists = [](const std::string &fpath) {
    return (access(fpath.c_str(), F_OK) == 0);
  };

  if (!file_exists(fpath)) {
    std::cerr << "File " << fpath << " not found on disk." << '\n';
  }

  io::MmapFile file(fpath);
  Type type = DeduceEnumType<Scalar>::value;
  Tensor tensor(type, shape, name);

  const auto *data = reinterpret_cast<const Scalar *>(file.data());
  size_t size_on_disk = file.size();
  size_t size_elements = tensor.size();
  size_t size_expected = sizeof(Scalar) * size_elements;

  if (size_on_disk != size_expected) {
    fprintf(stderr,
            "Mismatch in load size (%zu) vs (%zu x %zu = %zu) expected size "
            "detected\n",
            size_on_disk, sizeof(Scalar), size_elements, size_expected);
  }

  std::copy(data, data + size_elements, tensor.data<Scalar>());
  return tensor;
}

#define SLIMT_TENSOR_FROM_FILE_EXPLICIT(type) \
  template Tensor tensor_from_file<type>(     \
      const std::string &fpath, const Shape &shape, const std::string &name)

#define SLIMT_QTENSOR_FROM_FILE_EXPLICIT(type, quant_type) \
  template std::tuple<Tensor, quant_type>                  \
  quantized_tensor_from_file<type, quant_type>(            \
      const std::string &fpath, const Shape &shape, const std::string &name)

SLIMT_TENSOR_FROM_FILE_EXPLICIT(float);
SLIMT_TENSOR_FROM_FILE_EXPLICIT(int);
SLIMT_TENSOR_FROM_FILE_EXPLICIT(int8_t);
SLIMT_TENSOR_FROM_FILE_EXPLICIT(uint32_t);

SLIMT_QTENSOR_FROM_FILE_EXPLICIT(int8_t, float);

#undef SLIMT_TENSOR_FROM_FILE_EXPLICIT

template <class Scalar>
Scalar AverageMeter<Scalar>::value() {
  return running_avg_;
}

template <class Scalar>
void AverageMeter<Scalar>::record(Scalar point) {
  Scalar n = count_;
  auto np1 = static_cast<Scalar>(count_ + 1);
  running_avg_ = (n * running_avg_ + point) / np1;
  count_++;
}

const char *stringify(bool flag) {
  // Converts 0/1 to "false"/"true"
  return flag ? "true" : "false";
}

template class AverageMeter<float>;

}  // namespace slimt
