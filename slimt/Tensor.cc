#include "slimt/Tensor.hh"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/Macros.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"

namespace slimt {

size_t size_in_bytes(Type type) {
  size_t scalar_size = 0;
  // NOLINTBEGIN
  // clang-format off
  switch (type) {
    case Type::i8 : scalar_size = sizeof(int8_t); break;
    case Type::ig8: scalar_size = sizeof(int8_t); break;
    case Type::f32: scalar_size = sizeof(float);  break;
    case Type::i32: scalar_size = sizeof(int32_t);  break;
    case Type::u32: scalar_size = sizeof(uint32_t);  break;

  }
  // clang-format on
  // NOLINTEND
  return scalar_size;
}

bool operator==(const Shape &lhs, const Shape &rhs) {
  if (lhs.dims().size() != rhs.dims().size()) return false;
  for (size_t i = 0; i < lhs.dims().size(); i++) {
    if (lhs.dims()[i] != rhs.dims()[i]) return false;
  }
  return true;
}

Shape::Shape(std::vector<uint64_t> dims) : dims_(std::move(dims)) {
  recompute_dims();
}

void Shape::recompute_dims() {
  elements_ = 1;
  for (auto dim : dims_) {
    elements_ *= dim;
  }
}

uint64_t Shape::elements() const { return elements_; }

Shape Shape::transpose(int x, int y) const {
  if (x < 0) x += dims_.size();
  if (y < 0) y += dims_.size();

  std::vector<uint64_t> transposed_dims = dims();
  std::swap(transposed_dims[x], transposed_dims[y]);
  return Shape(std::move(transposed_dims));
}

void Shape::resize(uint64_t size) { dims_.resize(size); }

std::ostream &operator<<(std::ostream &out, const Shape &shape) {
  out << "Shape(";
  for (size_t i = 0; i < shape.dims_.size(); i++) {
    if (i != 0) {
      out << "x";
    }
    out << shape.dims_[i];
  }
  out << ")";
  return out;
}

uint64_t Shape::dim(int idx) const {
  if (idx < 0) {
    idx += dims_.size();
  }
  return dims_[idx];
}

void Tensor::load(View view, Type type, Shape shape, std::string name) {
  view_ = view;
  view_.size = size_in_bytes(type) * shape.elements();
  shape_ = std::move(shape);
  type_ = type;
  name_ = std::move(name);
}

uint64_t Tensor::dim(int index) const { return shape_.dim(index); }

std::string to_string(Type type) {
  switch (type) {
    // clang-format off
#define CASE(_type) case Type::_type: return #_type
    CASE(ig8);
    CASE(i8);
    CASE(f32);
    CASE(i32);
    CASE(u32);
#undef CASE
    // clang-format on
  }

  auto number = static_cast<uint64_t>(type);
  return "Unknown" + std::to_string(number);
}

Aligned Tensor::allocate(Type type, const Shape &shape, size_t alignment) {
  size_t scalar_size = size_in_bytes(type);
  size_t num_elements = shape.elements();
  return Aligned(alignment, num_elements * scalar_size);
}

Tensor::Tensor(Type type, Shape shape, std::string name)
    : aligned_(allocate(type, shape)),
      view_{aligned_.data(), aligned_.size()},
      type_(type),
      shape_(std::move(shape)),
      name_(std::move(name)) {}

Tensor Tensor::like(const std::string &name) const {
  return Tensor(type_, shape_, name);
}

Tensor Tensor::clone(const std::string &name) const {
  Tensor t = like(name.empty() ? name_ : name);
  const char *in = reinterpret_cast<const char *>(view_.data);
  char *out = reinterpret_cast<char *>(t.view_.data);
  std::copy(in, in + view_.size, out);
  return t;
}

Tensor Tensor::transpose_2d() {
  Tensor transposed(type_, shape_.transpose(0, 1), name_ + "_transpose");
  switch (type_) {
    case Type::f32:
      transpose_10(data<float>(), shape_.dim(-2), shape_.dim(-1),
                   transposed.data<float>());
      break;
    case Type::i8:
    case Type::ig8:
      transpose_10(data<int8_t>(), shape_.dim(-2), shape_.dim(-1),
                   transposed.data<int8_t>());
      break;
    case Type::i32:
      transpose_10(data<int>(), shape_.dim(-2), shape_.dim(-1),
                   transposed.data<int>());
      break;
    case Type::u32:
      transpose_10(data<uint32_t>(), shape_.dim(-2), shape_.dim(-1),
                   transposed.data<uint32_t>());
      break;
  }
  return transposed;
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor) {
  out << "Tensor(";
  out << tensor.name_ << ", ";
  out << ((tensor.standalone()) ? "standalone" : "view");
  out << ", ";
  out << to_string(tensor.type_);
  out << ", ";
  out << tensor.shape_;

  if (std::getenv("SLIMT_DEBUG")) {
    out << ", ";
    switch (tensor.type_) {
      case Type::i8:
      case Type::ig8:
        print_ndarray(out, tensor.data<int8_t>(), tensor.shape_.dims());
        break;
      case Type::f32:
        print_ndarray(out, tensor.data<float>(), tensor.shape_.dims());
        break;
      case Type::i32:
        print_ndarray(out, tensor.data<int>(), tensor.shape_.dims());
        break;
      case Type::u32:
        print_ndarray(out, tensor.data<uint32_t>(), tensor.shape_.dims());
        break;
    }
  }
  out << ")";
  return out;
}

bool operator==(const Tensor &lhs, const Tensor &rhs) {
  // Can't always rely on size, because sometimes we do aligned loads. So
  // something that is 256 bytes could only be 16 bytes w.r.t actual elements.
  // if (lhs.view_.size != rhs.view_.size) return false;
  if (lhs.type() != rhs.type()) return false;
  if (lhs.shape() != rhs.shape()) return false;
  const void *lhs_ptr = lhs.data<char>();
  const void *rhs_ptr = rhs.data<char>();

  auto message = [&](size_t position, auto l, auto r, float eps) {
    std::cerr << lhs.name() << " and " << rhs.name();
    std::cerr << "(" << to_string(lhs.type()) << ")";
    std::cerr << "\n differs at position " << position << ": ";
    std::cerr << "[" << std::scientific << l << "] ";
    std::cerr << "[" << std::scientific << r << "] ";
    std::cerr << "\n Δ: " << eps << " | \nbit: ";
    std::bitset<32> bl(l), br(r);  // NOLINT
    std::cerr << "\n " << bl << "\n " << br << "\n";
  };

  // Special cause for float32.
  // Can use this when suspect inconsistent values.
  const char *env_eps = std::getenv("SLIMT_EPS");
  if (env_eps != nullptr and lhs.type() == Type::f32) {  // NOLINT
    size_t size = lhs.size();
    const auto *l = lhs.data<float>();
    const auto *r = rhs.data<float>();

    float eps = std::stof(env_eps);

    SLIMT_TRACE(mse(lhs, rhs));
    for (size_t i = 0; i < size; i++) {
      float diff = std::abs(*l - *r);
      if (diff > eps) {
        SLIMT_TRACE2(diff, eps);
        int *il = (int *)l;  // NOLINT
        int *ir = (int *)r;  // NOLINT
        message(i, *il, *ir, diff);
        return false;
      }
      ++l, ++r;
    }
    return true;
  }

  size_t size_in_memory = std::min(lhs.view().size, rhs.view().size);
  int retval = memcmp(lhs_ptr, rhs_ptr, size_in_memory);
  // -1, 0 +1 if < = > respectively C-API, so.
  bool eq = (retval == 0);
  if (not eq) {
    const auto *l = lhs.data<char>();
    const auto *r = rhs.data<char>();
    for (size_t i = 0; i < size_in_memory; i++) {
      float nan = std::numeric_limits<float>::quiet_NaN();
      if (*l != *r) {
        message(i, int(*l), int(*r), nan);  // NOLINT
      }
      ++l, ++r;
    }
  }
  return eq;
}

}  // namespace slimt
