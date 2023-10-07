#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/Types.hh"

namespace slimt {

// NOLINTBEGIN
enum class Type {
  i8,   //
  ig8,  //
  i32,  //
  u32,  //
  f32,  //
};
// NOLINTEND
//
size_t size_in_bytes(Type type);

// clang-format off
template <Type ScalarType> struct DeduceNative;

template <>                struct DeduceNative <Type::f32> { using Type = float;    };
template <>                struct DeduceNative <Type::i32> { using Type = int32_t;  };
template <>                struct DeduceNative <Type::u32> { using Type = uint32_t; };
template <>                struct DeduceNative <Type::i8>  { using Type = int8_t;   };
template <>                struct DeduceNative <Type::ig8> { using Type = int8_t;   };

// NOLINTBEGIN
template <class Scalar> struct DeduceEnumType;

template <> struct DeduceEnumType<float>      { static constexpr Type value = Type::f32; };
template <> struct DeduceEnumType<int>        { static constexpr Type value = Type::i32; };
template <> struct DeduceEnumType<int8_t>     { static constexpr Type value = Type::i8;  };
template <> struct DeduceEnumType<uint32_t>   { static constexpr Type value = Type::u32;  };
// NOLINTEND
// clang-format on

class Shape {
 public:
  Shape() = default;
  explicit Shape(std::vector<uint64_t> dims);
  uint64_t elements() const;
  void resize(uint64_t size);
  uint64_t *data() { return dims_.data(); }
  uint64_t dim(int idx) const;
  const std::vector<uint64_t> &dims() const { return dims_; }
  size_t size() const { return dims_.size(); };

  Shape transpose(int x, int y) const;
  friend bool operator==(Shape &lhs, Shape &rhs);
  friend std::ostream &operator<<(std::ostream &out, const Shape &shape);

  void set_dim(int idx, int value) {
    if (idx < 0) idx += dims_.size();
    dims_[idx] = value;
    recompute_dims();
  }

  template <class Iterator>
  void set(Iterator *begin, Iterator *end) {
    dims_.resize(std::distance(begin, end));
    std::copy(begin, end, dims_.begin());
    recompute_dims();
  }

 private:
  void recompute_dims();

  size_t elements_ = 0;
  std::vector<uint64_t> dims_;
};

class Tensor {
 public:
  Tensor(Type type, Shape shape, std::string name);

  Tensor() = default;
  void load(View view, Type type, Shape shape, std::string name);

  static Aligned allocate(Type type, const Shape &shape,
                          size_t alignment = kAlignWidth);

  template <class Scalar>
  Scalar *data() {
    return reinterpret_cast<Scalar *>(view_.data);
  }
  template <class Scalar>
  const Scalar *data() const {
    return reinterpret_cast<Scalar *>(view_.data);
  }
  template <class Scalar>
  Scalar item() {
    return *(data<Scalar>());
  }

  template <class Scalar>
  Scalar *begin() {
    return data<Scalar>();
  }

  template <class Scalar>
  Scalar *end() {
    size_t bsize = size_in_bytes(type_) * shape().elements();
    return reinterpret_cast<Scalar *>(data<char>() + bsize);
  }

  template <class Scalar>
  void fill_in_place(Scalar value) {
    std::fill(data<Scalar>(), data<Scalar>() + size(), value);
  }

  bool standalone() const { return aligned_.data() != nullptr; }
  size_t size() const { return shape_.elements(); }
  uint64_t dim(int index);
  Shape &shape() { return shape_; }
  const Shape &shape() const { return shape_; }
  Type type() const { return type_; }
  const std::string &name() const { return name_; }

  Tensor like(const std::string &name) const;

  // This method is added omitting copy constructor so it's explicit it's a
  // copy.
  Tensor clone(const std::string &name = "") const;
  Tensor transpose_2d();

  friend bool operator==(Tensor &lhs, Tensor &rhs);
  friend std::ostream &operator<<(std::ostream &out, const Tensor &tensor);

 private:
  Aligned aligned_;
  View view_;
  Type type_;
  Shape shape_;
  std::string name_;
};

bool operator==(Shape &lhs, Shape &rhs);
bool operator==(Tensor &lhs, Tensor &rhs);

std::string to_string(Type type);

}  // namespace slimt
