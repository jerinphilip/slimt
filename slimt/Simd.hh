#include "slimt/simd/avx2.h"
#include "slimt/simd/sse.h"

namespace slimt {

struct F32x8 {
 public:
  using Scalar = float;
  using Register = __m256;
  static constexpr size_t kWidth = 8;
  F32x8() = default;
  // NOLINTBEGIN
  // clang-tidy complains explicit constructure, but this is intended to
  // interchange comfortably between float, Register and F32x8.
  F32x8(const Register& value) : register_(value) {}

  // Register _mm_set1_ps(float) copies value into all slots
  F32x8(const float& value) : register_(_mm256_set1_ps(value)) {}

  operator const Register&() const { return register_; }
  operator Register&() { return register_; }
  // NOLINTEND

  F32x8& operator=(const float& value) {
    register_ = _mm256_set1_ps(value);
    return *this;
  }

  float operator[](size_t i) const {
    // potentially undefined, but efficient.
    // In practice __m128 is an array of floats.
    const auto* begin = reinterpret_cast<const float*>(&register_);
    return *(begin + i);
  }

 private:
  Register register_;
};

struct F32x4 {
 public:
  using Scalar = float;
  using Register = __m128;
  static constexpr size_t kWidth = 4;
  F32x4() = default;
  // NOLINTBEGIN
  // clang-tidy complains explicit constructure, but this is intended to
  // interchange comfortably between float, Register and F32x4.
  F32x4(const Register& value) : register_(value) {}

  // Register _mm_set1_ps(float) copies value into all slots
  F32x4(const float& value) : register_(_mm_set1_ps(value)) {}

  operator const Register&() const { return register_; }
  operator Register&() { return register_; }
  // NOLINTEND

  F32x4& operator=(const float& value) {
    register_ = _mm_set1_ps(value);
    return *this;
  }

  float operator[](size_t i) const {
    // potentially undefined, but efficient.
    // In practice __m128 is an array of floats.
    const auto* begin = reinterpret_cast<const float*>(&register_);
    return *(begin + i);
  }

 private:
  Register register_;
};

template <class Type>
struct Ops;

template <>
struct Ops<F32x8> {
  using Scalar = F32x8::Scalar;
  using Register = F32x8::Register;

  // clang-format off
  static F32x8 exp(const F32x8& x)                     { return exp256_ps(x); }
  static F32x8 relu(const F32x8& x)                    { return max(0.0F, x); }

  static F32x8 max(const F32x8& lhs, const F32x8& rhs) { return _mm256_max_ps(lhs, rhs); }
  static F32x8 sub(const F32x8& lhs, const F32x8& rhs) { return _mm256_sub_ps(lhs, rhs); }
  static F32x8 add(const F32x8& lhs, const F32x8& rhs) { return _mm256_add_ps(lhs, rhs); }
  static F32x8 mul(const F32x8& lhs, const F32x8& rhs) { return _mm256_mul_ps(lhs, rhs); }
  static F32x8 div(const F32x8& lhs, const F32x8& rhs) { return _mm256_div_ps(lhs, rhs); }
  // clang-format on

  static F32x8 sigmoid(const F32x8& x) {
    F32x8 e = exp(x);
    return div(e, add(1.0F, e));
  }

  struct Reduce {
    static Scalar max(const F32x8& x) {
      Scalar accumulator = x[0];
      for (size_t i = 1; i < F32x8::kWidth; ++i) {
        accumulator = accumulator > x[i] ? accumulator : x[i];
      }
      return accumulator;
    }
    static Scalar sum(const F32x8& x) {
      Scalar accumulator = x[0];
      for (size_t i = 1; i < F32x8::kWidth; ++i) {
        accumulator += x[i];
      }
      return accumulator;
    }
  };
};

template <>
struct Ops<F32x4> {
  using Scalar = F32x4::Scalar;
  using Register = F32x4::Register;

  // clang-format off
  static F32x4 exp(const F32x4& x)                     { return exp_ps(x); }
  static F32x4 relu(const F32x4& x)                    { return max(0.0F, x); }

  static F32x4 max(const F32x4& lhs, const F32x4& rhs) { return _mm_max_ps(lhs, rhs); }
  static F32x4 sub(const F32x4& lhs, const F32x4& rhs) { return _mm_sub_ps(lhs, rhs); }
  static F32x4 add(const F32x4& lhs, const F32x4& rhs) { return _mm_add_ps(lhs, rhs); }
  static F32x4 mul(const F32x4& lhs, const F32x4& rhs) { return _mm_mul_ps(lhs, rhs); }
  static F32x4 div(const F32x4& lhs, const F32x4& rhs) { return _mm_div_ps(lhs, rhs); }
  //clang-format on

  static F32x4 sigmoid(const F32x4& x) {
    F32x4 e = exp(x);
    return div(e, add(1.0F, e));
  }

  struct Reduce {
    static Scalar max(const F32x4& x) {
      Scalar accumulator = x[0];
      for (size_t i = 1; i < F32x4::kWidth; ++i) {
        accumulator = accumulator > x[i]? accumulator : x[i];
      }
      return accumulator;
    }
    static Scalar sum(const F32x4& x) {
      Scalar accumulator = x[0];
      for (size_t i = 1; i < F32x4::kWidth; ++i) {
        accumulator += x[i];
      }
      return accumulator;
    }
  };
};

}  // namespace slimt
