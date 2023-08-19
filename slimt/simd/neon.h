
#include <arm_neon.h>
#include <cstddef>

namespace slimt {

template <> struct VDatum<VExt::w4> {
 public:
  using Scalar = float;
  using Register = float32x4_t;
  static constexpr size_t kWidth = 4;
  VDatum() = default;
  // NOLINTBEGIN
  VDatum(const Register& register_value) : register_(register_value) {}
  // __m128 _mm_set1_ps(float) copies value into all slots, vdupq_n_f32 is it's
  // NEON equivalent.
  VDatum(const float& register_value)
      : register_(vdupq_n_f32(register_value)) {}

  operator const Register&() const { return register_; }
  operator Register&() { return register_; }
  // NOLINTEND

  VDatum& operator=(const float& value) {
    register_ = vdupq_n_f32(value);
    return *this;
  }

  float operator[](size_t i) const {
    return *(((float *)&register_) +
             i);  // potentially undefined, but efficient. In
                  // practice __m128 is an array of floats
  }

 private:
  // NEON uses 128-bit SIMD registers, same as SSE. We are copying this class
  // and locally aliasing Register to float32x4_t, which is the NEON
  // equivalent.
  Register register_;
};

template <enum VExt> struct Ops;

template <> struct Ops<VExt::w4> {
  using Datum = VDatum<VExt::w4>;
  using Scalar = Datum::Scalar;
  using Register = Datum::Register;

  // clang-format off
  static Datum exp(const Datum& x)                     { return exp256_ps(x); }
  static Datum relu(const Datum& x)                    { return max(0.0F, x); }

  static Datum max(const Datum& lhs, const Datum& rhs) { return vmaxq_f32(lhs, rhs);}
  static Datum sub(const Datum& lhs, const Datum& rhs) { return vsubq_f32(lhs, rhs); }
  static Datum add(const Datum& lhs, const Datum& rhs) { return vaddq_f32(lhs, rhs); }
  static Datum mul(const Datum& lhs, const Datum& rhs) { return vmulq_f32(lhs, rhs); }
  static Datum div(const Datum& lhs, const Datum& rhs) { return vdivq_f32(lhs, rhs); }
  // clang-format on

  static Datum sigmoid(const Datum &x) {
    Datum e = exp(x);
    return div(e, add(1.0F, e));
  }

  struct Reduce {
    static Scalar max(const Datum &x) {
      Scalar accumulator = x[0];
      for (size_t i = 1; i < Datum::kWidth; ++i) {
        accumulator = accumulator > x[i] ? accumulator : x[i];
      }
      return accumulator;
    }
    static Scalar sum(const Datum &x) {
      Scalar accumulator = x[0];
      for (size_t i = 1; i < Datum::kWidth; ++i) {
        accumulator += x[i];
      }
      return accumulator;
    }
  };
};
} // namespace slimt
