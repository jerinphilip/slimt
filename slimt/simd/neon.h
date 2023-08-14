
#include <arm_neon.h>

namespace slimt {

template <>
struct VDatum<VExt::w4> {
 public:
  using Register = float32x4_t;
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
    return *(((float*)&register_) +
             i);  // potentially undefined, but efficient. In
                  // practice __m128 is an array of floats
  }

 private:
  // NEON uses 128-bit SIMD registers, same as SSE. We are copying this class
  // and locally aliasing Register to float32x4_t, which is the NEON
  // equivalent.
  Register register_;
};

}  // namespace slimt
