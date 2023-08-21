#include <arm_neon.h>

#include <cstddef>

using v4sf = float32x4_t;  // vector of 4 float
using v4su = uint32x4_t;   // vector of 4 uint32
using v4si = int32x4_t;    // vector of 4 uint32
                           //
#define c_exp_hi 88.3762626647949f
#define c_exp_lo (-88.3762626647949f)

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 (-2.12194440e-4)

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
v4sf exp_ps(v4sf x) {
  v4sf tmp;
  v4sf fx;

  v4sf one = vdupq_n_f32(1);
  x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
  x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  /* if greater, substract 1 */
  v4su mask = vcgtq_f32(tmp, fx);
  mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
  v4sf z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  static const float cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1,
                                        c_cephes_exp_p2, c_cephes_exp_p3,
                                        c_cephes_exp_p4, c_cephes_exp_p5};
  v4sf y = vld1q_dup_f32(cephes_exp_p + 0);
  v4sf c1 = vld1q_dup_f32(cephes_exp_p + 1);
  v4sf c2 = vld1q_dup_f32(cephes_exp_p + 2);
  v4sf c3 = vld1q_dup_f32(cephes_exp_p + 3);
  v4sf c4 = vld1q_dup_f32(cephes_exp_p + 4);
  v4sf c5 = vld1q_dup_f32(cephes_exp_p + 5);

  y = vmulq_f32(y, x);
  z = vmulq_f32(x, x);
  y = vaddq_f32(y, c1);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c2);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c3);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c4);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c5);

  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, one);

  /* build 2^n */
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
  mm = vshlq_n_s32(mm, 23);
  v4sf pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}

namespace slimt {

template <>
struct VDatum<VExt::w4> {
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

template <enum VExt>
struct Ops;

template <>
struct Ops<VExt::w4> {
  using Datum = VDatum<VExt::w4>;
  using Scalar = Datum::Scalar;
  using Register = Datum::Register;

  // clang-format off
  static Datum exp(const Datum& x)                     { return exp_ps(x); }
  static Datum relu(const Datum& x)                    { return max(0.0F, x); }

  static Datum max(const Datum& lhs, const Datum& rhs) { return vmaxq_f32(lhs, rhs);}
  static Datum sub(const Datum& lhs, const Datum& rhs) { return vsubq_f32(lhs, rhs); }
  static Datum add(const Datum& lhs, const Datum& rhs) { return vaddq_f32(lhs, rhs); }
  static Datum mul(const Datum& lhs, const Datum& rhs) { return vmulq_f32(lhs, rhs); }
  static Datum div(const Datum& lhs, const Datum& rhs) { return vdivq_f32(lhs, rhs); }
  // clang-format on

  static Datum sigmoid(const Datum& x) {
    Datum e = exp(x);
    return div(e, add(1.0F, e));
  }

  struct Reduce {
    static Scalar max(const Datum& x) {
      Scalar accumulator = x[0];
      for (size_t i = 1; i < Datum::kWidth; ++i) {
        accumulator = accumulator > x[i] ? accumulator : x[i];
      }
      return accumulator;
    }
    static Scalar sum(const Datum& x) {
      Scalar accumulator = x[0];
      for (size_t i = 1; i < Datum::kWidth; ++i) {
        accumulator += x[i];
      }
      return accumulator;
    }
  };
};
}  // namespace slimt
