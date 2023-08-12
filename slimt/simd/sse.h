#include <xmmintrin.h>

#ifdef _MSC_VER /* visual c++ */
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#ifndef USE_SSE2  // MSVC doesn't allow us to compile MME anyways
#define USE_SSE2  // so just hardcore disable it
#endif
#else /* gcc or icc */
#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))
#endif

// NOLINTBEGIN
/* __m128 is ugly to write */
typedef __m128 v4sf;  // vector of 4 float (sse1)

#ifdef USE_SSE2
#include <emmintrin.h>
typedef __m128i v4si;  // vector of 4 int (sse2)
#else
typedef __m64 v2si;  // vector of 2 int (mmx)
#endif

/* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val)                                   \
  static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { \
      (float)Val, (float)Val, (float)Val, (float)Val}
#define _PI32_CONST(Name, Val)                                               \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = {Val, Val, Val, \
                                                              Val}
#define _PS_CONST_TYPE(Name, Type, Val) \
  static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
// _PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

// _PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
// _PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

// _PI32_CONST(1, 1);
// _PI32_CONST(inv1, ~1);
// _PI32_CONST(2, 2);
// _PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, -1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, -1.2420140846E-1);
_PS_CONST(cephes_log_p4, +1.4249322787E-1);
_PS_CONST(cephes_log_p5, -1.6668057665E-1);
_PS_CONST(cephes_log_p6, +2.0000714765E-1);
_PS_CONST(cephes_log_p7, -2.4999993993E-1);
_PS_CONST(cephes_log_p8, +3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);

#ifndef USE_SSE2
typedef union xmm_mm_union {
  __m128 xmm;
  __m64 mm[2];
} xmm_mm_union;

#define COPY_XMM_TO_MM(xmm_, mm0_, mm1_) \
  {                                      \
    xmm_mm_union u;                      \
    u.xmm = xmm_;                        \
    mm0_ = u.mm[0];                      \
    mm1_ = u.mm[1];                      \
  }

#define COPY_MM_TO_XMM(mm0_, mm1_, xmm_) \
  {                                      \
    xmm_mm_union u;                      \
    u.mm[0] = mm0_;                      \
    u.mm[1] = mm1_;                      \
    xmm_ = u.xmm;                        \
  }

#endif  // USE_SSE2

/* natural logarithm computed for 4 simultaneous float
   return NaN for x <= 0
   */
static inline v4sf log_ps(v4sf x) {
#ifdef USE_SSE2
  v4si emm0;
#else
  v2si mm0, mm1;
#endif
  v4sf one = *(v4sf*)_ps_1;

  v4sf invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

  x = _mm_max_ps(x, *(v4sf*)_ps_min_norm_pos); /* cut off denormalized stuff */

#ifndef USE_SSE2
  /* part 1: x = frexpf(x, &e); */
  COPY_XMM_TO_MM(x, mm0, mm1);
  mm0 = _mm_srli_pi32(mm0, 23);
  mm1 = _mm_srli_pi32(mm1, 23);
#else
  emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
#endif
  /* keep only the fractional part */
  x = _mm_and_ps(x, *(v4sf*)_ps_inv_mant_mask);
  x = _mm_or_ps(x, *(v4sf*)_ps_0p5);

#ifndef USE_SSE2
  /* now e=mm0:mm1 contain the really base-2 exponent */
  mm0 = _mm_sub_pi32(mm0, *(v2si*)_pi32_0x7f);
  mm1 = _mm_sub_pi32(mm1, *(v2si*)_pi32_0x7f);
  v4sf e = _mm_cvtpi32x2_ps(mm0, mm1);
  _mm_empty(); /* bye bye mmx */
#else
  emm0 = _mm_sub_epi32(emm0, *(v4si*)_pi32_0x7f);
  v4sf e = _mm_cvtepi32_ps(emm0);
#endif

  e = _mm_add_ps(e, one);

  /* part2:
     if( x < SQRTHF ) {
     e -= 1;
     x = x + x - 1.0;
     } else { x = x - 1.0; }
     */
  v4sf mask = _mm_cmplt_ps(x, *(v4sf*)_ps_cephes_SQRTHF);
  v4sf tmp = _mm_and_ps(x, mask);
  x = _mm_sub_ps(x, one);
  e = _mm_sub_ps(e, _mm_and_ps(one, mask));
  x = _mm_add_ps(x, tmp);

  v4sf z = _mm_mul_ps(x, x);

  v4sf y = *(v4sf*)_ps_cephes_log_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p5);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p6);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p7);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p8);
  y = _mm_mul_ps(y, x);

  y = _mm_mul_ps(y, z);

  tmp = _mm_mul_ps(e, *(v4sf*)_ps_cephes_log_q1);
  y = _mm_add_ps(y, tmp);

  tmp = _mm_mul_ps(z, *(v4sf*)_ps_0p5);
  y = _mm_sub_ps(y, tmp);

  tmp = _mm_mul_ps(e, *(v4sf*)_ps_cephes_log_q2);
  x = _mm_add_ps(x, y);
  x = _mm_add_ps(x, tmp);
  x = _mm_or_ps(x, invalid_mask);  // negative arg will be NAN
  return x;
}

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS_CONST(cephes_exp_C1, 0.693359375);
_PS_CONST(cephes_exp_C2, -2.12194440e-4);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1);

static inline v4sf exp_ps(v4sf x) {
  v4sf tmp = _mm_setzero_ps(), fx;
#ifdef USE_SSE2
  v4si emm0;
#else
  v2si mm0, mm1;
#endif
  v4sf one = *(v4sf*)_ps_1;

  x = _mm_min_ps(x, *(v4sf*)_ps_exp_hi);
  x = _mm_max_ps(x, *(v4sf*)_ps_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm_mul_ps(x, *(v4sf*)_ps_cephes_LOG2EF);
  fx = _mm_add_ps(fx, *(v4sf*)_ps_0p5);

  /* how to perform a floorf with SSE: just below */
#ifndef USE_SSE2
  /* step 1 : cast to int */
  tmp = _mm_movehl_ps(tmp, fx);
  mm0 = _mm_cvttps_pi32(fx);
  mm1 = _mm_cvttps_pi32(tmp);
  /* step 2 : cast back to float */
  tmp = _mm_cvtpi32x2_ps(mm0, mm1);
#else
  emm0 = _mm_cvttps_epi32(fx);
  tmp = _mm_cvtepi32_ps(emm0);
#endif
  /* if greater, substract 1 */
  v4sf mask = _mm_cmpgt_ps(tmp, fx);
  mask = _mm_and_ps(mask, one);
  fx = _mm_sub_ps(tmp, mask);

  tmp = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C1);
  v4sf z = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C2);
  x = _mm_sub_ps(x, tmp);
  x = _mm_sub_ps(x, z);

  z = _mm_mul_ps(x, x);

  v4sf y = *(v4sf*)_ps_cephes_exp_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p5);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, x);
  y = _mm_add_ps(y, one);

  /* build 2^n */
#ifndef USE_SSE2
  z = _mm_movehl_ps(z, fx);
  mm0 = _mm_cvttps_pi32(fx);
  mm1 = _mm_cvttps_pi32(z);
  mm0 = _mm_add_pi32(mm0, *(v2si*)_pi32_0x7f);
  mm1 = _mm_add_pi32(mm1, *(v2si*)_pi32_0x7f);
  mm0 = _mm_slli_pi32(mm0, 23);
  mm1 = _mm_slli_pi32(mm1, 23);

  v4sf pow2n;
  COPY_MM_TO_XMM(mm0, mm1, pow2n);
  _mm_empty();
#else
  emm0 = _mm_cvttps_epi32(fx);
  emm0 = _mm_add_epi32(emm0, *(v4si*)_pi32_0x7f);
  emm0 = _mm_slli_epi32(emm0, 23);
  v4sf pow2n = _mm_castsi128_ps(emm0);
#endif
  y = _mm_mul_ps(y, pow2n);
  return y;
}
// NOLINTEND

namespace slimt {

template <>
struct VDatum<VExt::w4> {
 public:
  using Scalar = float;
  using Register = __m128;
  static constexpr size_t kWidth = 4;
  VDatum() = default;
  // NOLINTBEGIN
  // clang-tidy complains explicit constructure, but this is intended to
  // interchange comfortably between float, Register and VDatum.
  VDatum(const Register& value) : register_(value) {}

  // Register _mm_set1_ps(float) copies value into all slots
  VDatum(const float& value) : register_(_mm_set1_ps(value)) {}

  operator const Register&() const { return register_; }
  operator Register&() { return register_; }
  // NOLINTEND

  VDatum& operator=(const float& value) {
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

enum class VExt;

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

      static Datum max(const Datum& lhs, const Datum& rhs) { return _mm_max_ps(lhs, rhs); }
      static Datum sub(const Datum& lhs, const Datum& rhs) { return _mm_sub_ps(lhs, rhs); }
      static Datum add(const Datum& lhs, const Datum& rhs) { return _mm_add_ps(lhs, rhs); }
      static Datum mul(const Datum& lhs, const Datum& rhs) { return _mm_mul_ps(lhs, rhs); }
      static Datum div(const Datum& lhs, const Datum& rhs) { return _mm_div_ps(lhs, rhs); }
      //clang-format on

      static Datum sigmoid(const Datum& x) {
        Datum e = exp(x);
        return div(e, add(1.0F, e));
      }

      struct Reduce {
        static Scalar max(const Datum& x) {
          Scalar accumulator = x[0];
          for (size_t i = 1; i < Datum::kWidth; ++i) {
            accumulator = accumulator > x[i]? accumulator : x[i];
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
}
