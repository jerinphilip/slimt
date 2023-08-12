
#include <immintrin.h>
#include <xmmintrin.h>

using v8sf = __m256;
using v8si = __m256i;
using v4si = __m128i;

// NOLINTBEGIN

typedef union imm_xmm_union {
  v8si imm;
  v4si xmm[2];
} imm_xmm_union;

#ifdef _MSC_VER
#define ALIGN32_BEG __declspec(align(32))
#define ALIGN32_END
#else /* gcc or icc */
#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))
#endif

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_)  \
  {                                          \
    ALIGN32_BEG imm_xmm_union u ALIGN32_END; \
    u.imm = imm_;                            \
    xmm0_ = u.xmm[0];                        \
    xmm1_ = u.xmm[1];                        \
  }

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_)  \
  {                                          \
    ALIGN32_BEG imm_xmm_union u ALIGN32_END; \
    u.xmm[0] = xmm0_;                        \
    u.xmm[1] = xmm1_;                        \
    imm_ = u.imm;                            \
  }

#define AVX2_BITOP_USING_SSE2(fn)                        \
  static inline v8si avx2_mm256_##fn(v8si x, int a) {    \
    /* use SSE2 instruction to perform the bitop AVX2 */ \
    v4si x1, x2;                                         \
    v8si ret;                                            \
    COPY_IMM_TO_XMM(x, x1, x2);                          \
    x1 = _mm_##fn(x1, a);                                \
    x2 = _mm_##fn(x2, a);                                \
    COPY_XMM_TO_IMM(x1, x2, ret);                        \
    return (ret);                                        \
  }

#define AVX2_INTOP_USING_SSE2(fn)                                     \
  static inline v8si avx2_mm256_##fn(v8si x, v8si y) {                \
    /* use SSE2 instructions to perform the AVX2 integer operation */ \
    v4si x1, x2;                                                      \
    v4si y1, y2;                                                      \
    v8si ret;                                                         \
    COPY_IMM_TO_XMM(x, x1, x2);                                       \
    COPY_IMM_TO_XMM(y, y1, y2);                                       \
    x1 = _mm_##fn(x1, y1);                                            \
    x2 = _mm_##fn(x2, y2);                                            \
    COPY_XMM_TO_IMM(x1, x2, ret);                                     \
    return (ret);                                                     \
  }

AVX2_BITOP_USING_SSE2(slli_epi32);  // NOLINT
AVX2_INTOP_USING_SSE2(add_epi32);   // NOLINT

#define _PI32_CONST256(Name, Val)                                  \
  static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = { \
      Val, Val, Val, Val, Val, Val, Val, Val}

#define _PS256_CONST(Name, Val)                                   \
  static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = { \
      (float)Val, (float)Val, (float)Val, (float)Val,             \
      (float)Val, (float)Val, (float)Val, (float)Val}

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);
_PS256_CONST(1, 1);
_PS256_CONST(0p5, 0.5);
_PI32_CONST256(0x7f, 0x7f);

// NOLINTNEXTLINE
inline v8sf exp256_ps(v8sf x) {
  v8sf tmp = _mm256_setzero_ps(), fx;
  v8si imm0;
  v8sf one = *(v8sf*)_ps256_1;

  x = _mm256_min_ps(x, *(v8sf*)_ps256_exp_hi);
  x = _mm256_max_ps(x, *(v8sf*)_ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_LOG2EF);
  fx = _mm256_add_ps(fx, *(v8sf*)_ps256_0p5);

  /* how to perform a floorf with SSE: just below */
  // imm0 = _mm256_cvttps_epi32(fx);
  // tmp  = _mm256_cvtepi32_ps(imm0);

  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  // v8sf mask = _mm256_cmpgt_ps(tmp, fx);
  v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  tmp = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C1);
  v8sf z = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x, x);

  v8sf y = *(v8sf*)_ps256_cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = avx2_mm256_add_epi32(imm0, *(v8si*)_pi32_256_0x7f);
  imm0 = avx2_mm256_slli_epi32(imm0, 23);
  v8sf pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
// NOLINTEND

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

}  // namespace slimt
