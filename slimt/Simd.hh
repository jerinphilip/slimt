#include <cstddef>
#include <cstdint>

namespace slimt {

// NOLINTBEGIN
enum class VExt {
  w0,  //
  w1,  //
  w4,  //
  w8,  //
};

enum class VectorOp {

};

// NOLINTEND
template <enum VExt>
struct VDatum;

template <enum VExt>
struct Ops;

}  // namespace slimt

// Naive implementation

#if defined(USE_AVX2) && defined(SLIMT_SIMD)
#include "slimt/simd/avx2.h"
#define VEXT_W8_AVAILABLE

namespace slimt {
using F32x8 = VDatum<VExt::w8>;
}

#endif

#if defined(USE_SSE2) && defined(SLIMT_SIMD)
#include "slimt/simd/sse.h"
#define VEXT_W4_AVAILABLE

namespace slimt {
using F32x4 = VDatum<VExt::w4>;
}
#endif

#if defined(USE_NEON) && defined(SLIMT_SIMD)
#include "slimt/simd/neon.h"
#define VEXT_W4_AVAILABLE
namespace slimt {
using F32x4 = VDatum<VExt::w4>;
}
#endif

namespace slimt::vext {

template <VExt Width>
void add(const float* a, const float* b, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::add(va[i], vb[i]);
  }
}

template <VExt Width>
void sub(const float* a, const float* b, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::sub(va[i], vb[i]);
  }
}

template <VExt Width>
void mul(const float* a, const float* b, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::mul(va[i], vb[i]);
  }
}

template <VExt Width>
void relu(const float* a, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::relu(va[i]);
  }
}

template <VExt Width>
void sigmoid(const float* a, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::sigmoid(va[i]);
  }
}

template <VExt Width>
void softmax(const float* _logits, size_t batch_size, size_t num_classes,
             float* _out) {
  using Element = VDatum<Width>;
  const auto* logits = reinterpret_cast<const Element*>(_logits);
  auto* out = reinterpret_cast<Element*>(_out);
  int rows = batch_size;
  int cols = num_classes / Element::kWidth;  // operating with fewer columns.

  for (int j = 0; j < rows; ++j) {
    // p is probability, which is computed from logits.
    Element* p = out + j * cols;
    const Element* logit = logits + j * cols;

    // Compute maximum.
    Element max_value = logit[0];
    for (int i = 1; i < cols; ++i) {
      max_value = Ops<Width>::max(max_value, logit[i]);
    }

    // if ElementType is a complex type, e.g. float32x8, find the max of
    // these 8 values
    typename Ops<Width>::Scalar max_value_scalar =
        Ops<Width>::Reduce::max(max_value);
    Element max_value_projected(max_value_scalar);

    // Find numerically stable sumexp, after shifting values by maximum.
    Element vsum(0.0F);
    for (int i = 0; i < cols; ++i) {
      Element shifted = Ops<Width>::sub(logit[i], max_value_projected);
      Element exp_x = Ops<Width>::exp(shifted);
      vsum = Ops<Width>::add(vsum, exp_x);
      p[i] = exp_x;
    }

    // if Register is a complex type, e.g. float32x8, sum these 8 values
    typename Ops<Width>::Scalar sums = Ops<Width>::Reduce::sum(vsum);
    Element sums_value_projected(sums);

    for (int i = 0; i < cols; ++i) {
      p[i] = Ops<Width>::div(p[i], sums_value_projected);
    }
  }
}

}  // namespace slimt::vext
