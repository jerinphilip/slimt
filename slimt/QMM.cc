#include "slimt/QMM.hh"

#include <cassert>
#include <cmath>

#ifdef SLIMT_HAS_INTGEMM
#include "intgemm/callbacks/configs.h"
#include "intgemm/intgemm.h"

namespace slimt::qmm::detail {
constexpr Provider kAutoProvider = Provider::Intgemm;
}
#include "slimt/qmm/Intgemm.inl.hh"
#endif

#ifdef SLIMT_HAS_RUY
#include "ruy/ruy.h"
namespace slimt::qmm::detail {
constexpr Provider kAutoProvider = Provider::Ruy;
}
#include "slimt/qmm/Ruy.inl.hh"
#endif

#ifdef SLIMT_HAS_GEMMOLOGY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "gemmology/gemmology.h"
namespace slimt::qmm::detail {
constexpr Provider kAutoProvider = Provider::Gemmology;
}
#include "slimt/qmm/Gemmology.inl.hh"
#pragma GCC diagnostic pop
#endif

#ifdef SLIMT_HAS_GEMMOLOGY
#endif

namespace slimt::qmm {
Tensor affine(Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name) {
  using detail::affine;
  using detail::kAutoProvider;
  return affine<kAutoProvider>(x, W, b, a_quant, b_quant, name);
}

Tensor affine_with_select(Tensor& x, const Tensor& W, const Tensor& b,
                          float a_quant, float b_quant,
                          const std::vector<uint32_t>& indices,
                          const std::string& name) {
  using detail::affine_with_select;
  using detail::kAutoProvider;
  return affine_with_select<kAutoProvider>(x, W, b, a_quant, b_quant, indices,
                                           name);
}

Tensor dot(Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name) {
  using detail::dot;
  using detail::kAutoProvider;
  return dot<kAutoProvider>(x, W, a_quant, b_quant, name);
}

void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows) {
  using detail::kAutoProvider;
  using detail::prepare_weight_transposed;
  prepare_weight_transposed<kAutoProvider>(weights, prepared,
                                           quantization_multiplier, cols, rows);
}

void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols) {
  using detail::kAutoProvider;
  using detail::prepare_weight_quantized_transposed;
  prepare_weight_quantized_transposed<kAutoProvider>(input, output, rows, cols);
}

}  // namespace slimt::qmm
