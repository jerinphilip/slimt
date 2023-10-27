#include "slimt/QMM.hh"

#ifdef SLIMT_HAS_INTGEMM
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt::qmm::detail {
constexpr Provider kAutoProvider = Provider::Intgemm;
}
// NOLINTNEXTLINE: The C++ file inclusion is intended.
#include "slimt/qmm/Intgemm.inl.cc"
#endif

#ifdef SLIMT_HAS_RUY

namespace slimt::qmm::detail {
constexpr Provider kAutoProvider = Provider::Ruy;
}
// NOLINTNEXTLINE: The C++ file inclusion is intended.
#include "slimt/qmm/Ruy.inl.cc"
#endif

#ifdef SLIMT_HAS_GEMMOLOGY

namespace slimt::qmm::detail {
constexpr Provider kAutoProvider = Provider::Gemmology;
}
// NOLINTNEXTLINE: The C++ file inclusion is intended.
#include "slimt/qmm/Gemmology.inl.cc"
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
