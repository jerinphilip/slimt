#include <cstddef>

#include "slimt/Tensor.hh"

#ifdef __SSE__

#include "3rd-party/intgemm/intgemm/intgemm.h"
namespace slimt {

Tensor intgemm_affine(Tensor& x, Tensor& W, Tensor& b, float a_quant,
                      float b_quant, const std::string& name = "");
Tensor intgemm_affine_with_select(Tensor& x, Tensor& W, Tensor& b,
                                  float a_quant, float b_quant,
                                  const std::vector<uint32_t>& indices,
                                  const std::string& name = "");
Tensor intgemm_dot(Tensor& x, Tensor& W, float a_quant, float b_quant,
                   const std::string& name = "");

inline void PrepareBTransposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows) {
  intgemm::Int8::PrepareBTransposed(weights, prepared, quantization_multiplier,
                                    cols, rows);
}

inline void PrepareBQuantizedTransposed(const int8_t* input, int8_t* output,
                                        size_t rows, size_t cols) {
  intgemm::Int8::PrepareBQuantizedTransposed(input, output, rows, cols);
}

}  // namespace slimt
#else
#error "Shouldn't be here."
#endif
