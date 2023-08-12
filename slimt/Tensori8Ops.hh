#include <cstddef>

#include "slimt/Tensor.hh"

namespace slimt::i8 {

enum class i8xi8 {
  kNone,
  kIntgemm,  //
  kRuy       //
};

template <enum i8xi8>
Tensor affine(Tensor& x, Tensor& W, Tensor& b, float a_quant, float b_quant,
              const std::string& name = "");

template <enum i8xi8>
Tensor affine_with_select(Tensor& x, Tensor& W, Tensor& b, float a_quant,
                          float b_quant, const std::vector<uint32_t>& indices,
                          const std::string& name = "");

template <enum i8xi8>
Tensor dot(Tensor& x, Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

template <enum i8xi8>
void PrepareBTransposed(const float* weights, int8_t* prepared,
                        float quantization_multiplier, size_t cols,
                        size_t rows);
template <enum i8xi8>
void PrepareBQuantizedTransposed(const int8_t* input, int8_t* output,
                                 size_t rows, size_t cols);

constexpr i8xi8 kBi8xi8 = i8xi8::kIntgemm;

}  // namespace slimt::i8
