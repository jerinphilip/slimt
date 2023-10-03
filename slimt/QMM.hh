#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt::qmm {

constexpr float kInt8Maxf = 127.0F;

namespace detail {

enum class Provider {
  None,      //
  Intgemm,   //
  Ruy,       //
  Gemmology  //
};

template <enum Provider>
Tensor affine(Tensor& x, Tensor& W, Tensor& b, float a_quant, float b_quant,
              const std::string& name = "");

template <enum Provider>
Tensor affine_with_select(Tensor& x, Tensor& W, Tensor& b, float a_quant,
                          float b_quant, const std::vector<uint32_t>& indices,
                          const std::string& name = "");

template <enum Provider>
Tensor dot(Tensor& x, Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

template <enum Provider>
void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows);
template <enum Provider>
void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols);

#ifdef SLIMT_HAS_INTGEMM
constexpr Provider kAutoProvider = Provider::Intgemm;
#endif

#ifdef SLIMT_HAS_RUY
constexpr Provider kAutoProvider = Provider::Ruy;
#endif

#ifdef SLIMT_HAS_GEMMOLOGY
constexpr Provider kAutoProvider = Provider::Gemmology;
#endif
}  // namespace detail

Tensor affine(Tensor& x, Tensor& W, Tensor& b, float a_quant, float b_quant,
              const std::string& name = "");

Tensor affine_with_select(Tensor& x, Tensor& W, Tensor& b, float a_quant,
                          float b_quant, const std::vector<uint32_t>& indices,
                          const std::string& name = "");

Tensor dot(Tensor& x, Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows);
void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols);

}  // namespace slimt::qmm
