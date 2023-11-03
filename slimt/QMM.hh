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
Tensor affine(const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name = "");

template <enum Provider>
Tensor affine_with_select(const Tensor& x, const Tensor& W, const Tensor& b,
                          float a_quant, float b_quant,
                          const std::vector<uint32_t>& indices,
                          const std::string& name = "");

template <enum Provider>
Tensor dot(const Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

template <enum Provider>
void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows);
template <enum Provider>
void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols);

}  // namespace detail

Tensor affine(const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name = "");

Tensor affine_with_select(const Tensor& x, const Tensor& W, const Tensor& b,
                          float a_quant, float b_quant,
                          const std::vector<uint32_t>& indices,
                          const std::string& name = "");

Tensor dot(const Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows);
void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols);

}  // namespace slimt::qmm
