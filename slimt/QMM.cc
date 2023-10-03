#include "slimt/QMM.hh"

#include <cassert>

#ifdef SLIMT_HAS_INTGEMM
#include "intgemm/callbacks/configs.h"
#include "intgemm/intgemm.h"
#endif

#ifdef SLIMT_HAS_RUY
#include "ruy.h"
#endif

#ifdef SLIMT_HAS_GEMMOLOGY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "gemmology/gemmology.h"

#pragma GCC diagnostic pop
#endif

#include "slimt/Tensor.hh"

namespace slimt::qmm {
namespace detail {

#ifdef SLIMT_HAS_INTGEMM
template <>
Tensor affine_with_select<Provider::Intgemm>(
    Tensor& x, Tensor& W, Tensor& b, float a_quant, float b_quant,
    const std::vector<uint32_t>& indices, const std::string& name) {
  // Naming is to simplify thinking with the intgemm API below.
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT
  Tensor& bias = b;

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = A_cols;
  // SLIMT_TRACE3(x.shape(), W.shape(), b.shape());

  // Check widths are same, making matrix multiplication viable.
  assert(A_cols == B_rows);

  // Prepare Activations (A).
  Tensor prepared_A(Type::i8, A.shape(), "quantized_acts");  // NOLINT
  intgemm::Int8Shift::PrepareA(                              //
      A.data<float>(), prepared_A.data<int8_t>(),            //
      a_quant,                                               //
      A_rows, width                                          //
  );

  // Prepare bias
  Tensor prepared_bias(Type::f32, bias.shape(), "prepared_bias");
  constexpr float kMax8bit = kInt8Maxf;
  float a_alpha = kMax8bit / a_quant;
  float b_alpha = kMax8bit / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / kMax8bit;
  auto prepare_bias_callback = intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
      bias_unquant_multiplier, bias.data<float>(),  //
      prepared_bias.data<float>()                   //
  );

  intgemm::Int8Shift::PrepareBias(  //
      B.data<int8_t>(),             //
      width, B_cols,                //
      prepare_bias_callback         //
  );

  // Select before multiply?
  // NOLINTNEXTLINE
  Tensor selected_B(Type::i8, Shape({width, indices.size()}), "selected_B");
  const uint32_t* indices_begin = indices.data();
  const uint32_t* indices_end = indices.data() + indices.size();

  intgemm::Int8::SelectColumnsB(B.data<int8_t>(), selected_B.data<int8_t>(),
                                B_rows, indices_begin, indices_end);

  // Select bias accordingly.
  Tensor selected_bias(Type::f32, Shape({indices.size()}), "selected_bias");
  auto* selected_bias_ptr = selected_bias.data<float>();
  for (uint32_t index : indices) {
    *(selected_bias_ptr) = *(prepared_bias.data<float>() + index);
    ++selected_bias_ptr;
  }

  // Multiply y = A * B + bias (affine)
  // Set y's shape replacing last dimension with the feature-dim B is projecting
  // onto (B_cols).
  Shape out_shape = x.shape();
  out_shape.set_dim(-1, indices.size());

  Tensor y(Type::f32, out_shape, (name.empty() ? x.name() : name));
  size_t selected_B_cols = selected_B.dim(-1);  // NOLINT

  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  auto multiply_callback = intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
      unquant_multiplier, selected_bias.data<float>(), y.data<float>());
  intgemm::Int8Shift::Multiply(                              //
      prepared_A.data<int8_t>(), selected_B.data<int8_t>(),  //
      A_rows, width, selected_B_cols,                        //
      multiply_callback                                      //
  );

  return y;
}

template <>
Tensor affine<Provider::Intgemm>(Tensor& x, Tensor& W, Tensor& b, float a_quant,
                                 float b_quant, const std::string& name) {
  // Naming is to simplify thinking with the intgemm API below.
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT
  Tensor& bias = b;

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = A_cols;
  // SLIMT_TRACE3(x.shape(), W.shape(), b.shape());

  // Check widths are same, making matrix multiplication viable.
  (void)B_rows;
  assert(A_cols == B_rows);

  // Prepare Activations (A).
  Tensor prepared_A(Type::i8, A.shape(), "quantized_acts");  // NOLINT
  intgemm::Int8Shift::PrepareA(                              //
      A.data<float>(), prepared_A.data<int8_t>(),            //
      a_quant,                                               //
      A_rows, width                                          //
  );

  // Prepare bias
  Tensor prepared_bias(Type::f32, bias.shape(), "prepared_bias");
  float a_alpha = kInt8Maxf / a_quant;
  float b_alpha = kInt8Maxf / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / kInt8Maxf;
  auto prepare_bias_callback = intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
      bias_unquant_multiplier, bias.data<float>(),  //
      prepared_bias.data<float>()                   //
  );

  intgemm::Int8Shift::PrepareBias(  //
      B.data<int8_t>(),             //
      width, B_cols,                //
      prepare_bias_callback         //
  );

  // Multiply y = A * B + bias (affine)
  // Set y's shape replacing last dimension with the feature-dim B is projecting
  // onto (B_cols).
  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);

  Tensor y(Type::f32, out_shape, (name.empty() ? x.name() : name));

  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  auto multiply_callback = intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
      unquant_multiplier, prepared_bias.data<float>(), y.data<float>());
  intgemm::Int8Shift::Multiply(                     //
      prepared_A.data<int8_t>(), B.data<int8_t>(),  //
      A_rows, width, B_cols,                        //
      multiply_callback                             //
  );

  return y;
}

template <>
Tensor dot<Provider::Intgemm>(Tensor& x, Tensor& W, float a_quant,
                              float b_quant, const std::string& name) {
  // Naming is to simplify thinking with the intgemm API below.
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = A_cols;
  // SLIMT_TRACE3(x.shape(), W.shape(), b.shape());

  // Check widths are same, making matrix multiplication viable.
  (void)B_rows;
  assert(A_cols == B_rows);

  // Prepare Activations (A).
  Tensor prepared_A(Type::i8, A.shape(), "quantized_acts");  // NOLINT
  intgemm::Int8Shift::PrepareA(                              //
      A.data<float>(), prepared_A.data<int8_t>(),            //
      a_quant,                                               //
      A_rows, width                                          //
  );

  // Prepare bias

  // Fake bias, all elements are zero.
  Tensor bias(x.type(), Shape({1, B_cols}), "zero_bias");
  bias.fill_in_place(0.0F);

  Tensor prepared_bias(Type::f32, bias.shape(), "prepared_bias");
  float a_alpha = kInt8Maxf / a_quant;
  float b_alpha = kInt8Maxf / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / kInt8Maxf;
  auto prepare_bias_callback = intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
      bias_unquant_multiplier, bias.data<float>(),  //
      prepared_bias.data<float>()                   //
  );

  intgemm::Int8Shift::PrepareBias(  //
      B.data<int8_t>(),             //
      width, B_cols,                //
      prepare_bias_callback         //
  );

  //
  // Multiply y = A * B  (dot)
  // Set y's shape replacing last dimension with the feature-dim B is projecting
  // onto (B_cols).
  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);

  Tensor y(Type::f32, out_shape, (name.empty() ? x.name() : name));

  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  auto multiply_callback = intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
      unquant_multiplier, prepared_bias.data<float>(), y.data<float>());
  intgemm::Int8Shift::Multiply(                     //
      prepared_A.data<int8_t>(), B.data<int8_t>(),  //
      A_rows, width, B_cols,                        //
      multiply_callback                             //
  );

  return y;
}

template <>
void prepare_weight_transposed<Provider::Intgemm>(const float* weights,
                                                  int8_t* prepared,
                                                  float quantization_multiplier,
                                                  size_t cols, size_t rows) {
  intgemm::Int8::PrepareBTransposed(weights, prepared, quantization_multiplier,
                                    cols, rows);
}

template <>
void prepare_weight_quantized_transposed<Provider::Intgemm>(const int8_t* input,
                                                            int8_t* output,
                                                            size_t rows,
                                                            size_t cols) {
  intgemm::Int8::PrepareBQuantizedTransposed(input, output, rows, cols);
}

#endif

#ifdef SLIMT_HAS_RUY
namespace detail {

using Index = uint64_t;

void quantize(const float* input, float scale, Index rows, Index width,
              int8_t* output) {
  const Index size = rows * width;
  for (size_t i = 0; i < size; i++) {
    // Round to nearest after multiplying with scale.
    float value = roundf(scale * input[i]);

    // Since float can store bigger values, we threshold anything that's gone
    // higher and can't fit in int8.
    value = std::max<float>(-kInt8Maxf, value);
    value = std::min<float>(kInt8Maxf, value);

    // Finally a static cast.
    output[i] = static_cast<int8_t>(value);
  };
}

template <class Scalar>
void transpose(const Scalar* input, Index rows, Index cols, Scalar* output) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      output[j * rows + i] = input[i * cols + j];
    }
  }
}

void unquantize(const int32_t* input, float unquant_multiplier, Index rows_A,
                Index cols_B, float* output) {
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_B; j++) {
      Index idx = i * cols_B + j;
      output[idx] = (input[idx] * unquant_multiplier);
    }
  }
}

void unquantizeAddBias(const int32_t* input, const float* input_bias_prepared,
                       float unquant_multiplier, Index rows_A, Index cols_B,
                       float* output) {
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_B; j++) {
      Index idx = i * cols_B + j;
      output[idx] = (input[idx] * unquant_multiplier) + input_bias_prepared[j];
    }
  }
}

}  // namespace detail

// Ruy.
template <>
Tensor affine<Provider::Ruy>(Tensor& x, Tensor& W, Tensor& b, float a_quant,
                             float b_quant, const std::string& name) {
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT
  Tensor& bias = b;

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = B_rows;

  (void)name;
  // Prepare A: Quantize from f32 -> i8
  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT

  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  ruy::Context context;
  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(A_rows, width, ruy::Order::kRowMajor,
                        lhs.mutable_layout());
  lhs.set_data(prepared_A.data<int8_t>());

  // PrepareB: ?
  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, B_cols, ruy::Order::kColMajor,
                        rhs.mutable_layout());
  rhs.set_data(W.data<int8_t>());

  // PrepareBias: ?
  // Actualyl there is no need.
  Tensor& prepared_bias = bias;

  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(A_rows, B_cols, ruy::Order::kRowMajor,
                        dst.mutable_layout());

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, out_shape, name + "_out");  // NOLINT
  dst.set_data(AB.data<int32_t>());

  // Multiply C = AB;
  // When Dst is int32, mul_params is unused.
  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  // Unquantizes, then adds bias in a single statement on the output.
  Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantizeAddBias(AB.data<int32_t>(), prepared_bias.data<float>(),
                            unquant_multiplier, A_rows, B_cols,
                            y.data<float>());
  return y;
}

template <>
Tensor affine_with_select<Provider::Ruy>(Tensor& x, Tensor& W, Tensor& b,
                                         float a_quant, float b_quant,
                                         const std::vector<uint32_t>& indices,
                                         const std::string& name) {
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT
  Tensor& bias = b;

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = B_rows;

  (void)name;
  // Prepare A: Quantize from f32 -> i8
  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT

  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  ruy::Context context;
  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(A_rows, width, ruy::Order::kRowMajor,
                        lhs.mutable_layout());
  lhs.set_data(prepared_A.data<int8_t>());

  // PrepareB: Select
  Tensor selected_B(Type::i8, Shape({width, indices.size()}),  // NOLINT
                    "selected_B");

  // SelectColumnsB, but inlined?
  // B_prepared is expected to be col-major, for our implementation via ruy. If
  // col-major we can memcpy the respective column entries as they're
  // sequential. There are width = rows entries.
  auto B_data = B.data<int8_t>();            // NOLINT
  auto sB_data = selected_B.data<int8_t>();  // NOLINT
  for (size_t c = 0; c < indices.size(); ++c) {
    int8_t* sB_begin = &(sB_data[c * width]);         // NOLINT
    int8_t* B_begin = &(B_data[indices[c] * width]);  // NOLINT
    std::memcpy(sB_begin, B_begin, width);
  }

  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, indices.size(), ruy::Order::kColMajor,
                        rhs.mutable_layout());
  rhs.set_data(selected_B.data<int8_t>());

  // Once again, bias needn't be prepared. But needs to be selected.
  Tensor& prepared_bias = bias;
  Tensor selected_bias(Type::f32, Shape({indices.size()}), "selected_bias");
  auto* selected_bias_ptr = selected_bias.data<float>();
  for (uint32_t index : indices) {
    *(selected_bias_ptr) = *(prepared_bias.data<float>() + index);
    ++selected_bias_ptr;
  }

  // Multiply C = A select(B);
  // When Dst is int32, mul_params is unused.
  size_t selected_B_cols = selected_B.dim(-1);  // NOLINT
  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(A_rows, selected_B_cols, ruy::Order::kRowMajor,
                        dst.mutable_layout());

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, selected_B_cols);

  Tensor AB(Type::i32, out_shape, name + "_out");  // NOLINT
  dst.set_data(AB.data<int32_t>());

  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  // Unquantizes, then adds bias in a single statement on the output.
  Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantizeAddBias(AB.data<int32_t>(), prepared_bias.data<float>(),
                            unquant_multiplier, A_rows, selected_B_cols,
                            y.data<float>());
  return y;
}

template <>
Tensor dot<Provider::Ruy>(Tensor& x, Tensor& W, float a_quant, float b_quant,
                          const std::string& name) {
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = B_rows;

  (void)name;
  // Prepare A: Quantize from f32 -> i8
  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT

  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  ruy::Context context;
  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(A_rows, width, ruy::Order::kRowMajor,
                        lhs.mutable_layout());
  lhs.set_data(prepared_A.data<int8_t>());

  // PrepareB: ?
  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, B_cols, ruy::Order::kColMajor,
                        rhs.mutable_layout());
  rhs.set_data(W.data<int8_t>());

  // PrepareBias: ?
  // Actualyl there is no need.
  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(A_rows, B_cols, ruy::Order::kRowMajor,
                        dst.mutable_layout());

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, out_shape, name + "_out");  // NOLINT
  dst.set_data(AB.data<int32_t>());

  // Multiply C = AB;
  // When Dst is int32, mul_params is unused.
  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  // Unquantizes, then adds bias in a single statement on the output.
  Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantize(AB.data<int32_t>(), unquant_multiplier, A_rows, B_cols,
                     y.data<float>());
  return y;
}

template <>
void prepare_weight_transposed<Provider::Ruy>(const float* weights,
                                              int8_t* prepared,
                                              float quantization_multiplier,
                                              size_t cols, size_t rows) {
  detail::quantize(weights, quantization_multiplier, cols, rows, prepared);
}

template <>
void prepare_weight_quantized_transposed<Provider::Ruy>(const int8_t* input,
                                                        int8_t* output,
                                                        size_t rows,
                                                        size_t cols) {
  std::memcpy(output, input,
              /*count=*/sizeof(int8_t) * (rows * cols));
}
#endif

#ifdef SLIMT_HAS_GEMMOLOGY
template <>
Tensor affine_with_select<Provider::Gemmology>(
    Tensor& x, Tensor& W, Tensor& b, float a_quant, float b_quant,
    const std::vector<uint32_t>& indices, const std::string& name) {
  // Naming is to simplify thinking with the gemmology API below.
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT
  Tensor& bias = b;

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = A_cols;
  // SLIMT_TRACE3(x.shape(), W.shape(), b.shape());

  // Check widths are same, making matrix multiplication viable.
  assert(A_cols == B_rows);

  // Prepare Activations (A).
  Tensor prepared_A(Type::i8, A.shape(), "quantized_acts");  // NOLINT
  gemmology::Shift::PrepareA(                                //
      A.data<float>(), prepared_A.data<uint8_t>(),           //
      a_quant,                                               //
      A_rows, width                                          //
  );

  // Prepare bias
  Tensor prepared_bias(Type::f32, bias.shape(), "prepared_bias");
  constexpr float kMax8bit = kInt8Maxf;
  float a_alpha = kMax8bit / a_quant;
  float b_alpha = kMax8bit / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / kMax8bit;
  auto prepare_bias_callback =
      gemmology::callbacks::UnquantizeAndAddBiasAndWrite(
          bias_unquant_multiplier, bias.data<float>(),  //
          prepared_bias.data<float>()                   //
      );

  gemmology::Shift::PrepareBias(  //
      B.data<int8_t>(),           //
      width, B_cols,              //
      prepare_bias_callback       //
  );

  // Select before multiply?
  // NOLINTNEXTLINE
  Tensor selected_B(Type::i8, Shape({width, indices.size()}), "selected_B");
  const uint32_t* indices_begin = indices.data();
  const uint32_t* indices_end = indices.data() + indices.size();

  gemmology::SelectColumnsB(B.data<int8_t>(), selected_B.data<int8_t>(), B_rows,
                            indices_begin, indices_end);

  // Select bias accordingly.
  Tensor selected_bias(Type::f32, Shape({indices.size()}), "selected_bias");
  auto* selected_bias_ptr = selected_bias.data<float>();
  for (uint32_t index : indices) {
    *(selected_bias_ptr) = *(prepared_bias.data<float>() + index);
    ++selected_bias_ptr;
  }

  // Multiply y = A * B + bias (affine)
  // Set y's shape replacing last dimension with the feature-dim B is projecting
  // onto (B_cols).
  Shape out_shape = x.shape();
  out_shape.set_dim(-1, indices.size());

  Tensor y(Type::f32, out_shape, (name.empty() ? x.name() : name));
  size_t selected_B_cols = selected_B.dim(-1);  // NOLINT

  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  auto multiply_callback = gemmology::callbacks::UnquantizeAndAddBiasAndWrite(
      unquant_multiplier, selected_bias.data<float>(), y.data<float>());
  gemmology::Shift::Multiply(                                 //
      prepared_A.data<uint8_t>(), selected_B.data<int8_t>(),  //
      A_rows, width, selected_B_cols,                         //
      multiply_callback                                       //
  );

  return y;
}

template <>
Tensor affine<Provider::Gemmology>(Tensor& x, Tensor& W, Tensor& b,
                                   float a_quant, float b_quant,
                                   const std::string& name) {
  // Naming is to simplify thinking with the gemmology API below.
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT
  Tensor& bias = b;

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = A_cols;
  // SLIMT_TRACE3(x.shape(), W.shape(), b.shape());

  // Check widths are same, making matrix multiplication viable.
  (void)B_rows;
  assert(A_cols == B_rows);

  // Prepare Activations (A).
  Tensor prepared_A(Type::i8, A.shape(), "quantized_acts");  // NOLINT
  gemmology::Shift::PrepareA(                                //
      A.data<float>(), prepared_A.data<uint8_t>(),           //
      a_quant,                                               //
      A_rows, width                                          //
  );

  // Prepare bias
  Tensor prepared_bias(Type::f32, bias.shape(), "prepared_bias");
  float a_alpha = kInt8Maxf / a_quant;
  float b_alpha = kInt8Maxf / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / kInt8Maxf;
  auto prepare_bias_callback =
      gemmology::callbacks::UnquantizeAndAddBiasAndWrite(
          bias_unquant_multiplier, bias.data<float>(),  //
          prepared_bias.data<float>()                   //
      );

  gemmology::Shift::PrepareBias(  //
      B.data<int8_t>(),           //
      width, B_cols,              //
      prepare_bias_callback       //
  );

  // Multiply y = A * B + bias (affine)
  // Set y's shape replacing last dimension with the feature-dim B is projecting
  // onto (B_cols).
  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);

  Tensor y(Type::f32, out_shape, (name.empty() ? x.name() : name));

  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  auto multiply_callback = gemmology::callbacks::UnquantizeAndAddBiasAndWrite(
      unquant_multiplier, prepared_bias.data<float>(), y.data<float>());
  gemmology::Shift::Multiply(                        //
      prepared_A.data<uint8_t>(), B.data<int8_t>(),  //
      A_rows, width, B_cols,                         //
      multiply_callback                              //
  );

  return y;
}

template <>
Tensor dot<Provider::Gemmology>(Tensor& x, Tensor& W, float a_quant,
                                float b_quant, const std::string& name) {
  // Naming is to simplify thinking with the gemmology API below.
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = A_cols;
  // SLIMT_TRACE3(x.shape(), W.shape(), b.shape());

  // Check widths are same, making matrix multiplication viable.
  (void)B_rows;
  assert(A_cols == B_rows);

  // Prepare Activations (A).
  Tensor prepared_A(Type::i8, A.shape(), "quantized_acts");  // NOLINT
  gemmology::Shift::PrepareA(                                //
      A.data<float>(), prepared_A.data<uint8_t>(),           //
      a_quant,                                               //
      A_rows, width                                          //
  );

  // Prepare bias

  // Fake bias, all elements are zero.
  Tensor bias(x.type(), Shape({1, B_cols}), "zero_bias");
  bias.fill_in_place(0.0F);

  Tensor prepared_bias(Type::f32, bias.shape(), "prepared_bias");
  float a_alpha = kInt8Maxf / a_quant;
  float b_alpha = kInt8Maxf / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / kInt8Maxf;
  auto prepare_bias_callback =
      gemmology::callbacks::UnquantizeAndAddBiasAndWrite(
          bias_unquant_multiplier, bias.data<float>(),  //
          prepared_bias.data<float>()                   //
      );

  gemmology::Shift::PrepareBias(  //
      B.data<int8_t>(),           //
      width, B_cols,              //
      prepare_bias_callback       //
  );

  //
  // Multiply y = A * B  (dot)
  // Set y's shape replacing last dimension with the feature-dim B is projecting
  // onto (B_cols).
  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);

  Tensor y(Type::f32, out_shape, (name.empty() ? x.name() : name));

  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  auto multiply_callback = gemmology::callbacks::UnquantizeAndAddBiasAndWrite(
      unquant_multiplier, prepared_bias.data<float>(), y.data<float>());
  gemmology::Shift::Multiply(                        //
      prepared_A.data<uint8_t>(), B.data<int8_t>(),  //
      A_rows, width, B_cols,                         //
      multiply_callback                              //
  );

  return y;
}

template <>
void prepare_weight_transposed<Provider::Gemmology>(
    const float* weights, int8_t* prepared, float quantization_multiplier,
    size_t cols, size_t rows) {
  gemmology::PrepareBTransposed(weights, prepared, quantization_multiplier,
                                cols, rows);
}

template <>
void prepare_weight_quantized_transposed<Provider::Gemmology>(
    const int8_t* input, int8_t* output, size_t rows, size_t cols) {
  gemmology::PrepareBQuantizedTransposed(input, output, rows, cols);
}

#endif

}  // namespace detail
   //
Tensor affine(Tensor& x, Tensor& W, Tensor& b, float a_quant, float b_quant,
              const std::string& name) {
  using detail::affine;
  using detail::kAutoProvider;
  return affine<kAutoProvider>(x, W, b, a_quant, b_quant, name);
}

Tensor affine_with_select(Tensor& x, Tensor& W, Tensor& b, float a_quant,
                          float b_quant, const std::vector<uint32_t>& indices,
                          const std::string& name) {
  using detail::affine_with_select;
  using detail::kAutoProvider;
  return affine_with_select<kAutoProvider>(x, W, b, a_quant, b_quant, indices,
                                           name);
}

Tensor dot(Tensor& x, Tensor& W, float a_quant, float b_quant,
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
