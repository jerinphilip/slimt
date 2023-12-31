#include <gemmology/gemmology_fwd.h>

#include <xsimd/xsimd.hpp>

#if defined(USE_AVX512)
#define GEMMOLOGY_SUPPORTED_ARCHS \
  xsimd::arch_list<xsimd::avx512bw, xsimd::avx2, xsimd::ssse3, xsimd::sse2>
#elif defined(USE_AVX2)
#define GEMMOLOGY_SUPPORTED_ARCHS \
  xsimd::arch_list<xsimd::avx2, xsimd::ssse3, xsimd::sse2>
#elif defined(USE_SSSE3)
#define GEMMOLOGY_SUPPORTED_ARCHS xsimd::arch_list<xsimd::ssse3, xsimd::sse2>
#elif defined(USE_SSE2)
#define GEMMOLOGY_SUPPORTED_ARCHS xsimd::arch_list<xsimd::sse2>
#elif defined(USE_NEON) and defined(XSIMD_WITH_NEON64)
#define GEMMOLOGY_SUPPORTED_ARCHS xsimd::arch_list<xsimd::neon64>
#else
#error no supported architecture
#endif

// Dispatch *at runtime* based on run-time hardware and compile-time
// architectures.
//
// FIXME: Ideally we would not run the dispatch code at each function call.
#define GEMMOLOGY_DISPATCH(FUNCTION)                                       \
  xsimd::dispatch<GEMMOLOGY_SUPPORTED_ARCHS>([](auto arch, auto... args) { \
    return gemmology::Engine<decltype(arch)>::FUNCTION(args...);           \
  })

namespace slimt::qmm::detail {

template <>
Tensor affine_with_select<Provider::Gemmology>(
    const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
    float b_quant, const std::vector<uint32_t>& indices,
    const std::string& name) {
  // Naming is to simplify thinking with the gemmology API below.
  const Tensor& A = x;  // NOLINT
  const Tensor& B = W;  // NOLINT
  const Tensor& bias = b;

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
  auto PrepareA = GEMMOLOGY_DISPATCH(Shift::PrepareA);       // NOLINT
  PrepareA(                                                  //
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

  auto PrepareBias = GEMMOLOGY_DISPATCH(Shift::PrepareBias);  // NOLINT
  PrepareBias(                                                //
      B.data<int8_t>(),                                       //
      width, B_cols,                                          //
      prepare_bias_callback                                   //
  );

  // Select before multiply?
  // NOLINTNEXTLINE
  Tensor selected_B(Type::i8, Shape({width, indices.size()}), "selected_B");
  const uint32_t* indices_begin = indices.data();
  const uint32_t* indices_end = indices.data() + indices.size();

  auto SelectColumnsB = GEMMOLOGY_DISPATCH(SelectColumnsB);  //  NOLINT
  SelectColumnsB(B.data<int8_t>(), selected_B.data<int8_t>(), B_rows,
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
  auto Multiply = GEMMOLOGY_DISPATCH(Shift::Multiply);        // NOLINT
  Multiply(                                                   //
      prepared_A.data<uint8_t>(), selected_B.data<int8_t>(),  //
      A_rows, width, selected_B_cols,                         //
      multiply_callback                                       //
  );

  return y;
}

template <>
Tensor affine<Provider::Gemmology>(const Tensor& x, const Tensor& W,
                                   const Tensor& b, float a_quant,
                                   float b_quant, const std::string& name) {
  // Naming is to simplify thinking with the gemmology API below.
  const Tensor& A = x;  // NOLINT
  const Tensor& B = W;  // NOLINT
  const Tensor& bias = b;

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
  auto PrepareA = GEMMOLOGY_DISPATCH(Shift::PrepareA);       // NOLINT
  PrepareA(                                                  //
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

  auto PrepareBias = GEMMOLOGY_DISPATCH(Shift::PrepareBias);  // NOLINT
  PrepareBias(                                                //
      B.data<int8_t>(),                                       //
      width, B_cols,                                          //
      prepare_bias_callback                                   //
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
  auto Multiply = GEMMOLOGY_DISPATCH(Shift::Multiply);  // NOLINT
  Multiply(                                             //
      prepared_A.data<uint8_t>(), B.data<int8_t>(),     //
      A_rows, width, B_cols,                            //
      multiply_callback                                 //
  );

  return y;
}

template <>
Tensor dot<Provider::Gemmology>(const Tensor& x, const Tensor& W, float a_quant,
                                float b_quant, const std::string& name) {
  // Naming is to simplify thinking with the gemmology API below.
  const Tensor& A = x;  // NOLINT
  const Tensor& B = W;  // NOLINT

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
  auto PrepareA = GEMMOLOGY_DISPATCH(Shift::PrepareA);       // NOLINT
  PrepareA(                                                  //
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

  auto PrepareBias = GEMMOLOGY_DISPATCH(Shift::PrepareBias);  // NOLINT
  PrepareBias(                                                //
      B.data<int8_t>(),                                       //
      width, B_cols,                                          //
      prepare_bias_callback                                   //
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
  auto Multiply = GEMMOLOGY_DISPATCH(Shift::Multiply);  // NOLINT
  Multiply(                                             //
      prepared_A.data<uint8_t>(), B.data<int8_t>(),     //
      A_rows, width, B_cols,                            //
      multiply_callback                                 //
  );

  return y;
}

template <>
void prepare_weight_transposed<Provider::Gemmology>(
    const float* weights, int8_t* prepared, float quantization_multiplier,
    size_t cols, size_t rows) {
  auto PrepareBTransposed = GEMMOLOGY_DISPATCH(PrepareBTransposed);  // NOLINT
  PrepareBTransposed(weights, prepared, quantization_multiplier, cols, rows);
}

template <>
void prepare_weight_quantized_transposed<Provider::Gemmology>(
    const int8_t* input, int8_t* output, size_t rows, size_t cols) {
  // NOLINTNEXTLINE
  auto PrepareBQuantizedTransposed =
      GEMMOLOGY_DISPATCH(PrepareBQuantizedTransposed);
  PrepareBQuantizedTransposed(input, output, rows, cols);
}

}  // namespace slimt::qmm::detail
