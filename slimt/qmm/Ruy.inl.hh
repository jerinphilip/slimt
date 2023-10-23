
namespace slimt::qmm::detail {

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

// Ruy.
template <>
Tensor affine<Provider::Ruy>(Tensor& x, const Tensor& W, const Tensor& b,
                             float a_quant, float b_quant,
                             const std::string& name) {
  Tensor& A = x;        // NOLINT
  const Tensor& B = W;  // NOLINT
  const Tensor& bias = b;

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
  const Tensor& prepared_bias = bias;

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
Tensor affine_with_select<Provider::Ruy>(Tensor& x, const Tensor& W,
                                         const Tensor& b, float a_quant,
                                         float b_quant,
                                         const std::vector<uint32_t>& indices,
                                         const std::string& name) {
  Tensor& A = x;        // NOLINT
  const Tensor& B = W;  // NOLINT
  const Tensor& bias = b;

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
    int8_t* sB_begin = &(sB_data[c * width]);               // NOLINT
    const int8_t* B_begin = &(B_data[indices[c] * width]);  // NOLINT
    std::memcpy(sB_begin, B_begin, width);
  }

  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, indices.size(), ruy::Order::kColMajor,
                        rhs.mutable_layout());
  rhs.set_data(selected_B.data<int8_t>());

  // Once again, bias needn't be prepared. But needs to be selected.
  const Tensor& prepared_bias = bias;
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
Tensor dot<Provider::Ruy>(Tensor& x, const Tensor& W, float a_quant,
                          float b_quant, const std::string& name) {
  Tensor& A = x;        // NOLINT
  const Tensor& B = W;  // NOLINT

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
}  // namespace slimt::qmm::detail
