
#include "slimt/TensorOps.hh"

#include <cblas.h>

#include <cassert>
#include <cmath>
#include <iostream>

#include "3rd-party/intgemm/intgemm/intgemm.h"
#include "slimt/Simd.hh"
#include "slimt/Utils.hh"

#if defined(_MSC_VER)
#define MARIAN_FFAST_MATH_BEGIN __pragma(float_control(precise, off, push))
#define MARIAN_FFAST_MATH_END __pragma(float_control(pop))
#elif defined(__clang__)
#define MARIAN_FFAST_MATH_BEGIN _Pragma("float_control(precise, off, push)")
#define MARIAN_FFAST_MATH_END _Pragma("float_control(pop)")
#elif defined(__GNUC__)
// Also available as __attribute__((optimize("-ffast-math"))) but done as
// pragmas for consistency
#define MARIAN_FFAST_MATH_BEGIN \
  _Pragma("GCC push_options") _Pragma("GCC optimize(\"-ffast-math\")")
#define MARIAN_FFAST_MATH_END _Pragma("GCC pop_options")
#endif

namespace slimt {

Tensor index_select(Tensor& x, Tensor& indices,
                    const std::string& name /*= "selected"*/) {
  uint64_t sequence_length = indices.dim(-1);
  uint64_t batch_size = indices.dim(-2);

  uint64_t x_cols = x.dim(-1);
  uint64_t x_rows = x.dim(-2);

  // index_select... really, rows_select
  Shape selected_shape = Shape({batch_size, sequence_length, x_cols});
  Tensor selected(x.type(), selected_shape, name);

  auto* x_ptr = x.data<float>();
  auto* selected_ptr = selected.data<float>();
  auto* indices_ptr = indices.data<int>();
  index_select(x_ptr, indices_ptr, batch_size, sequence_length, x_cols, x_rows,
               selected_ptr);
  return selected;
}

void modify_mask_for_pad_tokens_in_attention(float* mask, size_t size) {
  // Adopted from:
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L132
  float f16_lowest = std::numeric_limits<float>::lowest() / 2.0F;
  float minus_inf = std::max(f16_lowest, -99999999.0F);
  for (size_t i = 0; i < size; i++) {
    float& x = mask[i];
    x = (1.0F - x) * minus_inf;
  }
}

template <class Scalar>
void transpose_10(const Scalar* in, size_t rows, size_t cols, Scalar* out) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out[j * rows + i] = in[i * cols + j];
    }
  }
}

// NOLINTBEGIN
#define SLIMT_TRANSPOSE_10_EXPLICIT(Type)                                    \
  template void transpose_10<Type>(const Type* in, size_t rows, size_t cols, \
                                   Type* out);
// NOLINTEND

SLIMT_TRANSPOSE_10_EXPLICIT(float);
SLIMT_TRANSPOSE_10_EXPLICIT(int);
SLIMT_TRANSPOSE_10_EXPLICIT(uint32_t);
SLIMT_TRANSPOSE_10_EXPLICIT(int8_t);
#undef SLIMT_TRANSPOSE_10_EXPLICIT

void transpose_120(const float* in, size_t dim2, size_t dim1, size_t dim0,
                   float* out) {
  // Adapted from _0213 case at:
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/tensors/cpu/tensor_operators.cpp#L198
  size_t cols = dim0;
  size_t rows = dim2 * dim1;

  for (size_t j = 0; j < rows; ++j) {
    size_t src = j;
    size_t dst = j / dim1 + (j % dim1) * dim2;

    const float* in_row = in + src * cols;
    float* out_row = out + dst * cols;

    // mostly for fast forward computation
    std::copy(in_row, in_row + cols, out_row);
  }
}

void transpose_3120(const float* in, size_t dim3, size_t dim2, size_t dim1,
                    size_t dim0, float* out) {
  // Adapted once again, from.
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/tensors/cpu/tensor_operators.cpp#L199
  size_t cols = dim0;
  size_t rows = dim3 * dim2 * dim1;

  size_t rest = rows / (dim2 * dim1);

  for (size_t k = 0; k < rest; ++k) {
    size_t shift = k * dim1 * dim2;
    for (size_t j = 0; j < dim1 * dim2; ++j) {
      size_t src = j + shift;
      size_t dst = j / dim1 + (j % dim1) * dim2 + shift;

      const float* in_row = in + src * cols;
      float* out_row = out + dst * cols;

      for (size_t i = 0; i < cols; ++i) {
        out_row[i] = in_row[i];
      }
    }
  }
}

template <class Element>
void vectorized_add(const float* a, const float* b, size_t size, float* c) {
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Element>::add(va[i], vb[i]);
  }
}

void add(const float* a, const float* b, size_t size, float* c) {
  if (size % F32x8::kWidth == 0) {
    vectorized_add<F32x8>(a, b, size, c);
    return;
  }

  if (size % F32x4::kWidth == 0) {
    vectorized_add<F32x4>(a, b, size, c);
    return;
  }

  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

template <class Element>
void vectorized_sub(const float* a, const float* b, size_t size, float* c) {
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Element>::sub(va[i], vb[i]);
  }
}

void sub(const float* a, const float* b, size_t size, float* c) {
  if (size % F32x8::kWidth == 0) {
    vectorized_sub<F32x8>(a, b, size, c);
    return;
  }

  if (size % F32x4::kWidth == 0) {
    vectorized_sub<F32x4>(a, b, size, c);
    return;
  }

  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] - b[i];
  }
}

template <class Element>
void vectorized_relu(const float* a, size_t size, float* c) {
  const auto* va = reinterpret_cast<const Element*>(a);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Element>::relu(va[i]);
  }
}

void relu(const float* a, size_t size, float* c) {
  if (size % F32x8::kWidth == 0) {
    vectorized_relu<F32x8>(a, size, c);
    return;
  }

  if (size % F32x4::kWidth == 0) {
    vectorized_relu<F32x4>(a, size, c);
    return;
  }

  for (size_t i = 0; i < size; i++) {
    c[i] = std::max<float>(0.0F, a[i]);
  }
}

template <class Element>
void vectorized_mul(const float* a, const float* b, size_t size, float* c) {
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Element>::mul(va[i], vb[i]);
  }
}

void mul(const float* a, const float* b, size_t size, float* c) {
  if (size % F32x8::kWidth == 0) {
    vectorized_mul<F32x8>(a, b, size, c);
    return;
  }

  if (size % F32x4::kWidth == 0) {
    vectorized_mul<F32x4>(a, b, size, c);
    return;
  }

  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}

void mul_scalar(const float* a, float scalar, size_t size, float* c) {
  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] * scalar;
  }
}

template <class Element>
void vectorized_sigmoid(const float* a, size_t size, float* c) {
  const auto* va = reinterpret_cast<const Element*>(a);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Element>::sigmoid(va[i]);
  }
}

void sigmoid(const float* a, size_t size, float* c) {
  if (size % F32x8::kWidth == 0) {
    vectorized_sigmoid<F32x8>(a, size, c);
    return;
  }

  if (size % F32x4::kWidth == 0) {
    vectorized_sigmoid<F32x4>(a, size, c);
    return;
  }

  for (size_t i = 0; i < size; i++) {
    float x = std::exp(a[i]);
    c[i] = x / (1 + x);
  }
}

void index_select(const float* source, const int* indices, uint64_t batch_size,
                  uint64_t sequence_length, uint64_t embed_dim,
                  uint64_t vocab_size, float* out) {
  // https://github.com/jerinphilip/marian/blob/8c4170fa08c46df1cf4c987e493b7a3772c380b3/src/tensors/cpu/tensor_operators.cpp#L554
  (void)vocab_size;  // We may use vocab_size to bounds-check token.

  // We can just do this as BT x E, but meh.
  for (uint64_t batch_id = 0; batch_id < batch_size; batch_id++) {
    for (uint64_t token_id = 0; token_id < sequence_length; token_id++) {
      // out [b, t, :] = vocab[t, :]
      uint64_t token = indices[batch_id * sequence_length + token_id];
      const float* embedding = source + token * embed_dim;
      float* target = out + (batch_id * sequence_length + token_id) * embed_dim;
      std::copy(embedding, embedding + embed_dim, target);
    }
  }
}

void sinusoidal_signal(int start, size_t sequence_length, size_t embed_dim,
                       float* out) {
  // Imported from:
  // https://github.com/jerinphilip/marian/blob/8c4170fa08c46df1cf4c987e493b7a3772c380b3/src/graph/node_initializers.cpp#L216
  float num_timescales = static_cast<float>(embed_dim) / 2;
  float log_timescale_increment = std::log(10000.0F) / (num_timescales - 1.0F);

  for (size_t p = start; p < sequence_length + start; ++p) {
    for (int i = 0; i < num_timescales; ++i) {
      float v = p * std::exp(i * -log_timescale_increment);
      size_t offset = (p - start) * embed_dim + i;

      size_t idx_sin = offset;
      out[idx_sin] = std::sin(v);

      size_t id_cos = offset + static_cast<int>(num_timescales);
      out[id_cos] = std::cos(v);
    }
  }
}

void add_positional_embedding(const float* word_embedding,
                              const float* position_signal, uint64_t batch_size,
                              uint64_t sequence_length, uint64_t embed_dim,
                              float* out) {
  size_t cols = sequence_length * embed_dim;
  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    size_t offset = batch_id * cols;

    const float* data = word_embedding + offset;
    float* out_data = out + offset;

    add(data, position_signal, cols, out_data);
  }
}

template <class Element>
void vectorized_softmax(const float* _logits, size_t batch_size,
                        size_t num_classes, float* _out) {
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
      max_value = Ops<Element>::max(max_value, logit[i]);
    }

    // if ElementType is a complex type, e.g. float32x8, find the max of
    // these 8 values
    typename Ops<Element>::Scalar max_value_scalar =
        Ops<Element>::Reduce::max(max_value);
    Element max_value_projected(max_value_scalar);

    // Find numerically stable sumexp, after shifting values by maximum.
    Element vsum(0.0F);
    for (int i = 0; i < cols; ++i) {
      Element shifted = Ops<Element>::sub(logit[i], max_value_projected);
      Element exp_x = Ops<Element>::exp(shifted);
      vsum = Ops<Element>::add(vsum, exp_x);
      p[i] = exp_x;
    }

    // if Register is a complex type, e.g. float32x8, sum these 8 values
    typename Ops<Element>::Scalar sums = Ops<Element>::Reduce::sum(vsum);
    Element sums_value_projected(sums);

    for (int i = 0; i < cols; ++i) {
      p[i] = Ops<Element>::div(p[i], sums_value_projected);
    }
  }
}

void softmax(float* logits, size_t batch_size, size_t num_classes, float* out) {
  if (num_classes % 8 == 0) {
    vectorized_softmax<F32x8>(logits, batch_size, num_classes, out);
    return;
  }

  if (num_classes % 4 == 0) {
    vectorized_softmax<F32x4>(logits, batch_size, num_classes, out);
    return;
  }

  for (size_t i = 0; i < batch_size; i++) {
    float* xs = logits + i * num_classes;

    // Numerically stable algorithm. Compute maximimum among logits first.
    float max_value = std::numeric_limits<float>::lowest();
    for (size_t j = 0; j < num_classes; j++) {
      max_value = std::max<float>(max_value, xs[j]);
    }

    float sumexp = 0;
    for (size_t j = 0; j < num_classes; j++) {
      sumexp += std::exp(xs[j] - max_value);
    }

    for (size_t j = 0; j < num_classes; j++) {
      float p = std::exp(xs[j] - max_value) / sumexp;
      out[i * num_classes + j] = p;
    }
  }
}

// NOLINTBEGIN
enum class Provider {
  BLAS,
};
// NOLINTEND

template <enum Provider>
void matrix_multiply(              //
    bool trans_a, bool trans_b,    //
    size_t m, size_t n, size_t k,  //
    float alpha,                   //
    const float* A, size_t lda,    //
    const float* B, size_t ldb,    //
    float beta,                    //
    float* C, size_t ldc           //
);

template <>
void matrix_multiply<Provider::BLAS>(  //
    bool trans_a, bool trans_b,        //
    size_t m, size_t n, size_t k,      //
    float alpha,                       //
    const float* A, size_t lda,        //
    const float* B, size_t ldb,        //
    float beta,                        //
    float* C, size_t ldc) {
  // clang-format off
  //
  //  4. m
  //     Specifies the number of rows of the matrix op(A) and of the matrix C.
  //     The value of m at least zero.
  //  5. n
  //     Specifies the number of columns of the matrix op(B) and the number of
  //     columns of the matrix C. The value of n at least zero.
  //  6. k
  //     Specifies the number of columns of the matrix op(A) and the number of
  //     rows of the matrix op(B). The value of k at least zero.
  //
  //  9. lda
  //                    |  transa=CblasNoTrans   |  transa=CblasTrans 
  //      CblasColMajor | lda at least max(1, m).|   lda at least max(1, k)
  //      CblasRowMajor | lda at least max(1, k) |   lda at least max(1, m).
  //
  // 11. ldb
  //                    | transb=CblasNoTrans     | transb=CblasTrans
  //     CblasColMajorA | ldb at least max(1, k). | ldb at least max(1, n).
  //     CblasRowMajor  | ldb at least max(1, n). | ldb at least max(1, k).
  //
  // 14. ldc
  //
  //      CblasColMajor | ldc must be at least max(1, m).
  //      CblasRowMajor | ldc must be at least max(1, n).
  //
  // clang-format on

  CBLAS_TRANSPOSE c_trans_a = trans_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE c_trans_b = trans_b ? CblasTrans : CblasNoTrans;

  // Consider matrices A [rows_a x cols_a], B[rows_b x cols_b]
  // A and B are not necessarily compatible for multiplication.
  //
  // op(A) * op(B) must be compatible for matrix multiplication, where op is
  // transpose or no-transpose (identity) indicated by bools trans_a and
  // trans_b.

  cblas_sgemm(                              //
      CblasRowMajor, c_trans_a, c_trans_b,  // Layout, op(A), op(B)
      m, n, k,                              //
      alpha,                                //
      A, lda, B, ldb,                       //
      beta,                                 //
      C, ldc                                //
  );
}

void batch_matrix_multiply(const float* A, const float* B, size_t batch_size,
                           size_t rows_a, size_t cols_a, size_t rows_b,
                           size_t cols_b, bool trans_a, bool trans_b,
                           float alpha, float* C) {
  // Let's assume compatible and assign m, k, n, then modify in case trans_a
  // or trans_b is applied. Eventually op(A): m x k, op(B): l x n, check is
  // k == l.
  size_t m = rows_a;
  size_t k = cols_a;

  size_t l = rows_b;
  size_t n = cols_b;

  // If ops are performed, we just swap dimensions to remain compatible.
  if (trans_a) std::swap(m, k);
  if (trans_b) std::swap(l, n);

  assert(k == l);

  // The LDA parameter in BLAS is effectively the stride of the matrix as it
  // is laid out in linear memory.
  // https://stackoverflow.com/a/8209290/4565794

  // In row-major storage, which this library uses, stride is columns.
  size_t lda = cols_a;
  size_t ldb = cols_b;  //
  size_t ldc = n;       // m x k, k x n, Expecting m x n. Therefore ldc = n;

  // These are strides for batch elements that are individual matrices.
  // Each element in the batch are apart by the size of the matrix, which
  // becomes stride here computed as rows*cols.
  size_t stride_a = m * k;
  size_t stride_b = k * n;
  size_t stride_c = m * n;

  float beta = 0.0;

  for (size_t i = 0; i < batch_size; ++i) {
    const float* a = A + i * stride_a;
    const float* b = B + i * stride_b;
    float* c = C + i * stride_c;
    matrix_multiply<Provider::BLAS>(  //
        trans_a, trans_b,             //
        m, n, k,                      //
        alpha,                        //
        a, lda, b, ldb,               //
        beta,                         //
        c, ldc                        //
    );
  }
}

void batch_add_vector(const float* A, const float* x, size_t batch_size,
                      size_t size, float* out) {
  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    size_t offset = (batch_id * size);

    const float* data = A + offset;
    float* out_data = out + offset;
    add(data, x, size, out_data);
  }
}

MARIAN_FFAST_MATH_BEGIN
void layer_norm(const float* in, const float* scale, const float* bias,
                float eps, size_t rows, size_t cols, size_t scale_stride,
                size_t bias_stride, bool has_bias, float* out) {
  // LayerNorm
  //
  //   y =    x − E[x]      γ  +  β
  //       ---------------
  //       sqrt(Var[x]+ϵ))
  //
  // Implementation lifted from:
  // https://github.com/browsermt/marian-dev/blob/7cf2159bc4e9c0c337aa38270081d941c9e59c26/src/tensors/cpu/tensor_operators.cpp#L1103

#pragma omp parallel for
  for (size_t j = 0; j < rows; ++j) {
    const float* x = in + j * cols;
    float* y = out + j * cols;

    // Compute E[x] (mean)
    float sum = 0.0F;
#pragma omp simd reduction(+ : sum)
    for (size_t i = 0; i < cols; ++i) {
      sum += x[i];
    }
    float mean = sum / cols;

    // Compute Std[X] = sqrt . Var[X]
    float square_sum_centered = 0.0F;
#pragma omp simd reduction(+ : square_sum_centered)
    for (size_t i = 0; i < cols; ++i) {
      float v = x[i] - mean;
      square_sum_centered += v * v;
    }

    float sigma = std::sqrt(square_sum_centered / cols + eps);

    // Normalize from sample estimate (E[X], Var[X}) and parameters learned
    // during the course of learning - scale and bias.

#pragma omp simd
    for (size_t i = 0; i < cols; ++i) {
      size_t s = scale_stride * i;
      float t = scale[s] * ((x[i] - mean) / sigma);
      if (has_bias) {
        size_t b = bias_stride * i;
        t += bias[b];
      }

      y[i] = t;
    }
  }
}

MARIAN_FFAST_MATH_END
Tensor intgemm_affine_with_select(Tensor& x, Tensor& W, Tensor& b,
                                  float a_quant, float b_quant,
                                  const std::vector<uint32_t>& indices,
                                  const std::string& name) {
  // Naming is to simplify thinking with the intgemm API below.
  Tensor& A = x;  // NOLINT
  Tensor& B = W;  // NOLINT
  Tensor& bias = b;

  VERIFY_MATCH(
      A, "var_586-cpu-int8_1x1x2x256_none_shifted-rhs0-float32_1x1x2x256.bin");


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
  VERIFY_MATCH(bias,
               "var_582-cpu-float32_1x32000_decoder_ff_logit_out_b_Prepared-"
               "rhs0-float32_1x32000_decoder_ff_logit_out_b.bin");
  VERIFY_MATCH(B,
               "var_582-cpu-float32_1x32000_decoder_ff_logit_out_b_Prepared-"
               "rhs1-intgemm8_256x32000_Wemb.bin");
  float a_alpha = 127.0F / a_quant;
  float b_alpha = 127.0F / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / 127.0F;
  auto prepare_bias_callback = intgemm::callbacks::UnquantizeAndAddBiasAndWrite(
      bias_unquant_multiplier, bias.data<float>(),  //
      prepared_bias.data<float>()                   //
  );

  intgemm::Int8Shift::PrepareBias(  //
      B.data<int8_t>(),             //
      width, B_cols,                //
      prepare_bias_callback         //
  );

  VERIFY_MATCH(
      prepared_bias,
      "var_582-cpu-float32_1x32000_decoder_ff_logit_out_b_Prepared-lhs.bin");

  // Select before multiply?
  // NOLINTNEXTLINE
  Tensor selected_B(Type::i8, Shape({256, indices.size()}), "selected_B");
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

Tensor intgemm_affine(Tensor& x, Tensor& W, Tensor& b, float a_quant,
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
  float a_alpha = 127.0F / a_quant;
  float b_alpha = 127.0F / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / 127.0F;
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

Tensor intgemm_dot(Tensor& x, Tensor& W, float a_quant, float b_quant,
                   const std::string& name) {
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
  float a_alpha = 127.0F / a_quant;
  float b_alpha = 127.0F / b_quant;

  float bias_unquant_multiplier = (-1.0F * (a_alpha * b_alpha)) / 127.0F;
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

float mse(Tensor& x, Tensor& y) {
  assert(x.type() == Type::f32);
  assert(y.type() == Type::f32);
  assert(x.size() == y.size());

  auto* p = x.data<float>();
  auto* q = y.data<float>();
  float sum = 0;
  for (size_t i = 0; i < x.size(); i++) {
    float x = (*p) - (*q);
    sum += (x * x);
    ++p, ++q;
  }
  return sum / x.size();
}

Tensor transpose_3120(Tensor& x) {
  Tensor y(x.type(), x.shape().transpose(-3, -2), x.name() + "transpose12");
  size_t d3 = x.dim(-3);
  size_t d2 = x.dim(-2);
  size_t d1 = x.dim(-1);
  size_t rest = x.size() / (d3 * d2 * d1);
  transpose_3120(x.data<float>(), rest, d3, d2, d1, y.data<float>());
  return y;
}

Tensor relu(Tensor& x) {
  assert(x.type() == Type::f32);
  Tensor y = x.like(x.name() + "_relu");
  relu(x.data<float>(), x.size(), y.data<float>());
  return y;
}

Tensor sigmoid(Tensor& x) {
  assert(x.type() == Type::f32);
  Tensor y = x.like(x.name() + "_sigmoid");
  sigmoid(x.data<float>(), x.size(), y.data<float>());
  return y;
}

Tensor add(Tensor& x, Tensor& y) {
  assert(x.type() == Type::f32);
  assert(x.size() == y.size());
  Tensor x_plus_y = x.like("x_plus_y");
  add(x.data<float>(), y.data<float>(), y.size(), x_plus_y.data<float>());
  return x_plus_y;
}

Tensor sub(Tensor& x, Tensor& y) {
  assert(x.type() == Type::f32);
  assert(x.size() == y.size());
  Tensor x_plus_y = x.like("x_plus_y");
  sub(x.data<float>(), y.data<float>(), y.size(), x_plus_y.data<float>());
  return x_plus_y;
}

Tensor mul(Tensor& x, Tensor& y) {
  assert(x.type() == Type::f32);
  assert(x.size() == y.size());
  Tensor x_plus_y = x.like("x_times_y");
  mul(x.data<float>(), y.data<float>(), y.size(), x_plus_y.data<float>());
  return x_plus_y;
}

}  // namespace slimt
