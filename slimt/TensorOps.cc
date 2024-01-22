#include "slimt/TensorOps.hh"

#include <cstddef>
#include <cstdint>
#include <string>

#include "slimt/Tensor.hh"

#ifdef SLIMT_HAS_BLAS

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <cblas.h>

#ifdef __cplusplus
}
#endif  // __cplusplus

#else  // SLIMT_HAS_BLAS
#include "ruy/ruy.h"
#endif  // SLIMT_HAS_BLAS

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

namespace slimt {

inline float sigmoid(float x) {
  return x > 0 ? (1.0F / (1.0F + std::exp(-x)))
               : (std::exp(x) / (1.0F + std::exp(x)));
}

Tensor index_select(const Tensor& x, const Tensor& indices,
                    const std::string& name /*= "selected"*/) {
  uint64_t sequence_length = indices.dim(-1);
  uint64_t batch_size = indices.dim(-2);

  uint64_t x_cols = x.dim(-1);
  uint64_t x_rows = x.dim(-2);

  // index_select... really, rows_select
  Shape selected_shape = Shape({batch_size, sequence_length, x_cols});
  Tensor selected(x.type(), selected_shape, name);

  const auto* x_ptr = x.data<float>();
  auto* selected_ptr = selected.data<float>();
  const auto* indices_ptr = indices.data<int>();
  index_select(x_ptr, indices_ptr, batch_size, sequence_length, x_cols, x_rows,
               selected_ptr);
  return selected;
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
                                   Type* out)
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

void add(const float* a, const float* b, size_t size, float* c) {
#ifdef VEXT_W8_AVAILABLE
  if (size % VDatum<VExt::w8>::kWidth == 0) {
    vext::add<VExt::w8>(a, b, size, c);
    return;
  }
#endif

#ifdef VEXT_W4_AVAILABLE
  if (size % F32x4::kWidth == 0) {
    vext::add<VExt::w4>(a, b, size, c);
    return;
  }
#endif

  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

void sub(const float* a, const float* b, size_t size, float* c) {
#ifdef VEXT_W8_AVAILABLE
  if (size % VDatum<VExt::w8>::kWidth == 0) {
    vext::sub<VExt::w8>(a, b, size, c);
    return;
  }
#endif

#ifdef VEXT_W4_AVAILABLE
  if (size % F32x4::kWidth == 0) {
    vext::sub<VExt::w4>(a, b, size, c);
    return;
  }
#endif

  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] - b[i];
  }
}

void relu(const float* a, size_t size, float* c) {
#ifdef VEXT_W8_AVAILABLE
  if (size % VDatum<VExt::w8>::kWidth == 0) {
    vext::relu<VExt::w8>(a, size, c);
    return;
  }

#endif
#ifdef VEXT_W4_AVAILABLE
  if (size % F32x4::kWidth == 0) {
    vext::relu<VExt::w4>(a, size, c);
    return;
  }
#endif

  for (size_t i = 0; i < size; i++) {
    c[i] = std::max<float>(0.0F, a[i]);
  }
}

void mul(const float* a, const float* b, size_t size, float* c) {
#ifdef VEXT_W8_AVAILABLE
  if (size % VDatum<VExt::w8>::kWidth == 0) {
    vext::mul<VExt::w8>(a, b, size, c);
    return;
  }
#endif

#ifdef VEXT_W4_AVAILABLE
  if (size % F32x4::kWidth == 0) {
    vext::mul<VExt::w4>(a, b, size, c);
    return;
  }
#endif
  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}

void mul_scalar(const float* a, float scalar, size_t size, float* c) {
  for (size_t i = 0; i < size; i++) {
    c[i] = a[i] * scalar;
  }
}

void sigmoid(const float* a, size_t size, float* c) {
#ifdef VEXT_W8_AVAILABLE
  if (size % VDatum<VExt::w8>::kWidth == 0) {
    vext::sigmoid<VExt::w8>(a, size, c);
    return;
  }
#endif
#ifdef VEXT_W4_AVAILABLE
  if (size % F32x4::kWidth == 0) {
    vext::sigmoid<VExt::w4>(a, size, c);
    return;
  }
#endif

  for (size_t i = 0; i < size; i++) {
    c[i] = sigmoid(a[i]);
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
  float log_timescale_increment =
      std::log(10000.0F) / (num_timescales - 1.0F);  // NOLINT

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

void softmax(float* logits, size_t batch_size, size_t num_classes, float* out) {
#ifdef VEXT_W8_AVAILABLE
  if (num_classes % VDatum<VExt::w8>::kWidth == 0) {
    vext::softmax<VExt::w8>(logits, batch_size, num_classes, out);
    return;
  }
#endif
#ifdef VEXT_W4_AVAILABLE
  if (num_classes % VDatum<VExt::w4>::kWidth == 0) {
    vext::softmax<VExt::w4>(logits, batch_size, num_classes, out);
    return;
  }
#endif

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
  Ruy,
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

#ifdef SLIMT_HAS_BLAS
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

constexpr Provider kChosenProvider = Provider::BLAS;

#else
template <>
inline void matrix_multiply<Provider::Ruy>(  //
    bool transA, bool transB,                //
    size_t M, size_t N, size_t K,            //
    float alpha,                             //
    const float* A, size_t lda,              //
    const float* B, size_t ldb,              //
    float beta,                              //
    float* C, size_t ldc) {
  (void)lda;
  (void)ldb;
  (void)ldc;
  ruy::Context context;

  // If we need to transpose, we can swap dimensions in layout claim the matrix
  // is just column-major. Set ordering so transpose.
  const auto orderA = (transA ? ruy::Order::kColMajor : ruy::Order::kRowMajor);
  const auto orderB = (transB ? ruy::Order::kColMajor : ruy::Order::kRowMajor);

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(M, K, orderA, lhs.mutable_layout());
  lhs.set_data(A);

  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(K, N, orderB, rhs.mutable_layout());
  rhs.set_data(B);

  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(M, N, ruy::Order::kRowMajor, dst.mutable_layout());

  if (beta == 0) {
    // For beta = 0, we want to avoid the additional allocation. This is a
    // large amount of our inference use-cases. sgemm is called with `beta` for
    // accumulating gradients in backpropogation, which is 0.0 during
    // inference.

    dst.set_data(C);
    ruy::MulParams<float, float> mul_params;
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);

    if (alpha != 1.0) {
      // Write out C as C = alpha * [op(A) * op(B)] + beta * C
      // Can we expect the compiler to autovectorize this?
      // TODO: Come back and explicitly use SIMD.
      const size_t size = M * N;
      const float* opA_opB = C;  // Alias.
#pragma clang loop vectorize(enable) interleave(enable)
      for (size_t i = 0; i < size; i++) {
        C[i] = alpha * opA_opB[i];
      }
    }
  } else {
    // @jerinphilip has not yet been able to find a ruy primitive that does in
    // place addition to obtain full gemm.
    //
    // Safe bet is to make an additional allocation to store the result of
    // multiply  and use the existing values in C.
    //
    // See also: https://github.com/google/ruy/issues/307

    Aligned intermediate(64, M * N * sizeof(float));
    auto* imd_data = reinterpret_cast<float*>(intermediate.data());
    dst.set_data(imd_data);
    ruy::MulParams<float, float> mul_params;
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);

    // Write out C as C = alpha * [op(A) * op(B)] + beta * C
    // Can we expect the compiler to autovectorize this?
    // TODO: Come back and explicitly use SIMD.
    const size_t size = M * N;
    const float* opA_opB = imd_data;
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < size; i++) {
      C[i] = alpha * opA_opB[i] + beta * C[i];
    }
  }
}

constexpr Provider kChosenProvider = Provider::Ruy;

#endif

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
    matrix_multiply<kChosenProvider>(  //
        trans_a, trans_b,              //
        m, n, k,                       //
        alpha,                         //
        a, lda, b, ldb,                //
        beta,                          //
        c, ldc                         //
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

void layer_norm(const float* in, const float* scale, const float* bias,
                float eps, size_t rows, size_t cols, float* out) {
  // LayerNorm
  //
  //   y =    x − E[x]      γ  +  β
  //       ---------------
  //       sqrt(Var[x]+ϵ))
  //
  // Implementation lifted from:
  // https://github.com/browsermt/marian-dev/blob/7cf2159bc4e9c0c337aa38270081d941c9e59c26/src/tensors/cpu/tensor_operators.cpp#L1103

  for (size_t j = 0; j < rows; ++j) {
    const float* x = in + j * cols;
    float* y = out + j * cols;

    // Compute E[x] (mean)
    float sum = 0.0F;
    for (size_t i = 0; i < cols; ++i) {
      sum += x[i];
    }
    float mean = sum / cols;

    // Compute Std[X] = sqrt . Var[X]
    float square_sum_centered = 0.0F;
    for (size_t i = 0; i < cols; ++i) {
      float v = x[i] - mean;
      square_sum_centered += v * v;
    }

    float sigma = std::sqrt(square_sum_centered / cols + eps);

    // Normalize from sample estimate (E[X], Var[X}) and parameters learned
    // during the course of learning - scale and bias.

    for (size_t i = 0; i < cols; ++i) {
      y[i] = scale[i] * ((x[i] - mean) / sigma) + bias[i];
    }
  }
}

float mse(const Tensor& x, const Tensor& y) {
  assert(x.type() == Type::f32);
  assert(y.type() == Type::f32);
  assert(x.size() == y.size());

  const auto* p = x.data<float>();
  const auto* q = y.data<float>();
  float sum = 0;
  for (size_t i = 0; i < x.size(); i++) {
    float x = (*p) - (*q);
    sum += (x * x);
    ++p, ++q;
  }
  return sum / x.size();
}

Tensor transpose_3120(const Tensor& x) {
  Tensor y(x.type(), x.shape().transpose(-3, -2), x.name() + "transpose12");
  size_t d3 = x.dim(-3);
  size_t d2 = x.dim(-2);
  size_t d1 = x.dim(-1);
  size_t rest = x.size() / (d3 * d2 * d1);
  transpose_3120(x.data<float>(), rest, d3, d2, d1, y.data<float>());
  return y;
}

Tensor relu(const Tensor& x) {
  assert(x.type() == Type::f32);
  Tensor y = x.like(x.name() + "_relu");
  relu(x.data<float>(), x.size(), y.data<float>());
  return y;
}

Tensor sigmoid(const Tensor& x) {
  assert(x.type() == Type::f32);
  Tensor y = x.like(x.name() + "_sigmoid");
  sigmoid(x.data<float>(), x.size(), y.data<float>());
  return y;
}

Tensor add(const Tensor& x, const Tensor& y) {
  assert(x.type() == Type::f32);
  assert(x.size() == y.size());
  Tensor x_plus_y = x.like("x_plus_y");
  add(x.data<float>(), y.data<float>(), y.size(), x_plus_y.data<float>());
  return x_plus_y;
}

Tensor sub(const Tensor& x, const Tensor& y) {
  assert(x.type() == Type::f32);
  assert(x.size() == y.size());
  Tensor x_plus_y = x.like("x_plus_y");
  sub(x.data<float>(), y.data<float>(), y.size(), x_plus_y.data<float>());
  return x_plus_y;
}

Tensor mul(const Tensor& x, const Tensor& y) {
  assert(x.type() == Type::f32);
  assert(x.size() == y.size());
  Tensor x_plus_y = x.like("x_times_y");
  mul(x.data<float>(), y.data<float>(), y.size(), x_plus_y.data<float>());
  return x_plus_y;
}

Tensor layer_norm(const Tensor& x, const Tensor& scale, const Tensor& bias,
                  float EPS /*= 1e-6F*/) {
  Tensor y = x.like("ln_out");
  size_t cols = x.dim(-1);
  size_t rows = x.size() / cols;

  layer_norm(x.data<float>(),                          //
             scale.data<float>(), bias.data<float>(),  //
             EPS, rows, cols, y.data<float>());
  return y;
}

Tensor operator+(const Tensor& x, const Tensor& y) { return add(x, y); }
Tensor operator-(const Tensor& x, const Tensor& y) { return sub(x, y); }
Tensor operator*(const Tensor& x, const Tensor& y) { return mul(x, y); }

Tensor highway(const Tensor& x, const Tensor& y, const Tensor& g) {
  // f(t) = σ(Wt . x(t) + bf )
  Tensor c_t = x.like("highway_out");

  assert(x.size() == y.size());
  assert(y.size() == g.size());
  const auto* tx = x.data<float>();
  const auto* ty = y.data<float>();
  const auto* tg = g.data<float>();
  auto* out = c_t.data<float>();
  size_t size = x.size();

  for (size_t i = 0; i < size; i++) {
    float sg = sigmoid(tg[i]);
    float vx = tx[i];
    float vy = ty[i];
    out[i] = sg * vx + (1.0F - sg) * vy;
  }

  return c_t;
}

}  // namespace slimt
