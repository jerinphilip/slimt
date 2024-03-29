#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt {

Tensor index_select(const Tensor& x, const Tensor& indices,
                    const std::string& name = "selected");

void modify_mask_for_pad_tokens_in_attention(float* mask, size_t size);

template <class Scalar>
void transpose_10(const Scalar* in, size_t rows, size_t cols, Scalar* out);

void transpose_120(const float* in, size_t dim2, size_t dim1, size_t dim0,
                   float* out);

void transpose_3120(const float* in, size_t dim3, size_t dim2, size_t dim1,
                    size_t dim0, float* out);

void add(const float* a, const float* b, size_t size, float* c);
void mul(const float* a, const float* b, size_t size, float* c);

void mul_scalar(const float* a, float scalar, size_t size, float* c);

// Given source [
void index_select(const float* source, const int* indices, uint64_t batch_size,
                  uint64_t sequence_length, uint64_t embed_dim,
                  uint64_t vocab_size, float* out);

void sinusoidal_signal(int start, size_t sequence_length, size_t embed_dim,
                       float* out);

void add_positional_embedding(const float* word_embedding, const float* signal,
                              uint64_t batch_size, uint64_t sequence_length,
                              uint64_t embed_dim, float* out);

void softmax(float* logits, size_t batch_size, size_t num_classes, float* out);

void batch_matrix_multiply(const float* A, const float* B, size_t batch_size,
                           size_t rows_a, size_t cols_a, size_t rows_b,
                           size_t cols_b, bool trans_a, bool trans_b,
                           float alpha, float* C);

void batch_add_vector(const float* A, const float* x, size_t batch_size,
                      size_t size, float* out);

void layer_norm(const float* in, const float* scale, const float* bias,
                float eps, size_t rows, size_t cols, float* out);

Tensor transpose_3120(const Tensor& x);
float mse(const Tensor& x, const Tensor& y);
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor add(const Tensor& x, const Tensor& y);
Tensor sub(const Tensor& x, const Tensor& y);
Tensor mul(const Tensor& x, const Tensor& y);

Tensor operator+(const Tensor& x, const Tensor& y);
Tensor operator-(const Tensor& x, const Tensor& y);
Tensor operator*(const Tensor& x, const Tensor& y);

Tensor layer_norm(const Tensor& x, const Tensor& scale, const Tensor& bias,
                  float EPS = 1e-6F);  // NOLINT

Tensor fast_select(Tensor& source, const std::vector<uint32_t>& indices);
Tensor highway(const Tensor& x, const Tensor& y, const Tensor& g);

}  // namespace slimt
