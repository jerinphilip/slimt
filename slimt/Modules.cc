#include "slimt/Modules.hh"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "slimt/QMM.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"

namespace slimt {

float retrieve_quantization_multiplier(const Tensor &W) {
  const auto *b_end = W.end<int8_t>();
  float b_quant = *(reinterpret_cast<const float *>(b_end));
  return b_quant;
}

std::tuple<Tensor, Tensor> scaled_dot_product_attention(const Tensor &q,
                                                        const Tensor &k,
                                                        const Tensor &v,
                                                        const Tensor &mask) {
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L228

  // attn = softmax((q . k^T)/d_k) . v
  size_t batch_size = q.dim(-4);
  size_t num_heads = q.dim(-3);
  size_t query_length = q.dim(-2);
  size_t dim_head = q.dim(-1);

  size_t value_length = v.dim(-2);

  // Compute QKT
  Shape shape({batch_size, num_heads, query_length, value_length});
  Tensor qkt(q.type(), shape, "qkt");

  // scaling to avoid extreme values due to matrix multiplication
  float d_k = 1.0F / std::sqrt(dim_head);
  size_t reinterpreted_batch_size = batch_size * num_heads;
  batch_matrix_multiply(                               //
      q.data<float>(), k.data<float>(),                //
      reinterpreted_batch_size,                        //
      query_length, dim_head, value_length, dim_head,  //
      /*trans_a=*/false, /*trans_b=*/true,             //
      /*alpha=*/d_k,                                   //
      qkt.data<float>()                                //
  );

  // SLIMT_TRACE(qkt.shape());
  // SLIMT_TRACE(mask.shape());

  // Add the mask for the tokens.
  // Add without transposing etc using stride.
  size_t batch_stride = (num_heads * query_length * value_length);
  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    float *data = qkt.data<float>() + batch_id * batch_stride;
    const float *mask_data = mask.data<float>() + batch_id * value_length;
    for (size_t offset = 0; offset < batch_stride; offset += value_length) {
      float *data_begin = data + offset;
      add(data_begin, mask_data, value_length, data_begin);
    }
  }

  // softmax (QKT/d_k)
  Tensor attn(v.type(), qkt.shape(), "sdpa_attn");
  softmax(qkt.data<float>(), reinterpreted_batch_size * query_length,
          value_length, attn.data<float>());

  // softmax (QKT/d_k) * V
  Tensor out(q.type(), q.shape(), "sdpa_out");
  batch_matrix_multiply(                                   //
      attn.data<float>(), v.data<float>(),                 //
      reinterpreted_batch_size,                            //
      query_length, value_length, value_length, dim_head,  //
      /*trans_a=*/false, /*trans_b=*/false,                //
      /*alpha =*/1.0,                                      //
      out.data<float>()                                    //
  );

  return std::make_tuple(std::move(out), std::move(attn));
}

Tensor split_heads(const Tensor &x, size_t num_heads) {
  size_t batch_size = x.dim(-3);
  size_t sequence_length = x.dim(-2);
  size_t feature_dim = x.dim(-1);
  size_t dim_head = feature_dim / num_heads;

  // Currently          [B x T x num_heads * dim_head]
  // Need to become     [num_heads, B, T, dim_head]
  //
  // So that T x T attention matrices are formed.
  //
  // This requires a reshaping. i.e, two axes has to be permuted.
  //
  // First, perceive input matrix as:
  //    [B x T  x num_heads x dim_head]
  //
  // Given the layout, this is easy, just a view will do, so add a 1 dimension.
  //
  // What remains is permute(1, 2), which transposes the T and num_heads
  // dimensions respectively. This will require changes to the underlying
  // storage.
  //
  // In marian, this is achieved by TransposeND
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/tensors/cpu/tensor_operators.cpp#L370
  assert(feature_dim % num_heads == 0);

  Shape shape({batch_size, sequence_length, num_heads, dim_head});
  // SLIMT_TRACE(x.shape());
  // SLIMT_TRACE(shape);

  Tensor y(x.type(), shape.transpose(-3, -2), x.name());

  transpose_3120(x.data<float>(), batch_size, sequence_length, num_heads,
                 dim_head, y.data<float>());

  // SLIMT_TRACE(y.shape());

  return y;
}

Tensor join_heads(const Tensor &x) {
  size_t batch_size = x.dim(-4);
  size_t num_heads = x.dim(-3);
  size_t sequence_length = x.dim(-2);
  size_t dim_depth = x.dim(-1);

  size_t dim_model = num_heads * dim_depth;
  Shape shape({batch_size, sequence_length, dim_model});
  Tensor y(x.type(), shape, "concat");

  // B x N x T x H -> B x T x N x H
  transpose_3120(x.data<float>(), batch_size, num_heads, sequence_length,
                 dim_depth, y.data<float>());

  return y;
}

Tensor affine(const Affine &parameters, const Tensor &x,
              const std::string &name /* = ""*/) {
  Tensor y = qmm::affine(                              //
      x,                                               //
      parameters.W, parameters.b,                      //
      parameters.quant.item<float>(),                  //
      retrieve_quantization_multiplier(parameters.W),  //
      name                                             //
  );
  return y;
}

Tensor affine_with_select(const Affine &parameters, const Tensor &x,
                          const std::vector<uint32_t> &indices,
                          const std::string &name /*= ""*/) {
  Tensor y = qmm::affine_with_select(                  //
      x,                                               //
      parameters.W, parameters.b,                      //
      parameters.quant.item<float>(),                  //
      retrieve_quantization_multiplier(parameters.W),  //
      indices,                                         //
      name                                             //
  );
  return y;
}

Tensor linear(const Linear &parameters, const Tensor &x,
              const std::string &name = "") {
  Tensor y = qmm::dot(                                 //
      x, parameters.W,                                 //
      parameters.quant.item<float>(),                  //
      retrieve_quantization_multiplier(parameters.W),  //
      name                                             //
  );
  return y;
}

Tensor SSRU::start_state(size_t batch_size) const {
  // auto start = graph->constant({1, 1, dimBatch, dim}, inits::zeros());
  size_t feature_dim = O_.W.dim(-1);
  Tensor start(Type::f32, Shape({batch_size, feature_dim}), "start");
  start.fill_in_place(0.0F);
  return start;
}

Tensor SSRU::forward(Tensor &state, const Tensor &x) const {
  // From Research to Production and Back: Ludicrously Fast Neural Machine
  // Translation (https://aclanthology.org/D19-5632.pdf) Section 3.1 describes
  // SSRU. SSRU is described by the following recurrent equations - which
  // is formulated using output (y), forget-gate (f), cell-states (c). for a
  // given input (x).
  //
  // f(t) = σ(Wt . x(t) + bf )
  // c(t) = f(t) ⊙  c(t−1) + (1 − ft) ⊙  Wx(t)
  // y(t) = ReLU(c(t));
  // h(t) = α LayerNorm( y(t) + x(t)) + β
  //
  // ⊙  indicates elementwise-multiplication.
  //
  // The notion of adding forget-gates dependent on the input to do alpha x +
  // beta y allowing the network to learn skip connections are described in
  // Highway Networks (https://arxiv.org/pdf/1505.00387.pdf)
  //
  // The term highway appears here because marian uses it in a similar capacity.
  // https://github.com/browsermt/marian-dev/blob/0f4196c767afd1070fbb20eb348a5777d0376283/src/tensors/cpu/tensor_operators.cpp#L1593
  //
  //       Wx(t)  is a linear operation (it's a linear transform).
  // Wfx(t) + bf  is an affine transform.

  Tensor &c = state;  // Load context from saved-state.

  // Forward parameter multiplications.
  Tensor f = affine(F_, x, "rnn_f");    // Forget gate? NOLINT
  Tensor Wxt = linear(O_, x, "rnn_o");  // NOLINT

  // https://github.com/browsermt/marian-dev/blob/77e886ae7ae6016981c6307c312650bf74b50487/src/rnn/cells.h#L1058
  // c(t) = f(t) ⊙  c(t−1) + (1 − ft) ⊙  Wx(t)
  // Tensor c_t = highway(c, f, Wxt);
  Tensor c_t = highway(c, Wxt, f);

  // https://github.com/browsermt/marian-dev/blob/77e886ae7ae6016981c6307c312650bf74b50487/src/rnn/cells.h#L1059
  // y(t) = ReLU(c(t));
  Tensor y = relu(c_t);

  // h(t) = α LayerNorm(y(t) + x(t)) + β
  Tensor h = ln_.forward(x + y);

  state = std::move(c_t);

  return h;
}

std::tuple<Tensor, Tensor> DecoderLayer::forward(const Tensor &encoder_out,
                                                 const Tensor &mask,
                                                 Tensor &state,
                                                 const Tensor &x) const {
  Tensor decoder_out = rnn_.forward(state, x);

  // Assign query, key, value for cross-attention.
  const Tensor &q = decoder_out;
  const Tensor &k = encoder_out;
  const Tensor &v = encoder_out;

  // TODO(@jerinphilip), this will be called over and over.
  auto [out, attn] = attention_.forward(q, k, v, mask);

  Tensor ffn1_out = ffn_[0].forward(out);
  Tensor ffn1_acts = relu(ffn1_out);
  Tensor ffn2_out = ffn_[1].forward(ffn1_acts);
  Tensor y = add(ffn2_out, out);

  // Post Norm
  Tensor normalized_ffn_out = ffn_ffn_.forward(y);
  return std::make_tuple(std::move(normalized_ffn_out), std::move(attn));
}

EncoderLayer::EncoderLayer(size_t depth, size_t ffn_count, size_t num_heads)
    : depth_(depth), attention_("self", num_heads) {
  for (size_t i = 0; i < ffn_count; i++) {
    ffn_.emplace_back(i + 1);
  }
}

DecoderLayer::DecoderLayer(size_t depth, size_t ffn_count, size_t num_heads)
    : depth_(depth), attention_("context", num_heads) {
  for (size_t i = 0; i < ffn_count; i++) {
    ffn_.emplace_back(i + 1);
  }
}

FFN::FFN(size_t depth) : depth_(depth) {}

Tensor FFN::forward(const Tensor &x) const {
  Tensor y = affine(O_, x, "ffn" + std::to_string(depth_));
  return y;
}

Tensor LayerNorm::forward(const Tensor &x) const {
  Tensor y = layer_norm(x, scale_, bias_);
  return y;
}

std::tuple<Tensor, Tensor> Attention::forward(const Tensor &q, const Tensor &k,
                                              const Tensor &v,
                                              const Tensor &mask) const {
  // We have a B x T x H sequence coming in, for q, k and v.
  Tensor yq = affine(Q_, q, "q");
  Tensor yk = affine(K_, k, "k");
  Tensor yv = affine(V_, v, "v");

  // Split heads for query, keys and values.
  Tensor split_yq = split_heads(yq, num_heads_);
  Tensor split_yk = split_heads(yk, num_heads_);
  Tensor split_yv = split_heads(yv, num_heads_);

  // Apply individual scaled-dot-product-attention (SDPA)
  auto [attn_out, attn] =
      scaled_dot_product_attention(split_yq, split_yk, split_yv, mask);

  // Join heads.
  Tensor out = join_heads(attn_out);

  // Project to output size.
  Tensor yo = affine(O_, out, "o");

  // Add and norm
  const Tensor &x = q;
  Tensor x_plus_y(x.type(), x.shape(), "x+y_(residual)");

  add(x.data<float>(), yo.data<float>(), yo.size(), x_plus_y.data<float>());

  Tensor y = ln_.forward(x_plus_y);

  return std::make_tuple(std::move(y), std::move(attn));
}

std::tuple<Tensor, Tensor> EncoderLayer::forward(const Tensor &x,
                                                 const Tensor &mask) const {
  // TODO(fill code):
  auto [out, attention] = attention_.forward(x, x, x, mask);

  Tensor ffn1_out = ffn_[0].forward(out);
  Tensor ffn1_acts = relu(ffn1_out);
  Tensor ffn2_out = ffn_[1].forward(ffn1_acts);
  Tensor y = add(ffn2_out, out);
  // Post Norm
  Tensor normalized_ffn_out = ffn_ffn_.forward(y);

  return std::make_tuple(std::move(normalized_ffn_out), std::move(attention));
}

void EncoderLayer::register_parameters(const std::string &prefix,
                                       ParameterMap &parameters) {
  std::string encoder_prefix = prefix + "encoder_l" + std::to_string(depth_);
  attention_.register_parameters(encoder_prefix, parameters);
  for (FFN &ffn : ffn_) {
    ffn.register_parameters(encoder_prefix, parameters);
  }

  ffn_ffn_.register_parameters(encoder_prefix + "_ffn_ffn", parameters);
}

void DecoderLayer::register_parameters(const std::string &prefix,
                                       ParameterMap &parameters) {
  std::string decoder_prefix = prefix + "decoder_l" + std::to_string(depth_);
  attention_.register_parameters(decoder_prefix, parameters);
  for (FFN &ffn : ffn_) {
    ffn.register_parameters(decoder_prefix, parameters);
  }
  rnn_.register_parameters(decoder_prefix, parameters);
  ffn_ffn_.register_parameters(decoder_prefix + "_ffn_ffn", parameters);
}

Attention::Attention(std::string name, size_t num_heads)
    : name_(std::move(name)), num_heads_(num_heads) {}

void Attention::register_parameters(const std::string &prefix,
                                    ParameterMap &parameters) {
  auto register_affine = [&](const std::string &suffix, Affine &affine) {
    std::string local_prefix = prefix + ("_" + name_ + "_");
    parameters.emplace(local_prefix + "W" + suffix, &affine.W);
    parameters.emplace(local_prefix + "b" + suffix, &affine.b);
    parameters.emplace(local_prefix + "W" + suffix + "_QuantMultA",
                       &affine.quant);
  };

  register_affine("q", Q_);
  register_affine("k", K_);
  register_affine("v", V_);
  register_affine("o", O_);

  std::string wo_prefix = prefix + ("_" + name_ + "_") + "Wo";
  ln_.register_parameters(wo_prefix, parameters);
}

void FFN::register_parameters(const std::string &prefix,
                              ParameterMap &parameters) {
  // std::string param_name = prefix + "_ffn_W" + std::to_string(depth_);
  parameters.emplace(prefix + "_ffn_W" + std::to_string(depth_), &O_.W);
  parameters.emplace(prefix + "_ffn_b" + std::to_string(depth_), &O_.b);
  parameters.emplace(prefix + "_ffn_W" + std::to_string(depth_) + "_QuantMultA",
                     &O_.quant);
}

void SSRU::register_parameters(const std::string &prefix,
                               ParameterMap &parameters) {
  const std::string local_prefix = prefix + "_rnn_";
  parameters.emplace(local_prefix + "W", &O_.W);
  parameters.emplace(local_prefix + "W_QuantMultA", &O_.quant);

  parameters.emplace(local_prefix + "Wf", &F_.W);
  parameters.emplace(local_prefix + "bf", &F_.b);
  parameters.emplace(local_prefix + "Wf_QuantMultA", &F_.quant);

  ln_.register_parameters(local_prefix + "ffn", parameters);
}

void LayerNorm::register_parameters(const std::string &prefix,
                                    ParameterMap &parameters) {
  parameters.emplace(prefix + "_ln_bias", &bias_);
  parameters.emplace(prefix + "_ln_scale", &scale_);
}

}  // namespace slimt
