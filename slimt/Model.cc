#include "slimt/Model.hh"

#include <cassert>
#include <cmath>
#include <iostream>

#include "slimt/TensorOps.hh"
#include "slimt/Tensori8Ops.hh"
#include "slimt/Utils.hh"

namespace slimt {

float retrieve_quantization_multiplier(Tensor &W) {
  auto *b_end = W.end<int8_t>();
  float b_quant = *(reinterpret_cast<float *>(b_end));
  return b_quant;
}

std::tuple<Tensor, Tensor> scaled_dot_product_attention(Tensor &q, Tensor &k,
                                                        Tensor &v,
                                                        Tensor &mask) {
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
    float *mask_data = mask.data<float>() + batch_id * value_length;
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

Tensor split_heads(Tensor &x, size_t num_heads) {
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

Tensor join_heads(Tensor &x) {
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

Tensor affine(Affine &parameters, Tensor &x, const std::string &name = "") {
  Tensor y = qmm::affine(                              //
      x,                                               //
      parameters.W, parameters.b,                      //
      parameters.quant.item<float>(),                  //
      retrieve_quantization_multiplier(parameters.W),  //
      name                                             //
  );
  return y;
}

Tensor affine_with_select(Affine &parameters, Tensor &x,
                          const std::vector<uint32_t> &indices,
                          const std::string &name = "") {
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

Tensor linear(Linear &parameters, Tensor &x, const std::string &name = "") {
  Tensor y = qmm::dot(                                 //
      x, parameters.W,                                 //
      parameters.quant.item<float>(),                  //
      retrieve_quantization_multiplier(parameters.W),  //
      name                                             //
  );
  return y;
}

Tensor FFN::forward(Tensor &x) {
  Tensor y = affine(O_, x, "ffn" + std::to_string(depth_));
  return y;
}

Tensor LayerNorm::forward(Tensor &x) {
  Tensor y = x.like("ln_out");
  size_t cols = x.dim(-1);
  size_t rows = x.size() / cols;

  // Currently this is hardcoded.
  // Not sure how to do it otherwise.
  constexpr float kEps = 1e-9;
  size_t scale_stride = 1;
  size_t bias_stride = 1;
  bool has_bias = true;

  layer_norm(x.data<float>(),                            //
             scale_.data<float>(), bias_.data<float>(),  //
             kEps, rows, cols,                           //
             scale_stride, bias_stride, has_bias,        //
             y.data<float>());

  return y;
}

std::tuple<Tensor, Tensor> Attention::forward(Tensor &q, Tensor &k, Tensor &v,
                                              Tensor &mask) {
  // We have a B x T x H sequence comoing in, for q, k and v.
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
  Tensor &x = q;
  Tensor x_plus_y(x.type(), x.shape(), "x+y_(residual)");

  add(x.data<float>(), yo.data<float>(), yo.size(), x_plus_y.data<float>());

  Tensor y = ln_.forward(x_plus_y);

  return std::make_tuple(std::move(y), std::move(attn));
}

std::tuple<Tensor, Tensor> EncoderLayer::forward(Tensor &x, Tensor &mask) {
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

void transform_embedding(Tensor &word_embedding, size_t start = 0) {
  // This is a pain, why does marian-transpose here, I do not get yet.

  uint64_t embed_dim = word_embedding.dim(-1);
  uint64_t sequence_length = word_embedding.dim(-2);
  uint64_t batch_size = word_embedding.dim(-3);

  auto *word_embedding_ptr = word_embedding.data<float>();

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L88
  mul_scalar(word_embedding_ptr, std::sqrt(static_cast<float>(embed_dim)),
             word_embedding.size(), word_embedding_ptr);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L105
  Tensor positional_embedding(word_embedding.type(),
                              Shape({sequence_length, embed_dim}),
                              "positional_embedding");
  auto *positional_embedding_ptr = positional_embedding.data<float>();
  sinusoidal_signal(start, sequence_length, embed_dim,
                    positional_embedding_ptr);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L109
  add_positional_embedding(word_embedding_ptr, positional_embedding_ptr,
                           batch_size, sequence_length, embed_dim,
                           word_embedding_ptr);
}

Model::Sentences Model::translate(Batch &batch) {
  Tensor &indices = batch.indices();
  Tensor &mask = batch.mask();

  // uint64_t batch_size = indices.dim(-2);
  // uint64_t sequence_length = indices.dim(-1);
  // uint64_t embed_dim = embedding_.dim(-1);

  Tensor word_embedding = index_select(embedding_, indices, "word_embedding");
  transform_embedding(word_embedding);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L570
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L133
  modify_mask_for_pad_tokens_in_attention(mask.data<float>(), mask.size());

  auto [x, attn] = encoder_[0].forward(word_embedding, mask);

  for (size_t i = 1; i < encoder_.size(); i++) {
    EncoderLayer &layer = encoder_[i];
    auto [y, attn_y] = layer.forward(x, mask);

    // VERIFY_MATCH(y,
    // "var_142-LayerNormalizationOp-float32_1x2x4x256-lhs.bin");

    // Overwriting x so that x is destroyed and we need lesser working memory.
    x = std::move(y);
  }

  Tensor &encoder_out = x;
  VERIFY_MATCH(encoder_out,
               "var_394-LayerNormalizationOp-float32_1x2x4x256-lhs.bin");
  return decoder_.decode(encoder_out, mask, batch.words());
}

Vocabulary::Words Decoder::greedy_sample(Tensor &logits,
                                         const Shortlist::Words &words,
                                         size_t batch_size) {
  Vocabulary::Words sampled_words;
  for (size_t i = 0; i < batch_size; i++) {
    auto *data = logits.data<float>();
    size_t max_index = 0;
    float max_value = data[0];
    size_t stride = words.size();
    for (size_t cls = 1; cls < stride; cls++) {
      float value = data[i * stride + cls];
      if (value > max_value) {
        max_index = cls;
        max_value = value;
      }
    }

    sampled_words.push_back(words[max_index]);
  }
  return sampled_words;
}

Decoder::Sentences Decoder::decode(Tensor &encoder_out, Tensor &mask,
                                   const Words &source) {
  // Prepare a shortlist for the entire batch.
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);

  Shortlist shortlist = shortlist_generator_.generate(source);

  std::vector<bool> complete(batch_size, false);
  auto record = [&complete](Vocabulary::Words &step,
                            Decoder::Sentences &sentences) {
    size_t finished = 0;
    for (size_t i = 0; i < step.size(); i++) {
      if (not complete[i]) {
        sentences[i].push_back(step[i]);
        complete[i] = (step[i] == 0);
      }
      finished += static_cast<int>(complete[i]);
    }
    return sentences.size() - finished;
  };

  // Initialize a first step.
  Decoder::Sentences sentences(batch_size);

  Vocabulary::Words previous_slice = {};
  set_start_state(batch_size);
  Tensor decoder_out = step(encoder_out, mask, previous_slice);

  Tensor logits =
      affine_with_select(output_, decoder_out, shortlist.words(), "logits");

  previous_slice = greedy_sample(logits, shortlist.words(), batch_size);
  record(previous_slice, sentences);

  size_t remaining = sentences.size();
  size_t max_seq_length = 1.5 * source_sequence_length;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    Tensor decoder_out = step(encoder_out, mask, previous_slice);

    Tensor logits =
        affine_with_select(output_, decoder_out, shortlist.words(), "logits");

    previous_slice = greedy_sample(logits, shortlist.words(), batch_size);
    remaining = record(previous_slice, sentences);
  }

  return sentences;
}

Model::Model(Tag tag, std::vector<io::Item> &&items,
             ShortlistGenerator &&shortlist_generator)
    : tag_(tag),
      items_(std::move(items)),
      decoder_(                                //
          Config::tiny11::decoder_layers,      //
          Config::tiny11::feed_forward_depth,  //
          embedding_,                          //
          std::move(shortlist_generator)       //
      ) {
  for (size_t i = 0; i < Config::tiny11::encoder_layers; i++) {
    encoder_.emplace_back(i + 1, Config::tiny11::feed_forward_depth);
  }

  load_parameters_from_items();
  (void)tag_;  // Apparently tag not used. This should fix.
}

void SSRU::set_start_state(size_t batch_size) {
  // auto start = graph->constant({1, 1, dimBatch, dim}, inits::zeros());
  size_t feature_dim = O_.W.dim(-1);
  Tensor start(Type::f32, Shape({batch_size, feature_dim}), "start");
  start.fill_in_place(0.0F);
  state_ = std::move(start);
}

Tensor SSRU::forward(Tensor &x) {
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

  // f(t) = σ(Wt . x(t) + bf )

  Tensor &c = state_;  // Load context from saved-state.

  Tensor f_out = affine(F_, x, "rnn_f");  // Forget gate?
  Tensor f = sigmoid(f_out);

  // c(t) = f(t) ⊙  c(t−1) + (1 − ft) ⊙  Wx(t)
  Tensor Wxt = linear(O_, x, "rnn_o");  // NOLINT

  Tensor ones = f.like("ones");
  ones.fill_in_place(1.0F);

  Tensor g = sub(ones, f);
  Tensor c_arg1 = mul(f, c);
  Tensor c_arg2 = mul(g, Wxt);
  Tensor c_next = add(c_arg1, c_arg2);

  // y(t) = ReLU(c(t));
  Tensor y = relu(c_next);

  // h(t) = α LayerNorm(y(t) + x(t)) + β
  Tensor o = add(x, y);

  Tensor h = ln_.forward(o);

  state_ = std::move(c_next);

  return h;
}

std::tuple<Tensor, Tensor> DecoderLayer::forward(Tensor &encoder_out,
                                                 Tensor &mask, Tensor &x) {
  Tensor decoder_out = rnn_.forward(x);

  // Assign query, key, value for cross-attention.
  Tensor &q = decoder_out;
  Tensor &k = encoder_out;
  Tensor &v = encoder_out;

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

Decoder::Decoder(size_t decoders, size_t ffn_count, Tensor &embedding,
                 ShortlistGenerator &&shortlist_generator)
    : embedding_(embedding),
      shortlist_generator_(std::move(shortlist_generator)) {
  for (size_t i = 0; i < decoders; i++) {
    decoder_.emplace_back(i + 1, ffn_count);
  }
}

void Decoder::register_parameters(const std::string &prefix,
                                  ParameterMap &parameters) {
  parameters.emplace("Wemb_intgemm8", &output_.W);
  parameters.emplace("Wemb_QuantMultA", &output_.quant);
  parameters.emplace("decoder_ff_logit_out_b", &output_.b);
  for (DecoderLayer &layer : decoder_) {
    layer.register_parameters(prefix, parameters);
  }
}

Tensor Decoder::step(Tensor &encoder_out, Tensor &mask,
                     Decoder::Words &previous_step) {
  // Infer batch-size from encoder_out.
  size_t encoder_feature_dim = encoder_out.dim(-1);
  size_t source_sequence_length = encoder_out.dim(-2);
  size_t batch_size = encoder_out.dim(-3);

  (void)encoder_feature_dim;
  (void)source_sequence_length;

  // Trying to re-imagine:
  // https://github.com/browsermt/marian-dev/blob/f436b2b7528927333da1629a74fde3779c0a96dd/src/models/decoder.h#L67
  auto from_sentences = [this](Decoder::Words &previous_step,
                               size_t batch_size) {
    const std::string name = "target_embed";
    size_t embed_dim = embedding_.dim(-1);

    // If no words, generate one embedding with all 0s.
    if (previous_step.empty()) {
      size_t sequence_length = 1;
      Shape shape({batch_size, sequence_length, embed_dim});
      Tensor empty_embed(Type::f32, std::move(shape), name);
      empty_embed.fill_in_place(0.0F);

      Tensor mask(Type::f32, Shape({batch_size, sequence_length}),
                  "decoder_start_mask");
      mask.fill_in_place(0.0F);
      return std::make_tuple(std::move(empty_embed), std::move(mask));
    }

    size_t sequence_length = 1;
    Shape shape({batch_size, sequence_length});
    // Maybe move this to some new construct?
    Tensor indices(Type::i32, std::move(shape), name);
    int *data = indices.data<int>();
    for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
      data[batch_id] = previous_step[batch_id];
    }

    Tensor embedding = index_select(embedding_, indices);
    Tensor mask(Type::f32, Shape({batch_size, sequence_length}),
                "decoder_mask");
    return std::make_tuple(std::move(embedding), std::move(mask));
  };

  auto [decoder_embed, decoder_mask] =
      from_sentences(previous_step, batch_size);

  modify_mask_for_pad_tokens_in_attention(decoder_mask.data<float>(),
                                          decoder_mask.size());

  transform_embedding(decoder_embed);

  VERIFY_MATCH(encoder_out,
               "var_394-LayerNormalizationOp-float32_1x2x4x256-lhs.bin");

  auto [x, attn] = decoder_[0].forward(encoder_out, mask, decoder_embed);
  for (size_t i = 1; i < decoder_.size(); i++) {
    auto [y, _attn] = decoder_[i].forward(encoder_out, mask, x);
    x = std::move(y);
  }

  return std::move(x);
}

void Model::load_parameters_from_items() {
  // Get the parameterss from strings to target tensors to load.
  ParameterMap parameters;
  std::string prefix;
  register_parameters(prefix, parameters);

  auto debug = [&parameters]() {
    for (const auto &p : parameters) {
      std::cout << p.first << "\n";
    }
  };

  (void)debug;

  auto lookup = [&parameters](const std::string &name) -> Tensor * {
    auto query = parameters.find(name);
    if (query == parameters.end()) {
      return nullptr;
    }
    return query->second;
  };

  std::vector<std::string> missed;
  for (io::Item &item : items_) {
    Tensor *target = lookup(item.name);
    if (target) {
      // std::cerr << "Loading " << item << "\n";
      target->load(item.view, item.type, item.shape, item.name);
    } else {
      missed.push_back(item.name);
    }
  }

  for (std::string &entry : missed) {
    (void)entry;
    // std::cerr << "Missed " << entry << "\n";
  }
}

void Model::register_parameters(const std::string &prefix,
                                ParameterMap &parameters) {
  parameters.emplace("Wemb", &embedding_);
  for (EncoderLayer &layer : encoder_) {
    layer.register_parameters(prefix, parameters);
  }
  decoder_.register_parameters(prefix, parameters);
}

EncoderLayer::EncoderLayer(size_t depth, size_t ffn_count)
    : depth_(depth), attention_("self") {
  for (size_t i = 0; i < ffn_count; i++) {
    ffn_.emplace_back(i + 1);
  }
}

DecoderLayer::DecoderLayer(size_t depth, size_t ffn_count)
    : depth_(depth), attention_("context") {
  for (size_t i = 0; i < ffn_count; i++) {
    ffn_.emplace_back(i + 1);
  }
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

Attention::Attention(std::string name) : name_(std::move(name)) {}

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

FFN::FFN(size_t depth) : depth_(depth) {}

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
