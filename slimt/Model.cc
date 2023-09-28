#include "slimt/Model.hh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

#include "slimt/Modules.hh"
#include "slimt/QMM.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Utils.hh"

namespace slimt {

void transform_embedding(Tensor &word_embedding, size_t start /* = 0*/) {
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

Encoder::Encoder(const Config &config) {
  for (size_t i = 0; i < config.encoder_layers; i++) {
    encoder_.emplace_back(i + 1, config.feed_forward_depth,
                          config.attention_num_heads);
  }
}

Tensor Encoder::forward(Tensor &word_embedding, Tensor &mask) {
  auto [x, attn] = encoder_[0].forward(word_embedding, mask);

  for (size_t i = 1; i < encoder_.size(); i++) {
    EncoderLayer &layer = encoder_[i];
    auto [y, attn_y] = layer.forward(x, mask);

    // Overwriting x so that x is destroyed and we need lesser working memory.
    x = std::move(y);
  }
  return std::move(x);
}

void Encoder::register_parameters(const std::string &prefix,
                                  ParameterMap &parameters) {
  for (EncoderLayer &layer : encoder_) {
    layer.register_parameters(prefix, parameters);
  }
}

std::vector<Tensor> Decoder::start_states(size_t batch_size) {
  std::vector<Tensor> states;
  for (auto &layer : decoder_) {
    Tensor state = layer.start_state(batch_size);
    states.push_back(std::move(state));
  }
  return states;
}

Model::Model(const Config &config, View model)
    : items_(io::loadItems(model.data)),
      encoder_(config),  //
      decoder_(config, embedding_) {
  load_parameters();
}

Decoder::Decoder(const Config &config, Tensor &embedding)
    : embedding_(embedding) {
  for (size_t i = 0; i < config.decoder_layers; i++) {
    decoder_.emplace_back(i + 1, config.feed_forward_depth,
                          config.attention_num_heads);
  }
}

void Decoder::register_parameters(const std::string &prefix,
                                  ParameterMap &parameters) {
  // Somehow we have historically ended up with `none_QuantMultA` being used for
  // Wemb_QuantMultA.
  parameters.emplace("Wemb_intgemm8", &output_.W);
  parameters.emplace("none_QuantMultA", &output_.quant);
  parameters.emplace("decoder_ff_logit_out_b", &output_.b);

  for (DecoderLayer &layer : decoder_) {
    layer.register_parameters(prefix, parameters);
  }
}

std::tuple<Tensor, Tensor> Decoder::step(Tensor &encoder_out, Tensor &mask,
                                         std::vector<Tensor> &states,
                                         Words &previous_step,
                                         Words &shortlist) {
  // Infer batch-size from encoder_out.
  size_t encoder_feature_dim = encoder_out.dim(-1);
  size_t source_sequence_length = encoder_out.dim(-2);
  size_t batch_size = encoder_out.dim(-3);

  (void)encoder_feature_dim;
  (void)source_sequence_length;

  // Trying to re-imagine:
  // https://github.com/browsermt/marian-dev/blob/f436b2b7528927333da1629a74fde3779c0a96dd/src/models/decoder.h#L67
  auto from_sentences = [this](Words &previous_step, size_t batch_size) {
    const std::string name = "target_embed";
    size_t embed_dim = embedding_.dim(-1);

    // If no words, generate one embedding with all 0s.
    if (previous_step.empty()) {
      size_t sequence_length = 1;
      Shape shape({batch_size, sequence_length, embed_dim});
      Tensor empty_embed(Type::f32, std::move(shape), name);
      empty_embed.fill_in_place(0.0F);
      return empty_embed;
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
    return embedding;
  };

  Tensor decoder_embed = from_sentences(previous_step, batch_size);
  transform_embedding(decoder_embed);

  auto [x, attn] =
      decoder_[0].forward(encoder_out, mask, states[0], decoder_embed);
  for (size_t i = 1; i < decoder_.size(); i++) {
    auto [y, _attn] = decoder_[i].forward(encoder_out, mask, states[i], x);
    x = std::move(y);
  }

  Tensor logits = affine_with_select(output_, x, shortlist, "logits");
  return {std::move(logits), std::move(attn)};
}

void Model::load_parameters() {
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
      target->load(item.view, item.type, item.shape, item.name);
      parameters.erase(item.name);
    } else {
      missed.push_back(item.name);
    }
  }

  for (std::string &entry : missed) {
    std::cerr << "[warn] Failed to ingest expected load of " << entry << "\n";
  }
  for (auto &parameter : parameters) {
    std::cerr << "[warn] Failed to complete expected load of ";
    std::cerr << parameter.first << "\n";
  }
}

void Model::register_parameters(const std::string &prefix,
                                ParameterMap &parameters) {
  parameters.emplace("Wemb", &embedding_);
  encoder_.register_parameters(prefix, parameters);
  decoder_.register_parameters(prefix, parameters);
}

Words greedy_sample(Tensor &logits, const Words &words, size_t batch_size) {
  Words sampled_words;
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

}  // namespace slimt
