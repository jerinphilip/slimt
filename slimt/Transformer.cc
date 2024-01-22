#include "slimt/Transformer.hh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"
#include "slimt/Vocabulary.hh"

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

Encoder::Encoder(size_t layers, size_t num_heads, size_t feed_forward_depth) {
  for (size_t i = 0; i < layers; i++) {
    encoder_.emplace_back(i + 1, feed_forward_depth, num_heads);
  }
}

Tensor Encoder::forward(const Tensor &word_embedding,
                        const Tensor &mask) const {
  auto [x, attn] = encoder()[0].forward(word_embedding, mask);

  for (size_t i = 1; i < encoder_.size(); i++) {
    const EncoderLayer &layer = encoder()[i];
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

std::vector<Tensor> Decoder::start_states(size_t batch_size) const {
  std::vector<Tensor> states;
  for (const auto &layer : decoder_) {
    Tensor state = layer.start_state(batch_size);
    states.push_back(std::move(state));
  }
  return states;
}

Transformer::Transformer(size_t encoder_layers, size_t decoder_layers,
                         size_t num_heads, size_t feed_forward_depth,
                         View model)
    : items_(io::load_items(model.data)),
      encoder_(encoder_layers, num_heads, feed_forward_depth),  //
      decoder_(decoder_layers, num_heads, feed_forward_depth, embedding_) {
  load_parameters();
}

Decoder::Decoder(size_t layers, size_t num_heads, size_t feed_forward_depth,
                 const Tensor &embedding)
    : embedding_(embedding) {
  for (size_t i = 0; i < layers; i++) {
    decoder_.emplace_back(i + 1, feed_forward_depth, num_heads);
  }
}

void Decoder::register_parameters(const std::string &prefix,
                                  ParameterMap &parameters) {
  // Somehow we have historically ended up with `none_QuantMultA` being used for
  // Wemb_QuantMultA.
  // https://github.com/browsermt/marian-dev/blob/2be8344fcf2776fb43a7376284067164674cbfaf/scripts/alphas/extract_stats.py#L55
  // - none_QuantMultA is generated when used with shortlist
  // - Wemb_QuantMultA is generated when used without shortlist.
  parameters.emplace("Wemb_intgemm8", &output_.W);
  parameters.emplace("none_QuantMultA", &output_.quant);
  parameters.emplace("decoder_ff_logit_out_b", &output_.b);

  for (DecoderLayer &layer : decoder_) {
    layer.register_parameters(prefix, parameters);
  }
}

std::tuple<Tensor, Tensor> Decoder::step(
    const Tensor &encoder_out, const Tensor &mask, std::vector<Tensor> &states,
    const Words &previous_step, const std::optional<Words> &shortlist) const {
  // Infer batch-size from encoder_out.
  size_t encoder_feature_dim = encoder_out.dim(-1);
  size_t source_sequence_length = encoder_out.dim(-2);
  size_t batch_size = encoder_out.dim(-3);

  (void)encoder_feature_dim;
  (void)source_sequence_length;

  // Trying to re-imagine:
  // https://github.com/browsermt/marian-dev/blob/f436b2b7528927333da1629a74fde3779c0a96dd/src/models/decoder.h#L67
  auto from_sentences = [this](const Words &previous_step, size_t batch_size) {
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

  Tensor guided_alignment = std::move(attn);
  for (size_t i = 1; i < decoder_.size(); i++) {
    auto [y, _attn] = decoder_[i].forward(encoder_out, mask, states[i], x);
    x = std::move(y);
    if (i + 1 == decoder_.size()) {
      // Last decoder layer
      // https://github.com/marian-nmt/marian-dev/blob/53b0b0d7c83e71265fee0dd832ab3bcb389c6ec3/src/models/transformer.h#L826C31-L826C41
      guided_alignment = std::move(_attn);
    }
  }

  if (shortlist) {
    Tensor logits = affine_with_select(output_, x, *shortlist, "logits");
    return {std::move(logits), std::move(guided_alignment)};
  }

  Tensor logits = affine(output_, x, "logits");
  return {std::move(logits), std::move(guided_alignment)};
}

void Transformer::load_parameters() {
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

void Transformer::register_parameters(const std::string &prefix,
                                      ParameterMap &parameters) {
  parameters.emplace("Wemb", &embedding_);
  encoder_.register_parameters(prefix, parameters);
  decoder_.register_parameters(prefix, parameters);
}

namespace {

template <class T>
void topk_inspect(size_t batch_id, const Vocabulary &vocabulary, T *begin,
                  T *end, size_t k) {
  const T *data = begin;
  size_t size = end - begin;

  std::vector<size_t> ordering = argsort(begin, end);
  fprintf(stderr, "batch %zu | ", batch_id);
  Words words(size + 1, vocabulary.eos_id());
  for (size_t i = 0; i < k; i++) {
    size_t j = size - i - 1;
    words[i] = ordering[j];
    std::string decoded;
    vocabulary.decode({words[i], vocabulary.eos_id()}, decoded);
    fprintf(stderr, "%s (%zu, %.9g) ", decoded.c_str(), ordering[j],
            data[ordering[j]]);
  }
  fprintf(stderr, "\n");
}

template <class T>
void topk_inspect_with_words(size_t batch_id, const Vocabulary &vocabulary,
                             const Words &shortlist, T *begin, T *end,
                             size_t k) {
  const T *data = begin;
  size_t size = end - begin;

  std::vector<size_t> ordering = argsort(begin, end);
  fprintf(stderr, "batch %zu | ", batch_id);
  Words words(size + 1, vocabulary.eos_id());
  for (size_t i = 0; i < k; i++) {
    size_t j = size - i - 1;
    words[i] = shortlist[ordering[j]];
    std::string decoded;
    vocabulary.decode({words[i], vocabulary.eos_id()}, decoded);
    fprintf(stderr, "%s (%zu, %.9g) ", decoded.c_str(), ordering[j],
            data[ordering[j]]);
  }
  fprintf(stderr, "\n");
}

}  // namespace

Words greedy_sample(const Tensor &logits, const Vocabulary &vocabulary,
                    size_t batch_size) {
  Words sampled_words;
  size_t stride = vocabulary.size();
  for (size_t i = 0; i < batch_size; i++) {
    const auto *data = logits.data<float>();

    // Initialize: 0
    size_t cls = 0;
    size_t max_index = cls;
    float max_value = data[i * stride + cls];

    for (cls = 1; cls < stride; cls++) {
      float value = data[i * stride + cls];
      if (value > max_value) {
        max_index = cls;
        max_value = value;
      }
    }

    sampled_words.push_back(max_index);
    if (false) {  // NOLINT
      constexpr size_t kValue = 5;
      topk_inspect(i, vocabulary, data + i * stride, data + (i + 1) * stride,
                   kValue);
    }
  }
  return sampled_words;
}

Words greedy_sample_from_words(const Tensor &logits,
                               const Vocabulary &vocabulary, const Words &words,
                               size_t batch_size) {
  (void)vocabulary;
  size_t stride = words.size();
  Words sampled_words;
  for (size_t i = 0; i < batch_size; i++) {
    const auto *data = logits.data<float>();

    // Initialize: 0
    size_t cls = 0;
    size_t max_index = cls;
    float max_value = data[i * stride + cls];

    for (cls = 1; cls < stride; cls++) {
      float value = data[i * stride + cls];
      if (value > max_value) {
        max_index = cls;
        max_value = value;
      }
    }

    sampled_words.push_back(words[max_index]);
    if (false) {  // NOLINT
      constexpr size_t kValue = 5;
      topk_inspect_with_words(i, vocabulary, words, data + i * stride,
                              data + (i + 1) * stride, kValue);
    }
  }
  return sampled_words;
}

}  // namespace slimt
