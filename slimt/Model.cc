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

Sentences Model::translate(Batch &batch) {
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

    // Overwriting x so that x is destroyed and we need lesser working memory.
    x = std::move(y);
  }

  Tensor &encoder_out = x;
  return decoder_.decode(encoder_out, mask, batch.words());
}

Words Decoder::greedy_sample(Tensor &logits, const Words &words,
                             size_t batch_size) {
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

Sentences Decoder::decode(Tensor &encoder_out, Tensor &mask,
                          const Words &source) {
  // Prepare a shortlist for the entire batch.
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);

  Shortlist shortlist = shortlist_generator_.generate(source);
  const auto &indices = shortlist.words();
  // The following can be used to check if shortlist is going wrong.
  // std::vector<uint32_t> indices(vocabulary_.size());
  // std::iota(indices.begin(), indices.end(), 0);

  std::vector<bool> complete(batch_size, false);
  uint32_t eos = vocabulary_.eos_id();
  auto record = [eos, &complete](Words &step, Sentences &sentences) {
    size_t finished = 0;
    for (size_t i = 0; i < step.size(); i++) {
      if (not complete[i]) {
        complete[i] = (step[i] == eos);
        sentences[i].push_back(step[i]);
      }
      finished += static_cast<int>(complete[i]);
    }
    return sentences.size() - finished;
  };

  // Initialize a first step.
  Sentences sentences(batch_size);

  Words previous_slice = {};
  std::vector<Tensor> states = start_states(batch_size);
  Tensor decoder_out = step(encoder_out, mask, states, previous_slice);

  Tensor logits = affine_with_select(output_, decoder_out, indices, "logits");

  previous_slice = greedy_sample(logits, indices, batch_size);
  record(previous_slice, sentences);

  size_t remaining = sentences.size();
  size_t max_seq_length = max_target_length_factor_ * source_sequence_length;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    Tensor decoder_out = step(encoder_out, mask, states, previous_slice);

    Tensor logits = affine_with_select(output_, decoder_out, indices, "logits");

    previous_slice = greedy_sample(logits, indices, batch_size);
    remaining = record(previous_slice, sentences);
  }

  return sentences;
}

std::vector<Tensor> Decoder::start_states(size_t batch_size) {
  std::vector<Tensor> states;
  for (auto &layer : decoder_) {
    Tensor state = layer.start_state(batch_size);
    states.push_back(std::move(state));
  }
  return states;
}

Model::Model(Tag tag, Vocabulary &vocabulary, std::vector<io::Item> &&items,
             ShortlistGenerator &&shortlist_generator)
    : tag_(tag),
      items_(std::move(items)),
      decoder_(                                //
          Config::tiny11::decoder_layers,      //
          Config::tiny11::feed_forward_depth,  //
          vocabulary,                          //
          embedding_,                          //
          std::move(shortlist_generator)       //
      ) {
  for (size_t i = 0; i < Config::tiny11::encoder_layers; i++) {
    encoder_.emplace_back(i + 1, Config::tiny11::feed_forward_depth);
  }

  load_parameters_from_items();
  (void)tag_;  // Apparently tag not used. This should fix.
}

Decoder::Decoder(size_t decoders, size_t ffn_count, Vocabulary &vocabulary,
                 Tensor &embedding, ShortlistGenerator &&shortlist_generator)
    : vocabulary_(vocabulary),
      embedding_(embedding),
      shortlist_generator_(std::move(shortlist_generator)) {
  for (size_t i = 0; i < decoders; i++) {
    decoder_.emplace_back(i + 1, ffn_count);
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

Tensor Decoder::step(Tensor &encoder_out, Tensor &mask,
                     std::vector<Tensor> &states, Words &previous_step) {
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
      target->load(item.view, item.type, item.shape, item.name);
      parameters.erase(item.name);
    } else {
      missed.push_back(item.name);
    }
  }

  for (std::string &entry : missed) {
    std::cerr << "Failed to ingest expected load of " << entry << "\n";
  }
  for (auto &parameter : parameters) {
    std::cerr << "Failed to complete expected load of ";
    std::cerr << parameter.first << "\n";
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

}  // namespace slimt
