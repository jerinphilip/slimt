#pragma once
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Tensor.hh"
#include "slimt/Types.hh"

namespace slimt {

class Encoder {
 public:
  explicit Encoder(size_t layers, size_t num_heads, size_t feed_forward_depth);
  Tensor forward(const Tensor &embedding, const Tensor &mask) const;
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  const std::vector<EncoderLayer> &encoder() const { return encoder_; }

 private:
  std::vector<EncoderLayer> encoder_;
};

class Decoder {
 public:
  Decoder(size_t layers, size_t num_heads, size_t feed_forward_depth,
          const Tensor &embedding);

  void register_parameters(const std::string &prefix, ParameterMap &parameters);

  std::vector<Tensor> start_states(size_t batch_size) const;
  std::tuple<Tensor, Tensor> step(const Tensor &encoder_out, const Tensor &mask,
                                  std::vector<Tensor> &states,
                                  const Words &previous_step,
                                  const std::optional<Words> &shortlist) const;

 private:
  const Tensor &embedding_;
  std::vector<DecoderLayer> decoder_;
  Affine output_;
};

class Vocabulary;

Words greedy_sample(const Tensor &logits, const Vocabulary &vocabulary,
                    size_t batch_size);
Words greedy_sample_from_words(const Tensor &logits,
                               const Vocabulary &vocabulary, const Words &words,
                               size_t batch_size);

void transform_embedding(Tensor &word_embedding, size_t start = 0);

class Transformer {
 public:
  explicit Transformer(size_t encoder_layers, size_t decoder_layers,
                       size_t num_heads, size_t feed_forward_depth, View model);

  const Tensor &embedding() const { return embedding_; }
  const Encoder &encoder() const { return encoder_; }
  const Decoder &decoder() const { return decoder_; }

  // Relay calls.
  Tensor encode(const Tensor &embedding, const Tensor &mask) const;
  std::vector<Tensor> start_states(size_t batch_size) const;
  std::tuple<Tensor, Tensor> step(const Tensor &encoder_out, const Tensor &mask,
                                  std::vector<Tensor> &states,
                                  const Words &previous_step,
                                  const std::optional<Words> &shortlist) const;

 private:
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void load_parameters();

  std::vector<io::Item> items_;
  Tensor embedding_;
  Encoder encoder_;
  Decoder decoder_;
};

}  // namespace slimt
