#pragma once
#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Tensor.hh"

namespace slimt {

class Encoder {
 public:
  explicit Encoder(size_t layers, size_t num_heads, size_t feed_forward_depth);
  Tensor forward(Tensor &embedding, Tensor &mask);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);

 private:
  std::vector<EncoderLayer> encoder_;
};

class Decoder {
 public:
  Decoder(size_t layers, size_t num_heads, size_t feed_forward_depth,
          Tensor &embedding);

  void register_parameters(const std::string &prefix, ParameterMap &parameters);

  std::vector<Tensor> start_states(size_t batch_size);
  std::tuple<Tensor, Tensor> step(Tensor &encoder_out, Tensor &mask,
                                  std::vector<Tensor> &states,
                                  Words &previous_step, Words &shortlist);

 private:
  Tensor &embedding_;
  std::vector<DecoderLayer> decoder_;
  Affine output_;
};

Words greedy_sample(Tensor &logits, const Words &words, size_t batch_size);
void transform_embedding(Tensor &word_embedding, size_t start = 0);

class Transformer {
 public:
  explicit Transformer(size_t encoder_layers, size_t decoder_layers,
                       size_t num_heads, size_t feed_forward_depth, View model);

  Tensor &embedding() { return embedding_; }
  Encoder &encoder() { return encoder_; }
  Decoder &decoder() { return decoder_; }

 private:
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void load_parameters();

  std::vector<io::Item> items_;
  Tensor embedding_;
  Encoder encoder_;
  Decoder decoder_;
};

}  // namespace slimt