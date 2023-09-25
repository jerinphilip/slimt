#pragma once
#include <unordered_map>

#include "slimt/Batch.hh"
#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

struct Config {
  // NOLINTBEGIN
  size_t encoder_layers = 6;
  size_t decoder_layers = 2;
  size_t feed_forward_depth = 2;
  size_t tgt_length_limit_factor = 2;
  size_t attention_num_heads = 8;

  size_t max_words = 1024;
  size_t wrap_length = 128;
  // NOLINTEND
};

class Encoder {
 public:
  explicit Encoder(const Config &config);
  Tensor forward(Tensor &embedding, Tensor &mask);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);

 private:
  std::vector<EncoderLayer> encoder_;
};

class Decoder {
 public:
  Decoder(const Config &config, Vocabulary &vocabulary, Tensor &embedding,
          ShortlistGenerator &&shortlist_generator);

  void register_parameters(const std::string &prefix, ParameterMap &parameters);

  std::vector<Tensor> start_states(size_t batch_size);
  Tensor step(Tensor &encoder_out, Tensor &mask, std::vector<Tensor> &states,
              Words &previous_step, Words &shortlist);

 private:
  float tgt_length_limit_factor_;
  Vocabulary &vocabulary_;

  Tensor &embedding_;
  std::vector<DecoderLayer> decoder_;
  Affine output_;

  ShortlistGenerator shortlist_generator_;
};

Words greedy_sample(Tensor &logits, const Words &words, size_t batch_size);
void transform_embedding(Tensor &word_embedding, size_t start = 0);

class Model {
 public:
  explicit Model(const Config &config, Vocabulary &vocabulary,
                 io::Items &&items, ShortlistGenerator &&shortlist_generator);

  Config &config() { return config_; }
  Tensor &embedding() { return embedding_; }
  Encoder &encoder() { return encoder_; }
  Decoder &decoder() { return decoder_; }

 private:
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void load_parameters();

  Config config_;
  std::vector<io::Item> items_;
  Tensor embedding_;
  Encoder encoder_;
  Decoder decoder_;
};

}  // namespace slimt
