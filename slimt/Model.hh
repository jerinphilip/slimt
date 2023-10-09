#pragma once
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "slimt/Batch.hh"
#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/TextProcessor.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

struct Config {
  // NOLINTBEGIN
  size_t encoder_layers = 6;
  size_t decoder_layers = 2;
  size_t feed_forward_depth = 2;
  float tgt_length_limit_factor = 1.5;
  size_t attention_num_heads = 8;

  size_t max_words = 1024;
  size_t wrap_length = 128;

  size_t cache_size = 1024;
  size_t workers = 1;

  std::string prefix_path;
  std::string split_mode = "sentence";
  // NOLINTEND

  template <class App>
  void setup_onto(App &app) {
    // clang-format off
    // app.add_option("--encoder-layers", encoder_layers, "Number of encoder layers");
    // app.add_option("--decoder-layers", decoder_layers, "Number of decoder layers");
    // app.add_option("--ffn-depth", decoder_layers, "Number of feedforward layers");
    app.add_option("--limit-tgt", tgt_length_limit_factor, "Max length proportional to source target can have.");
    app.add_option("--max-words", max_words, "Maximum words in a batch.");
    app.add_option("--wrap-length", max_words, "Maximum length allowed for a sample, beyond which hard-wrap.");
    app.add_option("--split-mode", split_mode, "Split mode to go with for sentence-splitter.");
    app.add_option("--workers", workers, "Number of workers threads to launch for translating.");
    // clang-format on
  }
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
  Decoder(const Config &config, Tensor &embedding);

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
  explicit Transformer(const Config &config, View model);

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

template <class Field>
struct Package {
  Field model;
  Field vocabulary;
  Field shortlist;
};

class Model {
 public:
  explicit Model(const Config &config, const Package<std::string> &package);
  // explicit Model(const Config &config, const Package<View> &package);
  Config &config() { return config_; }
  Vocabulary &vocabulary() { return vocabulary_; }
  TextProcessor &processor() { return processor_; }
  Transformer &model() { return model_; }
  size_t id() const { return id_; }  // NOLINT
  ShortlistGenerator &shortlist_generator() { return shortlist_generator_; }

 private:
  size_t id_;
  Config config_;
  using Mmap = Package<io::MmapFile>;
  Mmap mmap_;
  // std::optional<Mmap> mmap_;
  Package<View> view_;

  Vocabulary vocabulary_;
  TextProcessor processor_;
  Transformer model_;
  ShortlistGenerator shortlist_generator_;
};

}  // namespace slimt
