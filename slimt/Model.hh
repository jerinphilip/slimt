#pragma once
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "slimt/Batch.hh"
#include "slimt/Config.hh"
#include "slimt/Shortlist.hh"
#include "slimt/TextProcessor.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

template <class Field>
struct Package {
  Field model;
  Field vocabulary;
  Field shortlist;
};

class Model {
 public:
  struct Config {
    size_t encoder_layers = 6;
    size_t decoder_layers = 2;
    size_t feed_forward_depth = 2;
    size_t num_heads = 8;
    std::string split_mode = "sentence";
    template <class App>
    void setup_onto(App &app) {
      // clang-format off
      app.add_option("--encoder-layers", encoder_layers, "Number of encoder layers");
      app.add_option("--decoder-layers", decoder_layers, "Number of decoder layers");
      app.add_option("--num-heads", num_heads, "Number of decoder layers");
      app.add_option("--ffn-depth", decoder_layers, "Number of feedforward layers");
      app.add_option("--split-mode", split_mode, "Split mode to go with for sentence-splitter.");
      // clang-format on
    }
  };

  explicit Model(const Config &config, const Package<std::string> &package);
  explicit Model(const Config &config, const Package<View> &package);

  Histories forward(Batch &batch);

  Config &config() { return config_; }
  Vocabulary &vocabulary() { return vocabulary_; }
  TextProcessor &processor() { return processor_; }
  Transformer &transformer() { return transformer_; }
  size_t id() const { return id_; }  // NOLINT
  ShortlistGenerator &shortlist_generator() { return shortlist_generator_; }

 private:
  Histories decode(Tensor &encoder_out, Batch &batch);

  size_t id_;
  Config config_;
  using Mmap = Package<io::MmapFile>;
  std::optional<Mmap> mmap_;
  Package<View> view_;

  Vocabulary vocabulary_;
  TextProcessor processor_;
  Transformer transformer_;
  ShortlistGenerator shortlist_generator_;
};

}  // namespace slimt
