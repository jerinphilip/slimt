#pragma once
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Export.hh"
#include "slimt/Io.hh"
#include "slimt/Shortlist.hh"
#include "slimt/TextProcessor.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

class Input;
class Tensor;

template <class Field>
struct Package {
  Field model;
  Field vocabulary;
  Field shortlist;
  Field ssplit;
};

class SLIMT_EXPORT Model {
 public:
  struct SLIMT_EXPORT Config {
    // NOLINTBEGIN
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
    // NOLINTEND
  };

  explicit Model(const Config &config, const Package<std::string> &package);
  explicit Model(const Config &config, const Package<View> &package);

  Histories forward(const Input &input) const;

  const Config &config() const { return config_; }
  const Vocabulary &vocabulary() const { return vocabulary_; }
  const TextProcessor &processor() const { return processor_; }
  const Transformer &transformer() const { return transformer_; }
  size_t id() const { return id_; }  // NOLINT
  const ShortlistGenerator &shortlist_generator() const {
    return shortlist_generator_;
  }

 private:
  Histories decode(const Tensor &encoder_out, const Input &input) const;

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
