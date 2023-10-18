#pragma once
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "slimt/Batch.hh"
#include "slimt/Config.hh"
#include "slimt/Modules.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
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
