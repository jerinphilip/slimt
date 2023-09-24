
#pragma once
#include "slimt/Model.hh"
#include "slimt/Response.hh"
#include "slimt/Splitter.hh"

namespace slimt {

struct Data {
  Aligned model;
  Aligned shortlist;
  Aligned vocabulary;
};

class Translator {
 public:
  struct Config {
    size_t max_words;
    size_t wrap_length;
    size_t tgt_length_limit_factor;
  };

  explicit Translator(Data data, Config config);
  Responses translate(std::vector<std::string> sources, const Options &options);

 private:
  TextProcessor processor_;
  Vocabulary vocabulary_;

  // Model related stuff.
  std::vector<io::Item> items_;
  Tensor embedding_;
  std::vector<EncoderLayer> encoder_;
  Decoder decoder_;
};

}  // namespace slimt
