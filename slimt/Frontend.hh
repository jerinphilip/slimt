
#pragma once
#include <optional>

#include "slimt/Model.hh"
#include "slimt/Response.hh"
#include "slimt/Splitter.hh"
#include "slimt/Types.hh"

namespace slimt {

struct Data {
  Aligned model;
  Aligned shortlist;
  Aligned vocabulary;
};

class Translator {
 public:
  Translator(const Config &config, View model, View shortlist, View vocabulary);
  Histories decode(Tensor &encoder_out, Tensor &mask, const Words &source);
  Histories forward(Batch &batch);
  Response translate(std::string source, const Options &options);

 private:
  Config config_;
  Vocabulary vocabulary_;
  TextProcessor processor_;
  Model model_;
  ShortlistGenerator shortlist_generator_;
  std::optional<TranslationCache> cache_;

  size_t id_ = 0;
  size_t model_id_ = 0;
};

}  // namespace slimt