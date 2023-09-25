
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
  explicit Translator(Data data, Config config);
  Histories decode(Tensor &encoder_out, Tensor &mask, const Words &source);
  Histories forward(Batch &batch);
  Response translate(std::string source, const Options &options);

 private:
  Config config_;
  TextProcessor processor_;
  Vocabulary vocabulary_;
  ShortlistGenerator shortlist_generator_;

  // Model related stuff.
  Model model_;
  size_t id_ = 0;
  size_t model_id_ = 0;
};

}  // namespace slimt
