
#pragma once
#include <cstddef>
#include <future>
#include <optional>
#include <string>
#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/Batcher.hh"
#include "slimt/Model.hh"
#include "slimt/Response.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Splitter.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {
class Batch;
class Tensor;
struct View;

struct Data {
  Aligned model;
  Aligned shortlist;
  Aligned vocabulary;
};

class Translator {
 public:
  Translator(const Config &config, View model, View shortlist, View vocabulary);
  Histories decode(Tensor &encoder_out, Tensor &mask, const Words &source,
                   const std::vector<size_t> &lengths);
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

class Async {
 public:
  Async(const Config &config, View model, View shortlist, View vocabulary);
  Response translate(std::string &source, const Options &options);

 private:
  Config config_;
  Vocabulary vocabulary_;
  TextProcessor processor_;
  Model model_;
  ShortlistGenerator shortlist_generator_;
  std::optional<TranslationCache> cache_;
  rd::Threadsafe<rd::Batcher> batcher_;
  Histories forward(Batch &batch);
  Histories decode(Tensor &encoder_out, Tensor &mask, const Words &source,
                   const std::vector<size_t> &lengths);
  std::vector<std::thread> workers_;

  size_t id_ = 0;

  size_t model_id_ = 0;
};

}  // namespace slimt
