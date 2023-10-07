
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
#include "slimt/TextProcessor.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

class Batch;
class Tensor;
struct View;

class Blocking {
 public:
  explicit Blocking(const Config &config);
  Response translate(Ptr<Model> &model, std::string source,
                     const Options &options);

 private:
  Config config_;
  std::optional<TranslationCache> cache_;

  size_t id_ = 0;
  size_t model_id_ = 0;
};

class Async {
 public:
  explicit Async(const Config &config);
  ~Async();

  std::future<Response> translate(Ptr<Model> &model, std::string source,
                                  const Options &options);

 private:
  Config config_;
  std::optional<TranslationCache> cache_;
  rd::Threadsafe<rd::AggregateBatcher> batcher_;
  std::vector<std::thread> workers_;

  size_t id_ = 0;
  size_t model_id_ = 0;
};

}  // namespace slimt
