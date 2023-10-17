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

class Blocking {
 public:
  explicit Blocking(const Config &config);
  std::vector<Response> translate(Ptr<Model> model,
                                  std::vector<std::string> sources,
                                  const Options &options);
  std::vector<Response> pivot(Ptr<Model> first, Ptr<Model> second,
                              std::vector<std::string> sources,
                              const Options &options);

 private:
  Config config_;
  std::optional<TranslationCache> cache_;
  size_t id_ = 0;
};

class Async {
 public:
  explicit Async(const Config &config);
  ~Async();

  std::future<Response> translate(Ptr<Model> model, std::string source,
                                  const Options &options);
  std::future<Response> pivot(Ptr<Model> first, Ptr<Model> second,
                              std::string source, const Options &options);

 private:
  Config config_;
  std::optional<TranslationCache> cache_;
  rd::Threadsafe<rd::AggregateBatcher> batcher_;
  std::vector<std::thread> workers_;

  size_t id_ = 0;
};

}  // namespace slimt
