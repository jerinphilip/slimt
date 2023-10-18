#pragma once
#include <cstddef>
#include <future>
#include <optional>
#include <string>
#include <vector>

#include "slimt/Batcher.hh"
#include "slimt/Cache.hh"
#include "slimt/Config.hh"
#include "slimt/Response.hh"
#include "slimt/Types.hh"

namespace slimt {

class Model;

class Blocking {
 public:
  explicit Blocking(const Config &config);
  std::vector<Response> translate(const Ptr<Model> &model,
                                  std::vector<std::string> sources,
                                  const Options &options);
  std::vector<Response> pivot(const Ptr<Model> &first, const Ptr<Model> &second,
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

  std::future<Response> translate(const Ptr<Model> &model, std::string source,
                                  const Options &options);
  std::future<Response> pivot(const Ptr<Model> &first, const Ptr<Model> &second,
                              std::string source, const Options &options);

 private:
  Config config_;
  std::optional<TranslationCache> cache_;
  rd::Threadsafe<rd::AggregateBatcher> batcher_;
  std::vector<std::thread> workers_;

  size_t id_ = 0;
};

}  // namespace slimt
