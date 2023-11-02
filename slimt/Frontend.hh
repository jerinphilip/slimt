#pragma once
#include <cstddef>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "slimt/Batcher.hh"
#include "slimt/Cache.hh"
#include "slimt/Export.hh"
#include "slimt/Response.hh"
#include "slimt/Types.hh"

namespace slimt {

class Model;
struct Options;
struct Response;

struct SLIMT_EXPORT Config {
  // NOLINTBEGIN
  size_t max_words = 1024;
  size_t cache_size = 1024;
  size_t workers = 1;
  float tgt_length_limit_factor = 1.5;
  size_t wrap_length = 128;
  // NOLINTEND

  template <class App>
  void setup_onto(App &app) {
    // clang-format off
    app.add_option("--limit-tgt", tgt_length_limit_factor, "Max length proportional to source target can have.");
    app.add_option("--max-words", max_words, "Maximum words in a batch.");
    app.add_option("--wrap-length", max_words, "Maximum length allowed for a sample, beyond which hard-wrap.");
    app.add_option("--workers", workers, "Number of workers threads to launch for translating.");
    // clang-format on
  }
};

class SLIMT_EXPORT Blocking {
 public:
  explicit Blocking(const Config &config);
  std::vector<Response> translate(const Ptr<Model> &model,
                                  std::vector<std::string> sources,
                                  const Options &options);
  std::vector<Response> pivot(const Ptr<Model> &first, const Ptr<Model> &second,
                              std::vector<std::string> sources,
                              const Options &options);

 private:
  size_t id() { return id_++; }

  Config config_;
  std::optional<TranslationCache> cache_;
  size_t id_ = 0;
};

class SLIMT_EXPORT Async {
 public:
  explicit Async(const Config &config);
  ~Async();

  std::future<Response> translate(const Ptr<Model> &model, std::string source,
                                  const Options &options);
  std::future<Response> pivot(const Ptr<Model> &first, const Ptr<Model> &second,
                              std::string source, const Options &options);

 private:
  size_t id() { return id_++; }

  Config config_;
  std::optional<TranslationCache> cache_;
  Threadsafe<AggregateBatcher> batcher_;
  std::vector<std::thread> workers_;

  size_t id_ = 0;
};

}  // namespace slimt
