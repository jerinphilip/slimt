#include "slimt/Frontend.hh"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Batcher.hh"
#include "slimt/HTML.hh"
#include "slimt/Input.hh"
#include "slimt/Model.hh"
#include "slimt/Request.hh"
#include "slimt/Response.hh"
#include "slimt/Search.hh"
#include "slimt/TextProcessor.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

namespace {

Input convert(const Batch &batch, uint32_t pad_id, float limit_factor) {
  const auto &segment_refs = batch.segment_refs();
  Input input(batch.size(), batch.max_length(), pad_id, limit_factor);
  for (const auto &segment_ref : segment_refs) {
    const Segment &segment = segment_ref.get();
    input.add(segment);
  }

  input.finalize();
  return input;
}

void exhaust(const Config &config, const Ptr<Model> &model, Batcher &batcher) {
  AverageMeter<float> wps;
  AverageMeter<float> occupancy;
  Batch batch = batcher.generate();
  while (!batch.empty()) {
    // convert between batches.
    Timer timer;
    Input input = convert(batch, model->vocabulary().pad_id(),
                          config.tgt_length_limit_factor);
    Histories histories = forward(model->transformer(), model->vocabulary(),
                                  model->shortlist_generator(), input);
    batch.complete(histories);
    batch = batcher.generate();

    auto elapsed = static_cast<float>(timer.elapsed());
    float sample_wps = input.words().size() / elapsed;
    wps.record(sample_wps);
    occupancy.record(input.occupancy());
  }
}

template <class Continuation>
Ptr<Request> make_request(size_t id, const Ptr<Model> &model,
                          std::optional<TranslationCache> &cache,
                          AnnotatedText &&annotated_text, Segments &&segments,
                          Continuation &&continuation) {
  auto request = std::make_shared<Request>(     //
      id, model->id(),                          //
      std::move(annotated_text),                //
      std::move(segments),                      //
      model->vocabulary(),                      //
      cache,                                    //
      std::forward<Continuation>(continuation)  //
  );
  return request;
}

}  // namespace

Blocking::Blocking(const Config &config) : config_(config) {}  // NOLINT

std::vector<Response> Blocking::translate(const Ptr<Model> &model,
                                          std::vector<std::string> sources,
                                          const Options &options) {
  Batcher batcher(config_.max_words, config_.wrap_length,
                  config_.tgt_length_limit_factor);

  std::vector<HTML> htmls;
  if (options.html) {
    htmls.reserve(sources.size());
    for (std::string &source : sources) {
      htmls.emplace_back(source);
    }
  }

  // Configure promises, and HTML
  std::vector<Promise> promises(sources.size());
  std::vector<Future> futures;
  futures.reserve(sources.size());

  for (size_t i = 0; i < sources.size(); i++) {
    std::string &source = sources[i];
    HTML *html = options.html ? &(htmls[i]) : nullptr;

    Promise &promise = promises[i];
    Future future = promise.get_future();
    futures.push_back(std::move(future));

    auto continuation = [&promise, html](Response &&response) {
      if (html) {
        html->restore(response);
      }
      promise.set_value(std::move(response));
      return nullptr;
    };

    const auto &processor = model->processor();
    auto [annotated, segments] =
        processor.process(std::move(source), config_.wrap_length);
    auto request = make_request(id(), model, cache_, std::move(annotated),
                                std::move(segments), continuation);

    batcher.enqueue(request);
  }

  exhaust(config_, model, batcher);

  std::vector<Response> responses;
  responses.reserve(futures.size());
  for (auto &future : futures) {
    future.wait();
    Response response = future.get();
    responses.push_back(std::move(response));
  }
  return responses;
}

std::vector<Response> Blocking::pivot(const Ptr<Model> &first,
                                      const Ptr<Model> &second,
                                      std::vector<std::string> sources,
                                      const Options &options) {
  std::vector<HTML> htmls;
  // Strip any existing HTML.
  if (options.html) {
    htmls.reserve(sources.size());
    for (auto &source : sources) {
      htmls.emplace_back(source);
    }
  }

  // Translate source to pivots.
  std::vector<Response> source_to_pivots;
  Options raw{
      .alignment = options.alignment,  //
      .html = false                    //
  };

  source_to_pivots = translate(first, std::move(sources), raw);

  // Translate pivots to targets, after we have outputs at pivot from first
  // round.
  std::vector<Response> responses(source_to_pivots.size());

  Batcher batcher(config_.max_words, config_.wrap_length,
                  config_.tgt_length_limit_factor);

  for (size_t i = 0; i < source_to_pivots.size(); i++) {
    Response &source_to_pivot = source_to_pivots[i];
    Response &response = responses[i];

    auto continuation = [&source_to_pivot,
                         &response](Response &&pivot_to_target) {
      Response combined =
          combine(std::move(source_to_pivot), std::move(pivot_to_target));
      response = std::move(combined);
      return nullptr;
    };

    const TextProcessor &processor = second->processor();
    auto [annotated, segments] = processor.process(source_to_pivot.target);
    auto request = make_request(id(), second, cache_, std::move(annotated),
                                std::move(segments), continuation);

    batcher.enqueue(request);
  }

  exhaust(config_, second, batcher);

  if (options.html) {
    for (size_t i = 0; i < responses.size(); i++) {
      htmls[i].restore(responses[i]);
    }
  }

  return responses;
}

Async::Async(const Config &config)
    : batcher_(config.max_words, config.wrap_length,
               config.tgt_length_limit_factor) {
  // Also creates consumers, starts listening.
  for (size_t i = 0; i < config.workers; i++) {
    workers_.emplace_back([this]() {
      auto [batch, model] = batcher_.generate();
      while (!batch.empty()) {
        // convert between batches.
        Input input = convert(batch, model->vocabulary().pad_id(),
                              config_.tgt_length_limit_factor);
        Histories histories = forward(model->transformer(), model->vocabulary(),
                                      model->shortlist_generator(), input);
        batch.complete(histories);
        auto [next_batch, next_model] = batcher_.generate();
        batch = std::move(next_batch);
        model = std::move(next_model);
      }
    });
  }
}

Handle Async::translate(const Ptr<Model> &model, std::string source,
                        const Options &options) {
  std::shared_ptr<HTML> html = nullptr;
  if (options.html) {
    html = std::make_shared<HTML>(source);
  }

  auto promise = std::make_shared<Promise>();
  auto future = promise->get_future();
  auto continuation = [html, promise](Response &&response) {
    if (html) {
      html->restore(response);
    }
    promise->set_value(std::move(response));
    return nullptr;
  };

  const TextProcessor &processor = model->processor();
  auto [annotated, segments] =
      processor.process(std::move(source), config_.wrap_length);
  auto request = make_request(id(), model, cache_, std::move(annotated),
                              std::move(segments), continuation);

  batcher_.enqueue(model, request);

  constexpr size_t parts = 1;  // NOLINT
  Handle handle(request, parts, std::move(future));
  return handle;
}

Handle Async::pivot(const Ptr<Model> &first, const Ptr<Model> &second,
                    std::string source, const Options &options) {
  Ptr<HTML> html = nullptr;
  if (options.html) {
    html = std::make_shared<HTML>(source);
  }

  // This is callback chaining or CPS due to async.
  auto promise = std::make_shared<Promise>();
  auto future = promise->get_future();

  auto continuation = [this, promise, second,
                       html](Response &&partial) -> Ptr<Request> {
    // https://stackoverflow.com/a/65606554/4565794
    // Move semantics only work on mutable lambdas, and can only be done once.
    // It's only once in our case, so issok.
    AnnotatedText intermediate = partial.target;
    auto joining_continuation =
        [source_to_pivot = std::move(partial), promise,
         html](Response &&pivot_to_target) mutable -> Ptr<Request> {
      // We have both Responses at this callback, source_to_pivot is moved in,
      // second half will be available when complete.
      Response response =
          combine(std::move(source_to_pivot), std::move(pivot_to_target));

      // Sentences should be consistent now, give way to client.
      if (html) {
        html->restore(response);
      }
      promise->set_value(std::move(response));
      return nullptr;
    };

    const TextProcessor &processor = second->processor();
    auto [annotated, segments] = processor.process(intermediate);

    auto request =
        make_request(id(), second, cache_, std::move(annotated),
                     std::move(segments), std::move(joining_continuation));

    batcher_.enqueue(second, request);
    return request;
  };

  const TextProcessor &processor = first->processor();
  auto [annotated, segments] =
      processor.process(std::move(source), config_.wrap_length);
  auto request = make_request(id(), first, cache_, std::move(annotated),
                              std::move(segments), continuation);

  batcher_.enqueue(first, request);

  constexpr size_t parts = 2;  // NOLINT
  Handle handle(request, parts, std::move(future));
  return handle;
}

Async::~Async() {
  batcher_.shutdown();
  for (std::thread &worker : workers_) {
    assert(worker.joinable());
    worker.join();
  }
  workers_.clear();
}

}  // namespace slimt
