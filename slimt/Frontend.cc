#include "slimt/Frontend.hh"

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>
#include <utility>

#include "slimt/Annotation.hh"
#include "slimt/Batch.hh"
#include "slimt/Batcher.hh"
#include "slimt/HTML.hh"
#include "slimt/Model.hh"
#include "slimt/Request.hh"
#include "slimt/Response.hh"
#include "slimt/ResponseBuilder.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"
#include "slimt/TextProcessor.hh"

namespace slimt {

namespace {

Batch convert(rd::Batch &rd_batch, uint32_t pad_id, size_t limit_factor) {
  const auto &segment_refs = rd_batch.segment_refs();
  Batch batch(rd_batch.size(), rd_batch.max_length(), pad_id, limit_factor);
  for (const auto &segment_ref : segment_refs) {
    Segment segment = segment_ref.get();
    batch.add(segment);
  }

  return batch;
}

void exhaust(const Config &config, const Ptr<Model> &model,
             rd::Batcher &batcher) {
  AverageMeter<float> wps;
  AverageMeter<float> occupancy;
  rd::Batch rd_batch = batcher.generate();
  while (!rd_batch.empty()) {
    // convert between batches.
    Timer timer;
    Batch batch = convert(rd_batch, model->vocabulary().pad_id(),
                          config.tgt_length_limit_factor);
    Histories histories = model->forward(batch);
    rd_batch.complete(histories);
    rd_batch = batcher.generate();

    auto elapsed = static_cast<float>(timer.elapsed());
    float batch_wps = batch.words().size() / elapsed;
    wps.record(batch_wps);
    occupancy.record(batch.occupancy());
  }
}

template <class Continuation>
Ptr<rd::Request> make_request(size_t id, const Ptr<Model> &model,
                              std::optional<TranslationCache> &cache,
                              AnnotatedText &&annotated_text,
                              Segments &&segments,
                              Continuation &&continuation) {
  ResponseBuilder response_builder(                                  //
      std::move(annotated_text),                                     //
      model->vocabulary(), std::forward<Continuation>(continuation)  //
  );

  auto request = std::make_shared<rd::Request>(  //
      id, model->id(),                           //
      std::move(segments),                       //
      std::move(response_builder),               //
      cache                                      //
  );
  return request;
}

}  // namespace

Blocking::Blocking(const Config &config) : config_(config) {}  // NOLINT

std::vector<Response> Blocking::translate(const Ptr<Model> &model,
                                          std::vector<std::string> sources,
                                          const Options &options) {
  rd::Batcher batcher(config_.max_words, config_.wrap_length,
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
    };

    auto &processor = model->processor();
    auto [annotated, segments] =
        processor.process(std::move(source), config_.wrap_length);
    auto request = make_request(id_, model, cache_, std::move(annotated),
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

  rd::Batcher batcher(config_.max_words, config_.wrap_length,
                      config_.tgt_length_limit_factor);

  for (size_t i = 0; i < source_to_pivots.size(); i++) {
    Response &source_to_pivot = source_to_pivots[i];
    Response &response = responses[i];

    auto continuation = [&source_to_pivot,
                         &response](Response &&pivot_to_target) {
      Response combined =
          combine(std::move(source_to_pivot), std::move(pivot_to_target));
      response = std::move(combined);
    };

    TextProcessor &processor = second->processor();
    auto [annotated, segments] = processor.process(source_to_pivot.target);
    auto request = make_request(id_, second, cache_, std::move(annotated),
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
      auto [rd_batch, model] = batcher_.generate();
      while (!rd_batch.empty()) {
        // convert between batches.
        Batch batch = convert(rd_batch, model->vocabulary().pad_id(),
                              config_.tgt_length_limit_factor);
        Histories histories = model->forward(batch);
        rd_batch.complete(histories);
        auto [next_batch, next_model] = batcher_.generate();
        rd_batch = std::move(next_batch);
        model = std::move(next_model);
      }
    });
  }
}

std::future<Response> Async::translate(const Ptr<Model> &model,
                                       std::string source,
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
  };

  TextProcessor &processor = model->processor();
  auto [annotated, segments] =
      processor.process(std::move(source), config_.wrap_length);
  auto request = make_request(id_, model, cache_, std::move(annotated),
                              std::move(segments), continuation);

  batcher_.enqueue(model, request);
  return future;
}
std::future<Response> Async::pivot(const Ptr<Model> &first,
                                   const Ptr<Model> &second, std::string source,
                                   const Options &options) {
  Ptr<HTML> html = nullptr;
  if (options.html) {
    html = std::make_shared<HTML>(source);
  }

  // This is callback chaining or CPS due to async.
  Promise promise;
  auto future = promise.get_future();

  auto continuation = [this, &promise, second,
                       html](Response &&source_to_pivot) {
    // https://stackoverflow.com/a/65606554/4565794
    // Move semantics only work on mutable lambdas, and can only be done once.
    // It's only once in our case, so issok.
    auto joining_continuation = [source_to_pivot = std::move(source_to_pivot),
                                 &promise,
                                 html](Response &&pivot_to_target) mutable {
      // We have both Responses at this callback, source_to_pivot is moved in,
      // second half will be available when complete.
      Response response =
          combine(std::move(source_to_pivot), std::move(pivot_to_target));

      // Sentences should be consistent now, give way to client.
      if (html) {
        html->restore(response);
      }
      promise.set_value(std::move(response));
    };

    TextProcessor &processor = second->processor();
    auto [annotated, segments] = processor.process(source_to_pivot.target);

    auto request =
        make_request(id_, second, cache_, std::move(annotated),
                     std::move(segments), std::move(joining_continuation));

    batcher_.enqueue(second, request);
  };

  TextProcessor &processor = first->processor();
  auto [annotated, segments] =
      processor.process(std::move(source), config_.wrap_length);
  auto request = make_request(id_, first, cache_, std::move(annotated),
                              std::move(segments), continuation);

  batcher_.enqueue(first, request);
  return future;
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
