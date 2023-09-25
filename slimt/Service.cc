#include "slimt/Service.hh"

#include <string>
#include <utility>

#include "slimt/Batch.hh"
#include "slimt/Types.hh"

namespace slimt {
namespace {

// Combines two responses with first.target == second.source mapping alignments
// etc accordingly. There are several constraints which are matched by only the
// pivoting workflow in <>Service source, therefore this function is not for
// external use and in a hidden namespace.
Response combine(Response &&first, Response &&second) {
  Response combined;

  // Compute alignment first using internal matrices and mappings.
  if (first.alignments.size()) {
    combined.alignments = remapAlignments(first, second);
  }

  combined.source = std::move(first.source);
  combined.target = std::move(second.target);

  return combined;
}

std::optional<TranslationCache> makeOptionalCache(size_t size,
                                                  size_t mutexBuckets) {
  return size > 0 ? std::make_optional<TranslationCache>(size, mutexBuckets)
                  : std::nullopt;
}

}  // namespace

Blocking::Blocking(const Blocking::Config &config)
    : config_(config),
      requestId_(0),
      batchingPool_(),
      cache_(makeOptionalCache(config.cacheSize, /*mutexBuckets = */ 1)),
      logger_(config.logger),
      workspace_(/*deviceId=*/0, config.workspaceSizeInMB) {}

std::vector<Response> Blocking::translateMultiple(
    std::shared_ptr<Model> model, std::vector<std::string> &&sources,
    const std::vector<Options> &options) {
  std::vector<HTML> htmls;
  for (size_t i = 0; i < sources.size(); i++) {
    htmls.emplace_back(std::move(sources[i]), options[i].HTML);
  }
  std::vector<Response> responses =
      translateMultipleRaw(model, std::move(sources), options);
  for (size_t i = 0; i < responses.size(); i++) {
    htmls[i].restore(responses[i]);
  }

  return responses;
}

std::vector<Response> Blocking::translateMultipleRaw(
    std::shared_ptr<Model> model, std::vector<std::string> &&sources,
    const std::vector<Options> &options) {
  std::vector<Response> responses;
  responses.resize(sources.size());

  for (size_t i = 0; i < sources.size(); i++) {
    auto callback = [i, &responses](Response &&response) {
      responses[i] = std::move(response);
    };  //
    Ptr<Request> request = model->makeRequest(
        requestId_++, std::move(sources[i]), callback, options[i], cache_);
    batchingPool_.enqueueRequest(model, request);
  }

  Batch batch;
  Ptr<Model> model{nullptr};
  while (batchingPool_.generateBatch(model, batch)) {
    model->translateBatch(workspace_, batch);
  }

  return responses;
}

std::vector<Response> Blocking::pivotMultiple(
    std::shared_ptr<Model> first, std::shared_ptr<Model> second,
    std::vector<std::string> &&sources, const std::vector<Options> &options) {
  std::vector<HTML> htmls;
  for (size_t i = 0; i < sources.size(); i++) {
    htmls.emplace_back(std::move(sources[i]), options[i].HTML);
  }

  // Translate source to pivots. This is same as calling translateMultiple.
  std::vector<Response> sourcesToPivots;
  sourcesToPivots = translateMultipleRaw(first, std::move(sources), options);

  // Translate pivots to targets, after we have outputs at pivot from first
  // round. We cannot use translateMultiple here because need consistency at
  // pivot on both sides.
  std::vector<Response> pivotsToTargets;
  pivotsToTargets.resize(sourcesToPivots.size());

  for (size_t i = 0; i < sourcesToPivots.size(); i++) {
    AnnotatedText intermediate =
        sourcesToPivots[i].target;  // We cannot eliminate this copy, as we need
                                    // two versions of intermediate. Holding it
                                    // in allows further use in makePivotRequest
    auto callback = [i, &pivotsToTargets](Response &&response) {
      pivotsToTargets[i] = std::move(response);
    };  //

    Ptr<Request> request = second->makePivotRequest(
        requestId_++, std::move(intermediate), callback, options[i], cache_);
    batchingPool_.enqueueRequest(second, request);
  }

  Batch batch;
  Ptr<Model> model{nullptr};
  while (batchingPool_.generateBatch(model, batch)) {
    model->translateBatch(workspace_, batch);
  }

  // Combine both sides. They're associated by indices.
  std::vector<Response> finalResponses;
  for (size_t i = 0; i < sourcesToPivots.size(); i++) {
    Response finalResponse =
        combine(std::move(sourcesToPivots[i]), std::move(pivotsToTargets[i]));
    finalResponses.push_back(std::move(finalResponse));
  }

  for (size_t i = 0; i < finalResponses.size(); i++) {
    htmls[i].restore(finalResponses[i]);
  }

  return finalResponses;
}

Async::Async(const Async::Config &config)
    : requestId_(0),
      config_(config),
      safeBatchingPool_(),
      cache_(makeOptionalCache(config_.cacheSize,
                               /*mutexBuckets=*/config_.numWorkers)),
      logger_(config.logger) {
  ABORT_IF(config_.numWorkers == 0,
           "Number of workers should be at least 1 in a threaded workflow");
  workers_.reserve(config_.numWorkers);
  for (size_t cpuId = 0; cpuId < config_.numWorkers; cpuId++) {
    workspaces_.emplace_back(cpuId, config.workspaceSizeInMB);
    workers_.emplace_back([cpuId, this] {
      // Consumer thread main-loop. Note that this is an infinite-loop unless
      // the monitor is explicitly told to shutdown, which happens in the
      // destructor for this class.
      Batch batch;
      Ptr<Model> model{nullptr};
      while (safeBatchingPool_.generateBatch(model, batch)) {
        model->translateBatch(workspaces_[cpuId], batch);
      }
    });
  }
}

void Async::clear() { safeBatchingPool_.clear(); }

Async::~Async() {
  safeBatchingPool_.shutdown();
  for (std::thread &worker : workers_) {
    assert(worker.joinable());
    worker.join();
  }
  workers_.clear();
}

void Async::pivot(std::shared_ptr<Model> first, std::shared_ptr<Model> second,
                  std::string &&source, CallbackType clientCallback,
                  const Options &options) {
  Ptr<HTML> html = std::make_shared<HTML>(std::move(source), options.HTML);
  // This is callback chaining or CPS due to async.

  // We create a callback which feeds the result of first into a second
  // translation (internalCallback), which is supplied with a callback
  // (joiningCallback) which merges both results and creates our final response.
  //

  auto internalCallback = [this, clientCallback, second, options,
                           html](Response &&sourceToPivot) {
    // We cannot eliminate the following copy, as we need two versions of
    // intermediate. Holding it in a copy allows moving the response into the
    // lambda below.

    AnnotatedText intermediate = sourceToPivot.target;

    // https://stackoverflow.com/a/65606554/4565794
    // Move semantics only work on mutable lambdas, and can only be done once.
    // It's only once in our case, so issok.
    auto joiningCallback = [this, sourceToPivot = std::move(sourceToPivot),
                            clientCallback,
                            html](Response &&pivotToTarget) mutable {
      // We have both Responses at this callback, sourceToPivot is moved in,
      // second half will be available when complete.
      Response finalResponse =
          combine(std::move(sourceToPivot), std::move(pivotToTarget));

      // Sentences should be consistent now, give way to client.
      html->restore(finalResponse);
      clientCallback(std::move(finalResponse));
    };

    // Second call.
    Ptr<Request> request =
        second->makePivotRequest(requestId_++, std::move(intermediate),
                                 joiningCallback, options, cache_);
    safeBatchingPool_.enqueueRequest(second, request);
  };

  // First call.
  translateRaw(first, std::move(source), internalCallback, options);
}

void Async::translate(std::shared_ptr<Model> model, std::string &&source,
                      CallbackType callback, const Options &options) {
  // Producer thread, a call to this function adds new work items. If batches
  // are available, notifies workers waiting.
  Ptr<HTML> html = std::make_shared<HTML>(std::move(source), options.HTML);
  auto internalCallback = [html, callback](Response &&response) {
    html->restore(response);
    callback(std::move(response));
  };

  translateRaw(model, std::move(source), internalCallback, options);
}

void Async::translateRaw(std::shared_ptr<Model> model, std::string &&source,
                         CallbackType callback, const Options &options) {
  // Producer thread, a call to this function adds new work items. If batches
  // are available, notifies workers waiting.
  Ptr<Request> request = model->makeRequest(requestId_++, std::move(source),
                                            callback, options, cache_);
  safeBatchingPool_.enqueueRequest(model, request);
}

}  // namespace slimt