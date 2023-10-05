#pragma once

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "slimt/Batch.hh"
#include "slimt/Request.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"

namespace slimt {

class Model;
template <class T>
struct HashPtr;

namespace rd {

class Batcher {
 public:
  explicit Batcher(                        //
      size_t max_words,                    //
      size_t wrap_length,                  //
      float tgt_length_limit_factor = 3.0  // NOLINT
  );

  // SegmentRef incorporates (tentative) notions of priority with each
  // sentence. This method inserts the sentence into the internal data-structure
  // which maintains priority among sentences from multiple concurrent requests.
  size_t enqueue(const Ptr<Request>& request);

  // Loads sentences with sentences compiled from (tentatively) multiple
  // requests optimizing for both padding and priority.
  Batch generate();

  // Removes any pending requests from the pool.
  void clear();

 private:
  size_t max_words_;
  std::vector<std::set<SegmentRef>> bucket_;
  size_t running_bucket_max_size_{0};
};

/// Aggregates request queueing and generation of batches from multiple
/// Models (Batchers within, specifically), thereby acting as an
/// intermediary to enable multiple translation model capability in
/// BlockingService and AsyncService.
///
/// A simple queue containing shared owning references to Models are
/// held here from which batches are generated on demand. Since a queue is
/// involved, the ordering is first-come first serve on requests except there
/// are leaks effectively doing priority inversion if an earlier request with
/// the same Model is pending to be consumed for translation.
//
/// Actual storage for the request and batch generation are within the
/// respective Models, which owns its own Batcher.
///
/// Matches API provided by Batcher except arguments additionally
/// parameterized by Model.
///
/// Note: This class is not thread-safe. You may use this class wrapped with
/// Threadsafe for a thread-safe equivalent of this class, if
/// needed.
class AggregateBatcher {
 public:
  /// Create an AggregateBatcher with (tentatively) global (across all
  /// Batchers) limits imposed here.
  AggregateBatcher();

  /// Enqueue an existing request onto model, also keep account of that this
  /// model and request are now pending.
  ///
  /// @param [in] model: Model to use in translation. A shared ownership to this
  /// model is accepted by this object to keep the model alive until translation
  /// is complete.
  /// @param [in] request: A request to be enqueued to model.
  /// @returns number of sentences added for translation.
  size_t enqueue(const Ptr<Model>& model, const Ptr<Request>& request);

  /// Generate a batch from pending requests, obtained from available
  /// Models.
  ///
  /// @param [out] model: Model
  /// @param [out] batch: Batch to write onto, which is consumed at translation
  /// elsewhere.
  /// @returns Number of sentences in the generated batch.
  Batch generate(Ptr<Model>& model);

  /// Clear the aggregate queue. Does not clear the underlying model/request
  /// pairs but the next call to `generate()` will return 0. (Unless
  /// `enqueue()` was called in the mean time.)
  void clear();

 private:
  std::unordered_set<std::shared_ptr<Model>, HashPtr<Model>> queue_;
};

/// The following mechanism operates in a multithreaded async-workflow guarding
/// access to the pushes to the structure keeping sentences bucketed by length
/// and sorted by priority.
///
/// This is a wrap of a producer-consumer queue implemented as a monitor, where
/// there is a mutex guarding the underlying data structure (BatcherType)
/// and (worker/consumer) threads waiting on a condition variable and the
/// queuing thread producing and notifying waiting threads (consumers) through
/// the same condition variable.
///
/// Originally written by for a single model (where items are produce: Request,
/// consume: Batch), converted to also work for multiple models where items are
/// produce: (Model, Request), consume: (TranlsationModel, Batch).
/// This is accomplished by template parameter packs.
///
/// Requires BatcherType to implement the following:
///
/// * produce: `size_t enqueue(...)` (returns number elements produced)
/// * consume: `size_t generate(...)` (returns number of elements available
/// to be consumed)

template <class BatcherType>
class Threadsafe {
 public:
  // Signals shut down of batching pool. After this no new requests can be
  // enqueued, but all enqueued requests will be processed. To prevent this from
  // happening, call `clear()` before `shutdown()`.
  template <class... Args>
  explicit Threadsafe(Args&&... args) : backend_(std::forward<Args>(args)...) {}

  ~Threadsafe() { shutdown(); }

  template <class... Args>
  void enqueue(Args&&... args) {
    std::unique_lock<std::mutex> lock(mutex_);
    assert(!shutdown_);
    enqueued_ += backend_.enqueue(std::forward<Args>(args)...);
    work_.notify_all();
  }

  void clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    backend_.clear();
    enqueued_ = 0;
  }

  void shutdown() {
    std::unique_lock<std::mutex> lock(mutex_);
    shutdown_ = true;
    work_.notify_all();
  }

  template <class... Args>
  Batch generate(Args&&... args) {
    std::unique_lock<std::mutex> lock(mutex_);
    work_.wait(lock, [this]() { return enqueued_ || shutdown_; });
    Batch batch = backend_.generate(std::forward<Args>(args)...);
    assert(!batch.empty() || shutdown_);
    enqueued_ -= batch.size();
    return batch;
  }

 private:
  BatcherType backend_;

  // Number of sentences in backend_;
  size_t enqueued_ = 0;

  // Are we shutting down?
  bool shutdown_ = false;

  // Lock on this object.
  std::mutex mutex_;

  // Signaled when there are sentences to translate.
  std::condition_variable work_;
};

}  // namespace rd

}  // namespace slimt
