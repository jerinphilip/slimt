#pragma once

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "slimt/Types.hh"

namespace slimt {

class Model;
class Request;

/// A SegmentRef provides a view to a segment within a Request. Existence
/// of this class allows the segments and associated information to be kept
/// within Request, while batching mechanism (BatchingPool) compiles Batch
/// from SegmentRef-s coming from different Requests.
class SegmentRef {
 public:
  SegmentRef(size_t, Ptr<Request>);

  /// Number of tokens in the segment this SegmentRef represents. Used to
  /// order by length in batching.
  size_t size() const;

  /// Accessor to the segment represented by the SegmentRef.
  Segment get() const;

  /// Forwards history to Request to set history corresponding to this
  /// SegmentRef.
  void complete(History history);

  friend bool operator<(const SegmentRef &a, const SegmentRef &b);

 private:
  size_t index_;
  Ptr<Request> request_;
};

using SegmentRefs = std::vector<SegmentRef>;

// An empty batch is poison.
class Batch {
 public:
  Batch() = default;
  void clear();

  size_t size() const { return segment_refs_.size(); }
  bool empty() const { return segment_refs_.empty(); }
  size_t max_length() const { return max_length_; }

  void add(const SegmentRef &segment_ref);

  // Accessors to read from a Batch. For use in BatchTranslator (consumer on a
  // PCQueue holding batches).
  //
  // segment_refs() are used to access segment_refs to construct marian internal
  // batch.
  const SegmentRefs &segment_refs() { return segment_refs_; }

  // On obtaining Histories after translating a batch, complete can be
  // called with Histories , which forwards the call to Request through
  // SegmentRef and triggers completion, by setting the promised value to
  // the future given to client.
  void complete(const Histories &histories);

  // Convenience function to log batch-statistics. size, max-length.
  void log();

 private:
  SegmentRefs segment_refs_;
  size_t token_count_ = 0;
  size_t max_length_ = 0;
};

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
  size_t enqueue(const Ptr<Request> &request);

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
  AggregateBatcher(                        //
      size_t max_words,                    //
      size_t wrap_length,                  //
      float tgt_length_limit_factor = 3.0  // NOLINT
  );

  /// Enqueue an existing request onto model, also keep account of that this
  /// model and request are now pending.
  ///
  /// @param [in] model: Model to use in translation. A shared ownership to
  /// this model is accepted by this object to keep the model alive until
  /// translation is complete.
  /// @param [in] request: A request to be enqueued to model.
  /// @returns number of sentences added for translation.
  size_t enqueue(const Ptr<Model> &model, const Ptr<Request> &request);

  /// Generate a batch from pending requests, obtained from available
  /// Models.
  ///
  /// @param [out] model: Model
  /// @param [out] batch: Batch to write onto, which is consumed at translation
  /// elsewhere.
  /// @returns Number of sentences in the generated batch.
  std::tuple<Batch, Ptr<Model>> generate();

  /// Clear the aggregate queue. Does not clear the underlying model/request
  /// pairs but the next call to `generate()` will return 0. (Unless
  /// `enqueue()` was called in the mean time.)
  void clear();

 private:
  /// Hashes a pointer to an object using the address the pointer points to. If
  /// two pointers point to the same address, they hash to the same value.
  /// Useful to put widely shared_ptrs of entities (eg: Model, Vocab, Shortlist)
  /// etc into containers which require the members to be hashable
  /// (std::unordered_set, std::unordered_map).
  struct Hash {
    size_t operator()(const std::shared_ptr<Model> &model) const;
  };

  std::unordered_set<std::shared_ptr<Model>, Hash> queue_;
  std::unordered_map<size_t, Batcher> batcher_;

  size_t max_words_;               //
  size_t wrap_length_;             //
  float tgt_length_limit_factor_;  //
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
  explicit Threadsafe(Args &&...args) : backend_(std::forward<Args>(args)...) {}

  ~Threadsafe() { shutdown(); }

  template <class... Args>
  void enqueue(Args &&...args) {
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
  auto generate() {
    std::unique_lock<std::mutex> lock(mutex_);
    work_.wait(lock, [this]() { return enqueued_ || shutdown_; });
    auto pack = backend_.generate();
    auto batch = std::get<0>(pack);
    assert(!batch.empty() || shutdown_);
    enqueued_ -= batch.size();
    return pack;
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

}  // namespace slimt
