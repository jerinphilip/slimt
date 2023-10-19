#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <future>
#include <memory>
#include <optional>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/ResponseBuilder.hh"
#include "slimt/Types.hh"

namespace slimt {

/// A Request is an internal representation used to represent a request after
/// processed by TextProcessor into segments constituted by marian::Words.
///
/// The batching mechanism (BatchingPool) draws from multiple Requests and
/// compiles segments into a batch. When a batch completes translation (at
/// BatchTranslator, intended in a different thread), backward propogation
/// happens through:
///
/// ```cpp
///   Batch::complete(...)
///       -> SegmentRef::complete(..)
///          -> Request::complete(...)
/// ```
///
/// When all segments in a Request are completed, responseBuilder is
/// triggered with the compiled Histories, to construct the Response
/// corresponding to the Request and set value of the promise which triggers
/// the future at client.
class Request {
 public:
  /// Constructs an internal representation of the Request identified by Id,
  /// processed Segments and accepts a callback (ResponseBuilder) which builds
  /// the Response upon completion of the Request.
  ///
  ///
  /// @param [in] Id: Identifier assigned to Request by Service.
  /// @param [in] model: Model for identifying a unique translation
  /// segment key (model, words in a segment) for cache.
  /// @param [in] segments: Each segment is a segment to be translated.
  /// @param [in] responseBuilder: Callback function (of ResponseBuilder type)
  /// to be triggered upon the completion of translation of all segments in a
  /// Request.
  /// @param [in] cache: Cache supplied externally to attempt to fetch
  /// translations or store them after completion for reuse later.
  Request(size_t Id, size_t model_id, Segments &&segments,
          ResponseBuilder &&responseBuilder,
          std::optional<TranslationCache> &cache);

  /// Obtain the count of tokens in the segment correponding to index. Used to
  /// insert segment from multiple requests into the corresponding size
  /// bucket.
  size_t word_count(size_t index) const;

  /// Obtain number of segments in a request.
  size_t segment_count() const;

  /// Obtains segment corresponding to index  to create a batch of segments
  /// among several requests.
  Segment segment(size_t index) const;

  /// For notions of priority among requests, used to enable std::set in
  /// BatchingPool.
  friend bool operator<(const Request &a, const Request &b);

  /// Processes a history obtained after translating in a heterogenous batch
  /// compiled from requests.
  void complete(size_t index, History history);

  bool is_prefilled_from_cache(size_t index) const {
    return histories_[index] != nullptr;
  }

 private:
  size_t id_;

  /// Model associated with this request
  size_t model_id_;

  /// Multiple translation-workers can concurrently access the same Request.
  /// The following atomic atomically operates on the variable holding
  /// segments remaining to be translated.
  std::atomic<int> counter_;

  /// segments_ hold the segments processed into Words which generated from
  /// input string.
  Segments segments_;

  /// histories_ is a buffer which eventually stores the translations of each
  /// segment in the corresponding index.
  Histories histories_;

  /// Constructing Response requires the vocabs_ used to generate Request.
  /// std::vector<Ptr<Vocab const>> *vocabs_;
  ResponseBuilder responseBuilder_;

  /// Cache used to hold segment translations. If nullopt, means no-caching.
  std::optional<TranslationCache> &cache_;
};

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

}  // namespace slimt
