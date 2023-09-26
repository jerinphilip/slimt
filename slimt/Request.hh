#pragma once

#include <cassert>
#include <future>
#include <optional>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/ResponseBuilder.hh"
#include "slimt/Types.hh"

namespace slimt::rd {

using TranslationCache = slimt::TranslationCache;

/// A Request is an internal representation used to represent a request after
/// processed by TextProcessor into units constituted by marian::Words.
///
/// The batching mechanism (BatchingPool) draws from multiple Requests and
/// compiles units into a batch. When a batch completes translation (at
/// BatchTranslator, intended in a different thread), backward propogation
/// happens through:
///
/// ```cpp
///   Batch::completeBatch(...)
///       -> Unit::complete(..)
///          -> Request::complete(...)
/// ```
///
/// When all units in a Request are completed, responseBuilder is
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
  /// unit key (model, words in a unit) for cache.
  /// @param [in] segments: Each segment is a unit to be translated.
  /// @param [in] responseBuilder: Callback function (of ResponseBuilder type)
  /// to be triggered upon the completion of translation of all units in a
  /// Request.
  /// @param [in] cache: Cache supplied externally to attempt to fetch
  /// translations or store them after completion for reuse later.
  Request(size_t Id, size_t model_id, Segments &&segments,
          ResponseBuilder &&responseBuilder,
          std::optional<TranslationCache> &cache);

  /// Obtain the count of tokens in the segment correponding to index. Used to
  /// insert unit from multiple requests into the corresponding size
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

  bool cacheHitPrefilled(size_t index) const {
    return histories_[index] != nullptr;
  }

 private:
  size_t id_;

  /// Model associated with this request
  size_t model_id_;

  /// Multiple translation-workers can concurrently access the same Request.
  /// The following atomic atomically operates on the variable holding
  /// units remaining to be translated.
  std::atomic<int> counter_;

  /// segments_ hold the units processed into Words which generated from
  /// input string.
  Segments segments_;

  /// histories_ is a buffer which eventually stores the translations of each
  /// segment in the corresponding index.
  Histories histories_;

  /// Constructing Response requires the vocabs_ used to generate Request.
  /// std::vector<Ptr<Vocab const>> *vocabs_;
  ResponseBuilder responseBuilder_;

  /// Cache used to hold unit translations. If nullopt, means no-caching.
  std::optional<TranslationCache> &cache_;
};

/// A Unit provides a view to a unit within a Request. Existence
/// of this class allows the units and associated information to be kept
/// within Request, while batching mechanism (BatchingPool) compiles Batch
/// from Unit-s coming from different Requests.
class Unit {
 public:
  Unit(size_t, Ptr<Request>);

  /// Number of tokens in the segment this Unit represents. Used to
  /// order by length in batching.
  size_t numTokens() const;

  /// Accessor to the segment represented by the Unit.
  Segment getUnderlyingSegment() const;

  /// Forwards history to Request to set history corresponding to this
  /// Unit.
  void complete(History history);

  friend bool operator<(const Unit &a, const Unit &b);

 private:
  size_t index_;
  Ptr<Request> request_;
};

using Units = std::vector<Unit>;

// An empty batch is poison.
class Batch {
 public:
  Batch() = default;
  void clear();

  size_t size() const { return units_.size(); }
  bool empty() const { return units_.empty(); }
  size_t max_length() const { return max_length_; }

  void add(const Unit &unit);

  // Accessors to read from a Batch. For use in BatchTranslator (consumer on a
  // PCQueue holding batches).
  //
  // units() are used to access units to construct marian internal
  // batch.
  const Units &units() { return units_; }

  // On obtaining Histories after translating a batch, completeBatch can be
  // called with Histories , which forwards the call to Request through
  // Unit and triggers completion, by setting the promised value to
  // the future given to client.
  void complete(const Histories &histories);

  // Convenience function to log batch-statistics. numTokens, max-length.
  void log();

 private:
  Units units_;
  size_t token_count_ = 0;
  size_t max_length_ = 0;
};

}  // namespace slimt::rd
