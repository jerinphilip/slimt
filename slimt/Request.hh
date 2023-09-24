#pragma once

#include <cassert>
#include <future>
#include <optional>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Cache.hh"
#include "slimt/Model.hh"
#include "slimt/ResponseBuilder.hh"
#include "slimt/Types.hh"

namespace slimt {

/// A Request is an internal representation used to represent a request after
/// processed by TextProcessor into sentences constituted by marian::Words.
///
/// The batching mechanism (BatchingPool) draws from multiple Requests and
/// compiles sentences into a batch. When a batch completes translation (at
/// BatchTranslator, intended in a different thread), backward propogation
/// happens through:
///
/// ```cpp
///   Batch::completeBatch(...)
///       -> RequestSentence::completeSentence(..)
///          -> Request::processHistory(...)
/// ```
///
/// When all sentences in a Request are completed, responseBuilder is
/// triggered with the compiled Histories, to construct the Response
/// corresponding to the Request and set value of the promise which triggers the
/// future at client.
class Request {
 public:
  /// Constructs an internal representation of the Request identified by Id,
  /// processed Segments and accepts a callback (ResponseBuilder) which builds
  /// the Response upon completion of the Request.
  ///
  ///
  /// @param [in] Id: Identifier assigned to Request by Service.
  /// @param [in] model: Model for identifying a unique translation
  /// unit key (model, words in a sentence) for cache.
  /// @param [in] segments: Each segment is a unit to be translated.
  /// @param [in] responseBuilder: Callback function (of ResponseBuilder type)
  /// to be triggered upon the completion of translation of all units in a
  /// Request.
  /// @param [in] cache: Cache supplied externally to attempt to fetch
  /// translations or store them after completion for reuse later.
  Request(size_t Id, const Model &model, Segments &&segments,
          ResponseBuilder &&responseBuilder,
          std::optional<TranslationCache> &cache);

  /// Obtain the count of tokens in the segment correponding to index. Used to
  /// insert sentence from multiple requests into the corresponding size bucket.
  size_t segmentTokens(size_t index) const;

  /// Obtain number of segments in a request.
  size_t numSegments() const;

  /// Obtains segment corresponding to index  to create a batch of segments
  /// among several requests.
  Segment getSegment(size_t index) const;

  /// For notions of priority among requests, used to enable std::set in
  /// BatchingPool.
  friend bool operator<(const Request &a, const Request &b);

  /// Processes a history obtained after translating in a heterogenous batch
  /// compiled from requests.
  void processHistory(size_t index, History history);

  bool cacheHitPrefilled(size_t index) const {
    return histories_[index] != nullptr;
  }

 private:
  size_t Id_;

  /// Model associated with this request
  const Model &model_;

  /// Multiple translation-workers can concurrently access the same Request. The
  /// following atomic atomically operates on the variable holding sentences
  /// remaining to be translated.
  std::atomic<int> counter_;

  /// segments_ hold the sentences processed into Words which generated from
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

/// A RequestSentence provides a view to a sentence within a Request. Existence
/// of this class allows the sentences and associated information to be kept
/// within Request, while batching mechanism (BatchingPool) compiles Batch from
/// RequestSentence-s coming from different Requests.
class RequestSentence {
 public:
  RequestSentence(size_t, Ptr<Request>);

  /// Number of tokens in the segment this RequestSentence represents. Used to
  /// order by length in batching.
  size_t numTokens() const;

  /// Accessor to the segment represented by the RequestSentence.
  Segment getUnderlyingSegment() const;

  /// Forwards history to Request to set history corresponding to this
  /// RequestSentence.
  void completeSentence(History history);

  friend bool operator<(const RequestSentence &a, const RequestSentence &b);

 private:
  size_t index_;
  Ptr<Request> request_;
};

using RequestSentences = std::vector<RequestSentence>;

// An empty batch is poison.
class RequestBatch {
 public:
  RequestBatch() = default;
  void clear() { sentences_.clear(); }

  size_t size() const { return sentences_.size(); }

  void add(const RequestSentence &sentence);

  // Accessors to read from a Batch. For use in BatchTranslator (consumer on a
  // PCQueue holding batches).
  //
  // sentences() are used to access sentences to construct marian internal
  // batch.
  const RequestSentences &sentences() { return sentences_; }

  // On obtaining Histories after translating a batch, completeBatch can be
  // called with Histories , which forwards the call to Request through
  // RequestSentence and triggers completion, by setting the promised value to
  // the future given to client.
  void complete(const Histories &histories);

  // Convenience function to log batch-statistics. numTokens, max-length.
  void log();

 private:
  RequestSentences sentences_;
};

}  // namespace slimt
