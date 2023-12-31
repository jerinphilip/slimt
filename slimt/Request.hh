#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <future>
#include <memory>
#include <optional>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

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
///          -> Request::process(...)
/// ```
///
/// When all segments in a Request are completed, response_builder is
/// triggered with the compiled Histories, to construct the Response
/// corresponding to the Request and set value of the promise which triggers
/// the future at client.
class Request {
 public:
  using Continuation = std::function<Ptr<Request>(Response &&response)>;
  /// Constructs an internal representation of the Request identified by Id,
  /// processed Segments and accepts a callback (ResponseBuilder) which builds
  /// the Response upon completion of the Request.
  Request(size_t id, size_t model_id, AnnotatedText &&source,
          Segments &&segments, const Vocabulary &vocabulary,
          std::optional<TranslationCache> &cache, Continuation &&continuation);

  /// Obtain the count of tokens in the segment correponding to index. Used to
  /// insert segment from multiple requests into the corresponding size
  /// bucket.
  size_t word_count(size_t index) const;

  /// Obtain number of segments in a request.
  size_t size() const;

  /// Obtains segment corresponding to index  to create a batch of segments
  /// among several requests.
  const Segment &segment(size_t index) const;

  /// For notions of priority among requests, used to enable std::set in
  /// BatchingPool.
  friend bool operator<(const Request &a, const Request &b);

  /// Processes a history obtained after translating in a heterogenous batch
  /// compiled from requests.
  void process(size_t index, History history);

  bool cached(size_t index) const;

  Ptr<Request> next() const { return next_; }

  std::pair<Fraction, Fraction> progress() const;

 private:
  void complete(Histories &&histories);

  /// Multiple translation-workers can concurrently access the same Request.
  /// The following atomic atomically operates on the variable holding
  /// segments remaining to be translated.
  std::atomic<int> counter_;

  /// Completed words, to measure wps.
  std::atomic<int> words_complete_;
  size_t words_total_;

  size_t id_;

  /// Model associated with this request
  size_t model_id_;

  // Source text.
  AnnotatedText source_;
  /// segments_ hold the segments processed into Words which generated from
  /// input string.
  Segments segments_;

  const Vocabulary &vocabulary_;

  /// histories_ is a buffer which eventually stores the translations of each
  /// segment in the corresponding index.
  Histories histories_;

  /// Cache used to hold segment translations. If nullopt, means no-caching.
  std::optional<TranslationCache> &cache_;

  Continuation continuation_;
  Ptr<Request> next_ = nullptr;
};

}  // namespace slimt
