#include "slimt/Request.hh"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "slimt/Cache.hh"
#include "slimt/Macros.hh"
#include "slimt/Response.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"

namespace slimt {

size_t cache_key(size_t model_id, const Words &words) {
  auto seed = model_id;
  for (size_t word : words) {
    hash_combine<size_t>(seed, word);
  }
  return seed;
}

// -----------------------------------------------------------------
Request::Request(size_t id, size_t model_id, AnnotatedText &&source,
                 Segments &&segments, const Vocabulary &vocabulary,
                 std::optional<TranslationCache> &cache,
                 Continuation &&continuation)
    : id_(id),
      model_id_(model_id),
      source_(std::move(source)),
      segments_(std::move(segments)),
      vocabulary_(vocabulary),
      cache_(cache),
      continuation_(std::move(continuation)) {
  counter_ = segments_.size();
  histories_.resize(segments_.size(), nullptr);

  // 1. If there are no segments_, we are never able to trigger the
  // response_builder calls from a different thread. This happens when the use
  // provides empty input, or the unit and subword preprocessing deems no
  // translatable units present. However, in this case we want an empty valid
  // response. There's no need to do any additional processing here.
  if (segments_.empty()) {
    postprocess(std::move(histories_));
  } else {
    counter_ = segments_.size();
    histories_.resize(segments_.size());

    // Word count book-keeping.
    word_count_ = 0;
    completed_word_count_ = 0;

    if (cache_) {
      // Iterate through segments, see if any can be prefilled from cache. If
      // prefilled, mark the particular segments as complete (non-empty
      // ProcessedSegmentRef). Also update accounting used elsewhere
      // (counter_) to reflect one less segment to translate.
      for (size_t idx = 0; idx < segments_.size(); idx++) {
        word_count_ += segments_[idx].size();
        size_t key = cache_key(model_id_, segment(idx));
        auto [found, history] = cache_->find(key);
        if (found) {
          histories_[idx] = history;
          --counter_;
          completed_word_count_ += segments_[idx].size();
        }
      }
      // 2. Also, if cache somehow manages to decrease all counter prefilling
      // histories, then we'd have to trigger ResponseBuilder as well. No
      // segments go into batching and therefore no complete triggers.
      if (counter_.load() == 0) {
        postprocess(std::move(histories_));
      }
    } else {
      for (const Segment &segment : segments_) {
        word_count_ += segment.size();
      }
    }
  }
}

size_t Request::segment_count() const { return segments_.size(); }

size_t Request::completed() const { return segment_count() - counter_.load(); }

size_t Request::word_count(size_t index) const {
  return (segments_[index].size());
}

const Segment &Request::segment(size_t index) const { return segments_[index]; }

void Request::process(size_t index, History history) {
  // Concurrently called by multiple workers as a history from translation is
  // ready. The container storing histories is set with the value obtained.

  // Fill in placeholder from History obtained by freshly translating. Since
  // this was a cache-miss to have got through, update cache if available to
  // store the result.
  histories_[index] = std::move(history);
  if (cache_) {
    size_t key = cache_key(model_id_, segment(index));
    cache_->store(key, histories_[index]);
  }

  completed_word_count_ += segments_[index].size();

  // In case this is last request in, completeRequest is called, which sets the
  // value of the promise.
  if (--counter_ == 0) {
    postprocess(std::move(histories_));
  }
}

void Request::postprocess(Histories &&histories) {
  SLIMT_ABORT_IF(source_.sentence_count() != histories.size(),
                 "Mismatch in source and translated sentences");
  Response response;

  // Move source_ into response.
  response.source = std::move(source_);
  // Reserving length at least as much as source_ seems like a reasonable
  // thing to do to avoid reallocations.
  response.target.text.reserve(response.source.text.size());

  for (size_t sentence_id = 0; sentence_id < histories.size(); sentence_id++) {
    Words words = histories[sentence_id]->target;
    std::string decoded;
    auto views = vocabulary_.decode(words, decoded, /*ignore_eos=*/false);

    // For each sentence, prepend the filler text between the corresponding
    // source-sentence and the source-sentence before.
    std::string_view pre = response.source.gap(sentence_id);
    response.target.append_sentence(pre, views.begin(), views.end());

    // If this is the last history to be decoded and translated-text
    // constructed, append the text till the end, which could be spaces or
    // empty.
    if (sentence_id + 1 == histories.size()) {
      response.target.append_ending_whitespace(
          response.source.gap(sentence_id + 1));
    }

    Alignment &alignment = histories[sentence_id]->alignment;
    response.alignments.push_back(std::move(alignment));
  }

  next_ = continuation_(std::move(response));
}

}  // namespace slimt
