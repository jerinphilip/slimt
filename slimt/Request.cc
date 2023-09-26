#include "slimt/Request.hh"

#include <string>

#include "slimt/Annotation.hh"
#include "slimt/Cache.hh"
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

namespace rd {

// -----------------------------------------------------------------
Request::Request(size_t Id, size_t model_id, Segments &&segments,
                 ResponseBuilder &&responseBuilder,
                 std::optional<TranslationCache> &cache)
    : id_(Id),
      model_id_(model_id),
      segments_(std::move(segments)),
      responseBuilder_(std::move(responseBuilder)),
      cache_(cache) {
  counter_ = segments_.size();
  histories_.resize(segments_.size(), nullptr);

  // 1. If there are no segments_, we are never able to trigger the
  // responseBuilder calls from a different thread. This happens when the use
  // provides empty input, or the unit and subword preprocessing deems no
  // translatable units present. However, in this case we want an empty valid
  // response. There's no need to do any additional processing here.
  if (segments_.empty()) {
    responseBuilder_(std::move(histories_));
  } else {
    counter_ = segments_.size();
    histories_.resize(segments_.size());

    if (cache_) {
      // Iterate through segments, see if any can be prefilled from cache. If
      // prefilled, mark the particular segments as complete (non-empty
      // ProcessedUnit). Also update accounting used elsewhere
      // (counter_) to reflect one less segment to translate.
      for (size_t idx = 0; idx < segments_.size(); idx++) {
        size_t key = cache_key(model_id_, segment(idx));
        auto [found, history] = cache_->find(key);
        if (found) {
          histories_[idx] = history;
          --counter_;
        }
      }
      // 2. Also, if cache somehow manages to decrease all counter prefilling
      // histories, then we'd have to trigger ResponseBuilder as well. No
      // segments go into batching and therefore no complete triggers.
      if (counter_.load() == 0) {
        responseBuilder_(std::move(histories_));
      }
    }
  }
}

size_t Request::segment_count() const { return segments_.size(); }

size_t Request::word_count(size_t index) const {
  return (segments_[index].size());
}

Segment Request::segment(size_t index) const { return segments_[index]; }

void Request::complete(size_t index, History history) {
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

  // In case this is last request in, completeRequest is called, which sets the
  // value of the promise.
  if (--counter_ == 0) {
    responseBuilder_(std::move(histories_));
  }
}

// ------------------------------------------------------------------

Unit::Unit(size_t index, Ptr<Request> request)
    : index_(index), request_(std::move(request)) {}

size_t Unit::numTokens() const { return (request_->word_count(index_)); }

void Unit::complete(History history) {
  // Relays complete into request's complete, using index
  // information.
  request_->complete(index_, std::move(history));
}

Segment Unit::getUnderlyingSegment() const {
  return request_->segment(index_);
}

bool operator<(const Request &a, const Request &b) {
  // Among Requests, only sequence id is used for obtaining priority.
  return a.id_ < b.id_;
}

bool operator<(const Unit &a, const Unit &b) {
  // Operator overload for usage in priority-queue / set.
  if (a.request_ == b.request_) {
    return a.index_ < b.index_;
  }
  return a.request_ < b.request_;
}

// ----------------------------------------------------------------------

void Batch::log() {
  (void)token_count_;
  LOG(info, "Batch(tokens={}, max-length={}, units_={})", token_count_,
      max_length_, units_.size());
}

void Batch::add(const Unit &unit) {
  units_.push_back(unit);
  token_count_ += unit.numTokens();
  max_length_ = std::max(max_length_, static_cast<size_t>(unit.numTokens()));
}

void Batch::complete(const Histories &histories) {
  for (size_t i = 0; i < units_.size(); i++) {
    units_[i].complete(histories[i]);
  }
}
void Batch::clear() {
  units_.clear();
  token_count_ = 0;
  max_length_ = 0;
}

}  // namespace rd

}  // namespace slimt
