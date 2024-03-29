#include "slimt/Batcher.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>

#include "slimt/Macros.hh"
#include "slimt/Model.hh"
#include "slimt/Request.hh"
#include "slimt/Types.hh"

namespace slimt {

// ------------------------------------------------------------------

SegmentRef::SegmentRef(size_t index, Ptr<Request> request)
    : index_(index), request_(std::move(request)) {}

size_t SegmentRef::size() const { return (request_->word_count(index_)); }

void SegmentRef::complete(History history) {
  // Relays complete into request's complete, using index
  // information.
  request_->process(index_, std::move(history));
}

const Segment& SegmentRef::get() const { return request_->segment(index_); }

bool operator<(const Request& a, const Request& b) {
  // Among Requests, only sequence id is used for obtaining priority.
  return a.id_ < b.id_;
}

bool operator<(const SegmentRef& a, const SegmentRef& b) {
  // Operator overload for usage in priority-queue / set.
  if (a.request_ == b.request_) {
    return a.index_ < b.index_;
  }
  return a.request_ < b.request_;
}

// ----------------------------------------------------------------------

void Batch::log() {
  (void)token_count_;
  LOG(info, "Batch(tokens=%zu max-length=%zu, segment_refs_=%zu)", token_count_,
      max_length_, segment_refs_.size());
}

void Batch::add(const SegmentRef& segment_ref) {
  segment_refs_.push_back(segment_ref);
  token_count_ += segment_ref.size();
  max_length_ = std::max<size_t>(max_length_, segment_ref.size());
}

void Batch::complete(const Histories& histories) {
  for (size_t i = 0; i < segment_refs_.size(); i++) {
    segment_refs_[i].complete(histories[i]);
  }
}

void Batch::clear() {
  segment_refs_.clear();
  token_count_ = 0;
  max_length_ = 0;
}

size_t AggregateBatcher::Hash::operator()(
    const std::shared_ptr<Model>& model) const {
  return std::hash<size_t>()(model->id());
}

Batcher::Batcher(size_t max_words, size_t wrap_length,
                 float tgt_length_limit_factor)
    : max_words_(max_words) {
  // For the time being, we add some slack, which only Batcher is aware of.
  // Since the TextProcessor still wraps at first request in, most of the
  // Batches generated will be under max-length break.
  //
  // In the unlikely event of a few sentences overflowing, this allows the
  // exceeding words to be put in the slack area. Very few batches are expected
  // to be generated at a higher length.
  size_t pivot_slack = wrap_length * tgt_length_limit_factor - wrap_length;
  bucket_.resize(wrap_length + pivot_slack + 1);

  SLIMT_ABORT_IF(bucket_.size() - 1 > max_words_,
                 "Fatal: wrap_length > max_words  will lead to sentences "
                 "longer than what can fit in a batch.");
}

Batch Batcher::generate() {
  // For now simply iterates on buckets and converts batches greedily.  This
  // has to be enhanced with optimizing over priority. The baseline
  // implementation should at least be as fast as marian's maxi-batch with full
  // corpus size as maxi-batch size.
  Batch batch;
  size_t padded_batch_size = 0;

  for (size_t length = 0; length <= running_bucket_max_size_; length++) {
    auto p = bucket_[length].begin();
    while (p != bucket_[length].end()) {
      padded_batch_size = (batch.size() + 1) * length;
      if (padded_batch_size <= max_words_) {
        auto q = p++;
        batch.add(*q);
        bucket_[length].erase(q);
      } else {
        // Check if elements exist
        assert(!batch.empty());
        return batch;
      }
    }
  }

  return batch;
}

size_t Batcher::enqueue(const Ptr<Request>& request) {
  size_t to_be_translated = 0;
  for (size_t i = 0; i < request->size(); i++) {
    if (!request->cached(i)) {
      SegmentRef sentence(i, request);
      size_t bucket_id = sentence.size();

      // Due to a workaround for pivoting, unless we can discipline the
      // vocabulary to get stronger static requirements, it is difficult to
      // rework the rest of the components. Instead, we allow dynamic growth
      // here. We let std::vector take care of the dynamic growth.
      // https://en.cppreference.com/w/cpp/container/vector/resize#Complexity
      if (bucket_id >= bucket_.size()) {
        bucket_.resize(bucket_id + 1);
      }

      bucket_[bucket_id].insert(sentence);
      running_bucket_max_size_ =
          std::max<size_t>(bucket_id, running_bucket_max_size_);

      to_be_translated += 1;
    }
  }

  return to_be_translated;
}

void Batcher::clear() {
  for (auto& item : bucket_) {
    item.clear();
  }
}

AggregateBatcher::AggregateBatcher(
    size_t max_words,                         //
    size_t wrap_length,                       //
    float tgt_length_limit_factor /*= 3.0 */  // NOLINT
    )
    : max_words_(max_words),
      wrap_length_(wrap_length),
      tgt_length_limit_factor_(tgt_length_limit_factor) {}

size_t AggregateBatcher::enqueue(const Ptr<Model>& model,
                                 const Ptr<Request>& request) {
  queue_.insert(model);
  size_t id = model->id();

  auto query = batcher_.find(id);
  if (query == batcher_.end()) {
    batcher_.emplace(               //
        std::piecewise_construct,   //
        std::forward_as_tuple(id),  //
        std::forward_as_tuple(max_words_, wrap_length_,
                              tgt_length_limit_factor_)  //
    );
  }

  query = batcher_.find(id);
  Batcher& batcher = query->second;
  size_t size = batcher.enqueue(request);
  return size;
}

std::tuple<Batch, Ptr<Model>> AggregateBatcher::generate() {
  while (!queue_.empty()) {
    auto model_iterator = queue_.begin();
    Ptr<Model> model = *model_iterator;
    auto query = batcher_.find(model->id());
    Batcher& batcher = query->second;
    Batch batch = batcher.generate();
    if (!batch.empty()) {
      return {std::move(batch), std::move(model)};
    }
    queue_.erase(model_iterator);
  }
  // Empty.
  Batch batch;
  return {std::move(batch), nullptr};
}

void AggregateBatcher::clear() { queue_.clear(); }

}  // namespace slimt
