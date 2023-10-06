#include "slimt/Batcher.hh"

#include <algorithm>
#include <cassert>
#include <utility>

#include "slimt/Macros.hh"
#include "slimt/Model.hh"
#include "slimt/Utils.hh"

namespace slimt::rd {

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
  for (size_t i = 0; i < request->segment_count(); i++) {
    if (!request->is_prefilled_from_cache(i)) {
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
  std::lock_guard guard(mutex_);
  auto query = batcher_.find(model->id());
  if (query == batcher_.end()) {
    batcher_.emplace(                        //
        std::piecewise_construct,            //
        std::forward_as_tuple(model->id()),  //
        std::forward_as_tuple(max_words_, wrap_length_,
                              tgt_length_limit_factor_)  //
    );
  }
  query = batcher_.find(model->id());
  Batcher& batcher = query->second;

  size_t size = batcher.enqueue(request);
  queue_.insert(model);
  return size;
}

std::tuple<Batch, Ptr<Model>> AggregateBatcher::generate() {
  while (!queue_.empty()) {
    auto model_iterator = queue_.begin();
    Ptr<Model> model = *model_iterator;
    std::lock_guard guard(mutex_);
    auto query = batcher_.find(model->id());
    Batcher& batcher = query->second;
    Batch batch = batcher.generate();
    if (!batch.empty()) {
      return {std::move(batch), std::move(model)};
    }
    // Try the next model's batching pool.
    queue_.erase(model_iterator);
  }
  // Empty.
  Batch batch;
  return {std::move(batch), nullptr};
}

void AggregateBatcher::clear() { queue_.clear(); }

}  // namespace slimt::rd
