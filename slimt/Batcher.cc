#include "slimt/Batcher.hh"

#include <cassert>

#include "slimt/Batch.hh"

namespace slimt {
Batcher::Batcher(size_t max_words, size_t wrap_length,
                 float tgt_length_limit_factor)
    : max_words_(max_words), running_bucket_max_size_(0) {
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
  batch.clear();
  size_t paddedBatchSize = 0;

  for (size_t length = 0; length <= running_bucket_max_size_; length++) {
    auto p = bucket_[length].begin();
    while (p != bucket_[length].end()) {
      paddedBatchSize = (batch.size() + 1) * length;
      if (paddedBatchSize <= max_words_) {
        auto q = p++;
        batch.add(*q);
        bucket_[length].erase(q);
      } else {
        // Check if elements exist
        assert(batch.size() > 0);
        return batch.size();
      }
    }
  }

  return batch.size();
}

size_t Batcher::enqueue(Ptr<Request> request) {
  size_t to_be_translated = 0;
  for (size_t i = 0; i < request->numSegments(); i++) {
    if (!request->cacheHitPrefilled(i)) {
      RequestSentence sentence(i, request);
      size_t bucket_id = sentence.numTokens();

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
  for (size_t length = 0; length < bucket_.size(); length++) {
    bucket_[length].clear();
  }
}

AggregateBatcher::AggregateBatcher() {
  // TODO(@jerinphilip): Set aggregate limits
}

size_t AggregateBatcher::enqueue(Ptr<Model> model, Ptr<Request> request) {
  size_t sentences_enqueued = model->enqueue(request);
  aggregateQueue_.insert(model);
  return sentences_enqueued;
}

size_t AggregateBatcher::generate(Ptr<Model>& model, Batch& batch) {
  while (!aggregateQueue_.empty()) {
    auto candidate_iterator = aggregateQueue_.begin();
    Ptr<Model> candidate = *candidate_iterator;
    size_t num_sentences = candidate->generate(batch);
    if (num_sentences > 0) {
      model = candidate;
      return num_sentences;
    }
    // Try the next model's batching pool.
    aggregateQueue_.erase(candidate_iterator);
  }
  return /*num_sentences=*/0;
}

void AggregateBatcher::clear() { aggregateQueue_.clear(); }

}  // namespace slimt
