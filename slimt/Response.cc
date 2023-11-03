#include "slimt/Response.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Request.hh"
#include "slimt/Types.hh"

namespace slimt {

// We're marginalizing q out of p(s | q) x p( q | t). However, we have different
// representations of q on source side to intermediate - p(s_i | q_j) and
// intermediate to target side - p(q'_j' | t_k).
//
// The matrix p(q'_j' | t_k) is rewritten into p(q_j | t_k) by means of
// spreading the probability in the former over bytes and collecting it at the
// ranges specified by latter, using a two pointer accumulation strategy.
Alignment transfer_through_characters(
    const std::vector<Range> &source_side_pivots,
    const std::vector<Range> &target_side_pivots,
    const Alignment &pivot_given_targets) {
  // Initialize an empty alignment matrix.
  Alignment remapped(pivot_given_targets.size(),
                     std::vector<float>(source_side_pivots.size(), 0.0F));

  size_t sq;
  size_t qt;
  for (sq = 0, qt = 0; sq < source_side_pivots.size() && qt < target_side_pivots.size();
       /*each branch inside increments either sq or qt or both, therefore the loop terminates */) {
    const auto &source_side_pivot = source_side_pivots[sq];
    const auto &target_side_pivot = target_side_pivots[qt];
    if (source_side_pivot.begin == target_side_pivot.begin &&
        source_side_pivot.end == target_side_pivot.end) {
      for (size_t t = 0; t < pivot_given_targets.size(); t++) {
        remapped[t][sq] += pivot_given_targets[t][qt];
      }

      // Perfect match, move pointer from both.
      sq++, qt++;
    } else {
      // Do we have overlap?
      size_t left = std::max(target_side_pivot.begin, source_side_pivot.begin);
      size_t right = std::min(target_side_pivot.end, source_side_pivot.end);

      assert(left < right);  // there should be overlap.

      size_t character_count = right - left;
      size_t probability_spread = target_side_pivot.size();
      for (size_t t = 0; t < pivot_given_targets.size(); t++) {
        remapped[t][sq] += character_count * pivot_given_targets[t][qt] /
                           static_cast<float>(probability_spread);
      }

      // Which one is ahead? sq or qt or both end at same point?
      if (source_side_pivot.end == target_side_pivot.end) {
        sq++;
        qt++;
      } else if (source_side_pivot.end > target_side_pivot.end) {
        qt++;
      } else {  // source_side_pivot.end < target_side_pivot.end
        sq++;
      }
    }
  }

  // The following is left in here for future debugging. Every token in source
  // is expected to have been processed in the above pipeline. We advance the
  // pivot-token index based on overlap with source-token. @jerinphilip is
  // worried about EOS not existing when people try weird 4-model things in the
  // future and would like to keep this check here.
  assert(sq == source_side_pivots.size());

  while (qt < target_side_pivots.size()) {
    // There is a case of EOS not being predicted. In this case the two pointer
    // algorithm will fail. The just author will redistribute the surplus among
    // subjects.

    // assert in DEBUG, that this is only EOS - occuring at the end and with
    // zero-surface.
    assert(qt == target_side_pivots.size() - 1 &&
           target_side_pivots[qt].size() == 0);
    for (size_t t = 0; t < pivot_given_targets.size(); t++) {
      float gift = pivot_given_targets[t][qt] / source_side_pivots.size();
      for (size_t sq = 0; sq < source_side_pivots.size(); sq++) {
        remapped[t][sq] += gift;
      }
    }

    qt++;
  }

#ifdef DEBUG
  // The following sanity check ensures when DEBUG is enabled that we have
  // successfully transferred all probabily mass available over pivot tokens
  // given a target token in our original input to the new remapped
  // representation.
  //
  // It's been discovered that floating point arithmetic before we get the
  // Alignment matrix can have values such that the distribution does not sum
  // upto 1.
  const float EPS = 1e-6;
  for (size_t t = 0; t < pivot_given_targets.size(); t++) {
    float sum = 0.0f, expectedSum = 0.0f;
    for (size_t qt = 0; qt < target_side_pivots.size(); qt++) {
      expectedSum += pivot_given_targets[t][qt];
    }
    for (size_t sq = 0; sq < source_side_pivots.size(); sq++) {
      sum += remapped[t][sq];
    }
    std::cerr << fmt::format(
                     "Sum @ token {} = {} to be compared with expected {}.", t,
                     sum, expectedSum)
              << std::endl;
    ABORT_IF(std::abs(sum - expectedSum) > EPS,
             "Haven't accumulated probabilities, re-examine");
  }
#endif  // DEBUG

  return remapped;
}

std::vector<Alignment> remap_alignments(const Response &first,
                                        const Response &second) {
  std::vector<Alignment> alignments;
  for (size_t sentence_id = 0; sentence_id < first.source.sentence_count();
       sentence_id++) {
    const Alignment &source_given_pivots = first.alignments[sentence_id];
    const Alignment &pivot_given_targets = second.alignments[sentence_id];

    // TODO(any): Allow range iterators and change algorithm, directly tapping
    // into AnnotatedText Extracts Ranges corresponding to a words
    // constituting a sentence from an annotation.
    auto extract_word_ranges = [](const AnnotatedText &annotatedText,
                                  size_t sentence_id) -> std::vector<Range> {
      size_t num_words = annotatedText.word_count(sentence_id);
      std::vector<Range> output;

      for (size_t i = 0; i < num_words; i++) {
        output.push_back(annotatedText.word_as_range(sentence_id, i));
      }
      return output;
    };

    auto source_side_pivots = extract_word_ranges(first.target, sentence_id);
    auto target_side_pivots = extract_word_ranges(second.source, sentence_id);

    // Reintrepret probability p(q'_j' | t_k) as p(q_j | t_k)
    Alignment remapped_pivot_given_targets = transfer_through_characters(
        source_side_pivots, target_side_pivots, pivot_given_targets);

    // Marginalize out q_j.
    // p(s_i | t_k) = \sum_{j} p(s_i | q_j) x p(q_j | t_k)
    size_t source_token_count = first.source.word_count(sentence_id);
    size_t target_token_count = second.target.word_count(sentence_id);
    Alignment output(target_token_count,
                     std::vector<float>(source_token_count, 0.0F));
    for (size_t idt = 0; idt < target_token_count; idt++) {
      for (size_t idq = 0; idq < source_side_pivots.size(); idq++) {
        for (size_t ids = 0; ids < source_token_count; ids++) {
          // Matrices are of form p(s | t) = P[t][s], hence idq appears on the
          // extremes.
          output[idt][ids] += source_given_pivots[idq][ids] *
                              remapped_pivot_given_targets[idt][idq];
        }
      }
    }

    alignments.push_back(output);
  }
  return alignments;
}

Response combine(Response &&first, Response &&second) {
  Response combined;

  // Compute alignment first using internal matrices and mappings.
  if (!first.alignments.empty()) {
    combined.alignments = remap_alignments(first, second);
  }

  combined.source = std::move(first.source);
  combined.target = std::move(second.target);

  return combined;
}

std::pair<size_t, size_t> Handle::words() const {
  return std::make_pair(request_->completed_word_count(),
                        request_->word_count());
}

std::pair<size_t, size_t> Handle::segments() const {
  return std::make_pair(request_->completed(), request_->segment_count());
}

}  // namespace slimt
