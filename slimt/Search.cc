#include "slimt/Search.hh"

#include <cstdint>

#include "slimt/TensorOps.hh"

namespace slimt {

Greedy::Greedy(const Transformer &transformer, const Vocabulary &vocabulary,
               const std::optional<ShortlistGenerator> &shortlist_generator)
    : transformer_(transformer),
      vocabulary_(vocabulary),
      shortlist_generator_(shortlist_generator) {}

BeamSearch::BeamSearch(
    const Transformer &transformer, const Vocabulary &vocabulary,
    const std::optional<ShortlistGenerator> &shortlist_generator)
    : transformer_(transformer),
      vocabulary_(vocabulary),
      shortlist_generator_(shortlist_generator) {}

Histories Greedy::generate(const Input &input) {
  Tensor mask = input.mask().clone();

  // uint64_t batch_size = indices.dim(-2);
  // uint64_t sequence_length = indices.dim(-1);
  // uint64_t embed_dim = embedding_.dim(-1);

  Tensor word_embedding =
      index_select(transformer_.embedding(), input.indices(), "word_embedding");
  transform_embedding(word_embedding);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L570
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L133
  Tensor encoder_out = transformer_.encoder().forward(word_embedding, mask);

  std::optional<Words> indices = std::nullopt;
  if (shortlist_generator_) {
    Shortlist shortlist = shortlist_generator_->generate(input.words());
    indices = shortlist.words();
  }
  // The following can be used to check if shortlist is going wrong.
  // std::vector<uint32_t> indices(vocabulary_.size());
  // std::iota(indices.begin(), indices.end(), 0);

  // Prepare a shortlist for the entire input.
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);
  size_t max_seq_length = input.limit_factor() * source_sequence_length;
  Words previous_slice = {};
  std::vector<Tensor> states = transformer_.decoder_start_states(batch_size);

  // Initialize a first step.

  GenerationStep step(input.lengths(), std::move(encoder_out), std::move(mask),
                      std::move(previous_slice), std::move(indices),
                      std::move(states), max_seq_length, vocabulary_.eos_id(),
                      batch_size);

  auto [logits, attn] =
      transformer_.step(step.encoder_out(), step.mask(), step.states(),
                        step.previous(), step.shortlist());

  if (step.shortlist()) {
    previous_slice = greedy_sample_from_words(logits, vocabulary_,
                                              *step.shortlist(), batch_size);
  } else {
    previous_slice = greedy_sample(logits, vocabulary_, batch_size);
  }

  step.update(std::move(previous_slice), attn);

  for (size_t i = 1; i < max_seq_length && !step.complete() > 0; i++) {
    auto [logits, attn] =
        transformer_.step(step.encoder_out(), step.mask(), step.states(),
                          step.previous(), step.shortlist());
    if (step.shortlist()) {
      previous_slice =
          greedy_sample_from_words(logits, vocabulary_, *indices, batch_size);
    } else {
      previous_slice = greedy_sample(logits, vocabulary_, batch_size);
    }
    step.update(std::move(previous_slice), attn);
  }

  Histories histories = step.finish();
  return histories;
}

GenerationStep::GenerationStep(const std::vector<size_t> &input_lengths,
                               Tensor &&encoder_out, Tensor &&mask,
                               Words &&previous,
                               std::optional<Words> &&shortlist,
                               std::vector<Tensor> &&states,
                               size_t max_seq_length, uint32_t eos_id,
                               size_t batch_size)
    : input_lengths_(input_lengths),
      encoder_out_(std::move(encoder_out)),
      mask_(std::move(mask)),
      states_(std::move(states)),
      previous_(std::move(previous)),
      shortlist_(std::move(shortlist)),
      max_seq_length_(max_seq_length),
      result_(eos_id, batch_size) {}

Result::Result(uint32_t eos_id, size_t batch_size)
    : eos_id_(eos_id),
      complete_(batch_size),
      sentences_(batch_size),
      alignments_(batch_size) {}

size_t Result::record(const Words &step) {
  size_t finished = 0;
  for (size_t i = 0; i < step.size(); i++) {
    if (not complete_[i]) {
      complete_[i] = (step[i] == eos_id_);
      sentences_[i].push_back(step[i]);
    }
    finished += static_cast<int>(complete_[i]);
  }
  return sentences_.size() - finished;
}

void Result::update_alignment(const Tensor &attn,
                              const std::vector<size_t> &input_lengths) {
  const auto *data = attn.data<float>();
  // B x H x 1 (T) x S
  size_t batch_size = attn.dim(-4);
  size_t num_heads = attn.dim(-3);
  size_t slice = attn.dim(-2);
  size_t source_length = attn.dim(-1);

  // https://github.com/marian-nmt/marian-dev/blob/53b0b0d7c83e71265fee0dd832ab3bcb389c6ec3/src/models/transformer.h#L214-L232
  for (size_t id = 0; id < batch_size; id++) {
    // Copy the elements into the particular alignment index.
    size_t head_id = 0;
    if (!complete_[id]) {
      size_t batch_stride = (num_heads * slice * source_length);
      size_t head_stride = (slice * source_length);
      const float *alignment = data + id * batch_stride + head_id * head_stride;
      size_t length = input_lengths[id];
      Distribution distribution(length);
      std::copy(alignment, alignment + length, distribution.data());
      alignments_[id].push_back(std::move(distribution));
    }
  }
}

void GenerationStep::update(Words &&step, const Tensor &attn) {
  previous_ = std::move(step);

  result_.update_alignment(attn, input_lengths_);
  remaining_ = result_.record(previous_);
}

Histories Result::consume() {
  Histories histories;
  for (size_t i = 0; i < sentences_.size(); i++) {
    Hypothesis hypothesis{
        .target = std::move(sentences_[i]),     //
        .alignment = std::move(alignments_[i])  //
    };
    auto history = std::make_shared<Hypothesis>(std::move(hypothesis));
    histories.push_back(std::move(history));
  }
  return histories;
}

}  // namespace slimt
