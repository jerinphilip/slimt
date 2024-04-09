#include "slimt/Search.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "slimt/Input.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"

namespace slimt {

namespace {

std::tuple<Tensor, Tensor> encode(const Transformer &transformer,
                                  const Input &input) {
  Tensor mask = input.mask().clone();
  Tensor word_embedding =
      index_select(transformer.embedding(), input.indices(), "word_embedding");
  transform_embedding(word_embedding);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L570
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L133
  Tensor encoder_out = transformer.encoder().forward(word_embedding, mask);
  return std::make_tuple(std::move(encoder_out), std::move(mask));
}

// Project parts
Tensor project(const Tensor &in, size_t count) {
  // Form output tensor shape
  Shape out_shape = in.shape();
  size_t leading_dim = out_shape.dim(0);
  size_t elements = out_shape.elements();
  size_t width = elements / leading_dim;
  out_shape.set_dim(0, leading_dim * count);
  Tensor out(in.type(), out_shape, in.name() + "_projected");

  auto *out_data = out.data<float>();
  const auto *in_data = in.data<float>();

  // TODO(@jerinphilip): Need to verify.
  for (size_t sample_id = 0; sample_id < leading_dim; sample_id++) {
    for (size_t replica_id = 0; replica_id < count; replica_id++) {
      const float *src = in_data + sample_id * width;
      float *dest = out_data + (sample_id * count + replica_id) * width;
      const size_t bytes = width * sizeof(float);
      std::memcpy(dest, src, bytes);
    }
  }

  return out;
}

}  // namespace

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
  auto [encoder_out, mask] = encode(transformer_, input);

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

  for (size_t i = 0; i < max_seq_length && !step.complete(); i++) {
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

NBest BeamSearch::generate(const Input &input, size_t beam_size) {
  auto [encoder_out, mask] = encode(transformer_, input);

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

  // Project variables for beam-search
  std::vector<size_t> input_lengths_projected;
  input_lengths_projected.reserve(input.lengths().size() * beam_size);
  for (const auto &input_length : input.lengths()) {
    for (size_t beam_id = 0; beam_id < beam_size; beam_id++) {
      input_lengths_projected.push_back(input_length);
    }
  }
  Tensor encoder_out_projected = project(encoder_out, beam_size);
  Tensor mask_projected = project(input.mask(), beam_size);
  std::vector<Tensor> states_projected;
  for (auto &state : states) {
    Tensor state_projected = project(state, beam_size);
    states_projected.push_back(std::move(state_projected));
  }

  // Initialize a first step.
  GenerationStep step(input_lengths_projected, std::move(encoder_out_projected),
                      std::move(mask_projected), std::move(previous_slice),
                      std::move(indices), std::move(states_projected),
                      max_seq_length, vocabulary_.eos_id(), batch_size);

  for (size_t i = 0; i < max_seq_length && !step.complete(); i++) {
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
  }

  Histories histories = step.finish();

  NBest nbest;
  return nbest;
}

}  // namespace slimt
