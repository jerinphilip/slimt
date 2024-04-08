#include "slimt/Search.hh"

#include "slimt/TensorOps.hh"

namespace slimt {

namespace {

void update_alignment(const std::vector<size_t> &lengths,
                      const std::vector<bool> &finished, const Tensor &attn,
                      Alignments &alignments) {
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
    if (!finished[id]) {
      size_t batch_stride = (num_heads * slice * source_length);
      size_t head_stride = (slice * source_length);
      const float *alignment = data + id * batch_stride + head_id * head_stride;
      size_t length = lengths[id];
      Distribution distribution(length);
      std::copy(alignment, alignment + length, distribution.data());
      alignments[id].push_back(std::move(distribution));
    }
  }
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

  std::vector<bool> complete(batch_size, false);
  uint32_t eos = vocabulary_.eos_id();
  auto record = [eos, &complete](Words &step, Sentences &sentences) {
    size_t finished = 0;
    for (size_t i = 0; i < step.size(); i++) {
      if (not complete[i]) {
        complete[i] = (step[i] == eos);
        sentences[i].push_back(step[i]);
      }
      finished += static_cast<int>(complete[i]);
    }
    return sentences.size() - finished;
  };

  // Initialize a first step.
  Sentences sentences(batch_size);
  Alignments alignments(sentences.size());
  GenerationStep step(std::move(encoder_out), std::move(mask),
                      std::move(previous_slice), std::move(indices),
                      std::move(states), max_seq_length);

  auto [logits, attn] =
      transformer_.step(step.encoder_out(), step.mask(), step.states(),
                        step.previous(), step.shortlist());

  if (step.shortlist()) {
    previous_slice = greedy_sample_from_words(logits, vocabulary_,
                                              *step.shortlist(), batch_size);
  } else {
    previous_slice = greedy_sample(logits, vocabulary_, batch_size);
  }

  update_alignment(input.lengths(), complete, attn, alignments);
  record(previous_slice, sentences);

  size_t remaining = sentences.size();
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    auto [logits, attn] =
        transformer_.step(step.encoder_out(), step.mask(), step.states(),
                          step.previous(), step.shortlist());
    if (indices) {
      previous_slice =
          greedy_sample_from_words(logits, vocabulary_, *indices, batch_size);
    } else {
      previous_slice = greedy_sample(logits, vocabulary_, batch_size);
    }
    update_alignment(input.lengths(), complete, attn, alignments);
    remaining = record(previous_slice, sentences);
  }

  Histories histories;
  for (size_t i = 0; i < sentences.size(); i++) {
    Hypothesis hypothesis{
        .target = std::move(sentences[i]),     //
        .alignment = std::move(alignments[i])  //
    };
    auto history = std::make_shared<Hypothesis>(std::move(hypothesis));
    histories.push_back(std::move(history));
  }

  return histories;
}

GenerationStep::GenerationStep(Tensor &&encoder_out, Tensor &&mask,
                               Words &&previous,
                               std::optional<Words> &&shortlist,
                               std::vector<Tensor> &&states,
                               size_t max_seq_length)
    : encoder_out_(std::move(encoder_out)),
      mask_(std::move(mask)),
      states_(std::move(states)),
      previous_(std::move(previous)),
      shortlist_(shortlist),
      max_seq_length_(max_seq_length) {}

}  // namespace slimt
