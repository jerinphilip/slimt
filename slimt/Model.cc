#include "slimt/Model.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/Input.hh"
#include "slimt/Io.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

namespace {

size_t model_id = 0;

Package<io::MmapFile> mmap_from(const Package<std::string> &package) {
  return {
      .model = io::MmapFile(package.model),            //
      .vocabulary = io::MmapFile(package.vocabulary),  //
      .shortlist = io::MmapFile(package.shortlist),    //
  };
}

Package<View> view_from(const Package<io::MmapFile> &mmap) {
  return {
      .model = {mmap.model.data(), mmap.model.size()},                 //
      .vocabulary = {mmap.vocabulary.data(), mmap.vocabulary.size()},  //
      .shortlist = {mmap.shortlist.data(), mmap.shortlist.size()},     //
  };
}

}  // namespace

Model::Model(const Config &config, const Package<View> &package)
    : id_(model_id++),
      config_(config),
      view_(package),
      vocabulary_(package.vocabulary),
      processor_(config.split_mode, vocabulary_, Aligned()),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, package.model),
      shortlist_generator_(package.shortlist, vocabulary_, vocabulary_) {}

Model::Model(const Config &config, const Package<std::string> &package)
    : id_(model_id++),
      config_(config),
      mmap_(mmap_from(package)),
      view_(view_from(*mmap_)),
      vocabulary_(view_.vocabulary),
      processor_(config.split_mode, vocabulary_, Aligned()),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, view_.model),
      shortlist_generator_(view_.shortlist, vocabulary_, vocabulary_) {}

namespace {
void update_alignment(const std::vector<size_t> &lengths,
                      const std::vector<bool> &finished, Tensor &attn,
                      Alignments &alignments) {
  auto *data = attn.data<float>();
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
      float *alignment = data + id * batch_stride + head_id * head_stride;
      size_t length = lengths[id];
      Distribution distribution(length);
      std::copy(alignment, alignment + length, distribution.data());
      alignments[id].push_back(std::move(distribution));
    }
  }
}
}  // namespace

Histories Model::decode(const Tensor &encoder_out, const Input &input) const {
  // Prepare a shortlist for the entire input.
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);

  Shortlist shortlist = shortlist_generator_.generate(input.words());
  const Words &indices = shortlist.words();
  // The following can be used to check if shortlist is going wrong.
  // std::vector<uint32_t> indices(vocabulary_.size());
  // std::iota(indices.begin(), indices.end(), 0);

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

  const Decoder &decoder = transformer_.decoder();
  Words previous_slice = {};
  std::vector<Tensor> states = decoder.start_states(batch_size);
  auto [logits, attn] =
      decoder.step(encoder_out, input.mask(), states, previous_slice, indices);

  previous_slice = greedy_sample(logits, indices, batch_size);
  update_alignment(input.lengths(), complete, attn, alignments);
  record(previous_slice, sentences);

  size_t remaining = sentences.size();
  size_t max_seq_length = input.limit_factor() * source_sequence_length;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    auto [logits, attn] = decoder.step(encoder_out, input.mask(), states,
                                       previous_slice, indices);
    previous_slice = greedy_sample(logits, indices, batch_size);
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

Histories Model::forward(const Input &input) const {
  const Tensor &indices = input.indices();
  const Tensor &mask = input.mask();

  // uint64_t batch_size = indices.dim(-2);
  // uint64_t sequence_length = indices.dim(-1);
  // uint64_t embed_dim = embedding_.dim(-1);

  Tensor word_embedding =
      index_select(transformer_.embedding(), indices, "word_embedding");
  transform_embedding(word_embedding);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L570
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L133
  Tensor encoder_out = transformer_.encoder().forward(word_embedding, mask);
  Histories histories = decode(encoder_out, input);
  return histories;
}
}  // namespace slimt
