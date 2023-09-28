
#include "slimt/Frontend.hh"

#include "slimt/Batcher.hh"
#include "slimt/Request.hh"
#include "slimt/Response.hh"
#include "slimt/TensorOps.hh"

namespace slimt {

namespace {

Batch convert(rd::Batch &rd_batch) {
  const auto &segment_refs = rd_batch.segment_refs();
  Batch batch(rd_batch.size(), rd_batch.max_length(), 0);
  for (const auto &segment_ref : segment_refs) {
    Segment segment = segment_ref.get();
    batch.add(segment);
  }

  return batch;
}

void update_alignment(Tensor &attn, Alignments &alignments) {
  auto *data = attn.data<float>();
  // B x H x 1 (T) x S
  size_t batch_size = attn.dim(-4);
  size_t num_heads = attn.dim(-3);
  size_t slice = attn.dim(-2);
  size_t source_length = attn.dim(-1);

  // https://github.com/marian-nmt/marian-dev/blob/53b0b0d7c83e71265fee0dd832ab3bcb389c6ec3/src/models/transformer.h#L214-L232
  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    // Copy the elements into the particular alignment index.
    // size_t head_id = 0; (unused)
    size_t batch_stride = (num_heads * slice * source_length);
    float *alignment = data + batch_id * batch_stride;
    Distribution distribution(source_length);
    std::copy(alignment, alignment + source_length, distribution.data());
    alignments[batch_id].push_back(std::move(distribution));
  }
}

}  // namespace

Translator::Translator(const Config &config, View model, View shortlist,
                       View vocabulary)
    : config_(config),
      vocabulary_(vocabulary.data, vocabulary.size),
      processor_(config.wrap_length, config.split_mode, vocabulary_,
                 config.prefix_path),
      model_(config, model),
      shortlist_generator_(shortlist.data, shortlist.size, vocabulary_,
                           vocabulary_) {}

Histories Translator::decode(Tensor &encoder_out, Tensor &mask,
                             const Words &source) {
  // Prepare a shortlist for the entire batch.
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);

  Shortlist shortlist = shortlist_generator_.generate(source);
  Words indices = shortlist.words();
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

  Decoder &decoder = model_.decoder();
  Words previous_slice = {};
  std::vector<Tensor> states = decoder.start_states(batch_size);
  auto [logits, attn] =
      decoder.step(encoder_out, mask, states, previous_slice, indices);

  update_alignment(attn, alignments);

  previous_slice = greedy_sample(logits, indices, batch_size);
  record(previous_slice, sentences);

  size_t remaining = sentences.size();
  size_t max_seq_length =
      config_.tgt_length_limit_factor * source_sequence_length;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    auto [logits, attn] =
        decoder.step(encoder_out, mask, states, previous_slice, indices);
    update_alignment(attn, alignments);
    previous_slice = greedy_sample(logits, indices, batch_size);
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

Histories Translator::forward(Batch &batch) {
  Tensor &indices = batch.indices();
  Tensor &mask = batch.mask();

  // uint64_t batch_size = indices.dim(-2);
  // uint64_t sequence_length = indices.dim(-1);
  // uint64_t embed_dim = embedding_.dim(-1);

  Tensor word_embedding =
      index_select(model_.embedding(), indices, "word_embedding");
  transform_embedding(word_embedding);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L570
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L133
  modify_mask_for_pad_tokens_in_attention(mask.data<float>(), mask.size());
  Tensor encoder_out = model_.encoder().forward(word_embedding, mask);
  Histories histories = decode(encoder_out, mask, batch.words());
  return histories;
}

Response Translator::translate(std::string source, const Options &options) {
  // Create a request
  std::optional<HTML> html = std::nullopt;
  if (options.HTML) {
    html.emplace(source);
  }
  auto [annotated_source, segments] = processor_.process(std::move(source));

  std::promise<Response> promise;
  auto future = promise.get_future();

  ResponseBuilder response_builder(options, std::move(annotated_source),
                                   vocabulary_, std::move(promise));

  auto request = std::make_shared<rd::Request>(  //
      id_, model_id_,                            //
      std::move(segments),                       //
      std::move(response_builder),               //
      cache_                                     //
  );

  rd::Batcher batcher(config_.max_words, config_.wrap_length,
                      config_.tgt_length_limit_factor);
  batcher.enqueue(request);

  rd::Batch rd_batch = batcher.generate();
  while (!rd_batch.empty()) {
    // convert between batches.
    Batch batch = convert(rd_batch);
    Histories histories = forward(batch);
    rd_batch.complete(histories);
    rd_batch = batcher.generate();
  }

  future.wait();
  Response response = future.get();
  if (html) {
    html->restore(response);
  }
  return response;
}

}  // namespace slimt
