
#include "slimt/Frontend.hh"

#include "slimt/Batcher.hh"
#include "slimt/Request.hh"
#include "slimt/Response.hh"
#include "slimt/TensorOps.hh"

namespace slimt {

namespace {

Batch convert(rd::Batch &rd_batch) {
  const auto &units = rd_batch.units();
  Batch batch(rd_batch.size(), rd_batch.max_length(), 0);
  for (const auto &unit : units) {
    Segment segment = unit.getUnderlyingSegment();
    batch.add(segment);
  }

  return batch;
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

  Decoder &decoder = model_.decoder();
  Words previous_slice = {};
  std::vector<Tensor> states = decoder.start_states(batch_size);
  Tensor logits =
      decoder.step(encoder_out, mask, states, previous_slice, indices);

  previous_slice = greedy_sample(logits, indices, batch_size);
  record(previous_slice, sentences);

  size_t remaining = sentences.size();
  size_t max_seq_length =
      config_.tgt_length_limit_factor * source_sequence_length;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    Tensor logits =
        decoder.step(encoder_out, mask, states, previous_slice, indices);

    previous_slice = greedy_sample(logits, indices, batch_size);
    remaining = record(previous_slice, sentences);
  }

  Histories histories;
  Alignments alignments(sentences.size());
  for (size_t i = 0; i < sentences.size(); i++) {
    Hypothesis hypothesis{
        .target = sentences[i],     //
        .alignment = alignments[i]  //
    };
    auto history = std::make_shared<Hypothesis>();
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
  AnnotatedText annotated_source;
  Segments segments;
  processor_.process(std::move(source), annotated_source, segments);

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
  return future.get();
}

}  // namespace slimt
