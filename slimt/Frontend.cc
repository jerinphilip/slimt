#include "slimt/Frontend.hh"

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>
#include <utility>

#include "slimt/Batch.hh"
#include "slimt/Batcher.hh"
#include "slimt/HTML.hh"
#include "slimt/Request.hh"
#include "slimt/Response.hh"
#include "slimt/ResponseBuilder.hh"
#include "slimt/Tensor.hh"
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

Histories decode(Tensor &encoder_out, Batch &batch,
                 const size_t &tgt_length_limit_factor, Transformer &model,
                 const Word &eos_id, ShortlistGenerator &shortlist_generator) {
  // Prepare a shortlist for the entire batch.
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);

  Shortlist shortlist = shortlist_generator.generate(batch.words());
  Words indices = shortlist.words();
  // The following can be used to check if shortlist is going wrong.
  // std::vector<uint32_t> indices(vocabulary_.size());
  // std::iota(indices.begin(), indices.end(), 0);

  std::vector<bool> complete(batch_size, false);
  uint32_t eos = eos_id;
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

  Decoder &decoder = model.decoder();
  Words previous_slice = {};
  std::vector<Tensor> states = decoder.start_states(batch_size);
  auto [logits, attn] =
      decoder.step(encoder_out, batch.mask(), states, previous_slice, indices);

  previous_slice = greedy_sample(logits, indices, batch_size);
  update_alignment(batch.lengths(), complete, attn, alignments);
  record(previous_slice, sentences);

  size_t remaining = sentences.size();
  size_t max_seq_length = tgt_length_limit_factor * source_sequence_length;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    auto [logits, attn] = decoder.step(encoder_out, batch.mask(), states,
                                       previous_slice, indices);
    previous_slice = greedy_sample(logits, indices, batch_size);
    update_alignment(batch.lengths(), complete, attn, alignments);
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

Histories forward(Batch &batch, const size_t &tgt_length_limit_factor,
                  Transformer &model_, const Word &eos_id,
                  ShortlistGenerator &shortlist_generator) {
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
  Histories histories = decode(encoder_out, batch, tgt_length_limit_factor,
                               model_, eos_id, shortlist_generator);
  return histories;
}

}  // namespace

Blocking::Blocking(const Config &config) : config_(config) {}  // NOLINT

std::vector<Response> Blocking::translate(Ptr<Model> &model,
                                          std::vector<std::string> sources,
                                          const Options &options) {
  rd::Batcher batcher(config_.max_words, config_.wrap_length,
                      config_.tgt_length_limit_factor);

  // Configure promises, and HTML
  std::vector<Promise> promises(sources.size());

  std::vector<HTML> htmls;
  if (options.html) {
    htmls.reserve(sources.size());
  }

  if (options.html) {
    for (std::string &source : sources) {
      htmls.emplace_back(source);
    }
  }

  std::vector<Future> futures;
  futures.reserve(sources.size());

  for (size_t i = 0; i < sources.size(); i++) {
    std::string &source = sources[i];
    HTML *html = options.html ? &htmls[i] : nullptr;

    auto [annotated_source, segments] =
        model->processor().process(std::move(source));

    Promise &promise = promises[i];
    Future future = promise.get_future();
    futures.push_back(std::move(future));

    auto continuation = [&promise, html](Response &&response) {
      if (html) {
        html->restore(response);
      }
      promise.set_value(std::move(response));
    };

    ResponseBuilder response_builder(                 //
        options, std::move(annotated_source),         //
        model->vocabulary(), std::move(continuation)  //
    );

    auto request = std::make_shared<rd::Request>(  //
        id_, model->id(),                          //
        std::move(segments),                       //
        std::move(response_builder),               //
        cache_                                     //
    );

    batcher.enqueue(request);
  }

  AverageMeter<float> wps;
  AverageMeter<float> occupancy;
  rd::Batch rd_batch = batcher.generate();
  while (!rd_batch.empty()) {
    // convert between batches.
    Timer timer;
    Batch batch = convert(rd_batch);
    Histories histories =
        forward(batch, config_.tgt_length_limit_factor, model->transformer(),
                model->vocabulary().eos_id(), model->shortlist_generator());
    rd_batch.complete(histories);
    rd_batch = batcher.generate();

    auto elapsed = static_cast<float>(timer.elapsed());
    float batch_wps = batch.words().size() / elapsed;
    wps.record(batch_wps);
    occupancy.record(batch.occupancy());
  }

  std::vector<Response> responses;
  responses.reserve(futures.size());
  for (auto &future : futures) {
    future.wait();
    Response response = future.get();
    responses.push_back(std::move(response));
  }
  return responses;
}

std::vector<Response> Blocking::pivot(Ptr<Model> &first, Ptr<Model> &second,
                                      std::vector<std::string> sources,
                                      const Options &options) {
  std::vector<HTML> htmls;

  // Strip any existing HTML.
  for (auto &source : sources) {
    if (options.html) {
      htmls.emplace_back(source);
    }
  }

  // Translate source to pivots.
  std::vector<Response> source_to_pivots;
  Options raw{
      .alignment = options.alignment,  //
      .html = false                    //
  };

  source_to_pivots = translate(first, std::move(sources), raw);

  // Translate pivots to targets, after we have outputs at pivot from first
  // round.
  std::vector<Response> responses(source_to_pivots.size());

  rd::Batcher batcher(config_.max_words, config_.wrap_length,
                      config_.tgt_length_limit_factor);

  for (size_t i = 0; i < source_to_pivots.size(); i++) {
    Response &source_to_pivot = source_to_pivots[i];
    Response &response = responses[i];

    auto continuation = [&source_to_pivot,
                         &response](Response &&pivot_to_target) {
      Response combined =
          combine(std::move(source_to_pivot), std::move(pivot_to_target));
      response = std::move(combined);
    };

    // We cannot eliminate this copy, as we need two versions of intermediate.
    std::string target = source_to_pivots[i].target.text;
    auto [annotated_source, segments] =
        second->processor().process(std::move(target));

    ResponseBuilder response_builder(options, std::move(annotated_source),
                                     second->vocabulary(),
                                     std::move(continuation));

    Ptr<rd::Request> request = std::make_shared<rd::Request>(  //
        id_, second->id(),                                     //
        std::move(segments),                                   //
        std::move(response_builder),                           //
        cache_                                                 //
    );
    batcher.enqueue(request);
  }

  AverageMeter<float> wps;
  AverageMeter<float> occupancy;
  rd::Batch rd_batch = batcher.generate();
  while (!rd_batch.empty()) {
    // convert between batches.
    Timer timer;
    Batch batch = convert(rd_batch);
    Histories histories =
        forward(batch, config_.tgt_length_limit_factor, second->transformer(),
                second->vocabulary().eos_id(), second->shortlist_generator());
    rd_batch.complete(histories);
    rd_batch = batcher.generate();

    auto elapsed = static_cast<float>(timer.elapsed());
    float batch_wps = batch.words().size() / elapsed;
    wps.record(batch_wps);
    occupancy.record(batch.occupancy());
  }

  if (options.html) {
    for (size_t i = 0; i < responses.size(); i++) {
      htmls[i].restore(responses[i]);
    }
  }

  return responses;
}

Async::Async(const Config &config)
    : batcher_(config.max_words, config.wrap_length,
               config.tgt_length_limit_factor) {
  // Also creates consumers, starts listening.
  for (size_t i = 0; i < config.workers; i++) {
    workers_.emplace_back([this]() {
      auto [rd_batch, model] = batcher_.generate();
      while (!rd_batch.empty()) {
        // convert between batches.
        Batch batch = convert(rd_batch);
        Histories histories = forward(
            batch, config_.tgt_length_limit_factor, model->transformer(),
            model->vocabulary().eos_id(), model->shortlist_generator());
        rd_batch.complete(histories);
        auto [next_batch, next_model] = batcher_.generate();
        rd_batch = std::move(next_batch);
        model = std::move(next_model);
      }
    });
  }
}

std::future<Response> Async::translate(Ptr<Model> &model, std::string source,
                                       const Options &options) {
  std::shared_ptr<HTML> html = nullptr;
  if (options.html) {
    html = std::make_shared<HTML>(source);
  }

  auto [annotated_source, segments] =
      model->processor().process(std::move(source));

  auto promise = std::make_shared<Promise>();
  auto future = promise->get_future();
  auto continuation = [html, promise](Response &&response) {
    if (html) {
      html->restore(response);
    }
    promise->set_value(std::move(response));
  };

  ResponseBuilder response_builder(                 //
      options, std::move(annotated_source),         //
      model->vocabulary(), std::move(continuation)  //
  );

  auto request = std::make_shared<rd::Request>(  //
      id_, model->id(),                          //
      std::move(segments),                       //
      std::move(response_builder),               //
      cache_                                     //
  );

  batcher_.enqueue(model, request);
  return future;
}
std::future<Response> Async::pivot(Ptr<Model> &first, Ptr<Model> &second,
                                   std::string source, const Options &options) {
  Ptr<HTML> html = nullptr;
  if (options.html) {
    html = std::make_shared<HTML>(source);
  }

  // This is callback chaining or CPS due to async.
  Promise promise;
  auto future = promise.get_future();

  auto continuation = [this, &promise, second,
                       html](Response &&source_to_pivot) {
    // We cannot eliminate the following copy, as we need two versions of
    // intermediate. Holding it in a copy allows moving the response into the
    // lambda below.
    AnnotatedText intermediate = source_to_pivot.target;

    // https://stackoverflow.com/a/65606554/4565794
    // Move semantics only work on mutable lambdas, and can only be done once.
    // It's only once in our case, so issok.
    auto joining_continuation = [source_to_pivot = std::move(source_to_pivot),
                                 &promise,
                                 html](Response &&pivotToTarget) mutable {
      // We have both Responses at this callback, source_to_pivot is moved in,
      // second half will be available when complete.
      Response response =
          combine(std::move(source_to_pivot), std::move(pivotToTarget));

      // Sentences should be consistent now, give way to client.
      if (html) {
        html->restore(response);
      }
      promise.set_value(std::move(response));
    };

    // Second call.
    std::string pivot = intermediate.text;
    auto [annotated_pivot, segments] =
        second->processor().process(std::move(pivot));
    Options raw{.html = false};
    ResponseBuilder response_builder(                          //
        raw, std::move(annotated_pivot),                       //
        second->vocabulary(), std::move(joining_continuation)  //
    );

    auto request = std::make_shared<rd::Request>(
        id_, second->id(),                                 //
        std::move(segments), std::move(response_builder),  //
        cache_                                             //
    );

    batcher_.enqueue(second, request);
  };

  auto [annotated_source, segments] =
      first->processor().process(std::move(source));

  ResponseBuilder response_builder(                 //
      options, std::move(annotated_source),         //
      first->vocabulary(), std::move(continuation)  //
  );

  auto request = std::make_shared<rd::Request>(  //
      id_, first->id(),                          //
      std::move(segments),                       //
      std::move(response_builder),               //
      cache_                                     //
  );

  batcher_.enqueue(first, request);
  return future;
}

Async::~Async() {
  batcher_.shutdown();
  for (std::thread &worker : workers_) {
    assert(worker.joinable());
    worker.join();
  }
  workers_.clear();
}

}  // namespace slimt
