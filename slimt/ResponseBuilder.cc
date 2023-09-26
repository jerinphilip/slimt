#include "slimt/ResponseBuilder.hh"

#include "slimt/Response.hh"

namespace slimt {

void ResponseBuilder::operator()(Histories &&histories) {
  // TODO(jerinphilip) load Options into options and turn build
  // functions on or off.
  // options_ is unused, but we can try something here.
  SLIMT_ABORT_IF(source_.sentence_count() != histories.size(),
                 "Mismatch in source and translated sentences");
  Response response;

  // Move source_ into response.
  response.source = std::move(source_);
  // Reserving length at least as much as source_ seems like a reasonable
  // thing to do to avoid reallocations.
  response.target.text.reserve(response.source.text.size());

  for (size_t sentence_id = 0; sentence_id < histories.size(); sentence_id++) {
    Words words = histories[sentence_id]->target;
    std::string decoded;
    auto views = vocabulary_.decode(words, decoded, /*ignore_eos=*/false);

    // For each sentence, prepend the filler text between the corresponding
    // source-sentence and the source-sentence before.
    std::string_view pre = response.source.gap(sentence_id);
    response.target.appendSentence(pre, views.begin(), views.end());

    // If this is the last history to be decoded and translated-text
    // constructed, append the text till the end, which could be spaces or
    // empty.
    if (sentence_id + 1 == histories.size()) {
      response.target.appendEndingWhitespace(
          response.source.gap(sentence_id + 1));
    }
  }

  promise_.set_value(std::move(response));
}

}  // namespace slimt
