#include "slimt/ResponseBuilder.hh"

#include "slimt/Response.hh"

namespace slimt {

void ResponseBuilder::operator()(Histories &&histories) {
  // TODO(jerinphilip) load Options into options and turn build
  // functions on or off.
  // options_ is unused, but we can try something here.
  SLIMT_ABORT_IF(source_.numSentences() != histories.size(),
                 "Mismatch in source and translated sentences");
  Response response;

  // Move source_ into response.
  response.source = std::move(source_);
  // Reserving length at least as much as source_ seems like a reasonable
  // thing to do to avoid reallocations.
  response.target.text.reserve(response.source.text.size());

  for (size_t sentence_idx = 0; sentence_idx < histories.size();
       sentence_idx++) {
    // TODO(jerin): Change hardcode of nBest = 1

    Words words /*= std::get<0>(result)*/;

    auto [decoded, views] = vocabulary_.decode(words, /*ignore_eos=*/false);

    // For each sentence, prepend the filler text between the corresponding
    // source-sentence and the source-sentence before.
    std::string_view pre = response.source.gap(sentence_idx);
    response.target.appendSentence(pre, views.begin(), views.end());

    // If this is the last history to be decoded and translated-text
    // constructed, append the text till the end, which could be spaces or
    // empty.
    if (sentence_idx + 1 == histories.size()) {
      response.target.appendEndingWhitespace(
          response.source.gap(sentence_idx + 1));
    }
  }

  promise_.set_value(std::move(response));
}

}  // namespace slimt