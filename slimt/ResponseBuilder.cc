#include "slimt/ResponseBuilder.hh"

#include "slimt/Response.hh"

namespace slimt {

void ResponseBuilder::operator()(Histories &&histories) const {
  // TODO(jerinphilip) load Options into options and turn build
  // functions on or off.
  // options_ is unused, but we can try something here.
  SLIMT_ABORT_IF(source_.numSentences() != histories.size(),
                 "Mismatch in source and translated sentences");
  Response response;

  // Move source_ into response.
  response.source = std::move(source_);

  // Should be after source is set
  buildTranslatedText(histories, response);

  if (options_.alignment || options_.HTML) {
    buildAlignments(histories, response);
  }

  callback_(std::move(response));
}

void ResponseBuilder::buildAlignments(Histories &histories,
                                      Response &response) {
  (void)histories;
  (void)response;
  /*
  for (auto &history : histories) {
    // TODO(jerin): Change hardcode of nBest = 1
    NBestList onebest = history->nBest(1);

    Result result = onebest[0];  // Expecting only one result;
    Words words = std::get<0>(result);
    auto hyp = std::get<1>(result);
    auto softAlignment = hyp->tracebackAlignment();
    response.alignments.push_back(std::move(softAlignment));
  }
  */
}

void ResponseBuilder::buildTranslatedText(Histories &histories,
                                          Response &response) const {
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
}

}  // namespace slimt
