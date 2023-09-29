#include "slimt/Splitter.hh"

#include <string>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Macros.hh"

namespace slimt {

namespace {
ug::ssplit::SentenceStream::splitmode string2splitmode(const std::string &m) {
  using splitmode = ug::ssplit::SentenceStream::splitmode;
  if (m == "sentence") {
    return splitmode::one_sentence_per_line;
  }
  if (m == "paragraph") {
    return splitmode::one_paragraph_per_line;
  }
  if (m == "wrapped_text") {
    return splitmode::wrapped_text;
  }
  SLIMT_ABORT(
      "Unknown ssplitmode {}, Please choose one of "
      "{sentence,paragraph,wrapped_text}");
}

ug::ssplit::SentenceSplitter load_splitter(const std::string &prefix_path) {
  // Temporarily supports empty, will be removed when mozilla passes
  // prefix_path
  ug::ssplit::SentenceSplitter splitter;
  if (!prefix_path.empty()) {
    LOG(info, "Loading protected prefixes for sentence splitting from {}",
        prefix_path);
    splitter.load(prefix_path);
  } else {
    LOG(warn,
        "Missing list of protected prefixes for sentence splitting. "
        "Set with --ssplit-prefix-file.");
  }
  return splitter;
}

ug::ssplit::SentenceSplitter load_splitter(const Aligned &memory) {
  // Temporarily supports empty, will be removed when mozilla passes memory
  ug::ssplit::SentenceSplitter splitter;
  if (!memory.empty()) {
    std::string_view serialized(memory.begin(), memory.size());
    splitter.loadFromSerialized(serialized);
  }
  return splitter;
}

}  // namespace

Segment TextProcessor::tokenize(
    const std::string_view &segment,
    std::vector<std::string_view> &word_ranges) const {
  auto [words, views] = vocabulary_.encode(segment, /*add_eos=*/false);
  word_ranges.reserve(word_ranges.size() +
                      distance(views.begin(), views.end()));
  word_ranges.insert(word_ranges.end(), views.begin(), views.end());
  return words;
}

TextProcessor::TextProcessor(size_t wrap_length, const std::string &mode,
                             const Vocabulary &vocabulary,
                             const std::string &prefix_path)
    : wrap_length_(wrap_length),
      ssplit_mode_(string2splitmode(mode)),
      vocabulary_(vocabulary),
      ssplit_(load_splitter(prefix_path)) {}

TextProcessor::TextProcessor(size_t wrap_length, const std::string &mode,
                             const Vocabulary &vocabulary,
                             const Aligned &memory)
    : wrap_length_(wrap_length),
      ssplit_mode_(string2splitmode(mode)),
      vocabulary_(vocabulary) {
  // This is not the best of the solutions at the moment, but is consistent
  // with // what happens among other structures like model, vocabulary or
  // shortlist.  First, we check if the bytearray is empty. If not, we load
  // from ByteArray.  In case empty, the string based loader which reads from
  // file is called.  However, ssplit allows for not supplying
  // ssplit-prefix-file where-in the purely regular expression based splitter
  // is activated.
  //
  // For now, we allow not supplying an ssplit-prefix-file.

  SLIMT_ABORT_IF(memory.empty(),
                 "ssplit: Empty blob supplied for initialization.");
  ssplit_ = load_splitter(memory);
}

std::tuple<AnnotatedText, Segments> TextProcessor::process(
    std::string &&input) const {
  AnnotatedText source(std::move(input));
  Segments segments;
  std::string_view input_converted(source.text.data(), source.text.size());
  auto sentence_stream =
      ug::ssplit::SentenceStream(input_converted, ssplit_, ssplit_mode_);

  std::string_view sentence_string_piece;

  while (sentence_stream >> sentence_string_piece) {
    std::string_view sentence(sentence_string_piece.data(),
                              sentence_string_piece.size());

    std::vector<std::string_view> word_ranges;
    Segment segment = tokenize(sentence, word_ranges);

    // There are some cases where SentencePiece or vocab returns no words
    // after normalization. 0 prevents any empty entries from being added.
    if (!segment.empty()) {
      // Wrap segment into sentences of at most wrap_length_ tokens and
      // tell source about them.
      wrap(segment, word_ranges, segments, source);
    }
  }
  return {std::move(source), std::move(segments)};
}

void TextProcessor::wrap(Segment &segment,
                         std::vector<std::string_view> &word_ranges,
                         Segments &segments, AnnotatedText &source) const {
  // There's an EOS token added to the words, manually.
  // SentencePiece/marian-vocab is set to not append EOS. Marian requires EOS to
  // be at the end as a marker to start translating. So while we're supplied
  // wrap_length_ from outside, we need to ensure there's space for EOS in
  // each wrapped segment.
  Word eos_id = vocabulary_.eos_id();
  size_t wrap_step_size = wrap_length_ - 1;

  for (size_t offset = 0; offset < segment.size(); offset += wrap_step_size) {
    auto start = segment.begin() + offset;

    // Restrict the range within bounds.
    size_t left = segment.size() - offset;
    size_t diff = std::min(wrap_step_size, left);

    segments.emplace_back(start, start + diff);
    segments.back().push_back(eos_id);

    auto astart = word_ranges.begin() + offset;

    // Construct a part vector of std::string_view representing wrapped segment,
    // use the last std::string_view to create an EOS std::string_view manually.
    std::vector<std::string_view> partial_word_ranges(astart, astart + diff);
    std::string_view &last = partial_word_ranges.back();
    const char *end = last.data() + last.size();
    partial_word_ranges.emplace_back(end, 0);
    // diff > 0
    source.record_existing_sentence(partial_word_ranges.begin(),
                                    partial_word_ranges.end(), astart->data());
  }
}

void TextProcessor::process_from_annotation(AnnotatedText &source,
                                          Segments &segments) const {
  std::string text = source.text;
  AnnotatedText replacement(std::move(text));

  for (size_t s = 0; s < source.sentence_count(); s++) {
    // This is our sentence_stream
    Range sentence_range = source.sentence_as_range(s);

    // Fool tokenization using Ranges into looking at replacement. They're
    // same, so okay.
    std::string_view sentence{&replacement.text[sentence_range.begin],
                              sentence_range.size()};

    std::vector<std::string_view> word_ranges;
    Segment segment = tokenize(sentence, word_ranges);

    // Manually add EoS
    Word eos_id = vocabulary_.eos_id();
    segment.push_back(eos_id);

    if (!word_ranges.empty()) {
      std::string_view &last =
          word_ranges.back();  // this is a possible segfault if
                               // word_ranges is empty. So guard.
      const char *end = last.data() + last.size();
      word_ranges.emplace_back(end, 0);
    } else {
      const char *end = sentence.data() + sentence.size();
      word_ranges.emplace_back(end, 0);
    }

    segments.push_back(std::move(segment));
    replacement.record_existing_sentence(word_ranges.begin(), word_ranges.end(),
                                         word_ranges.begin()->data());
  }

  source = replacement;
}

}  // namespace slimt
