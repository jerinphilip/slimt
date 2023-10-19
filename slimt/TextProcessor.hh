#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/Splitter.hh"
#include "slimt/Types.hh"

namespace slimt {
class Aligned;
class AnnotatedText;
class Vocabulary;

class TextProcessor {
  /// TextProcessor handles loading the sentencepiece vocabulary and also
  /// contains an instance of sentence-splitter based on ssplit.
  ///
  /// Used in Service to convert an incoming blob of text to a vector of
  /// sentences (vector of words). In addition, the Ranges of the
  /// source-tokens in unnormalized text are provided as string_views.
 public:
  /// Construct TextProcessor from options, vocabs and prefix-file supplied as a
  /// bytearray. For other parameters, see the path based constructor. Note:
  /// This falls back to string based loads if memory is null, this behaviour
  /// will be deprecated in the future.
  ///
  /// @param [in] memory: ssplit-prefix-file contents in memory, passed as a
  /// bytearray.
  TextProcessor(const std::string &mode, const Vocabulary &vocabulary,
                const Aligned &memory);

  std::tuple<AnnotatedText, Segments> process(std::string &&input,
                                              size_t wrap_length) const;
  std::tuple<AnnotatedText, Segments> process(AnnotatedText &source) const;

 private:
  /// Tokenizes an input string, returns Words corresponding. Loads the
  /// corresponding byte-ranges into word_ranges.
  Segment tokenize(const std::string_view &segment,
                   std::vector<std::string_view> &word_ranges) const;

  /// Wrap into sentences of at most wrap_length tokens and add to source.
  void wrap(Segment &segment, std::vector<std::string_view> &word_ranges,
            Segments &segments, AnnotatedText &source,
            size_t wrap_length) const;

  /// Mode of splitting, can be line ('\n') based, paragraph based, also
  /// supports a wrapped mode.
  slimt::SentenceStream::splitmode ssplit_mode_;

  const Vocabulary &vocabulary_;  ///< Vocabularies used to tokenize a sentence
  /// SentenceSplitter compatible with moses sentence-splitter
  slimt::Splitter ssplit_;
};

}  // namespace slimt
