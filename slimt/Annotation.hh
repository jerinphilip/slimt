#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "slimt/Types.hh"

namespace slimt {

/// Annotation expresses sentence and token boundary information as ranges of
/// bytes in a string, but does not itself own the string.
///
/// See also AnnotatedText, which owns Annotation and the string. AnnotatedText
/// wraps these Range functions to provide a std::string_view interface.
///
/// Text is divided into gaps (whitespace between sentences) and sentences like
/// so:
///   gap sentence gap sentence gap
/// Because gaps appear at the beginning and end of the text, there's always
/// one more gap than there are sentences.
///
/// The entire text is a unbroken sequence of tokens (i.e. the end of a token
/// is the beginning of the next token).  A gap is exactly one token containing
/// whatever whitespace is between the sentences.  A sentence is a sequence of
/// tokens.
///
/// Since we are using SentencePiece, a token can include whitespace.  The term
/// "word" is used, somewhat incorrectly, as a synonym of token.
///
/// A gap can be empty (for example there may not have been whitespace at the
/// beginning).  A sentence can also be empty (typically the translation system
/// produced empty output).  That's fine, these are just empty ranges as you
/// would expect.

class Annotation {
 public:
  /// Initially an empty string.  Populated by AnnotatedText.
  Annotation() {
    token_begin_.push_back(0);
    token_begin_.push_back(0);
    gap_.push_back(0);
  }

  size_t sentence_count() const { return gap_.size() - 1; }

  /// Returns number of words in the sentence identified by `sentence_id`.
  size_t word_count(size_t sentence_id) const {
    return gap_[sentence_id + 1] - gap_[sentence_id] - 1 /* minus the gap */;
  }

  /// Returns a Range representing `word_id` in sentence indexed by
  /// `sentence_id`. `word_id` follows 0-based indexing, and should be less
  /// than
  /// `.word_count()` for `sentence_id` for defined behaviour.
  Range word(size_t sentence_id, size_t word_id) const {
    size_t token_idx = gap_[sentence_id] + 1 + word_id;
    return Range{token_begin_[token_idx], token_begin_[token_idx + 1]};
  }

  /// Returns a Range representing sentence corresponding to `sentence_id`.
  /// `sentence_id` follows 0-based indexing, and behaviour is defined only
  /// when less than `.sentence_count()`.
  Range sentence(size_t sentence_id) const {
    return Range{
        token_begin_[gap_[sentence_id] + 1], /*end of whitespace before */
        token_begin_[gap_[sentence_id + 1]]  /*beginning of whitespace after */
    };
  }

  Range gap(size_t gap_idx) const {
    size_t token_idx = gap_[gap_idx];
    return Range{token_begin_[token_idx], token_begin_[token_idx + 1]};
  }

  void update(const std::vector<Range> &words) {
    if (words.empty()) {
      return;
    }

    token_begin_.clear();
    token_begin_.push_back(0);

    for (const auto &word : words) {
      token_begin_.push_back(word.begin);
    }
    token_begin_.push_back(words.back().end);
    // The last range is repated to denote EOS [sentence_length,
    // sentence_length].
    token_begin_.push_back(words.back().end);
  }

 private:
  friend class AnnotatedText;
  /// Map from token index to byte offset at which it begins.  Token i is:
  ///   [token_begin_[i], token_begin_[i+1])
  /// The vector is padded so that these indices are always valid, even at the
  /// end.  So tokens_begin_.size() is the number of tokens plus 1.
  std::vector<size_t> token_begin_;

  /// Indices of tokens that correspond to gaps between sentences.  These are
  /// indices into token_begin_.
  /// Gap g is byte range:
  ///   [token_begin_[gap_[w]], token_begin_[gap_[w]+1])
  /// Sentence s is byte range:
  ///   [token_begin_[gap_[s]+1], token_begin_[gap_[s+1]])
  /// A sentence does not include whitespace at the beginning or end.
  ///
  /// gap_.size() == sentence_count() + 1.
  ///
  /// Example: empty text "" -> just an empty gap.
  /// token_begin_ = {0, 0};
  /// gap_ = {0};
  ///
  /// Example: only space " " -> just a gap containing the space.
  /// token_begin_ = {0, 1};
  /// gap_ = {0};
  ///
  /// Example: one token "hi" -> empty gap, sentence with one token, empty gap
  /// token_begin_ = {0, 0, 2, 2};
  /// gap_ = {0, 2};
  std::vector<size_t> gap_;
};

/// AnnotatedText is effectively std::string text + Annotation, providing the
/// following additional desiderata.
///
/// 1. Access to processed std::string_views for convenience rather than
/// Ranges (which only provides index information).
///
/// 2. Transparently convert std::string_views into Ranges for the
/// Annotation referring to the text bound by this structure.
///
/// 3. Bind the text and annotations together, to move around as a meaningful
/// unit.
class AnnotatedText {
 public:
  std::string text;       ///< Blob of string elements in annotation refers to.
  Annotation annotation;  ///< sentence and (sub-) word annotations.

  /// Construct an empty AnnotatedText. This is useful when the target string or
  /// Ranges are not known yet, but the public members can be used to
  /// populate it. One use-case, when translated-text is created decoding from
  /// histories and the Ranges only known after the string has been
  /// constructed.
  AnnotatedText() = default;

  /// Construct moving in a string (for efficiency purposes, copying string
  /// constructor is disallowed).
  explicit AnnotatedText(std::string &&text);

  /// Appends a sentence to the existing text and transparently rebases
  /// std::string_views.  Since this tracks only prefix, remember
  /// append_ending_whitespace.
  /// The std::string_views must not already be in text.
  void append_sentence(std::string_view prefix,
                       std::vector<std::string_view>::iterator tokens_begin,
                       std::vector<std::string_view>::iterator tokens_end);

  /// Append the whitespace at the end of input. std::string_view must not be in
  /// text.
  void append_ending_whitespace(std::string_view whitespace);
  void update(const std::vector<Range> &words);
  void to(Encoding encoding);

  /// Package the existence of a sentence that is already in text.  The
  /// iterators are over std::string_views for each token that must be in text
  /// already.  This function must be called to record sentences in order.
  /// Normally the beginning of the sentence can be inferred from
  /// tokens_begin->data() but the tokens could be empty, so sentence_begin is
  /// required to know where the sentence is.
  void record_existing_sentence(
      std::vector<std::string_view>::iterator tokens_begin,
      std::vector<std::string_view>::iterator tokens_end,
      const char *sentence_begin);

  /// Returns the number of sentences in the annotation structure.
  size_t sentence_count() const { return annotation.sentence_count(); }

  /// Returns number of words in the sentece identified by sentence_id.
  size_t word_count(size_t sentence_id) const {
    return annotation.word_count(sentence_id);
  }

  /// Returns a std::string_view representing word_id in sentence_id
  std::string_view word(size_t sentence_id, size_t word_id) const {
    return as_view(annotation.word(sentence_id, word_id));
  }

  /// Returns a std::string_view representing sentence corresponding to
  /// sentence_id.
  std::string_view sentence(size_t sentence_id) const {
    return as_view(annotation.sentence(sentence_id));
  }

  /// Returns the std::string_view of the gap between two sentences in the
  /// container.
  ///
  /// More precisely where `i = sentence_id, N = sentence_count()` for brevity:
  ///
  /// * For `i = 0`: The gap between the start of text and the 0th sentence.
  /// * For `i = 1...N-1`, returns the text comprising of the gap
  ///   between the `i`-th and `i+1`-th sentence.
  /// * For `i = N`, the gap between the last (N-1th) sentence and end of
  ///   text.
  /// @param sentence_id: Can be between `[0, sentence_count()]`.
  std::string_view gap(size_t sentence_id) const {
    return as_view(annotation.gap(sentence_id));
  }

  /// Returns a Range representing word_id in sentence_id
  Range word_as_range(size_t sentence_id, size_t word_id) const {
    return annotation.word(sentence_id, word_id);
  }

  /// Returns a Range representing sentence corresponding to sentence_id.
  Range sentence_as_range(size_t sentence_id) const {
    return annotation.sentence(sentence_id);
  }

  /// Utility function to call `fun` on each word (subword token effectively) in
  /// an `AnnotatedText`. `fun` is called with the `Range`, the
  /// `std::string_view` with the word, and a `bool` to indicate whether it is
  /// the last word in the `AnnotatedText`, which is also the ending whitespace
  /// slot of AnnotatedText.
  template <typename Fun>
  AnnotatedText apply(Fun fun) const {
    AnnotatedText out;

    for (size_t sentence_id = 0; sentence_id < sentence_count();
         ++sentence_id) {
      std::string sentence;
      std::vector<Range> tokens;

      std::string prefix =
          fun(annotation.gap(sentence_id), gap(sentence_id), false);

      for (size_t word_id = 0; word_id < word_count(sentence_id); ++word_id) {
        std::string token = fun(word_as_range(sentence_id, word_id),
                                word(sentence_id, word_id), false);
        tokens.push_back(
            Range{sentence.size(), sentence.size() + token.size()});
        sentence += token;
      }

      // Convert our Ranges to std::string_views since that's what
      // append_sentence expects
      std::vector<std::string_view> views(tokens.size());
      std::transform(tokens.begin(), tokens.end(), views.begin(),
                     [&](const Range &range) {
                       return std::string_view(sentence.data() + range.begin,
                                               range.size());
                     });

      out.append_sentence(prefix, views.begin(), views.end());
    }

    out.append_ending_whitespace(
        fun(annotation.gap(sentence_count()), gap(sentence_count()), true));

    return out;
  }

 private:
  std::string_view as_view(const Range &range) const {
    return std::string_view(text.data() + range.begin, range.size());
  }
  Encoding encoding_ = Encoding::Byte;
};

class WordIterator {
 public:
  explicit WordIterator(const AnnotatedText &annotated)
      : annotated_(annotated) {}
  Range &operator*();
  Range *operator->();
  WordIterator &operator++();
  bool has_next();

 private:
  const AnnotatedText &annotated_;
  size_t sentence_idx_ = 0;
  size_t word_idx_ = 0;
  Range range_;
};

// Returns a sequence length for a UTF-8 multi-byte sequence starting with the
// character. Continuation bytes return 0. 1-byte, 2-byte, 3-byte, 4-byte
// multisequences return their respective length for the start character.
int utf8_sequence_length(char c);

}  // namespace slimt
