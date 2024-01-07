#include "Annotation.hh"

#include <cassert>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Macros.hh"
#include "slimt/Types.hh"

namespace slimt {

AnnotatedText::AnnotatedText(std::string &&t) : text(std::move(t)) {
  // Treat the entire text as a gap that record_existing_sentence will break.
  annotation.token_begin_.back() = text.size();
}

void AnnotatedText::append_sentence(
    std::string_view prefix, std::vector<std::string_view>::iterator begin,
    std::vector<std::string_view>::iterator end) {
  assert(annotation.token_begin_.back() == text.size());

  // prefix is just end of the previous one.
  append_ending_whitespace(prefix);

  // Appending sentence text.
  std::size_t offset = text.size();
  for (auto token = begin; token != end; ++token) {
    offset += token->size();
    annotation.token_begin_.push_back(offset);
  }
  if (begin != end) {
    text.append(begin->data(), (end - 1)->data() + (end - 1)->size());
    assert(offset == text.size());  // Tokens should be contiguous.
  }

  // Add the gap after the sentence.  This is empty for now, but will be
  // extended with append_ending_whitespace or another append_sentence.
  annotation.gap_.push_back(annotation.token_begin_.size() - 1);
  annotation.token_begin_.push_back(offset);
}

void AnnotatedText::append_ending_whitespace(std::string_view whitespace) {
  text.append(whitespace.data(), whitespace.size());
  annotation.token_begin_.back() = text.size();
}
void AnnotatedText::update(const std::vector<size_t> &token_begin) {
  annotation.update(token_begin);
}

void AnnotatedText::record_existing_sentence(
    std::vector<std::string_view>::iterator begin,
    std::vector<std::string_view>::iterator end, const char *sentence_begin) {
  assert(sentence_begin >= text.data());
  assert(sentence_begin <= text.data() + text.size());
  assert(begin == end || sentence_begin == begin->data());
  assert(!annotation.token_begin_.empty());
  assert(annotation.token_begin_.back() == text.size());
  // Clip off size token ending.
  annotation.token_begin_.pop_back();
  for (auto i = begin; i != end; ++i) {
    assert(i->data() >= text.data());                            // In range.
    assert(i->data() + i->size() <= text.data() + text.size());  // In range
    assert(i + 1 == end ||
           i->data() + i->size() == (i + 1)->data());  // Contiguous
    annotation.token_begin_.push_back(i->data() - text.data());
  }
  // Gap token after sentence.
  annotation.gap_.push_back(annotation.token_begin_.size());
  if (begin != end) {
    annotation.token_begin_.push_back((end - 1)->data() + (end - 1)->size() -
                                      text.data());
  } else {
    // empty sentence.
    annotation.token_begin_.push_back(sentence_begin - text.data());
  }
  // Add back size token ending.
  annotation.token_begin_.push_back(text.size());
}

void AnnotatedText::to(Encoding encoding) {
  if (encoding == encoding_) {
    return;
  }

  if (encoding_ == Encoding::UTF8 && encoding == Encoding::Byte) {
    // Encoding::UTF8 -> Encoding::Byte
    std::vector<size_t> token_begin;

    size_t byte_idx = 0;
    size_t utf8_idx = 0;
    size_t token_begin_idx = 0;

    while (token_begin_idx < annotation.token_begin_.size() &&
           utf8_idx == annotation.token_begin_[token_begin_idx]) {
      token_begin.push_back(byte_idx);
      token_begin_idx++;
    }

    const char *marker = text.data();
    while (marker != text.data() + text.size()) {
      int sequence_length = utf8_sequence_length(*marker);
      utf8_idx += 1;
      marker += sequence_length;
      byte_idx += sequence_length;

      while (token_begin_idx < annotation.token_begin_.size() &&
             utf8_idx == annotation.token_begin_[token_begin_idx]) {
        token_begin.push_back(byte_idx);
        token_begin_idx++;
      }
    }

    annotation.update(token_begin);
    encoding_ = Encoding::Byte;
  } else if (encoding_ == Encoding::Byte && encoding == Encoding::UTF8) {
    // Encoding::Byte -> Encoding::UTF8
    std::vector<size_t> token_begin;

    // We have indices into two views of the same string. One is bytes, the
    // other is utf8 encoded. We want to convert what is bytes to utf8.
    //
    // Iterate through byte-indices. The skip is a unicode multi-byte character
    // sequence.
    size_t byte_idx = 0;
    size_t utf8_idx = 0;
    size_t token_begin_idx = 0;

    while (token_begin_idx < annotation.token_begin_.size() &&
           annotation.token_begin_[token_begin_idx] == byte_idx) {
      token_begin.push_back(utf8_idx);
      token_begin_idx++;
    }

    while (byte_idx < text.size()) {
      char c = text[byte_idx];
      int sequence_length = utf8_sequence_length(c);
      if (sequence_length > 0) {
        // If not a continuation-character (i.e, start of any multi-byte
        // sequence), the unicode index increments by 1.
        ++utf8_idx;
        byte_idx += sequence_length;
      } else {
        // Conti
        byte_idx++;
      }

      // If byte_idx is token_begin_idx, then be sure to add the corresponding
      // utf8_idx
      while (token_begin_idx < annotation.token_begin_.size() &&
             annotation.token_begin_[token_begin_idx] == byte_idx) {
        token_begin.push_back(utf8_idx);
        token_begin_idx++;
      }
    }

    annotation.update(token_begin);
    encoding_ = Encoding::UTF8;
  } else {
    SLIMT_ABORT("Unimplemented");
  }
}

int utf8_sequence_length(char c) {
  // char is 8 bit (1 byte). "xxxxxxxx". Per UTF-8 Encoding rules:
  //   * first 1 bit  is  "0xxxxxxx" => start of a 1-byte character.
  //   * first 3 bits are "110xxxxx" => start of a 2-byte sequence.
  //   * first 4 bits are "1110xxxx" => start of a 3-byte sequence.
  //   * first 5 bits are "11110xxx" => start of a 4-byte sequence.

  // Check leading bit. 0x80 = 1000000, & gets us first bit.
  if ((c & 0x80) == 0) {  // NOLINT
    return 1;
  }

  if ((c & 0xE0) == 0xC0) {  // NOLINT
    // Check leading 3-bits. 0xE0 = 11100000, & gets us the first 3-bits.
    return 2;
  } else if ((c & 0xF0) == 0xE0) {  // NOLINT
    // Check leading 4-bits. 0xF0 = 11110000, & gets us the first 4-bits.
    return 3;
  } else if ((c & 0xF8) == 0xF0) {  // NOLINT
    // Check leading 5-bits. 0xF8 = 11111000, & gets us the first 5-bits.
    return 4;
  }

  // Not a valid start of a multi-byte sequence. Possibly a UTF-8 continuation
  // character.
  return 0;
}

WordIterator &WordIterator::operator++() {
  ++word_idx_;
  if (word_idx_ >= annotated_.word_count(sentence_idx_)) {
    ++sentence_idx_;
    word_idx_ = 0;
  }
  return *this;
}

Range &WordIterator::operator*() {
  range_ = annotated_.word_as_range(sentence_idx_, word_idx_);
  return range_;
}

Range *WordIterator::operator->() {
  range_ = annotated_.word_as_range(sentence_idx_, word_idx_);
  return &range_;
}

bool WordIterator::has_next() {
  return sentence_idx_ < annotated_.sentence_count() &&
         word_idx_ < annotated_.word_count(sentence_idx_);
}
}  // namespace slimt
