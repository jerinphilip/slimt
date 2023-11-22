#include "Annotation.hh"

#include <cassert>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Macros.hh"

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
void AnnotatedText::update(const std::vector<Range> &words) {
  annotation.update(words);
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

  if (encoding == Encoding::Byte && encoding_ == Encoding::UTF8) {
    WordIterator current(*this);

    std::vector<Range> words;

    size_t byte_idx = 0;
    Range byte{.begin = byte_idx, .end = 0};

    const char *marker = text.data();
    for (; current.has_next(); ++current) {
      byte.begin = byte_idx;

      for (size_t idx = (*current).begin; idx != (*current).end; idx++) {
        ++marker;
        ++byte_idx;

        if ((*marker & 0xc0) == 0x80) {
          ++byte_idx;
          ++marker;
        }
      }
      byte.end = byte_idx;
      words.push_back(byte);
    }

    annotation.update(words);
    encoding_ = Encoding::Byte;
  } else if (encoding == Encoding::UTF8 && encoding_ == Encoding::Byte) {
    WordIterator current(*this);

    std::vector<Range> words;

    size_t utf8_idx = 0;
    size_t byte_idx = 0;
    Range utf8{
        //
        .begin = utf8_idx,  //
        .end = 0            //
    };
    // This loops run on the entire string.
    //
    // We have indices into two views of the same string. One is bytes, the
    // other is utf8 encoded. We want to convert what is bytes to utf8.
    //
    // For this, we traverse through the characters, checking for unicode
    // encoding and associated adjustments to the utf8_idx, keeping
    // correspondences with the byte_idx;
    for (const char c : text) {
      // current = [begin, end)
      if ((c & 0xc0) != 0x80)  // if is not utf-8 continuation character
        ++utf8_idx;

      ++byte_idx;

      if (byte_idx == (*current).end) {
        utf8.end = utf8_idx;

        // Push it to the list of (utf8) words.
        // We will use these to create the utf8 range annotation.
        words.push_back(utf8);
        ++current;

        utf8.begin = utf8_idx;
      }
    }
    annotation.update(words);
    encoding_ = Encoding::UTF8;
  } else {
    SLIMT_ABORT("Unimplemented");
  }
}

WordIterator &WordIterator::operator++() {
  ++word_idx_;
  if (word_idx_ >= annotated_.word_count(sentence_idx_)) {
    ++sentence_idx_;
    word_idx_ = 0;
  }
  return *this;
}

Range WordIterator::operator*() {
  range_ = annotated_.word_as_range(sentence_idx_, word_idx_);
  return range_;
}

bool WordIterator::has_next() {
  return sentence_idx_ < annotated_.sentence_count() &&
         word_idx_ < annotated_.word_count(sentence_idx_);
}
}  // namespace slimt
