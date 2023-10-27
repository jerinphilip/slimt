// https://www.codeproject.com/Articles/14076/Fast-and-Compact-HTML-XML-Scanner-Tokenizer
// BSD license

#include "slimt/XHScanner.hh"

#include <cctype>
#include <cstring>
#include <string_view>

namespace {

// Simple replacement for string_view.ends_with(compile-time C string)
template <typename CharType, size_t Len>
inline bool ends_with(markup::StringRef &str, const CharType (&suffix)[Len]) {
  size_t offset = str.size - (Len - 1);
  return offset <= str.size &&
         std::memcmp(str.data + offset, suffix, Len - 1) == 0;
}

inline bool equals_case_insensitive(const char *lhs, const char *rhs,
                                    size_t len) {
  for (size_t i = 0; i < len; ++i) {
    // cast to unsigned char otherwise std::tolower has undefined behaviour
    if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
        std::tolower(static_cast<unsigned char>(rhs[i])))
      return false;
  }

  return true;
}

// Alias for the above, but with compile-time known C string
template <size_t Len>
inline bool equals_case_insensitive(markup::StringRef &lhs,
                                    const char (&rhs)[Len]) {
  return lhs.size == Len - 1 && equals_case_insensitive(lhs.data, rhs, Len - 1);
}

template <typename CharType, size_t Len>
bool operator==(const markup::StringRef &str, const CharType (&str2)[Len]) {
  return str.size == Len - 1 && std::memcmp(str.data, str2, Len - 1) == 0;
}

template <size_t N>
constexpr size_t length(const char (&unused)[N]) {
  (void)unused;
  return N - 1;
}

}  // end namespace

namespace markup {

// case sensitive string equality test
// s_lowcase shall be lowercase string
std::string_view Scanner::value() const {
  return std::string_view(value_.data, value_.size);
}

std::string_view Scanner::attribute() const {
  return std::string_view(attribute_.data, attribute_.size);
}

std::string_view Scanner::tag() const {
  return std::string_view(tag_.data, tag_.size);
}

Scanner::TokenType Scanner::scan_body() {
  value_ = StringRef{input_.pos(), 0};

  start_ = input_.pos();

  switch (input_.peek()) {
    case '\0':
      return TT_EOF;
    case '<':
      return scan_tag();
    case '&':
      return scan_entity(TT_TEXT);
    default:
      break;
  }

  while (true) {
    switch (input_.peek()) {
      case '\0':
      case '<':
      case '&':
        return TT_TEXT;
      default:
        input_.consume();
        ++value_.size;
        break;
    }
  }
}

// Consumes one or closing bit of a tag:
//   <tag attr="value">...</tag>
//       |------------|
// Followed by:
// - scan_special if <script> or <style>
// - scan_body
// - another scan_head for the next attribute or end of open tag
// Returns:
// - TT_ATTRIBUTE if attribute is read
// - TT_TAG_END if self-closing tag
// - TT_ERROR if wrong character encountered
// - TT_EOF if unexpected end of input (will not return TT_ATTRIBUTE if
// attribute value wasn't finished yet)
// - TT_TAG_END through scan_special
// - TT_TEXT through scan_body
Scanner::TokenType Scanner::scan_attribute() {
  // Skip all whitespace between tag name or last attribute and next attribute
  // or '>'
  skip_whitespace();

  // Find end of tag name
  switch (input_.peek()) {
    case '>':
      input_.consume();

      // Treat some elements as opaque, e.g. <script>, <style>
      if (/*equals_case_insensitive(tag_, "title") ||*/
          equals_case_insensitive(tag_, "script") ||
          equals_case_insensitive(tag_, "style") ||
          equals_case_insensitive(tag_, "textarea") ||
          equals_case_insensitive(tag_, "iframe") ||
          equals_case_insensitive(tag_, "noembed") ||
          equals_case_insensitive(tag_, "noscript") ||
          equals_case_insensitive(tag_, "noframes")) {
        // script is special because we want to parse the attributes,
        // but not the content
        scanFun_ = &Scanner::scan_special;
        return scan_special();
      } else {  // NOLINT
        scanFun_ = &Scanner::scan_body;
        return scan_body();
      }
    case '/':
      input_.consume();
      if (input_.peek() == '>') {
        // self closing tag
        input_.consume();
        scanFun_ = &Scanner::scan_body;
        return TT_TAG_END;
      } else {  // NOLINT
        return TT_ERROR;
      }
    default:
      break;
  }

  attribute_ = StringRef{input_.pos(), 0};
  value_ = StringRef{nullptr, 0};

  // attribute name...
  while (input_.peek() != '=') {
    switch (input_.peek()) {
      case '\0':
        return TT_EOF;
      case '>':
        return TT_ATTRIBUTE;  // attribute without value (HTML style) at end of
                              // tag
      case '<':
        return TT_ERROR;
      default:
        if (skip_whitespace()) {
          if (input_.peek() == '=') {
            break;
          } else {                // NOLINT
            return TT_ATTRIBUTE;  // attribute without value (HTML style) but
                                  // not yet at end of tag
          }
        }
        input_.consume();
        ++attribute_.size;
        break;
    }
  }

  // consume '=' and any following whitespace
  input_.consume();
  skip_whitespace();
  // attribute value...

  char
      quote;  // Either '"' or '\'' depending on which quote we're searching for
  switch (input_.peek()) {
    case '"':
    case '\'':
      quote = input_.consume();
      value_ = StringRef{input_.pos(), 0};
      while (true) {
        if (input_.peek() == '\0') {
          return TT_ERROR;
        } else if (input_.peek() == quote) {  // NOLINT
          input_.consume();
          return TT_ATTRIBUTE;
        } else {
          input_.consume();
          ++value_.size;
        }
      }
      break;
    default:
      value_ = StringRef{input_.pos(), 0};

      while (true) {
        if (is_whitespace(input_.peek())) return TT_ATTRIBUTE;
        if (input_.peek() == '>')
          return TT_ATTRIBUTE;  // '>' will be consumed next round
        input_.consume();
        ++value_.size;
      }
      break;
  }

  // How did we end up here?!
  return TT_ERROR;
}

// scans tag name of open or closing tag
//   <tag attr="value">...</tag>
//   |--|                 |----|
// Emits:
// - TT_TAG_START if tag head is read
// - TT_COMMENT_START
// - TT_PROCESSING_INSTRUCTION_START
// - TT_CDATA_START
// - TT_ENTITY_START
// - TT_ERROR if unexpected character or end
Scanner::TokenType Scanner::scan_tag() {
  start_ = input_.pos();
  if (input_.consume() != '<') return TT_ERROR;

  bool is_tail = input_.peek() == '/';
  if (is_tail) input_.consume();

  tag_ = StringRef{input_.pos(), 0};

  while (input_.peek()) {
    if (skip_whitespace()) break;

    if (input_.peek() == '/' || input_.peek() == '>') break;

    input_.consume();
    ++tag_.size;

    // Note: these tests are executed at every char, thus eager.
    // "<?xml" will match on `tag_ == "?"`.
    if (tag_ == "!--") {
      scanFun_ = &Scanner::scan_comment;
      return TT_COMMENT_START;
    } else if (tag_ == "?") {  // NOLINT
      scanFun_ = &Scanner::scan_processing_instruction;
      return TT_PROCESSING_INSTRUCTION_START;
    }
  }

  if (!input_.peek()) return TT_EOF;

  if (is_tail) return input_.consume() == '>' ? TT_TAG_END : TT_ERROR;

  scanFun_ = &Scanner::scan_attribute;
  return TT_TAG_START;
}

Scanner::TokenType Scanner::scan_entity(TokenType parentTokenType) {
  // `entity` includes starting '&' and ending ';'
  start_ = input_.pos();
  StringRef entity{input_.pos(), 0};
  bool has_end = false;

  if (input_.consume() != '&') return TT_ERROR;

  ++entity.size;  // Account for the consumed '&'

  // Consume the entity
  while (input_.peek()) {
    if (input_.peek() == ';') {
      input_.consume();
      ++entity.size;
      has_end = true;
      break;
    } else if (!isalpha(input_.peek())) {  // NOLINT
      has_end = false;
      break;
    } else {
      input_.consume();
      ++entity.size;
    }
  }

  // If we can decode the entity, do so
  if (has_end && resolve_entity(entity, value_)) return parentTokenType;

  // Otherwise, just yield the whole thing undecoded, interpret it as text
  value_ = entity;
  return parentTokenType;
}

bool Scanner::resolve_entity(const StringRef &buffer, StringRef &decoded) {
  char lt = '<';
  char gt = '>';
  char amp = '&';
  char quot = '"';
  char apos = '\'';
  char nbsp = ' ';

  if (buffer == "&lt;") {
    decoded = StringRef{&lt, 1};
    return true;
  }
  if (buffer == "&gt;") {
    decoded = StringRef{&gt, 1};
    return true;
  }
  if (buffer == "&amp;") {
    decoded = StringRef{&amp, 1};
    return true;
  }
  if (buffer == "&quot;") {
    decoded = StringRef{&quot, 1};
    return true;
  }
  if (buffer == "&apos;") {
    decoded = StringRef{&apos, 1};
    return true;
  }
  if (buffer == "&nbsp;") {
    // TODO(any):
    // handle non-breaking spaces better than just converting them to spaces
    decoded = StringRef{&nbsp, 1};
    return true;
  }
  return false;
}

// skip whitespaces.
// returns how many whitespaces were skipped
size_t Scanner::skip_whitespace() {
  size_t skipped = 0;
  while (is_whitespace(input_.peek())) {
    input_.consume();
    ++skipped;
  }
  return skipped;
}

bool Scanner::is_whitespace(char c) {
  return c <= ' ' &&
         (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f');
}

Scanner::TokenType Scanner::scan_comment() {
  if (gotTail_) {
    start_ = input_.pos() - length("-->");  // minus "-->"
    scanFun_ = &Scanner::scan_body;
    gotTail_ = false;
    return TT_COMMENT_END;
  }

  start_ = input_.pos();
  value_ = StringRef{input_.pos(), 0};

  while (true) {
    if (input_.consume() == '\0') return TT_EOF;
    ++value_.size;

    if (ends_with(value_, "-->")) {
      gotTail_ = true;
      value_.size -= length("-->");
      break;
    }
  }
  return TT_DATA;
}

Scanner::TokenType Scanner::scan_processing_instruction() {
  if (gotTail_) {
    start_ = input_.pos() - length("?>");
    scanFun_ = &Scanner::scan_body;
    gotTail_ = false;
    return TT_PROCESSING_INSTRUCTION_END;
  }

  start_ = input_.pos();
  value_ = StringRef{input_.pos(), 0};

  while (true) {
    if (input_.consume() == '\0') return TT_EOF;
    ++value_.size;

    if (ends_with(value_, "?>")) {
      gotTail_ = true;
      value_.size -= length("?>");
      break;
    }
  }
  return TT_DATA;
}

Scanner::TokenType Scanner::scan_special() {
  if (gotTail_) {
    start_ = input_.pos() - (tag_.size + length("</>"));
    scanFun_ = &Scanner::scan_body;
    gotTail_ = false;
    return TT_TAG_END;
  }

  start_ = input_.pos();
  value_ = StringRef{input_.pos(), 0};

  while (true) {
    if (input_.consume() == '\0') return TT_EOF;
    ++value_.size;

    // Test for </tag>
    // TODO(any): no whitespaces allowed? Is that okay?
    if (value_.data[value_.size - 1] == '>' &&
        value_.size >= tag_.size + length("</>")) {
      // Test for the "</"" bit of "</tag>"
      size_t tag_start_position = value_.size - tag_.size - length("</>");
      if (std::memcmp(value_.data + tag_start_position, "</", length("</")) !=
          0)
        continue;

      // Test for the "tag" bit of "</tag>". Doing case insensitive compare
      // because <I>...</i> is okay.
      size_t tag_name_position =
          value_.size - tag_.size - length(">");  // end - tag>
      if (!equals_case_insensitive(value_.data + tag_name_position, tag_.data,
                                   tag_.size))
        continue;

      gotTail_ = true;
      value_.size -= tag_.size + length("</>");
      break;
    }
  }

  return TT_DATA;
}

}  // namespace markup
