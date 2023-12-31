// https://www.codeproject.com/Articles/14076/Fast-and-Compact-HTML-XML-Scanner-Tokenizer
// BSD license
//|
//| simple and fast XML/HTML scanner/tokenizer
//|
//| (C) Andrew Fedoniouk @ terrainformatica.com
//|
#include <cassert>
#include <cstring>
#include <string_view>

namespace markup {

struct InStream {
  const char *p;
  const char *begin;
  const char *end;
  explicit InStream(const char *src)
      : p(src), begin(src), end(src + strlen(src)) {}
  InStream(const char *begin, const char *end)
      : p(begin), begin(begin), end(end) {}
  char consume() { return p < end ? *p++ : 0; }
  char peek() const { return p < end ? *p : 0; }
  const char *pos() const { return p; }
};

// Think string_view, but with a mutable range
struct StringRef {
  const char *data;
  size_t size;
};

class Scanner {
 public:
  // NOLINTBEGIN
  // clang-format off
  enum TokenType {
    TT_ERROR = -1,
    TT_EOF = 0,

    TT_TAG_START,                     // <tag ...
                                      //     ^-- happens here
                                      //
    TT_TAG_END,                       // </tag>
                                      //       ^-- happens here
                                      // <tag ... />
                                      //            ^-- or here
                                      //
    TT_ATTRIBUTE,                     // <tag attr="value" >
                                      //                 ^-- happens here, attr_name() and value()
                                      //                     will be filled with 'attr' and 'value'.
                                      //
    TT_TEXT,                          // <tag>xxx</tag>
                                      //         ^-- happens here
                                      // <tag>foo &amp;&amp; bar</tag>
                                      //          ^---^----^----^-- and all of here as well
                                      // Comes after TT_TAG_START or as the first token if the input
                                      // begins with text instead of a root element.
                                      //
    TT_DATA,                          // <!-- foo -->
                                      //         ^-- here
                                      // <? ... ?>
                                      //       ^-- as well as here
                                      // <script>...</script>
                                      //            ^-- or here
                                      // <style>...</style>
                                      //           ^-- or here
                                      // comes after TT_COMMENT_START, TT_PI_START, or TT_TAG_START
                                      // if the tag was <script> or <style>.
                                      //
    TT_COMMENT_START,                 // <!-- foo -->
                                      //     ^-- happens here
                                      //
    TT_COMMENT_END,                   // <!-- foo -->
                                      //             ^-- happens here
                                      //
    TT_PROCESSING_INSTRUCTION_START,  // <?xml version="1.0?>
                                      //   ^-- happens here
                                      //
    TT_PROCESSING_INSTRUCTION_END,    // <?xml version="1.0?>
                                      //                     ^-- would you believe this happens here
  };
  // clang-format on
  // NOLINTEND

  explicit Scanner(InStream &is)
      : value_{nullptr, 0},
        tag_{nullptr, 0},
        attribute_{nullptr, 0},
        input_(is),
        scanFun_(&Scanner::scan_body) {}

  // get next token
  TokenType next() { return (this->*scanFun_)(); }

  // get value of TT_TEXT, TT_ATTR and TT_DATA
  std::string_view value() const;

  // get attribute name
  std::string_view attribute() const;

  // get tag name
  std::string_view tag() const;

  inline const char *start() const { return start_; }

 private:
  typedef TokenType (Scanner::*ScanPtr)();  // NOLINT

  // Consumes the text around and between tags
  TokenType scan_body();

  // Consumes name="attr"
  TokenType scan_attribute();

  // Consumes <!-- ... -->
  TokenType scan_comment();

  // Consumes <?name [attrs]?>
  TokenType scan_processing_instruction();

  // Consumes ...</style> and ...</script>
  TokenType scan_special();

  // Consumes <tagname and </tagname>
  TokenType scan_tag();

  // Consumes '&amp;' etc, emits parent_token_type
  TokenType scan_entity(TokenType parentTokenType);

  size_t skip_whitespace();

  static bool resolve_entity(const StringRef &buffer, StringRef &decoded);

  static bool is_whitespace(char c);

  StringRef value_;
  StringRef tag_;
  StringRef attribute_;

  InStream &input_;

  // Start position of a token.
  const char *start_ = nullptr;

  ScanPtr scanFun_;       // current 'reader'
                          //
  bool gotTail_ = false;  // aux flag used in scan_comment, scan_special,
                          // scan_processing_instruction
};
}  // namespace markup
