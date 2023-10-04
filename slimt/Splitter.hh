#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace slimt {

class Splitter {
 public:
  Splitter() = default;
  explicit Splitter(const std::string& prefix_file);

  void load(const std::string& fname);
  void load_from_serialized(std::string_view buffer);

  // Find next sentence boundary, return view for next sentence,
  // advance rest to reflect the rest of the text.
  std::string_view operator()(std::string_view* rest) const;

 private:
  using PrefixMap = std::map<std::string, int, std::less<>>;
  PrefixMap prefix_type_;

  // Return the prefix class of a prefix.
  // 0: not a prefix
  // 1: prefix
  // 2: prefix only in front of numbers
  int get_prefix_class(std::string_view piece) const;

  // auxiliary function to declare a prefix from a line in the prefix file
  void declare_prefix(std::string_view buffer);

  explicit Splitter(std::istream& istream);
};

class SentenceStream {
 public:
  enum class splitmode { OneSentencePerLine, OneParagraphPerLine, WrappedText };

  // @param text text to be split into sentences
  // @param splitter the actual sentence splitter
  // @param mode the split mode (one sentence/paragraph per line, wrapped text)
  // @param verify utf8?
  SentenceStream(std::string_view text, const Splitter& splitter,
                 splitmode mode, bool verify_utf8 = true);

  // @param data start of data
  // @param size size of data
  // @param splitter the actual sentence splitter
  // @param mode the split mode (one sentence/paragraph per line, wrapped text)
  // @param verify utf8?
  SentenceStream(const char* data, size_t size, const Splitter& splitter,
                 splitmode mode, bool verify_utf8 = true);

  int status() const;  // return status (pcre2 error code)
  const std::string& error_message() const;
  bool operator>>(std::string& snt);
  bool operator>>(std::string_view& snt);

 private:
  const char* cursor_;
  const char* const stop_;
  std::string_view paragraph_;
  splitmode mode_;
  const Splitter& splitter_;
  std::string error_message_;  // holds error message if UTF8 validation fails
  int status_;                 // holds prce2 error code
};

// Auxiliary function to print a chunk of text as a single line,
// replacing line breaks by blanks. This is faster than doing a
// global replacement in a string first.
std::ostream& single_line(
    std::ostream& out,           // destination stream
    std::string_view span,       // text span to be printed in a single line
    std::string_view end = "",   // stuff to put at end of line
    bool validate_utf = false);  // do we need to validate UTF8?

// Auxiliary function to stiore a chunk of text as a single line,
// replacing line breaks by blanks.
std::string& single_line(
    std::string& snt,            // destination stream
    std::string_view span,       // text span to be printed in a single line
    std::string_view end = "",   // stuff to put at end of line
    bool validate_utf = false);  // do we need to validate UTF8?

}  // namespace slimt
