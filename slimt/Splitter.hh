#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace slimt {

class Splitter {
  using prefix_map_t = std::map<std::string, int, std::less<>>;
  prefix_map_t prefix_type_;

  // Return the prefix class of a prefix.
  // 0: not a prefix
  // 1: prefix
  // 2: prefix only in front of numbers
  int get_prefix_class(std::string_view piece) const;

  // auxiliary function to declare a prefix from a line in the prefix file
  void declare_prefix(std::string_view buffer);

  explicit Splitter(std::istream& istream);

 public:
  Splitter();
  explicit Splitter(const std::string& prefix_file);

  void load(const std::string& fname);
  void loadFromSerialized(const std::string_view buffer);

  // Find next sentence boundary, return StringPiece for next sentence,
  // advance rest to reflect the rest of the text.
  std::string_view operator()(std::string_view* rest) const;
};  // end of class Splitter

class SentenceStream {
 public:
  enum class splitmode {
    one_sentence_per_line,
    one_paragraph_per_line,
    wrapped_text
  };

 private:
  const char* cursor_;
  const char* const stop_;
  std::string_view paragraph_;
  splitmode mode_;
  const Splitter& splitter_;
  std::string error_message_;  // holds error message if UTF8 validation fails
  int status_;                 // holds prce2 error code
 public:
  // @param text text to be split into sentences
  // @param splitter the actual sentence splitter
  // @param mode the split mode (one sentence/paragraph per line, wrapped text)
  // @param verify utf8?
  SentenceStream(std::string_view text, const Splitter& splitter,
                 splitmode mode, bool verify_utf8 = true);

  // @param data start of data
  // @param datasize size of data
  // @param splitter the actual sentence splitter
  // @param mode the split mode (one sentence/paragraph per line, wrapped text)
  // @param verify utf8?
  SentenceStream(const char* data, size_t datasize, const Splitter& splitter,
                 splitmode mode, bool verify_utf8 = true);

  //  bool OK() const; // return true if UTF8 verification succeeded
  int status() const;  // return status (pcre2 error code)
  const std::string& error_message() const;
  bool operator>>(std::string& snt);
  // bool operator>>(pcrecpp::StringPiece& snt);
  bool operator>>(std::string_view& snt);
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
