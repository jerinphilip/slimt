#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "slimt/Regex.hh"
#include "slimt/Splitter.hh"

namespace slimt {

std::string_view read_line(const char** start, const char* stop);

// Load a prefix file
void Splitter::load(const std::string& fname) {
  std::ifstream pfile(fname);
  std::string line;
  while (getline(pfile, line)) declare_prefix(line);

  // for debugging
  // for (auto m: prefix_type_) {
  //   std::cout << m.first << " " << m.second << std::endl;
  // }
}

void Splitter::declare_prefix(std::string_view buffer) {
  // parse a line from a prefix file and interpret it
  static Regex pat(R"(([^#\s]*)\s*(?:(#\s*NUMERIC_ONLY\s*#))?)", PCRE2_UTF);
  Match match(pat);
  if (pat.find(buffer, &match) > 0) {
    auto m1 = match[1];
    if (!m1.empty()) {
      std::string foo(m1.data(), m1.size());
      prefix_type_[foo] = !match[2].empty() ? 2 : 1;
      // for debugging:
      // std::cerr << foo << " " << (match[2].size() ? "N" : "") << std::endl;
    }
  }
}

void Splitter::load_from_serialized(const std::string_view buffer) {
  const char* start = buffer.data();
  const char* stop = start + buffer.size();
  for (std::string_view line = read_line(&start, stop); line.data();
       line = read_line(&start, stop)) {
    declare_prefix(line);
  }
}

Splitter::Splitter() = default;

Splitter::Splitter(const std::string& prefix_file) : Splitter() {
  if (!prefix_file.empty()) {
    load(prefix_file);
  }
}

// Auxiliary function to print a chunk of text as a single line,
// replacing line breaks by blanks. This is faster than doing a
// global replacement in a string first.
std::ostream& single_line(
    std::ostream& out,      // destination stream
    std::string_view span,  // text span to be printed in a single line
    std::string_view end,   // stuff to put at end of line
    bool validate_utf) {    // do we need to validate UTF8?
  static Regex pattern(R"(^\s*(.*)\R+\s*)", PCRE2_UTF);
  thread_local static Match match(pattern);
  int success =
      pattern.consume(&span, &match, validate_utf ? 0 : PCRE2_NO_UTF_CHECK);
  while (success > 0) {
    auto m = match[1];
    out.write(m.data(), m.size());
    out.write(" ", 1);
    success = pattern.consume(&span, &match, PCRE2_NO_UTF_CHECK);
  }
  out.write(span.data(), span.size());
  out.write(end.data(), end.size());
  return out;
}

// Auxiliary function to stiore a chunk of text as a single line in a string,
// replacing line breaks by blanks.
std::string& single_line(
    std::string& snt,       // destination stream
    std::string_view span,  // text span to be printed in a single line
    std::string_view end,   // stuff to put at end of line
    bool validate_utf) {    // do we need to validate UTF8?
  static Regex pattern(R"(^\s*(.*)\R+\s*)", PCRE2_UTF);
  thread_local static Match match(pattern);
  int success =
      pattern.consume(&span, &match, validate_utf ? 0 : PCRE2_NO_UTF_CHECK);
  snt.reserve(span.size());
  snt.clear();
  while (success > 0) {
    auto m = match[1];
    snt.append(m.data(), m.size());
    snt += ' ';
    success = pattern.consume(&span, &match, PCRE2_NO_UTF_CHECK);
  }
  snt.append(span.data(), span.size());
  snt.append(end.data(), end.size());
  return snt;
}

// return the prefix class of a prefix
// 0: not a prefix
// 1: prefix
// 2: prefix only in front of numbers
int Splitter::get_prefix_class(std::string_view piece) const {
  static Regex foo(".*\\s([^\\s]*)", PCRE2_DOTALL);
  static Match match(foo);
  if (foo.consume(&piece, &match, PCRE2_NO_UTF_CHECK) > 0) {
    piece = match[1];
  }
  auto m = prefix_type_.find(piece);
  // for debugging:
  // std::cout << piece << " " << (m == prefix_type_.end() ? 0 : m->second) <<
  // std::endl;
  return m == prefix_type_.end() ? 0 : m->second;
}

std::string_view Splitter::operator()(std::string_view* rest) const {
  // IMPORTANT: this operater does not do any UTF validation. If there's
  // broken UTF8 in the input, the operator may hang or crash. Doing
  // UTF8 validation is infeasible for long strings.
  // cf. http://www.pcre.org/current/doc/html/pcre2unicode.html

  static Regex whitespace_re("\\s*",
                             PCRE2_UTF | PCRE2_DOTALL | PCRE2_NEWLINE_ANY);

  // The chunker is the first step in sentence splitting.
  // It identifies candidate split points.
  //
  // Regarding \p{} below, see
  // https://www.pcre.org/current/doc/html/pcre2syntax.html#SEC5
  static Regex chunker_re(
      "\\s*"                      // whitespace
      "[^.?!։。？！]*?"           // non alphanumeric stuff
      "([\\p{L}\\p{Lo}\\p{N}]*)"  // 1: alphanumeric prefix of potential EOS
                                  // marker
      "([.?!։。？！]++)"          // 2: the potential EOS marker
      "("                         // 3: open group for trailing matter
      "['\")\\]’”\\p{Pf}]*"       // any "trailing matter"
      "(?:\\[[\\p{Nd}]+[\\p{Nd},\\s]*[\\p{Nd}]\\])?"  // footnote?
      "['\")\\]’”\\p{Pf}]*"  // any more "trailing matter"
      ")"                    // 3: close group for trailing matter
      "(\\s*)"               // 4: whitespace after
      "(?="                  // start look-ahead
      "([^\\s\\p{L}\\p{Lo}\\p{N}\\p{M}\\p{S}]*)"  // 5: sentence-initial punct.
      "\\s*"                                      // whitespace
      "([\\p{L}\\p{Lo}\\p{M}\\p{N}]*)"  // 6: leading letters or digits
      ")"                               // close look-ahead
      ,
      PCRE2_UTF | PCRE2_DOTALL | PCRE2_NEWLINE_ANY);

  // The following patterns are used to make heuristic decisions once a
  // potential split point has been identified.
  static const Regex lowercase("\\p{M}*\\p{Ll}", PCRE2_NO_UTF_CHECK);
  static Regex uppercase(R"(\p{M}*[\p{Lu}\p{Lt}])", PCRE2_NO_UTF_CHECK);
  static Regex digit("[\\p{Nd}\\p{Nl}]", PCRE2_NO_UTF_CHECK);
  static Regex letterother("\\p{M}*[\\p{Lo}]", PCRE2_NO_UTF_CHECK | PCRE2_UTF);

  // We need these to store match results:
  thread_local static Match whitespace_m(whitespace_re);
  thread_local static Match chunker_m(chunker_re);
  thread_local static Match lowercase_M(lowercase);
  thread_local static Match uppercase_M(uppercase);
  thread_local static Match digit_M(digit);

  thread_local static Match letterother_M(letterother);

  int success; /* stores the return value of pcre2_match() which is
                * called in Regex::find() / Regex::consume() */

  std::string_view snt;  // this will eventually be our return value

  whitespace_re.consume(rest, &whitespace_m, PCRE2_NO_UTF_CHECK);
  const char* snt_start = rest->data();
  const char* snt_end = rest->data() + rest->size();
  while ((success = chunker_re.consume(rest, &chunker_m, PCRE2_NO_UTF_CHECK)) >
         0) {
    auto whole_match = chunker_m[0];
    auto prefix = chunker_m[1];
    auto punct = chunker_m[2];             // punctuation
    auto tail = chunker_m[3];              // trailing punctuation
    auto whitespace_after = chunker_m[4];  // whitespace after
    // auto inipunct = Chunker_M[5];  // following symbols (not letters/digits)
    auto following_symbol =
        chunker_m[6];  // first letter or digit after whitespace

    // FOR DEBUGGING
    // std::cerr << "DEBUG\n" << prefix << "|"
    //           << punct << "|"
    //           << tail << "|"
    //           << whitespace_after <<"|"
    //           << inipunct << "|"
    //           << following_symbol << std::endl;

    // whitespace not required after ideographic full widths
    if (whitespace_after.empty() &&
        !(punct == "。" || punct == "！" || punct == "？")) {
      continue;
    } if (letterother.find(following_symbol, &letterother_M, 0,
                                PCRE2_ANCHORED) > 0) {
      // Finding a letterother is not cause for a non-break; (i.e we omit
      // continue)
    } else if (lowercase.find(following_symbol, &lowercase_M, 0,
                              PCRE2_ANCHORED) > 0) {
      // followed by lower case
      continue;
    } else if (uppercase.find(following_symbol, &uppercase_M, 0,
                              PCRE2_ANCHORED) > 0) {
      // followed by uppercase
      if (punct == "." &&
          get_prefix_class(prefix) != 0)  // preceded by nonbreaking prefix
        continue;
      if (punct.size() == 1 &&
          *snt_end == '.')  // preceded by abbreviation a.b.c
        continue;
    } else if (digit.find(following_symbol, &digit_M, 0, PCRE2_ANCHORED) > 0) {
      // std::cout << "Digit" << std::endl;
      // followed by digit
      if (punct == "." &&
          get_prefix_class(prefix) == 2)  // preceded by nonbreaking prefix
        continue;
    } else {
      // check for in-text ellipsis "[...]"
      if (punct == "..." &&
          (punct.data() - whole_match.data() > 1)  // not at the beginning
          && tail == "]" && *(punct.data() - 1) == '[') {
        continue;
      }
    }
    snt_end = whitespace_after.data();  // set end of sentence span
    break;
  }
  snt = std::string_view(snt_start, snt_end - snt_start);
  if (success < 1) {
    // Remove trailing whitespace:
    static Regex rtrim("(.*[^\\s])\\s*", PCRE2_NO_UTF_CHECK | PCRE2_DOTALL);
    thread_local static Match m(rtrim);
    if (rtrim.consume(&snt, &m, PCRE2_NO_UTF_CHECK) > 0) {
      snt = m[1];
    }
    *rest = std::string_view();
  }
  // if (success < -1) {
  //   PCRE2_UCHAR buffer[256];
  //   pcre2_get_error_message(success, buffer, sizeof(buffer));
  //   printf("%s\n", buffer);
  // }
  return snt;
}

// readLine gets pointers to start and stop of data instead of
// a std::string_view to be able to proccess chunks of data that
// exceed the size that a std::string_view can store (apparently int32_t).
// @TODO: verify: this dates back to working with pcrecpp::StringPiece.
// Update: apparently not true any more for absl::std::string_view!
// Returns a std::string_view to the line, not including the EOL character!
// So an empty line returns a std::string_view with size() == 0 and data() !=
// nullptr, At the end of the buffer, the return value has data() == nullptr.
std::string_view read_line(const char** start, const char* const stop) {
  std::string_view line;
  if (*start == stop) {  // no more data
    return line;
  }
  const char* c = *start;
  while (c < stop && *c != '\n') ++c;  // skip to next EOL
  const char* d = c;
  while (d-- > *start && *d == '\r')
    ;  // trim potential CR
  line = std::string_view(*start, ++d - *start);
  *start = (c == stop ? c : c + 1);
  return line;
}

// readParagraph gets pointers to start and stop of data instead of
// a std::string_view to be able to proccess chunks of data that
// exceed the size that a std::string_view can store (apparently int32_t).
// @TODO: verify: this dates back to working with pcrecpp::StringPiece.
std::string_view read_paragraph(const char** start, const char* const stop) {
  std::string_view par;
  if (*start == stop) {  // no more data
    return par;
  }
  const char* c = *start;
  const char* d;
  do {
    while (c < stop && *c != '\n') ++c;  // skip to next EOL
    d = c++;
    while (d < stop && (*d == '\n' || *d == '\r')) ++d;
  } while (d < stop && d == c);

  const char* e = --c;
  while (e-- > *start && *e == '\r')
    ;
  par = std::string_view(*start, ++e - *start);
  *start = (d < stop ? d : stop);
  return par;
}

SentenceStream::SentenceStream(std::string_view text, const Splitter& splitter,
                               splitmode mode, bool verify_utf8)
    : SentenceStream(text.data(), text.size(), splitter, mode, verify_utf8) {}

SentenceStream::SentenceStream(const char* data, size_t datasize,
                               const Splitter& splitter, splitmode mode,
                               bool verify_utf8)
    : cursor_(data), stop_(data + datasize), mode_(mode), splitter_(splitter) {
  static Regex r(".*", PCRE2_UTF);
  thread_local static Match m(r);

  if (verify_utf8) {
    // pre-flight verification: make sure it's well-formed UTF8
    int success = r.find(std::string_view(data, datasize), &m);
    if (success < 0) {
      auto offset = pcre2_get_startchar(m.match_data);
      PCRE2_UCHAR buffer[256];
      pcre2_get_error_message(success, buffer, sizeof(buffer));
      std::ostringstream msg;
      msg << "Invalid UTF at position " << offset << ": " << buffer;
      error_message_ = msg.str();
    }
  }
  if (mode == splitmode::OneParagraphPerLine) {
    paragraph_ = read_line(&cursor_, stop_);
  } else if (mode == splitmode::WrappedText) {
    paragraph_ = read_paragraph(&cursor_, stop_);
  }
}

const std::string& SentenceStream::error_message() const {
  return error_message_;
}

bool SentenceStream::operator>>(std::string_view& snt) {
  if (!error_message_.empty()) return false;

  if (paragraph_.empty() && cursor_ == stop_) {
    // We have reached the end of the data.
    return false;
  }

  if (mode_ == splitmode::OneSentencePerLine) {
    snt = read_line(&cursor_, stop_);
  } else if (paragraph_.empty()) {
    // No more sentences in this paragraph.
    // Read the next paragraph but for this call return
    // and empty sentence to indicate the end of this paragraph.
    snt = std::string_view();
    if (mode_ == splitmode::OneParagraphPerLine) {
      paragraph_ = read_line(&cursor_, stop_);
    } else {  // wrapped text
      paragraph_ = read_paragraph(&cursor_, stop_);
    }
  } else {
    snt = splitter_(&paragraph_);
  }
  return true;
};

bool SentenceStream::operator>>(std::string& snt) {
  std::string_view s;
  if (!((*this) >> s)) return false;
  single_line(snt, s, "", false);
  return true;
};

}  // namespace slimt
