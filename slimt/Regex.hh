#include <cassert>
#include <string>
#include <string_view>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <cstddef>
#include <cstdint>

namespace slimt {

// Inspired by https://github.com/luvit/pcre2/blob/master/src/pcre2demo.c
class Match;

class Regex {
 public:
  Regex(const std::string& pattern,  // pattern to be compiled
        uint32_t options             // pcre2 options for regex compilation
  );
  ~Regex();

  int find(std::string_view subj,  // the string (view) agains we are matching
           Match* M,               // where to store the results of the match
           size_t start = 0,       // where to start searching in the string
           uint32_t options = 0    // search options
  ) const;

  int consume(
      std::string_view* subj,  // the string (view) agains we are matching
      Match* M,                // where to store the results of the match
      uint32_t options = 0     // search options
  ) const;

  const pcre2_code* get_pcre2_code() const;  // return compiled regex
  std::string get_error_message() const;     // return error message
  // return true if pattern compiled successfully, false otherwise
  bool ok() const;

 private:
  // TODO(any): create name table for named groups
  // see https://github.com/luvit/pcre2/blob/master/src/pcre2demo.c
  std::string pattern_;

  PCRE2_SIZE error_offset_;
  int error_number_;
  pcre2_code* const re_;
};

class Match {
 public:
  pcre2_match_data* const match_data;  // stores matching offsets
  const char* data{nullptr};           // beginning of subject text span
  int num_matched_groups{0};
  std::string_view operator[](int i) const;
  explicit Match(const pcre2_code* re);
  explicit Match(const Regex& re);
  ~Match();
};

}  // namespace slimt
