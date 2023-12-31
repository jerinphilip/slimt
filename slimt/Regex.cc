#include "slimt/Regex.hh"

#include <pcre2.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace slimt {

Regex::Regex(const std::string& pattern, uint32_t options, uint32_t jit_options)
    : re_(pcre2_compile(PCRE2_SPTR(pattern.c_str()), /* the pattern */
                        PCRE2_ZERO_TERMINATED, /* pattern is zero-terminated */
                        options,               /* options */
                        &error_number_,        /* for error number */
                        &error_offset_,        /* for error offset */
                        nullptr))              /* use default compile context */
{
  uint32_t have_jit;
  pcre2_config(PCRE2_CONFIG_JIT, &have_jit);
  if (have_jit) {
    pcre2_jit_compile(re_, jit_options);
  }

  pattern_ = pattern;
}

std::string Regex::get_error_message() const {
  constexpr size_t kMaxBufferSize = 256;
  PCRE2_UCHAR buffer[kMaxBufferSize];
  pcre2_get_error_message(error_number_, buffer, sizeof(buffer));
  std::ostringstream msg;
  msg << "PCRE2 compilation failed at offset " << error_offset_ << ": "
      << buffer;
  return msg.str();
}

// return compiled regex
const pcre2_code* Regex::get_pcre2_code() const { return re_; }

int Regex::consume(
    std::string_view* subj,  // the string (view) agains we are matching
    Match* match,            // where to store the results of the match
    uint32_t options         // search options
) const {
  int success = find(*subj, match, 0, options | PCRE2_ANCHORED);
  if (success > 0) {
    subj->remove_prefix((*match)[0].size());
  }
  return success;
}

int Regex::find(
    std::string_view subj,  // the string (view) agains we are matching
    Match* match,           // where to store the results of the match
    size_t start,           // where to start searching in the string
    uint32_t options        // search options
) const {
  assert(start <= subj.size());
  int rc = pcre2_match(re_,                     /* the compiled pattern */
                       PCRE2_SPTR(subj.data()), /* the subject string */
                       subj.size(),             /* the length of the subject */
                       start,                   /* where to start */
                       options,                 /* options */
                       match->match_data, /* block for storing the result */
                       nullptr);          /* use default match context */
  match->data = rc > 0 ? subj.data() : nullptr;
  match->num_matched_groups = rc;
  return rc;  // returns the number of matched groups
}

bool Regex::ok() const { return re_ != nullptr; }

Regex::~Regex() { pcre2_code_free(re_); }

Match::Match(const Regex& re) : Match(re.get_pcre2_code()) {}

Match::Match(const pcre2_code* re)
    : match_data(pcre2_match_data_create_from_pattern(re, nullptr)) {}

Match::~Match() { pcre2_match_data_free(match_data); }

std::string_view Match::operator[](int i) const {
  PCRE2_SIZE* o = pcre2_get_ovector_pointer(match_data);
  assert(i <= num_matched_groups);
  i <<= 1;
  return std::string_view(data + o[i], o[i + 1] - o[i]);
}
}  // namespace slimt
