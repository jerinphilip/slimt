#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>

#include "sentencepiece_processor.h"
#include "slimt/Types.hh"

namespace slimt {

class Vocabulary {
 public:
  explicit Vocabulary(const std::string &fpath);
  explicit Vocabulary(View view);
  std::tuple<Words, Views> encode(const std::string_view &line,
                                  bool add_eos = false) const;
  Views decode(const Words &words, std::string &decoded,
               bool ignore_eos = true) const;

  Word pad_id() const { return std::max(0, processor_.pad_id()); }
  Word eos_id() const { return processor_.eos_id(); }
  size_t size() const { return processor_.GetPieceSize(); }

 private:
  sentencepiece::SentencePieceProcessor processor_;
};

}  // namespace slimt
