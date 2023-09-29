#pragma once

#include <string_view>

#include "slimt/Types.hh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "3rd-party/sentencepiece/src/builtin_pb/sentencepiece.pb.h"
#include "3rd-party/sentencepiece/src/sentencepiece_processor.h"
#include "3rd-party/sentencepiece/src/sentencepiece_trainer.h"
#pragma GCC diagnostic pop

namespace slimt {

class Vocabulary {
 public:
  explicit Vocabulary(const std::string &fpath);
  Vocabulary(void *data, size_t size);
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
