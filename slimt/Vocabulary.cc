#include "slimt/Vocabulary.hh"

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "sentencepiece.pb.h"
#pragma GCC diagnostic pop

namespace slimt {

Vocabulary::Vocabulary(View view) {
  absl::string_view serialized(reinterpret_cast<char *>(view.data), view.size);
  processor_.LoadFromSerializedProto(serialized);
}

Vocabulary::Vocabulary(const std::string &fpath) {
  // Load vocabulary
  processor_.Load(fpath);
}

std::tuple<Words, Views> Vocabulary::encode(const std::string_view &line,
                                            bool add_eos /* = false*/) const {
  absl::string_view a_line(line.data(), line.size());
  std::vector<absl::string_view> views;
  sentencepiece::SentencePieceText sentencepiece_text;

  processor_.Encode(a_line, &sentencepiece_text);
  const auto &pieces = sentencepiece_text.pieces();

  Words words;
  words.reserve(pieces.size() + static_cast<uint32_t>(add_eos));

  // Deprecation warning on the other iterator, so accessing via index.
  // Then it claims use range loop sigh.
  // NOLINTNEXTLINE
  size_t piece_count = static_cast<size_t>(pieces.size());
  for (size_t i = 0; i < piece_count; i++) {
    const auto &piece = pieces[i];
    Word word = piece.id();
    words.push_back(word);
    std::string_view view =
        line.substr(piece.begin(), piece.end() - piece.begin());
    views.emplace_back(view.data(), view.size());
  }

  if (add_eos) {
    uint32_t eos_id = processor_.eos_id();
    words.push_back(eos_id);
  }

  // Sentencepiece uses absl::string_view all around. Since C++ treats
  // absl::string_view and std::string_view as different classes despite the
  // same layout and intent, this conversion is inevitable, unless we want to
  // edit sentencepiece.
  std::vector<std::string_view> std_views;
  std_views.reserve(views.size());
  for (const auto &view : views) {
    std_views.emplace_back(view.data(), view.size());
  }

  return std::make_tuple(words, std_views);
}

Views Vocabulary::decode(const Words &words, std::string &decoded,
                         bool ignore_eos) const {
  sentencepiece::SentencePieceText sentencepiece_text;
  std::vector<std::string_view> views;

  // int. -1 could be pad_id()?
  std::vector<int> sentence;
  sentence.reserve(words.size());
  for (auto word : words) {
    sentence.push_back(word);
  }

  processor_.Decode(sentence, &sentencepiece_text);

  // Creates copy of string.
  decoded = std::move(sentencepiece_text.text());
  for (const auto &piece : sentencepiece_text.pieces()) {
    size_t size = piece.end() - piece.begin();
    std::string_view view(decoded.data() + piece.begin(), size);
    views.push_back(view);
  }

  if (ignore_eos) {
    views.pop_back();
  }

  return views;
}

}  // namespace slimt
