#pragma once
#include <cstdint>

#include "slimt/Batch.hh"
#include "slimt/Io.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

class Shortlist {
 public:
  explicit Shortlist(Words words) : words_(std::move(words)) {}
  const std::vector<Word>& words() const { return words_; }
  Word reverse_map(int idx) { return words_[idx]; }

  int try_forward_map(Word w_idx) {
    auto first = std::lower_bound(words_.begin(), words_.end(), w_idx);
    if (first != words_.end() && *first == w_idx) {
      // Check if element not less than w_idx has been found
      // and if equal to w_idx
      // return coordinate if found
      return static_cast<int>(std::distance(words_.begin(), first));
    }
    // return -1 if not found
    return -1;
  }

 private:
  // // [packed shortlist index] -> word index,
  // used to select columns from output embeddings
  std::vector<Word> words_;
};

class ShortlistGenerator {
 public:
  static constexpr uint64_t kMagic = 0xF11A48D5013417F5;
  static constexpr uint64_t kFrequent = 100;
  static constexpr uint64_t kBest = 100;

  // construct directly from buffer
  ShortlistGenerator(
      const void* data, size_t blob_size,      //
      Vocabulary& source, Vocabulary& target,  //
      size_t source_index = 0, size_t /*target_indx=*/ = 1,
      bool shared = false,  // Kept there for backward compatibility
      bool check = true);

  Shortlist generate(const Words& words) const;

 private:
  Vocabulary& source_;
  Vocabulary& target_;

  size_t source_index_;
  bool shared_{false};

  uint64_t first_num_{kFrequent};  // baked into binary header
  uint64_t best_num_{kBest};       // baked into binary header

  // shortlist is stored in a skip list
  // [&shortLists_[word_to_offset_[word]],
  // &shortlist[word_to_offset_[word+1]]) is a sorted array of word indices
  // in the shortlist for word
  // io::MmapFile mmap_file_;

  uint64_t word_to_offset_size_;
  uint64_t shortlist_size_;

  const uint64_t* word_to_offset_;
  const Word* shortlist_;

  struct Header {
    uint64_t magic;                // BINARY_SHORTLIST_MAGIC
    uint64_t checksum;             // hash([&first_num, eof]).
    uint64_t first_num;            // Limits used to create the shortlist.
    uint64_t best_num;             //
    uint64_t word_to_offset_size;  // Length of word_to_offset_ array.
    uint64_t shortlist_size;       // Length of short_lists_ array.
  };

  bool content_check();
  // load shortlist from buffer
  void load(const void* data, size_t blob_size, bool check = true);
};

}  // namespace slimt
