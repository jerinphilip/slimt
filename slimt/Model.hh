#include <unordered_map>

#include "slimt/Batch.hh"
#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

class Decoder {
 public:
  Decoder(size_t decoders, size_t ffn_count, Vocabulary &vocabulary,
          Tensor &embedding, ShortlistGenerator &&shortlist_generator);

  void register_parameters(const std::string &prefix, ParameterMap &parameters);

  Sentences decode(Tensor &encoder_out, Tensor &mask, const Words &source);

 private:
  Tensor step(Tensor &encoder_out, Tensor &mask, std::vector<Tensor> &states,
              Words &previous_step);

  static Words greedy_sample(Tensor &logits, const Words &words,
                             size_t batch_size);

  std::vector<Tensor> start_states(size_t batch_size);

  Vocabulary &vocabulary_;

  Tensor &embedding_;
  std::vector<DecoderLayer> decoder_;
  Affine output_;

  ShortlistGenerator shortlist_generator_;

  float max_target_length_factor_ = 1.5;  // FIXME(-1): HARDCODE
};

// Restrict the models that can be created by a few kinds.
enum class Tag {
  // NOLINTBEGIN (readability-identifier-naming)
  tiny11  //
  // NOLINTEND
};

struct Config {
  // NOLINTBEGIN (readability-identifier-naming)
  struct tiny11 {
    static constexpr size_t encoder_layers = 6;
    static constexpr size_t decoder_layers = 2;
    static constexpr size_t feed_forward_depth = 2;
  };
  // NOLINTEND
};

class Model {
 public:
  explicit Model(Tag tag, Vocabulary &vocabulary, std::vector<io::Item> &&items,
                 ShortlistGenerator &&shortlist_generator);

  Sentences translate(Batch &batch);

 private:
  void load_parameters_from_items();
  void register_parameters(const std::string &prefix, ParameterMap &parameters);

  Tag tag_;

  std::vector<io::Item> items_;
  Tensor embedding_;
  std::vector<EncoderLayer> encoder_;
  Decoder decoder_;
};

}  // namespace slimt
