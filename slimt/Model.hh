#include <unordered_map>

#include "slimt/Batch.hh"
#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

struct Config {
  // NOLINTBEGIN (readability-identifier-naming)
  size_t encoder_layers = 6;
  size_t decoder_layers = 2;
  size_t feed_forward_depth = 2;
  size_t tgt_length_limit_factor = 2;
  size_t attention_num_heads = 8;
  // NOLINTEND
};

class Decoder {
 public:
  Decoder(const Config &config, Vocabulary &vocabulary, Tensor &embedding,
          ShortlistGenerator &&shortlist_generator);

  void register_parameters(const std::string &prefix, ParameterMap &parameters);

  Sentences decode(Tensor &encoder_out, Tensor &mask, const Words &source);

 private:
  Tensor step(Tensor &encoder_out, Tensor &mask, std::vector<Tensor> &states,
              Words &previous_step);

  static Words greedy_sample(Tensor &logits, const Words &words,
                             size_t batch_size);

  std::vector<Tensor> start_states(size_t batch_size);

  float tgt_length_limit_factor_;
  Vocabulary &vocabulary_;

  Tensor &embedding_;
  std::vector<DecoderLayer> decoder_;
  Affine output_;

  ShortlistGenerator shortlist_generator_;
};

class Model {
 public:
  explicit Model(Config config, Vocabulary &vocabulary,
                 std::vector<io::Item> &&items,
                 ShortlistGenerator &&shortlist_generator);

  Sentences translate(Batch &batch);

 private:
  void load_parameters_from_items();
  void register_parameters(const std::string &prefix, ParameterMap &parameters);

  Config config_;

  std::vector<io::Item> items_;
  Tensor embedding_;
  std::vector<EncoderLayer> encoder_;
  Decoder decoder_;
};

}  // namespace slimt
