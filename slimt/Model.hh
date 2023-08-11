#include <unordered_map>

#include "slimt/Batch.hh"
#include "slimt/Io.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"

namespace slimt {

using ParameterMap = std::unordered_map<std::string, Tensor *>;

struct Affine {
  Tensor W, b;
  Tensor quant;
};

struct Linear {
  Tensor W;
  Tensor quant;
};

class LayerNorm {
 public:
  explicit LayerNorm() = default;
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  Tensor forward(Tensor &x);

 private:
  Tensor bias_;
  Tensor scale_;
};

class Attention {
 public:
  explicit Attention(std::string name);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  std::tuple<Tensor, Tensor> forward(Tensor &q, Tensor &k, Tensor &v,
                                     Tensor &mask);

 private:
  std::string name_;
  Affine Q_, K_, V_, O_;
  LayerNorm ln_;
  size_t num_heads_ = 8;
};

class SSRU {
 public:
  explicit SSRU() = default;
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  Tensor forward(Tensor &x);
  void set_start_state(size_t batch_size);

 private:
  Tensor state_;
  Affine F_;
  Linear O_;
  LayerNorm ln_;
};

class FFN {
 public:
  explicit FFN(size_t depth);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  Tensor forward(Tensor &x);

 private:
  Affine O_;
  size_t depth_;
};

class EncoderLayer {
 public:
  explicit EncoderLayer(size_t depth, size_t ffn_count);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  std::tuple<Tensor, Tensor> forward(Tensor &x, Tensor &mask);

 private:
  size_t depth_;
  Attention attention_;
  std::vector<FFN> ffn_;
  LayerNorm ffn_ffn_;
};

class DecoderLayer {
 public:
  explicit DecoderLayer(size_t depth, size_t ffn_count);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  std::tuple<Tensor, Tensor> forward(Tensor &encoder_out, Tensor &mask,
                                     Tensor &x);
  void set_start_state(size_t batch_size) { rnn_.set_start_state(batch_size); }

 private:
  size_t depth_;
  Attention attention_;
  SSRU rnn_;
  std::vector<FFN> ffn_;
  LayerNorm ffn_ffn_;
};

class Decoder {
 public:
  using Words = std::vector<uint32_t>;
  using Sentences = std::vector<Words>;
  Decoder(size_t decoders, size_t ffn_count, Tensor &embedding,
          ShortlistGenerator &&shortlist_generator);

  void register_parameters(const std::string &prefix, ParameterMap &parameters);

  Decoder::Sentences decode(Tensor &encoder_out, Tensor &mask,
                            const Words &source);
  Tensor step(Tensor &encoder_out, Tensor &mask, Words &previous_step);

  static Words greedy_sample(Tensor &logits, const Shortlist::Words &words,
                             size_t batch_size);

  void set_start_state(size_t batch_size) {
    for (auto &layer : decoder_) {
      layer.set_start_state(batch_size);
    }
  }

 private:
  Tensor &embedding_;
  std::vector<DecoderLayer> decoder_;
  Affine output_;
  ShortlistGenerator shortlist_generator_;
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
  using Words = std::vector<uint32_t>;
  using Sentences = std::vector<Words>;
  explicit Model(Tag tag, std::vector<io::Item> &&items,
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
