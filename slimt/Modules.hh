#pragma once
#include <unordered_map>

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
  size_t num_heads_ = 8;  // FIXME(-1): HARDCODE
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
  EncoderLayer(size_t depth, size_t ffn_count);
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

Tensor affine_with_select(Affine &parameters, Tensor &x,
                          const std::vector<uint32_t> &indices,
                          const std::string &name = "");

}  // namespace slimt
