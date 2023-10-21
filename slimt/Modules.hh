#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt {
using ParameterMap = std::unordered_map<std::string, Tensor *>;

struct Affine {
  Tensor W, b;  // NOLINT
  Tensor quant;
};

struct Linear {
  Tensor W;  // NOLINT
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
  explicit Attention(std::string name, size_t num_heads);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  std::tuple<Tensor, Tensor> forward(Tensor &q, Tensor &k, Tensor &v,
                                     Tensor &mask);

 private:
  std::string name_;
  Affine Q_, K_, V_, O_;
  LayerNorm ln_;
  size_t num_heads_;
};

class SSRU {
 public:
  explicit SSRU() = default;
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  Tensor forward(Tensor &state, Tensor &x);
  Tensor start_state(size_t batch_size) const;

 private:
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
  EncoderLayer(size_t depth, size_t ffn_count, size_t num_heads);
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
  explicit DecoderLayer(size_t depth, size_t ffn_count, size_t num_heads);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  std::tuple<Tensor, Tensor> forward(Tensor &encoder_out, Tensor &mask,
                                     Tensor &state, Tensor &x);
  Tensor start_state(size_t batch_size) const {
    return rnn_.start_state(batch_size);
  }

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
