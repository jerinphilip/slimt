#include <cstdint>
#include <limits>

#include "slimt/Input.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

class Result {
 public:
  Result(uint32_t eos_id, size_t batch_size);

  size_t record(const Words &step);
  void update_alignment(const Tensor &attn,
                        const std::vector<size_t> &input_lengths);

  Histories consume();

 private:
  uint32_t eos_id_;
  std::vector<bool> complete_;
  Sentences sentences_;
  Alignments alignments_;
};

class GenerationStep {
 public:
  GenerationStep(const std::vector<size_t> &input_lengths, Tensor &&encoder_out,
                 Tensor &&mask, Words &&previous,
                 std::optional<Words> &&shortlist, std::vector<Tensor> &&states,
                 size_t max_seq_length, uint32_t eos_id, size_t batch_size);

  const Tensor &encoder_out() const { return encoder_out_; }
  const Tensor &mask() const { return mask_; }
  const Words &previous() const { return previous_; }
  std::vector<Tensor> &states() { return states_; }
  const std::optional<Words> &shortlist() const { return shortlist_; }
  size_t time_step() const { return time_step_; }
  bool complete() const { return remaining_ == 0; }

  void update(Words &&step, const Tensor &attn);
  Histories finish() { return result_.consume(); }
  void prune(std::vector<size_t> indices);

 private:
  const std::vector<size_t> &input_lengths_;
  Tensor encoder_out_;
  Tensor mask_;
  std::vector<Tensor> states_;

  Words previous_;
  const std::optional<Words> &shortlist_;
  size_t time_step_ = 0;
  size_t max_seq_length_;
  size_t remaining_ = std::numeric_limits<size_t>::max();
  Result result_;
};

class Greedy {
 public:
  Greedy(const Transformer &transformer, const Vocabulary &vocabulary,
         const std::optional<ShortlistGenerator> &shortlist_generator);
  Histories generate(const Input &input);

 private:
  const Transformer &transformer_;
  const Vocabulary &vocabulary_;
  const std::optional<ShortlistGenerator> &shortlist_generator_;
};

using NBest = Histories;

class BeamSearch {
 public:
  BeamSearch(const Transformer &transformer, const Vocabulary &vocabulary,
             const std::optional<ShortlistGenerator> &shortlist_generator);

  NBest generate(const Input &input, size_t beam_size);

 private:
  const Transformer &transformer_;
  const Vocabulary &vocabulary_;
  const std::optional<ShortlistGenerator> &shortlist_generator_;
};

}  // namespace slimt
