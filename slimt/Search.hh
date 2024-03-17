#include "slimt/Input.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

// Holds state for a single time-step during the course of generation
// Allows pruning, and other book-keeping required to dynamically reduce matrix
// multiplications (N^3) at the overhead of (N^2) copy.
class GenerationStep {
 public:
  GenerationStep(Tensor &&encoder_out, Tensor &&mask, Words &&previous,
                 std::optional<Words> &&shortlist, std::vector<Tensor> &&states,
                 size_t max_seq_length);

  void prune(std::vector<size_t> indices);
  bool complete() const { return previous_.empty(); }

  const Tensor &encoder_out() const { return encoder_out_; }
  const Tensor &mask() const { return mask_; }
  const Words &previous() const { return previous_; }
  std::vector<Tensor> &states() { return states_; }
  const std::optional<Words> &shortlist() const { return shortlist_; }
  size_t time_step() const { return time_step_; }

  void advance(Words &&previous);

 private:
  Tensor encoder_out_;
  Tensor mask_;
  std::vector<Tensor> states_;

  Words previous_;
  const std::optional<Words> &shortlist_;
  size_t time_step_ = 0;
  size_t max_seq_length_;
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

using NBest = std::vector<Histories>;

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
