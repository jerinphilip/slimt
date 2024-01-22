#include "slimt/Input.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

struct Greedy {
  static Histories decode(
      const Transformer &transformer, const Vocabulary &vocabulary,
      const std::optional<ShortlistGenerator> &shortlist_generator,
      const Tensor &encoder_out, const Input &input);

  Histories forward(
      const Transformer &transformer, const Vocabulary &vocabulary,
      const std::optional<ShortlistGenerator> &shortlist_generator,
      const Input &input);
};

using NBest = std::vector<Histories>;

struct BeamSearch {
  NBest forward(const Transformer &transformer, const Vocabulary &vocabulary,
                const std::optional<ShortlistGenerator> &shortlist_generator,
                const Input &input);
};

}  // namespace slimt
