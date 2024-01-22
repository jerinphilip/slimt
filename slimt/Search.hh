#include "slimt/Input.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

Histories decode(const Transformer &transformer, const Vocabulary &vocabulary,
                 const std::optional<ShortlistGenerator> &shortlist_generator,
                 const Tensor &encoder_out, const Input &input);

Histories forward(const Transformer &transformer, const Vocabulary &vocabulary,
                  const std::optional<ShortlistGenerator> &shortlist_generator,
                  const Input &input);
struct Greedy {};
struct BeamSearch {};

}  // namespace slimt
