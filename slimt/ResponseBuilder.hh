#pragma once

#include <functional>
#include <future>
#include <optional>
#include <utility>

#include "slimt/Annotation.hh"
#include "slimt/Response.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

// For now we will work with this, to avoid complaints another structure is hard
// to operate with.

namespace slimt {
class Vocabulary;

/// ResponseBuilder is a callback functor. It is expected to be bound to a
/// Request after giving it the context of options, vocabulary and promise to
/// set. It constructs the Response and it's members based on options
/// (quality=on|off, alignments=on|off, mappings=on|off, splitmode=sentence |
/// paragraph).

class ResponseBuilder {
 public:
  using Continuation = std::function<void(Response &&response)>;
  /// @param [in] options: Options, indicating what to include
  /// or not in the response and any additional configurable parameters.
  /// @param [in] vocabulary: marian vocab object (used in decoding)
  /// @param [in] callback: callback with operates on the constructed Response.
  ResponseBuilder(AnnotatedText &&source, const Vocabulary &vocabulary,
                  Continuation &&continuation)
      : vocabulary_(vocabulary),
        source_(std::move(source)),
        continuation_(std::move(continuation)) {}

  /// Constructs and sets the promise of a Response object from obtained
  /// histories after translating.
  /// @param [in] histories: Histories obtained after translating the Request
  /// from which this functor is called.
  void operator()(Histories &&histories);

 private:
  // Data members are context/curried args for the functor.
  const Vocabulary &vocabulary_;  // vocabulary are required for decoding
                                  // and any source validation checks.
  AnnotatedText source_;

  Continuation continuation_;
};
}  // namespace slimt
