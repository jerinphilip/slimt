#pragma once

#include <functional>
#include <optional>

#include "slimt/Annotation.hh"
#include "slimt/HTML.hh"
#include "slimt/Macros.hh"
#include "slimt/Response.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

// For now we will work with this, to avoid complaints another structure is hard
// to operate with.

namespace slimt {

/// ResponseBuilder is a callback functor. It is expected to be bound to a
/// Request after giving it the context of options, vocabulary and promise to
/// set. It constructs the Response and it's members based on options
/// (quality=on|off, alignments=on|off, mappings=on|off, splitmode=sentence |
/// paragraph).

class ResponseBuilder {
 public:
  /// @param [in] options: Options, indicating what to include
  /// or not in the response and any additional configurable parameters.
  /// @param [in] vocabulary: marian vocab object (used in decoding)
  /// @param [in] callback: callback with operates on the constructed Response.
  /// @param [in] qualityEstimator: the QualityEstimator model that can be used
  /// to provide translation quality probability.
  ResponseBuilder(Options options, AnnotatedText &&source,
                  const Vocabulary &vocabulary,
                  std::function<void(Response &&)> callback)
      : options_(options),
        vocabulary_(vocabulary),
        source_(std::move(source)),
        callback_(std::move(callback)) {}

  /// Constructs and sets the promise of a Response object from obtained
  /// histories after translating.
  /// @param [in] histories: Histories obtained after translating the Request
  /// from which this functor is called.
  void operator()(Histories &&histories);

 private:
  /// Builds alignments from histories and writes onto response.
  /// @param histories [in]
  /// @param response [out]
  static void buildAlignments(Histories &histories, Response &response);

  /// Builds translated text and subword annotations and writes onto response.
  /// @param histories [in]
  /// @param response [out]
  void buildTranslatedText(Histories &histories, Response &response);

  // Data members are context/curried args for the functor.

  Options options_;
  const Vocabulary &vocabulary_;  // vocabulary are required for decoding
                                  // and any source validation checks.
  AnnotatedText source_;
  std::function<void(Response &&)>
      callback_;  //  To be set when callback triggered and
                  //  after Response constructed.
};
}  // namespace slimt
