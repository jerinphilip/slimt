
#include <cstddef>
#include <string>

#include "slimt/Batcher.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

namespace t12n {

template <class Field>
struct Package {
  Field model;
  Field vocabulary;
  Field shortlist;
};

}  // namespace t12n

// This class is not meant to be thread-safe.
class Transliterator {
 public:
  struct Config {
    // NOLINTBEGIN
    size_t encoder_layers = 6;
    size_t decoder_layers = 2;
    size_t feed_forward_depth = 2;
    size_t num_heads = 8;
    size_t max_words = 1024;
    size_t cache_size = 1024;
    float tgt_length_limit_factor = 1.5;
    size_t wrap_length = 128;

    template <class App>
    void setup_onto(App &app) {
      // clang-format off
      app.add_option("--encoder-layers", encoder_layers, "Number of encoder layers");
      app.add_option("--decoder-layers", decoder_layers, "Number of decoder layers");
      app.add_option("--num-heads", num_heads, "Number of decoder layers");
      app.add_option("--ffn-depth", decoder_layers, "Number of feedforward layers");
      app.add_option("--limit-tgt", tgt_length_limit_factor, "Max length proportional to source target can have.");
      app.add_option("--max-words", max_words, "Maximum words in a batch.");
      app.add_option("--wrap-length", max_words, "Maximum length allowed for a sample, beyond which hard-wrap.");
      // clang-format on
    }
    // NOLINTEND
  };

  Transliterator(const Config &config, const t12n::Package<View> &package);
  std::vector<std::string> transliterate(const std::string &source);

 private:
  static std::optional<ShortlistGenerator> make_shortlist_generator(
      View view, const Vocabulary &source, const Vocabulary &target);

  size_t id_ = 0;
  size_t model_id_ = 0;
  Config config_;

  using Mmap = t12n::Package<io::MmapFile>;
  std::optional<Mmap> mmap_;
  t12n::Package<View> view_;

  Vocabulary vocabulary_;
  std::optional<ShortlistGenerator> shortlist_generator_;
  Transformer transformer_;
  Batcher batcher_;
  std::optional<TranslationCache> cache_;
};

}  // namespace slimt
