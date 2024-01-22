
#include "slimt/Transliterator.hh"

#include "slimt/Request.hh"

namespace slimt {

namespace {
size_t model_id = 0;
}

namespace t12n {

Package<io::MmapFile> mmap_from(const Package<std::string> &package) {
  auto maybe_mmap = [](const std::string &path) {
    return path.empty() ? io::MmapFile() : io::MmapFile(path);
  };

  return {
      .model = maybe_mmap(package.model),            //
      .vocabulary = maybe_mmap(package.vocabulary),  //
      .shortlist = maybe_mmap(package.shortlist),    //
  };
}

Package<View> view_from(const Package<io::MmapFile> &mmap) {
  return {
      .model = {mmap.model.data(), mmap.model.size()},                 //
      .vocabulary = {mmap.vocabulary.data(), mmap.vocabulary.size()},  //
      .shortlist = {mmap.shortlist.data(), mmap.shortlist.size()},     //
  };
}

}  // namespace t12n

Transliterator::Transliterator(const Config &config,
                               const t12n::Package<View> &package)
    : id_(model_id++),
      view_(package),
      vocabulary_(package.vocabulary),
      shortlist_generator_(make_shortlist_generator(package.shortlist,
                                                    vocabulary_, vocabulary_)),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, package.model),
      batcher_(config_.max_words, config_.wrap_length,
               config_.tgt_length_limit_factor) {}

std::vector<std::string> Transliterator::transliterate(
    const std::string &source) {
  auto [words, views] = vocabulary_.encode(source);
  // Beam-search for multiple candidates, add multiple candidates.
  return {};
}

}  // namespace slimt
