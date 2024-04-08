
#include "slimt/Transliterator.hh"

#include "slimt/Input.hh"
#include "slimt/Request.hh"
#include "slimt/Search.hh"

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
    const std::string &source, size_t count) {
  auto [words, views] = vocabulary_.encode(source);

  // Generate input
  size_t batch_size = 1;
  Input input(batch_size, words.size(), vocabulary_.pad_id(),
              config_.tgt_length_limit_factor);
  input.add(words);
  input.finalize();

  BeamSearch search(transformer_, vocabulary_, shortlist_generator_);
  auto nbest = search.generate(input, count);

  std::vector<std::string> candidates;
  candidates.reserve(count);

  for (size_t i = 0; i < nbest.size(); i += count) {
    for (size_t j = 0; j < count; j++) {
      std::string decoded;
#if 0
      const History &history = nbest[i * batch_size + j];
      vocabulary_.decode(history->target, decoded);
      candidates.push_back(std::move(decoded));
#endif
    }
  }
  return candidates;
}

}  // namespace slimt
