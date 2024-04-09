#include "slimt/Model.hh"

#include <cstddef>
#include <optional>
#include <string>

#include "slimt/Aligned.hh"
#include "slimt/Io.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

namespace {

size_t model_id = 0;

Package<io::MmapFile> mmap_from(const Package<std::string> &package) {
  auto maybe_mmap = [](const std::string &path) {
    return path.empty() ? io::MmapFile() : io::MmapFile(path);
  };

  return {
      .model = maybe_mmap(package.model),            //
      .vocabulary = maybe_mmap(package.vocabulary),  //
      .shortlist = maybe_mmap(package.shortlist),    //
      .ssplit = maybe_mmap(package.ssplit),          //
  };
}

Package<View> view_from(const Package<io::MmapFile> &mmap) {
  return {
      .model = {mmap.model.data(), mmap.model.size()},                 //
      .vocabulary = {mmap.vocabulary.data(), mmap.vocabulary.size()},  //
      .shortlist = {mmap.shortlist.data(), mmap.shortlist.size()},     //
      .ssplit = {mmap.ssplit.data(), mmap.ssplit.size()},              //
  };
}

}  // namespace

Model::Model(const Config &config, const Package<View> &package)
    : id_(model_id++),
      config_(config),
      view_(package),
      vocabulary_(package.vocabulary),
      processor_(config.split_mode, vocabulary_, Aligned()),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, package.model),
      shortlist_generator_(make_shortlist_generator(
          package.shortlist, vocabulary_, vocabulary_)) {}

Model::Model(const Config &config, const Package<std::string> &package)
    : id_(model_id++),
      config_(config),
      mmap_(mmap_from(package)),
      view_(view_from(*mmap_)),
      vocabulary_(view_.vocabulary),
      processor_(config.split_mode, vocabulary_, Aligned()),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, view_.model),
      shortlist_generator_(make_shortlist_generator(
          view_.shortlist, vocabulary_, vocabulary_)) {}

namespace preset {
Model::Config tiny() {
  // NOLINTBEGIN
  Model::Config config{
      .encoder_layers = 6,      //
      .decoder_layers = 2,      //
      .feed_forward_depth = 2,  //
      .num_heads = 8,           //
      .split_mode = "sentence"  //
  };
  // NOLINTEND
  return config;
}

Model::Config base() {
  // NOLINTBEGIN
  Model::Config config{
      .encoder_layers = 6,      //
      .decoder_layers = 2,      //
      .feed_forward_depth = 2,  //
      .num_heads = 8,           //
      .split_mode = "sentence"  //
  };
  // NOLINTEND
  return config;
}

Model::Config nano() {
  // NOLINTBEGIN
  Model::Config config{
      .encoder_layers = 4,      //
      .decoder_layers = 2,      //
      .feed_forward_depth = 2,  //
      .num_heads = 8,           //
      .split_mode = "sentence"  //
  };
  // NOLINTEND
  return config;
}
}  // namespace preset

}  // namespace slimt
