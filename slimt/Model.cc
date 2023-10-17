#include "slimt/Model.hh"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "slimt/Modules.hh"
#include "slimt/TensorOps.hh"

namespace slimt {

namespace {

size_t model_id = 0;

Package<io::MmapFile> mmap_from(const Package<std::string> &package) {
  return {
      .model = io::MmapFile(package.model),            //
      .vocabulary = io::MmapFile(package.vocabulary),  //
      .shortlist = io::MmapFile(package.shortlist),    //
  };
}

Package<View> view_from(const Package<io::MmapFile> &mmap) {
  return {
      .model = {mmap.model.data(), mmap.model.size()},                 //
      .vocabulary = {mmap.vocabulary.data(), mmap.vocabulary.size()},  //
      .shortlist = {mmap.shortlist.data(), mmap.shortlist.size()},     //
  };
}

}  // namespace

Model::Model(const Config &config, const Package<View> &package)
    : id_(model_id++),
      config_(config),
      view_(package),
      vocabulary_(package.vocabulary),
      processor_(config.wrap_length, config.split_mode, vocabulary_,
                 config.prefix_path),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.attention_num_heads, config.feed_forward_depth,
                   package.model),
      shortlist_generator_(package.shortlist, vocabulary_, vocabulary_) {}

Model::Model(const Config &config, const Package<std::string> &package)
    : id_(model_id++),
      config_(config),
      mmap_(mmap_from(package)),
      view_(view_from(*mmap_)),
      vocabulary_(view_.vocabulary),
      processor_(config.wrap_length, config.split_mode, vocabulary_,
                 config.prefix_path),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.attention_num_heads, config.feed_forward_depth,
                   view_.model),
      shortlist_generator_(view_.shortlist, vocabulary_, vocabulary_) {}

}  // namespace slimt
