#include <fstream>
#include <iostream>
#include <sstream>

#include "3rd-party/CLI11.hpp"
#include "slimt/slimt.hh"

template <class Field>
struct Record {
  Field model;
  Field vocabulary;
  Field shortlist;
};

struct Options {
  Record<std::string> translator;
  std::string root;
  size_t max_tokens_per_batch = 1024;

  template <class App>
  void setup_onto(App &app) {
    // clang-format off
    app.add_option("--root", root, "Path to prefix other filenames to");
    app.add_option("--model", translator.model, "Path to model");
    app.add_option("--vocabulary", translator.vocabulary, "Path to vocabulary");
    app.add_option("--shortlist", translator.shortlist, "Path to shortlist");
    app.add_option("--max-tokens-per-batch", max_tokens_per_batch, "Path to shortlist");
    // clang-format on
  }
};

std::string prefix(const std::string &root, const std::string &basename) {
  return root + "/" + basename;
}

void run(const Options &options) {
  std::string indent = "  ";
  // clang-format off
  fprintf(stdout, "%s model: %s\n", indent.c_str(), options.translator.model.c_str());
  fprintf(stdout, "%s vocabulary: %s\n", indent.c_str(), options.translator.vocabulary.c_str());
  fprintf(stdout, "%s shortlist: %s\n", indent.c_str(), options.translator.shortlist.c_str());
  // clang-format on
  //
  using namespace slimt;  // NOLINT

  // Adjust paths.
  Record<std::string> adjusted{
      .model = prefix(options.root, options.translator.model),            //
      .vocabulary = prefix(options.root, options.translator.vocabulary),  //
      .shortlist = prefix(options.root, options.translator.shortlist)     //
  };

  Record<io::MmapFile> mmap{
      .model = io::MmapFile(adjusted.model),            //
      .vocabulary = io::MmapFile(adjusted.vocabulary),  //
      .shortlist = io::MmapFile(adjusted.shortlist),    //
  };

  // Tokenize into numeric-ids using sentencepiece.
  Vocabulary vocabulary(mmap.vocabulary.data(), mmap.vocabulary.size());
  ShortlistGenerator shortlist_generator(            //
      mmap.shortlist.data(), mmap.shortlist.size(),  //
      vocabulary, vocabulary                         //
  );

  // Load model
  auto items = io::loadItems(mmap.model.data());
  Config config;
  Translator translator(config, vocabulary, std::move(items),
                        std::move(shortlist_generator));

  using Sentences = std::vector<Words>;

  AverageMeter<float> wps;

  auto batch_and_translate = [&vocabulary, &model, &wps](    //
                                 Sentences &sentences,       //
                                 size_t max_sequence_length  //
                             ) {
    Timer timer;
    uint64_t batch_size = sentences.size();
    Batch batch(batch_size, max_sequence_length, vocabulary.pad_id());
    for (auto &sentence : sentences) {
      batch.add(sentence);
    }

    auto translated = model.translate(batch);
    for (auto &sentence : translated) {
      auto [result, views] = vocabulary.decode(sentence);
      std::cout << result << "\n";
    }
    size_t words_processed = max_sequence_length * sentences.size();
    float batch_wps = words_processed / timer.elapsed();
    wps.record(batch_wps);
  };

  std::string line;
  size_t max_sequence_length = 0;
  size_t token_count = 0;
  size_t line_no = 0;
  Sentences sentences;
  while (getline(std::cin, line)) {
    auto [words, views] = vocabulary.encode(line, /*add_eos =*/true);
    // std::cout << "Adding ";
    // for (const auto &view : views) {
    //   std::cout << "[" << view << "]";
    // }
    // std::cout << "\n";

    size_t candidate_max_sequence_length =
        std::max(words.size(), max_sequence_length);
    ++line_no;

    token_count = candidate_max_sequence_length * (line_no);
    if (token_count > options.max_tokens_per_batch) {
      batch_and_translate(sentences, max_sequence_length);
      sentences.clear();
      max_sequence_length = 0;
    }
    sentences.push_back(std::move(words));
    max_sequence_length = candidate_max_sequence_length;
  }

  // Overhang.
  if (!sentences.empty()) {
    batch_and_translate(sentences, max_sequence_length);
    sentences.clear();
  }

  fprintf(stdout, "wps: %f\n", wps.value());
}

int main(int argc, char *argv[]) {
  CLI::App app{"slimt"};
  std::string task;

  Options options;
  options.setup_onto(app);

  try {
    app.parse(argc, argv);
    run(options);
  } catch (const CLI::ParseError &e) {
    exit(app.exit(e));
  }

  return 0;
}
