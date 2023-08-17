#include <fstream>
#include <iostream>
#include <sstream>

#include "3rd-party/CLI11.hpp"
#include "slimt/slimt.hh"

template <class Field>
struct Record {
  Field model;
  Field vocab;
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
    app.add_option("--vocab", translator.vocab, "Path to vocab");
    app.add_option("--shortlist", translator.shortlist, "Path to shortlist");
    app.add_option("--max-tokens-per-batch", max_tokens_per_batch, "Path to shortlist");
    // clang-format on
  }
};

std::string prefix(const std::string &root, const std::string &basename) {
  return root + "/" + basename;
}

struct SlimtIO {
  FILE *in;
  FILE *out;
  FILE *err;
};

static SlimtIO sio{
    .in = stdin,    //
    .out = stdout,  //
    .err = stderr   //
};

void run(const Options &options) {
  std::string indent = "  ";
  // clang-format off
  fprintf(sio.err, "%s model: %s\n", indent.c_str(), options.translator.model.c_str());
  fprintf(sio.err, "%s vocab: %s\n", indent.c_str(), options.translator.vocab.c_str());
  fprintf(sio.err, "%s shortlist: %s\n", indent.c_str(), options.translator.shortlist.c_str());
  // clang-format on
  //
  using namespace slimt;  // NOLINT

  // Adjust paths.
  Record<std::string> adjusted{
      .model = prefix(options.root, options.translator.model),         //
      .vocab = prefix(options.root, options.translator.vocab),         //
      .shortlist = prefix(options.root, options.translator.shortlist)  //
  };

  Record<io::MmapFile> mmap{
      .model = io::MmapFile(adjusted.model),          //
      .vocab = io::MmapFile(adjusted.vocab),          //
      .shortlist = io::MmapFile(adjusted.shortlist),  //
  };

  // Tokenize into numeric-ids using sentencepiece.
  Vocabulary vocab(mmap.vocab.data(), mmap.vocab.size());
  ShortlistGenerator shortlist_generator(            //
      mmap.shortlist.data(), mmap.shortlist.size(),  //
      vocab, vocab                                   //
  );

  // Load model
  auto items = io::loadItems(mmap.model.data());
  Model model(Tag::tiny11, std::move(items), std::move(shortlist_generator));

  fprintf(sio.err, "%s eos_id: %u\n", indent.c_str(), vocab.eos_id());
  using Sentences = std::vector<Vocabulary::Words>;

  AverageMeter<float> wps;

  auto batch_and_translate = [&vocab, &model, &wps](         //
                                 Sentences &sentences,       //
                                 size_t max_sequence_length  //
                             ) {
    Timer timer;
    uint64_t batch_size = sentences.size();
    Batch batch(batch_size, max_sequence_length, vocab.pad_id());
    for (auto &sentence : sentences) {
      batch.add(sentence);
    }

    auto translated = model.translate(batch);
    for (auto &sentence : translated) {
      auto [result, views] = vocab.decode(sentence);
      fprintf(sio.out, "%s\n", result.c_str());
    }
    size_t words_processed = max_sequence_length * sentences.size();
    float batch_wps = words_processed / timer.elapsed();
    wps.record(batch_wps);
    fprintf(sio.err, "wps: %f | occupancy: %f\n", wps.value(),
            batch.occupancy());
  };

  std::string line;
  size_t max_sequence_length = 0;
  size_t token_count = 0;
  size_t line_no = 0;
  Sentences sentences;
  while (getline(std::cin, line)) {
    auto [words, views] = vocab.encode(line, /*add_eos =*/true);

    ++line_no;
    size_t candidate_max_sequence_length =
        std::max(words.size(), max_sequence_length);
    size_t candidate_token_count =
        candidate_max_sequence_length * (sentences.size() + 1);

    if (candidate_token_count > options.max_tokens_per_batch) {
      // Cleave off a batch.
      fprintf(sio.err, "seq_len x bsz = %zu x %zu\n", max_sequence_length,
              sentences.size());
      batch_and_translate(sentences, max_sequence_length);
      sentences.clear();
      // New stuff based on words.
      max_sequence_length = words.size();
      token_count = words.size();
      sentences.push_back(std::move(words));
    } else {
      max_sequence_length = candidate_max_sequence_length;
      token_count = candidate_token_count;
      sentences.push_back(std::move(words));
    }
  }

  // Overhang.
  if (!sentences.empty()) {
    batch_and_translate(sentences, max_sequence_length);
    sentences.clear();
  }

  fprintf(sio.err, "wps: %f\n", wps.value());
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
