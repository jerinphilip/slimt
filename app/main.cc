#include <fstream>
#include <iostream>
#include <sstream>

#include "3rd-party/CLI11.hpp"
#include "slimt/slimt.hh"

inline std::string read_from_stdin() {
  // Read a large input text blob from stdin
  std::ostringstream stream;
  stream << std::cin.rdbuf();
  std::string input = stream.str();
  return input;
}

template <class Field>
struct Record {
  Field model;
  Field vocabulary;
  Field shortlist;
};

struct Options {
  Record<std::string> translator;
  std::string root;
  bool html = false;
  size_t max_tokens_per_batch = 1024;  // NOLINT

  template <class App>
  void setup_onto(App &app) {
    // clang-format off
    app.add_option("--root", root, "Path to prefix other filenames to");
    app.add_option("--model", translator.model, "Path to model");
    app.add_option("--vocabulary", translator.vocabulary, "Path to vocabulary");
    app.add_option("--shortlist", translator.shortlist, "Path to shortlist");
    app.add_option("--max-tokens-per-batch", max_tokens_per_batch, "Path to shortlist");
    app.add_flag("--html", html, "Whether content is HTML");
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

  Record<View> view{
      .model = {mmap.model.data(), mmap.model.size()},                 //
      .vocabulary = {mmap.vocabulary.data(), mmap.vocabulary.size()},  //
      .shortlist = {mmap.shortlist.data(), mmap.shortlist.size()},     //
  };

  Config config;
  Translator translator(config, view.model, view.shortlist, view.vocabulary);
  std::string source = read_from_stdin();
  slimt::Options opts{
      .alignment = true,    //
      .HTML = options.html  //
  };
  Response response = translator.translate(source, opts);
  fprintf(stdout, "%s\n", response.target.text.c_str());

  // fprintf(stdout, "wps: %f\n", wps.value());
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
