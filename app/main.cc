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

struct Options {
  slimt::Package<std::string> translator;
  std::string root;
  bool async = false;
  bool html = false;
  bool version = false;
  slimt::Config config;

  template <class App>
  void setup_onto(App &app) {
    // clang-format off
    app.add_option("--root", root, "Path to prefix other filenames to");
    app.add_option("--model", translator.model, "Path to model");
    app.add_option("--vocabulary", translator.vocabulary, "Path to vocabulary");
    app.add_option("--shortlist", translator.shortlist, "Path to shortlist");
    app.add_flag("--version", version, "Display version");
    app.add_flag("--html", html, "Whether content is HTML");
    app.add_flag("--async", async, "Try async backend");
    config.setup_onto(app);
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
  Package<std::string> package{
      .model = prefix(options.root, options.translator.model),            //
      .vocabulary = prefix(options.root, options.translator.vocabulary),  //
      .shortlist = prefix(options.root, options.translator.shortlist)     //
  };

  // Sample user-operation.
  // We decide the user interface first, ideally nice, clean.
  // There are times when it won't match - EM.
  auto model = std::make_shared<Model>(options.config, package);

  if (options.async) {
    // Async operation.
    Async service(options.config);

    std::string source = read_from_stdin();
    slimt::Options opts{
        .alignment = true,    //
        .html = options.html  //
    };

    std::future<Response> future =
        service.translate(model, std::move(source), opts);

    Response response = future.get();
    fprintf(stdout, "%s\n", response.target.text.c_str());
  } else {
    // Blocking operation.
    Blocking service(options.config);

    std::string source = read_from_stdin();
    slimt::Options opts{
        .alignment = true,    //
        .html = options.html  //
    };

    auto responses = service.translate(model, {std::move(source)}, opts);
    fprintf(stdout, "%s\n", responses[0].target.text.c_str());
  }

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
