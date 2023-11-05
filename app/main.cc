#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "3rd-party/CLI11.hpp"
#include "slimt/Frontend.hh"
#include "slimt/Model.hh"
#include "slimt/Response.hh"

inline std::string read_from_stdin() {
  // Read a large input text blob from stdin
  std::ostringstream stream;
  stream << std::cin.rdbuf();
  std::string input = stream.str();
  return input;
}

struct Options {
  std::string root;
  slimt::Package<std::string> translator;

  std::string follow_root;
  slimt::Package<std::string> follow;
  size_t poll = 5;  // NOLINT

  slimt::Config service;
  slimt::Model::Config model;

  bool async = false;
  bool html = false;
  bool version = false;

  template <class App>
  void setup_onto(App &app) {
    // clang-format off
    app.add_option("--root", root, "Path to prefix other filenames to");
    app.add_option("--model", translator.model, "Path to model");
    app.add_option("--vocabulary", translator.vocabulary, "Path to vocabulary");
    app.add_option("--shortlist", translator.shortlist, "Path to shortlist");
    app.add_option("--ssplit", translator.ssplit, "Path to ssplit prefixes file.");

    app.add_option("--follow-root", follow_root, "Path to prefix other filenames to");
    app.add_option("--follow-model", follow.model, "Path to model");
    app.add_option("--follow-vocabulary", follow.vocabulary, "Path to vocabulary");
    app.add_option("--follow-shortlist", follow.shortlist, "Path to shortlist");
    app.add_option("--follow-ssplit", follow.ssplit, "Path to ssplit prefixes file.");

    app.add_option("--poll", poll, "Seconds to poll a long request to report");
    app.add_flag("--version", version, "Display version");
    app.add_flag("--html", html, "Whether content is HTML");
    app.add_flag("--async", async, "Try async backend");

    service.setup_onto(app);
    model.setup_onto(app);
    // clang-format on
  }
};

std::string prefix(const std::string &root, const std::string &basename) {
  return basename.empty() ? basename : root + "/" + basename;
}

void run(const Options &options) {
  std::string indent = "  ";
  // clang-format off
  fprintf(stdout, "%s model: %s\n", indent.c_str(), options.translator.model.c_str());
  fprintf(stdout, "%s vocabulary: %s\n", indent.c_str(), options.translator.vocabulary.c_str());
  fprintf(stdout, "%s shortlist: %s\n", indent.c_str(), options.translator.shortlist.c_str());
  fprintf(stdout, "%s ssplit: %s\n", indent.c_str(), options.translator.ssplit.c_str());
  // clang-format on
  //
  using namespace slimt;  // NOLINT

  // Adjust paths.
  auto package =
      [&](const std::string &root,
          const Package<std::string> &translator) -> Package<std::string> {
    return {
        .model = prefix(root, translator.model),            //
        .vocabulary = prefix(root, translator.vocabulary),  //
        .shortlist = prefix(root, translator.shortlist),    //
        .ssplit = prefix(root, translator.ssplit)           //
    };
  };

  // Sample user-operation.
  // We decide the user interface first, ideally nice, clean.
  // There are times when it won't match - EM.
  auto model = std::make_shared<Model>(
      options.model, package(options.root, options.translator));

  std::shared_ptr<Model> follow = nullptr;
  if (!options.follow_root.empty()) {
    follow = std::make_shared<Model>(
        options.model, package(options.follow_root, options.follow));
  }

  if (options.async) {
    // Async operation.
    Async service(options.service);

    std::string source = read_from_stdin();
    slimt::Options opts{
        .alignment = true,    //
        .html = options.html  //
    };

    Handle handle = (!follow)
                        ? service.translate(model, std::move(source), opts)
                        : service.pivot(model, follow, std::move(source), opts);

    auto report = [&handle]() {
      auto info = handle.info();
      auto percent = [](const Handle::Info &info) {
        auto decimal = [](const Fraction &v) {
          float ratio = (static_cast<float>(v.p) / static_cast<float>(v.q));
          return ratio;
        };

        const auto &value = info.words;
        const auto &parts = info.parts;

        float remaining = parts.q - parts.p;
        float completed = parts.p - 1;
        float unit = 100.0F / static_cast<float>(parts.q);
        return completed * unit + decimal(value) * unit;  // NOLINT
      };

      auto length = [](size_t value) {
        int count = 0;
        constexpr size_t kBase = 10;
        while (value) {
          value /= kBase;
          ++count;
        }
        return count;
      };

      int word_width = length(info.words.q);
      int segment_width = length(info.segments.q);
      int part_width = length(info.parts.q);
      fprintf(stderr,
              "Fraction %6.2lf %% [ wps %lf | part %*zu/%zu | words %*zu/%zu | "
              "segments %*zu/%zu ] \n",
              percent(info), info.wps,                         //
              part_width, info.parts.p, info.parts.q,          //
              word_width, info.words.p, info.words.q,          //
              segment_width, info.segments.p, info.segments.q  //
      );
    };

    std::chrono::seconds poll(options.poll);
    std::future_status status = handle.future().wait_for(poll);
    while (status == std::future_status::timeout) {
      report();
      status = handle.future().wait_for(poll);
    }

    report();

    Response response = handle.future().get();
    fprintf(stdout, "%s\n", response.target.text.c_str());
  } else {
    // Blocking operation.
    Blocking service(options.service);

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
