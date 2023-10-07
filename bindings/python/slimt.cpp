#include "slimt/slimt.hh"

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>
#include <string>
#include <vector>

namespace py = pybind11;

using slimt::Alignment;
using slimt::Alignments;
using slimt::AnnotatedText;
using slimt::Options;
using slimt::Range;
using slimt::Response;

using Service = slimt::Async;
using _Model = slimt::Model;
using Model = std::shared_ptr<_Model>;

PYBIND11_MAKE_OPAQUE(std::vector<Response>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(Alignments);

class ServicePyAdapter {
 public:
  ServicePyAdapter(const size_t numWorkers, const size_t cacheSize,
                   const std::string &logLevel)
      : service_(make_service(numWorkers, cacheSize, logLevel)) {
    // Set marian to throw exceptions instead of std::abort()
    marian::setThrowExceptionOnAbort(true);
  }

  std::vector<Response> translate(Model model, py::list &texts, bool html,
                                  bool qualityScores, bool alignment) {
    py::scoped_ostream_redirect outstream(
        std::cout,                                 // std::ostream&
        py::module_::import("sys").attr("stdout")  // Python output
    );
    py::scoped_ostream_redirect errstream(
        std::cerr,                                 // std::ostream&
        py::module_::import("sys").attr("stderr")  // Python output
    );

    py::call_guard<py::gil_scoped_release> gil_guard;

    std::vector<std::string> inputs;
    for (auto handle : texts) {
      inputs.push_back(py::str(handle));
    }

    // Prepare promises, save respective futures. Have callback's in async set
    // value to the promises.
    std::vector<std::future<Response>> futures;
    std::vector<std::promise<Response>> promises;
    promises.resize(inputs.size());

    Options options;
    options.HTML = html;
    options.qualityScores = qualityScores;
    options.alignment = alignment;

    for (size_t i = 0; i < inputs.size(); i++) {
      auto callback = [&promises, i](Response &&response) {
        promises[i].set_value(std::move(response));
      };

      service_.translate(model, std::move(inputs[i]), std::move(callback),
                         options);

      futures.push_back(std::move(promises[i].get_future()));
    }

    // Wait on all futures to be ready.
    std::vector<Response> responses;
    for (size_t i = 0; i < futures.size(); i++) {
      futures[i].wait();
      responses.push_back(std::move(futures[i].get()));
    }

    return responses;
  }

  std::vector<Response> pivot(Model first, Model second, py::list &texts,
                              bool html, bool qualityScores, bool alignment) {
    py::scoped_ostream_redirect outstream(
        std::cout,                                 // std::ostream&
        py::module_::import("sys").attr("stdout")  // Python output
    );
    py::scoped_ostream_redirect errstream(
        std::cerr,                                 // std::ostream&
        py::module_::import("sys").attr("stderr")  // Python output
    );

    py::call_guard<py::gil_scoped_release> gil_guard;

    std::vector<std::string> inputs;
    for (auto handle : texts) {
      inputs.push_back(py::str(handle));
    }

    Options options{
        .alignment = alignment,  //
        .html = html             //
    };

    // Prepare promises, save respective futures. Have callback's in async set
    // value to the promises.
    std::vector<std::future<Response>> futures;
    std::vector<std::promise<Response>> promises;
    promises.resize(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++) {
      auto callback = [&promises, i](Response &&response) {
        promises[i].set_value(std::move(response));
      };

      service_.pivot(first, second, std::move(inputs[i]), std::move(callback),
                     options);

      futures.push_back(std::move(promises[i].get_future()));
    }

    // Wait on all futures to be ready.
    std::vector<Response> responses;
    for (size_t i = 0; i < futures.size(); i++) {
      futures[i].wait();
      responses.push_back(std::move(futures[i].get()));
    }

    return responses;
  }

 private /*functions*/:
  static Service make_service(size_t numWorkers, size_t cacheSize,
                              const std::string &logLevel) {
    py::scoped_ostream_redirect outstream(
        std::cout,                                 // std::ostream&
        py::module_::import("sys").attr("stdout")  // Python output
    );
    py::scoped_ostream_redirect errstream(
        std::cerr,                                 // std::ostream&
        py::module_::import("sys").attr("stderr")  // Python output
    );

    py::call_guard<py::gil_scoped_release> gil_guard;

    Service::Config config;
    config.numWorkers = numWorkers;
    config.cacheSize = cacheSize;
    config.logger.level = logLevel;

    return Service(config);
  }

 private /*data*/:
  Service service_;
};

PYBIND11_MODULE(_bergamot, m) {
  m.doc() = "Bergamot pybind11 bindings";
  m.attr("__version__") = slimt::bergamotBuildVersion();
  py::class_<Range>(m, "Range")
      .def(py::init<>())
      .def_readonly("begin", &Range::begin)
      .def_readonly("end", &Range::end)
      .def("__repr__", [](const Range &range) {
        return "{" + std::to_string(range.begin) + ", " +
               std::to_string(range.end) + "}";
      });

  py::class_<AnnotatedText>(m, "AnnotatedText")
      .def(py::init<>())
      .def("word_count", &AnnotatedText::word_count)
      .def("sentence_count", &AnnotatedText::sentence_count)
      .def("word",
           [](const AnnotatedText &annotated_text, size_t sentence_id,
              size_t word_id) -> std::string {
             auto view = annotated_text.word(sentence_id, word_id);
             return std::string(view.data(), view.size());
           })
      .def("sentence",
           [](const AnnotatedText &annotated_text,
              size_t sentence_id) -> std::string {
             auto view = annotated_text.sentence(sentence_id);
             return std::string(view.data(), view.size());
           })
      .def("wordAsRange", &AnnotatedText::wordAsRange)
      .def("sentenceAsRange", &AnnotatedText::sentenceAsRange)
      .def_readonly("text", &AnnotatedText::text);

  py::class_<Response>(m, "Response")
      .def(py::init<>())
      .def_readonly("source", &Response::source)
      .def_readonly("target", &Response::target)
      .def_readonly("alignments", &Response::alignments);

  py::bind_vector<std::vector<std::string>>(m, "VectorString");
  py::bind_vector<std::vector<Response>>(m, "VectorResponse");

  py::bind_vector<std::vector<float>>(m, "VectorFloat");
  py::bind_vector<Alignment>(m, "Alignment");
  py::bind_vector<Alignments>(m, "Alignments");

  py::class_<ServicePyAdapter>(m, "Service")
      .def(py::init<size_t, size_t, const std::string &>(),
           py::arg("num_workers") = 1, py::arg("cache_size") = 0,
           py::arg("log_level") = "off")
      .def("translate", &ServicePyAdapter::translate, py::arg("model"),
           py::arg("texts"), py::arg("html") = false,
           py::arg("quality_scores") = false, py::arg("alignment") = false)
      .def("pivot", &ServicePyAdapter::pivot, py::arg("first"),
           py::arg("second"), py::arg("texts"), py::arg("html") = false,
           py::arg("quality_scores") = false, py::arg("alignment") = false);

  py::class_<_Model, std::shared_ptr<_Model>>(m, "Model")
      .def_static(
          "from_config",
          [](const std::string &config) {
            auto options = slimt::parseOptionsFromString(config);
            return std::make_shared<_Model>(options);
          },
          py::arg("config"))
      .def_static(
          "from_config_path",
          [](const std::string &configPath) {
            auto options = slimt::parseOptionsFromFilePath(configPath);
            return std::make_shared<_Model>(options);
          },
          py::arg("config_path"));
}
