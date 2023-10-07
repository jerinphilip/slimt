#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <translator/annotation.h>
#include <translator/parser.h>
#include <translator/project_version.h>
#include <translator/response.h>
#include <translator/response_options.h>
#include <translator/service.h>
#include <translator/translation_model.h>

#include <iostream>
#include <string>
#include <vector>

namespace py = pybind11;

using marian::bergamot::AnnotatedText;
using marian::bergamot::ByteRange;
using marian::bergamot::Response;
using marian::bergamot::ResponseOptions;
using Service = marian::bergamot::AsyncService;
using _Model = marian::bergamot::TranslationModel;
using Model = std::shared_ptr<_Model>;
using Alignment = std::vector<std::vector<float>>;
using Alignments = std::vector<Alignment>;

PYBIND11_MAKE_OPAQUE(std::vector<Response>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(Alignments);

class ServicePyAdapter {
 public:
  ServicePyAdapter(const size_t numWorkers, const size_t cacheSize, const std::string &logLevel)
      : service_(make_service(numWorkers, cacheSize, logLevel)) {
    // Set marian to throw exceptions instead of std::abort()
    marian::setThrowExceptionOnAbort(true);
  }

  std::vector<Response> translate(Model model, py::list &texts, bool html, bool qualityScores, bool alignment) {
    py::scoped_ostream_redirect outstream(std::cout,                                 // std::ostream&
                                          py::module_::import("sys").attr("stdout")  // Python output
    );
    py::scoped_ostream_redirect errstream(std::cerr,                                 // std::ostream&
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

    ResponseOptions options;
    options.HTML = html;
    options.qualityScores = qualityScores;
    options.alignment = alignment;

    for (size_t i = 0; i < inputs.size(); i++) {
      auto callback = [&promises, i](Response &&response) { promises[i].set_value(std::move(response)); };

      service_.translate(model, std::move(inputs[i]), std::move(callback), options);

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

  std::vector<Response> pivot(Model first, Model second, py::list &texts, bool html, bool qualityScores,
                              bool alignment) {
    py::scoped_ostream_redirect outstream(std::cout,                                 // std::ostream&
                                          py::module_::import("sys").attr("stdout")  // Python output
    );
    py::scoped_ostream_redirect errstream(std::cerr,                                 // std::ostream&
                                          py::module_::import("sys").attr("stderr")  // Python output
    );

    py::call_guard<py::gil_scoped_release> gil_guard;

    std::vector<std::string> inputs;
    for (auto handle : texts) {
      inputs.push_back(py::str(handle));
    }

    ResponseOptions options;
    options.HTML = html;
    options.qualityScores = qualityScores;
    options.alignment = alignment;

    // Prepare promises, save respective futures. Have callback's in async set
    // value to the promises.
    std::vector<std::future<Response>> futures;
    std::vector<std::promise<Response>> promises;
    promises.resize(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++) {
      auto callback = [&promises, i](Response &&response) { promises[i].set_value(std::move(response)); };

      service_.pivot(first, second, std::move(inputs[i]), std::move(callback), options);

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
  static Service make_service(size_t numWorkers, size_t cacheSize, const std::string &logLevel) {
    py::scoped_ostream_redirect outstream(std::cout,                                 // std::ostream&
                                          py::module_::import("sys").attr("stdout")  // Python output
    );
    py::scoped_ostream_redirect errstream(std::cerr,                                 // std::ostream&
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
  m.attr("__version__") = marian::bergamot::bergamotBuildVersion();
  py::class_<ByteRange>(m, "ByteRange")
      .def(py::init<>())
      .def_readonly("begin", &ByteRange::begin)
      .def_readonly("end", &ByteRange::end)
      .def("__repr__", [](const ByteRange &range) {
        return "{" + std::to_string(range.begin) + ", " + std::to_string(range.end) + "}";
      });

  py::class_<AnnotatedText>(m, "AnnotatedText")
      .def(py::init<>())
      .def("numWords", &AnnotatedText::numWords)
      .def("numSentences", &AnnotatedText::numSentences)
      .def("word",
           [](const AnnotatedText &annotatedText, size_t sentenceIdx, size_t wordIdx) -> std::string {
             auto view = annotatedText.word(sentenceIdx, wordIdx);
             return std::string(view.data(), view.size());
           })
      .def("sentence",
           [](const AnnotatedText &annotatedText, size_t sentenceIdx) -> std::string {
             auto view = annotatedText.sentence(sentenceIdx);
             return std::string(view.data(), view.size());
           })
      .def("wordAsByteRange", &AnnotatedText::wordAsByteRange)
      .def("sentenceAsByteRange", &AnnotatedText::sentenceAsByteRange)
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
      .def(py::init<size_t, size_t, const std::string &>(), py::arg("num_workers") = 1, py::arg("cache_size") = 0,
           py::arg("log_level") = "off")
      .def("translate", &ServicePyAdapter::translate, py::arg("model"), py::arg("texts"), py::arg("html") = false,
           py::arg("quality_scores") = false, py::arg("alignment") = false)
      .def("pivot", &ServicePyAdapter::pivot, py::arg("first"), py::arg("second"), py::arg("texts"),
           py::arg("html") = false, py::arg("quality_scores") = false, py::arg("alignment") = false);

  py::class_<_Model, std::shared_ptr<_Model>>(m, "Model")
      .def_static(
          "from_config",
          [](const std::string &config) {
            auto options = marian::bergamot::parseOptionsFromString(config);
            return marian::New<_Model>(options);
          },
          py::arg("config"))
      .def_static(
          "from_config_path",
          [](const std::string &configPath) {
            auto options = marian::bergamot::parseOptionsFromFilePath(configPath);
            return marian::New<_Model>(options);
          },
          py::arg("config_path"));
}
