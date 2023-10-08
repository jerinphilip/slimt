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
using slimt::Config;
using slimt::Options;
using slimt::Range;
using slimt::Response;

using Package = slimt::Package<std::string>;
using Service = slimt::Async;
using Model = slimt::Model;

PYBIND11_MAKE_OPAQUE(std::vector<Response>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(Alignments);

class PyService {
 public:
  PyService(const size_t workers, const size_t cache_size)
      : service_(make_service(workers, cache_size)) {
    // Set slimt to throw exceptions instead of std::abort()
    // slimt::setThrowExceptionOnAbort(true);
  }

  std::vector<Response> translate(std::shared_ptr<Model> model, py::list &texts,
                                  bool html) {
    py::scoped_ostream_redirect outstream(
        std::cout,                                 // std::ostream&
        py::module_::import("sys").attr("stdout")  // Python output
    );
    py::scoped_ostream_redirect errstream(
        std::cerr,                                 // std::ostream&
        py::module_::import("sys").attr("stderr")  // Python output
    );

    py::call_guard<py::gil_scoped_release> gil_guard;

    std::vector<std::string> sources;
    for (auto handle : texts) {
      sources.push_back(py::str(handle));
    }

    // Prepare promises, save respective futures. Have callback's in async set
    // value to the promises.
    std::vector<std::future<Response>> futures;

    Options options{
        .html = html,  //
    };

    for (auto &source : sources) {
      std::future<Response> future =
          service_.translate(model, std::move(source), options);

      futures.push_back(std::move(future));
    }

    // Wait on all futures to be ready.
    std::vector<Response> responses;
    for (auto &future : futures) {
      future.wait();
      responses.push_back(std::move(future.get()));
    }

    return responses;
  }

#if 0
  std::vector<Response> pivot(std::shared_ptr<Model> first, std::shared_ptr<Model> second, py::list &texts,
                              bool html) {
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
        .html = html  //
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

      service_.pivot(            //
          first, second,         //
          std::move(inputs[i]),  //
          std::move(callback),   //
          options                //
      );

      futures.push_back(std::move(promises[i].get_future()));
    }

    // Wait on all futures to be ready.
    std::vector<Response> responses;
    for (size_t i = 0; i < futures.size(); i++) {
      futures[i].wait();
      Response response = futures[i].get();
      responses.push_back(std::move(response));
    }

    return responses;
  }
#endif

 private:
  static Service make_service(size_t workers, size_t cache_size) {
    py::scoped_ostream_redirect outstream(
        std::cout,                                 // std::ostream&
        py::module_::import("sys").attr("stdout")  // Python output
    );
    py::scoped_ostream_redirect errstream(
        std::cerr,                                 // std::ostream&
        py::module_::import("sys").attr("stderr")  // Python output
    );

    py::call_guard<py::gil_scoped_release> gil_guard;

    Config config;
    config.workers = workers;
    config.cache_size = cache_size;

    return Service(config);
  }

  Service service_;
};

PYBIND11_MODULE(_slimt, m) {
  m.doc() = "slimt python bindings";
  m.attr("__version__") = slimt::version();
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
      .def("word_as_range", &AnnotatedText::word_as_range)
      .def("sentence_as_range", &AnnotatedText::sentence_as_range)
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
  py::class_<Package>(m, "Package")
      .def(py::init<>())
      .def_readwrite("model", &Package::model)
      .def_readwrite("vocabulary", &Package::vocabulary)
      .def_readwrite("shortlist", &Package::shortlist);

  py::class_<Config>(m, "Config").def(py::init<>());

  py::class_<PyService>(m, "Service")
      .def(py::init<size_t, size_t>(), py::arg("workers") = 1,
           py::arg("cache_size") = 0)
      .def("translate", &PyService::translate, py::arg("model"),
           py::arg("texts"), py::arg("html") = false)
#if 0
      .def("pivot", &PyService::pivot, py::arg("first"),
           py::arg("second"), py::arg("texts"), py::arg("html") = false)
#endif
      ;

  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(py::init<>([](const Config &config, const Package &package) {
             return std::make_shared<Model>(config, package);
           }),
           py::arg("config"), py::arg("package"));
}
