#include "slimt/slimt.hh"

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <iostream>
#include <string>
#include <vector>

namespace py = pybind11;

#define pystdout py::module_::import("sys").attr("stdout")
#define pystderr py::module_::import("sys").attr("stderr")

using slimt::Alignment;
using slimt::Alignments;
using slimt::AnnotatedText;
using ServiceConfig = slimt::Config;
using ModelConfig = slimt::Model::Config;
using slimt::Options;
using slimt::Range;
using slimt::Response;

using Package = slimt::Package<std::string>;
using Service = slimt::Async;
using Model = slimt::Model;

PYBIND11_MAKE_OPAQUE(std::vector<Response>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(Alignments);

class Redirect {
 public:
  Redirect() : out_(std::cout, pystdout), err_(std::cerr, pystderr) {}

 public:
  py::scoped_ostream_redirect out_;
  py::scoped_ostream_redirect err_;
};

class PyService {
 public:
  PyService(const size_t workers, const size_t cache_size)
      : service_(make_service(workers, cache_size)) {
    // Set slimt to throw exceptions instead of std::abort()
    // slimt::setThrowExceptionOnAbort(true);
  }

  std::vector<Response> translate(std::shared_ptr<Model> model, py::list &texts,
                                  bool html) {
    py::call_guard<py::gil_scoped_release> gil_guard;
    Redirect redirect;

    std::vector<std::string> sources;
    for (auto handle : texts) {
      sources.push_back(py::str(handle));
    }

    // Prepare promises, save respective futures. Have callback's in async set
    // value to the promises.
    using Handle = slimt::Handle;
    std::vector<Handle> handles;

    Options options{
        .html = html,  //
    };

    for (auto &source : sources) {
      Handle handle = service_.translate(model, std::move(source), options);
      handles.push_back(std::move(handle));
    }

    // Wait on all futures to be ready.
    std::vector<Response> responses;
    for (auto &handle : handles) {
      auto &future = handle.future();
      future.wait();
      Response response = future.get();
      change_ranges_to_utf8(response);
      responses.push_back(std::move(response));
    }

    return responses;
  }

  std::vector<Response> pivot(std::shared_ptr<Model> first,
                              std::shared_ptr<Model> second, py::list &texts,
                              bool html) {
    py::call_guard<py::gil_scoped_release> gil_guard;
    Redirect redirect;

    std::vector<std::string> sources;
    for (auto handle : texts) {
      sources.push_back(py::str(handle));
    }

    Options options{
        .html = html  //
    };

    using Handle = slimt::Handle;
    std::vector<Handle> handles;
    for (size_t i = 0; i < sources.size(); i++) {
      std::string &source = sources[i];
      Handle handle = service_.pivot(  //
          first, second,               //
          std::move(source),           //
          options                      //
      );

      handles.push_back(std::move(handle));
    }

    // Wait on all futures to be ready.
    std::vector<Response> responses;
    for (auto &handle : handles) {
      handle.future().wait();
      Response response = handle.future().get();
      responses.push_back(std::move(response));
    }

    return responses;
  }

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

    ServiceConfig config;
    config.workers = workers;
    config.cache_size = cache_size;

    return Service(config);
  }

  static void change_ranges_to_utf8(Response &response) {
    // Ranges are in C++ Byte Ranges.  Python however requires indices into
    // unicode strings. For example, the following is required without this
    // change.
    //
    // # Case 1: Byte Ranges
    //
    // source = response.source.text // "Hello world"
    // range = response.source.word_as_range(0, 0)
    //
    // #  Additional encoding step required to access
    // #  "byte" ranges.
    // bytes = source.encode("utf8")
    //
    // word_bytes = bytes[range.begin:range.end]
    // word = word_bytes.decode("utf8")
    //
    // # Case 2: Unicode (UTF-8 ranges)
    //
    // What we want to do, is omit the conversion step, by providing ranges
    // native to python.
    //
    // source = response.source.text // "Hello world"
    // range = response.source.word_as_range(0, 0)
    // word = source[range.begin:range.end]
    //
    //
    // Assert:
    //  word == response.source.word(0, 0)
    //  [python-slicing] = C++ string conversion
    //
    // Why? Diagnosis?
    //
    // pybind11 converts std::string, by default to unicode-strings (python).
    // However, since we manage ranges, they remain "byte ranges". We can
    // mitigate this problem, by patching the annotation's ranges to point to
    // unicode before returning the response.

    auto transform_to_utf8 =
        [](const AnnotatedText &annotated) -> std::vector<Range> {
      // f: Range [Bytes] -> Range[UTF8]
      const std::string &text = annotated.text;

      size_t sentence_idx = 0;
      size_t word_idx = 0;
      Range current = annotated.word_as_range(sentence_idx, word_idx);

      std::vector<Range> words;

      size_t utf8_idx = 0;
      size_t byte_idx = 0;
      Range utf8{
          //
          .begin = utf8_idx,  //
          .end = 0            //
      };
      // This loops run on the entire string.
      //
      // We have indices into two views of the same string. One is bytes, the
      // other is utf8 encoded. We want to convert what is bytes to utf8.
      //
      // For this, we traverse through the characters, checking for unicode
      // encoding and associated adjustments to the utf8_idx, keeping
      // correspondences with the byte_idx;
      for (const char c : text) {
        // current = [begin, end)
        if ((c & 0xc0) != 0x80)  // if is not utf-8 continuation character
          ++utf8_idx;

        ++byte_idx;

        if (byte_idx == current.end) {
          utf8.end = utf8_idx;

          // Push it to the list of (utf8) words.
          // We will use these to create the utf8 range annotation.
          words.push_back(utf8);

          ++word_idx;
          if (word_idx >= annotated.word_count(sentence_idx)) {
            ++sentence_idx;
            word_idx = 0;
          }

          current = annotated.word_as_range(sentence_idx, word_idx);

          utf8.begin = utf8_idx;
          utf8.end = -1;
        }
      }
      return words;
    };

    response.source.update_annotation(transform_to_utf8(response.source));
    response.target.update_annotation(transform_to_utf8(response.target));
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
      // .def("word",
      //      [](const AnnotatedText &annotated_text, size_t sentence_id,
      //         size_t word_id) -> std::string {
      //        auto view = annotated_text.word(sentence_id, word_id);
      //        return std::string(view.data(), view.size());
      //      })
      // .def("sentence",
      //      [](const AnnotatedText &annotated_text,
      //         size_t sentence_id) -> std::string {
      //        auto view = annotated_text.sentence(sentence_id);
      //        return std::string(view.data(), view.size());
      //      })
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
      .def(py::init<>([](std::string model, std::string vocabulary,
                         std::string shortlist) {
             return Package{.model = std::move(model),
                            .vocabulary = std::move(vocabulary),
                            .shortlist = std::move(shortlist)};
           }),
           py::arg("model"), py::arg("vocabulary"), py::arg("shortlist"))
      .def_readwrite("model", &Package::model)
      .def_readwrite("vocabulary", &Package::vocabulary)
      .def_readwrite("shortlist", &Package::shortlist);

  py::class_<ModelConfig>(m, "Config").def(py::init<>());

  py::class_<PyService>(m, "Service")
      .def(py::init<size_t, size_t>(), py::arg("workers") = 1,
           py::arg("cache_size") = 0)
      .def("translate", &PyService::translate, py::arg("model"),
           py::arg("texts"), py::arg("html") = false)
      .def("pivot", &PyService::pivot, py::arg("first"), py::arg("second"),
           py::arg("texts"), py::arg("html") = false);

  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(py::init<>([](const ModelConfig &config, const Package &package) {
             return std::make_shared<Model>(config, package);
           }),
           py::arg("config"), py::arg("package"));
}
