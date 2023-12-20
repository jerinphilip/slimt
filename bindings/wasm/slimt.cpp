#include <emscripten/bind.h>

#include <vector>

#include "slimt/Frontend.hh"
#include "slimt/Response.hh"

// This file is narrow enough and a source file), so we will import everything.
using namespace slimt;
using namespace emscripten;

// Binding code
EMSCRIPTEN_BINDINGS(range) {
  value_object<Range>("Range")
      .field("begin", &Range::begin)
      .field("end", &Range::end);
}

EMSCRIPTEN_BINDINGS(response) {
  class_<Response>("Response")
      .constructor<>()
      .function("size", &Response::size)
      .function("source", &Response::source)
      .function("target", &Response::target);

  register_vector<Response>("VectorResponse");
}

// Binding code
EMSCRIPTEN_BINDINGS(options) {
  value_object<Options>("Options")
      .field("alignment", &Options::alignment)
      .field("html", &Options::HTML);
  register_vector<Options>("VectorOptions");
}

EMSCRIPTEN_BINDINGS(aligned_memory) {
  class_<Aligned>("Aligned")
      .constructor<std::size_t, std::size_t>()
      .function("size", &Aligned::size)
      .function("as_bytes", [](Aligned& aligned) -> val {
        return val(typed_memory_view(aligned.size(), aligned.as<char>()));
      });

  register_vector<Aligned*>("AlignedList");
}

EMSCRIPTEN_BINDINGS(translation_model) {
  class_<Model>("Model").smart_ptr_constructor(
      "Model",
      [](const std::string& config,           //
         Aligned* model,                      //
         Aligned* shortlist,                  //
         std::vector<Aligned*> vocabularies,  //
         ) -> std::shared_ptr<Model> {
        // TODO(jerinphilip): Fix
        Package<View> view = {
            .model = View{.data = model->data(), size = model->size()},
            .vocabulary =
                View{.data = vocabularies->data(), size = vocabularies->size()},
            .shortlist =
                View{.data = shortlist->data(), size = shortlist->size()},
        };

        return std::make_shared<Model>(config, view);
      },
      allow_raw_pointers());
}

EMSCRIPTEN_BINDINGS(blocking_service) {
  class_<Blocking>("Blocking")
      .smart_ptr_constructor(
          "Blocking",
          auto [](const Config& config)->std::shared_ptr<Blocking> {
            return std::make_shared<Blocking>(config);
          })
      .function("translate", &Blocking::translate)
      .function("pivot", &Blocking::pivot);
  register_vector<std::string>("VectorString");
}
