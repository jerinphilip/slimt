#include <emscripten/bind.h>

#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/Frontend.hh"
#include "slimt/Model.hh"
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
      .field("html", &Options::html);
  register_vector<Options>("VectorOptions");
}

EMSCRIPTEN_BINDINGS(aligned_memory) {
  class_<Aligned>("Aligned")
      .constructor<std::size_t, std::size_t>()
      .function("size", &Aligned::size)
      .function("as_bytes", [](Aligned& aligned) -> val {
        return val(typed_memory_view(aligned.size(), aligned.begin()));
      });

  register_vector<Aligned*>("AlignedList");
}

EMSCRIPTEN_BINDINGS(translation_model) {
  class_<Model>("Model").smart_ptr_constructor(
      "Model",
      [](const Config& config,                //
         Aligned* model,                      //
         Aligned* shortlist,                  //
         std::vector<Aligned*> vocabularies,  //
         ) -> std::shared_ptr<Model> {
        // TODO(jerinphilip): Fix
        auto from_aligned = [](Aligned* aligned) -> View {
          return {
              .data = aligned->data(),  //
              .size = aligned->size()   //
          };
        };

        // Something needs to hold this.
        Package<View> view = {
            .model = from_aligned(model),                 //
            .vocabulary = from_aligned(vocabularies[0]),  //
            .shortlist = from_aligned(shortlist)          //
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
