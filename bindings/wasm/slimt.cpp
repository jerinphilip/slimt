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

val getByteArrayView(Aligned& aligned) {
  return val(typed_memory_view(aligned.size(), aligned.as<char>()));
}

EMSCRIPTEN_BINDINGS(aligned_memory) {
  class_<Aligned>("Aligned")
      .constructor<std::size_t, std::size_t>()
      .function("size", &Aligned::size)
      .function("getByteArrayView", &getByteArrayView);

  register_vector<Aligned*>("AlignedList");
}

// When source and target vocab files are same, only one memory object is passed
// from JS to avoid allocating memory twice for the same file. However, the
// constructor of the Service class still expects 2 entries in this case, where
// each entry has the shared ownership of the same Aligned object. This
// function prepares these smart pointer based Aligned objects for unique
// Aligned objects passed from JS.
std::vector<std::shared_ptr<Aligned>> prepareVocabsSmartMemories(
    std::vector<Aligned*>& vocabsMemories) {
  auto sourceVocabMemory =
      std::make_shared<Aligned>(std::move(*(vocabsMemories[0])));
  std::vector<std::shared_ptr<Aligned>> vocabsSmartMemories;
  vocabsSmartMemories.push_back(sourceVocabMemory);
  if (vocabsMemories.size() == 2) {
    auto targetVocabMemory =
        std::make_shared<Aligned>(std::move(*(vocabsMemories[1])));
    vocabsSmartMemories.push_back(std::move(targetVocabMemory));
  } else {
    vocabsSmartMemories.push_back(sourceVocabMemory);
  }
  return vocabsSmartMemories;
}

MemoryBundle prepareMemoryBundle(Aligned* modelMemory, Aligned* shortlistMemory,
                                 std::vector<Aligned*> uniqueVocabsMemories,
                                 Aligned* qualityEstimatorMemory) {
  MemoryBundle memoryBundle;
  memoryBundle.models.emplace_back(std::move(*modelMemory));
  memoryBundle.shortlist = std::move(*shortlistMemory);
  memoryBundle.vocabs =
      std::move(prepareVocabsSmartMemories(uniqueVocabsMemories));
  if (qualityEstimatorMemory != nullptr) {
    memoryBundle.qualityEstimatorMemory = std::move(*qualityEstimatorMemory);
  }

  return memoryBundle;
}

// This allows only shared_ptrs to be operational in JavaScript, according to
// emscripten.
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html#smart-pointers
std::shared_ptr<TranslationModel> TranslationModelFactory(
    const std::string& config, Aligned* model, Aligned* shortlist,
    std::vector<Aligned*> vocabs, Aligned* qualityEstimator) {
  MemoryBundle memoryBundle =
      prepareMemoryBundle(model, shortlist, vocabs, qualityEstimator);
  return std::make_shared<TranslationModel>(config, std::move(memoryBundle));
}

EMSCRIPTEN_BINDINGS(translation_model) {
  class_<TranslationModel>("TranslationModel")
      .smart_ptr_constructor("TranslationModel", &TranslationModelFactory,
                             allow_raw_pointers());
}

EMSCRIPTEN_BINDINGS(blocking_service_config) {
  value_object<BlockingService::Config>("BlockingServiceConfig")
      .field("cacheSize", &BlockingService::Config::cacheSize);
}

std::shared_ptr<BlockingService> BlockingServiceFactory(
    const BlockingService::Config& config) {
  auto copy = config;
  copy.logger.level = "critical";
  return std::make_shared<BlockingService>(copy);
}

EMSCRIPTEN_BINDINGS(blocking_service) {
  class_<Blocking>("Blocking")
      .smart_ptr_constructor("Blocking", &BlockingFactory)
      .function("translate", &Blocking::translate)
      .function("pivot", &Blocking::pivot);
  register_vector<std::string>("VectorString");
}
