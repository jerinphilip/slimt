add_library(
  slimt STATIC
  Aligned.cc
  Annotation.cc
  Batch.cc
  Batcher.cc
  Frontend.cc
  Io.cc
  Model.cc
  Modules.cc
  QMM.cc
  Request.cc
  Response.cc
  ResponseBuilder.cc
  Shortlist.cc
  Splitter.cc
  Tensor.cc
  TensorOps.cc
  Utils.cc
  Vocabulary.cc)

target_compile_options(slimt PRIVATE ${SLIMT_COMPILE_OPTIONS})
target_compile_definitions(slimt PRIVATE ${SLIMT_COMPILE_DEFINITIONS})
target_link_libraries(slimt PUBLIC ${SLIMT_EXTERNAL_LIBS})

if(WITH_ASAN)
  target_compile_options(slimt PUBLIC -fsanitize=address -fsanitize=undefined)
  target_link_options(slimt PUBLIC -fsanitize=address -fsanitize=undefined)
endif(WITH_ASAN)
