add_library(
  slimt STATIC
  Aligned.cc
  Annotation.cc
  Batch.cc
  Batcher.cc
  Frontend.cc
  HTML.cc
  Io.cc
  Model.cc
  Modules.cc
  QMM.cc
  Regex.cc
  Request.cc
  Response.cc
  ResponseBuilder.cc
  Shortlist.cc
  Splitter.cc
  Tensor.cc
  TensorOps.cc
  TextProcessor.cc
  Utils.cc
  Vocabulary.cc
  XHScanner.cc)

target_compile_options(slimt PRIVATE ${SLIMT_COMPILE_OPTIONS})
target_compile_definitions(slimt PRIVATE ${SLIMT_COMPILE_DEFINITIONS})
target_link_libraries(slimt PUBLIC ${SLIMT_EXTERNAL_LIBS})

if(WITH_ASAN)
  target_compile_options(slimt PUBLIC -fsanitize=address -fsanitize=undefined)
  target_link_options(slimt PUBLIC -fsanitize=address -fsanitize=undefined)
endif(WITH_ASAN)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Version.hh.in
               ${CMAKE_CURRENT_BINARY_DIR}/Version.hh @ONLY)

