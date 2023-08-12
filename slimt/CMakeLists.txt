add_library(
  slimt STATIC
  Aligned.cc
  Vocabulary.cc
  Io.cc
  Tensor.cc
  Model.cc
  TensorOps.cc
  Tensori8Ops.cc
  Batch.cc
  Utils.cc
  Shortlist.cc)

target_compile_options(slimt PRIVATE ${SLIMT_EXTRA_COMPILE_OPTIONS})
target_link_libraries(slimt PUBLIC ${SLIMT_EXTERNAL_LIBS})

if(WITH_ASAN)
  target_compile_options(slimt PUBLIC -fsanitize=address -fsanitize=undefined)
  target_link_options(slimt PUBLIC -fsanitize=address -fsanitize=undefined)
endif(WITH_ASAN)
