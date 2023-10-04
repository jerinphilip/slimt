set(SLIMT_PUBLIC_HEADERS
    Aligned.hh
    Annotation.hh
    Frontend.hh
    Model.hh
    Splitter.hh
    TextProcessor.hh
    slimt.hh)

set(SLIMT_SOURCES
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

set(SLIMT_LIBRIARIES slimt-static slimt-shared)

add_library(slimt-static STATIC ${SLIMT_SOURCES})
target_compile_options(slimt-static PRIVATE ${SLIMT_COMPILE_OPTIONS})
set_target_properties(slimt-static PROPERTIES OUTPUT_NAME "slimt")
target_compile_definitions(slimt-static PRIVATE ${SLIMT_COMPILE_DEFINITIONS})
target_link_libraries(slimt-static PUBLIC ${SLIMT_EXTERNAL_LIBS})

if(WITH_ASAN)
  target_compile_options(slimt PUBLIC -fsanitize=address -fsanitize=undefined)
  target_link_options(slimt PUBLIC -fsanitize=address -fsanitize=undefined)
endif(WITH_ASAN)

add_library(slimt-shared SHARED ${SLIMT_SOURCES})
target_compile_options(slimt-shared PRIVATE ${SLIMT_COMPILE_OPTIONS})
set_target_properties(slimt-shared PROPERTIES OUTPUT_NAME "slimt")
target_compile_definitions(slimt-static PRIVATE ${SLIMT_COMPILE_DEFINITIONS})
target_compile_definitions(slimt-shared PRIVATE ${SLIMT_COMPILE_DEFINITIONS})
target_link_libraries(slimt-shared PUBLIC ${SLIMT_EXTERNAL_LIBS})

if(WITH_ASAN)
  target_compile_options(slimt-shared PUBLIC -fsanitize=address
                                             -fsanitize=undefined)
  target_link_options(slimt-shared PUBLIC -fsanitize=address
                      -fsanitize=undefined)
endif(WITH_ASAN)

if(UNIX)
  install(FILES ${SLIMT_PUBLIC_HEADERS}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slimt)
  install(TARGETS ${SLIMT_LIBRIARIES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  message(STATUS "Exports: ${SLIMT_PUBLIC_HEADERS} ${SLIMT_LIBRIARIES}")
endif(UNIX)
