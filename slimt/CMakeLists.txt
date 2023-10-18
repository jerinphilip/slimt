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
    Transformer.cc
    Utils.cc
    Vocabulary.cc
    XHScanner.cc)

set(SLIMT_LIBRIARIES slimt-static slimt-shared)
add_library(slimt-static STATIC ${SLIMT_SOURCES})
add_library(slimt-shared SHARED ${SLIMT_SOURCES})

foreach(SLIMT_LIB IN LISTS SLIMT_LIBRIARIES)
  target_compile_options(${SLIMT_LIB} PRIVATE ${SLIMT_COMPILE_OPTIONS})
  set_target_properties(${SLIMT_LIB} PROPERTIES OUTPUT_NAME "slimt")
  target_compile_definitions(${SLIMT_LIB} PRIVATE ${SLIMT_COMPILE_DEFINITIONS})
  target_link_libraries(${SLIMT_LIB} PUBLIC ${SLIMT_EXTERNAL_LIBS})
  target_link_options(${SLIMT_LIB} PUBLIC ${SLIMT_LINK_OPTIONS})
endforeach()

if(UNIX)
  install(FILES ${SLIMT_PUBLIC_HEADERS}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slimt)
  install(TARGETS ${SLIMT_LIBRIARIES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  message(STATUS "Exports: ${SLIMT_PUBLIC_HEADERS} ${SLIMT_LIBRIARIES}")
endif(UNIX)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Version.hh.in
               ${CMAKE_CURRENT_BINARY_DIR}/Version.hh @ONLY)
