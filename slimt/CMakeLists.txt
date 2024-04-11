set(SLIMT_PUBLIC_HEADERS
    slimt.hh
    Aligned.hh
    Annotation.hh
    Batcher.hh
    Cache.hh
    Frontend.hh
    Io.hh
    Model.hh
    Modules.hh
    Response.hh
    Search.hh
    Shortlist.hh
    Splitter.hh
    Tensor.hh
    TextProcessor.hh
    Transformer.hh
    Transliterator.hh
    Types.hh
    Vocabulary.hh
    Utils.hh)

# Configure generated headers. Version.hh can be configured here.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Version.hh.in
               ${CMAKE_CURRENT_BINARY_DIR}/Version.hh @ONLY)

# Generated headers also include Export.hh, which is generated after the library
# target `slimt` is defined.
set(SLIMT_GENERATED_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/Version.hh
                            ${CMAKE_CURRENT_BINARY_DIR}/Export.hh)

set(SLIMT_PRIVATE_HEADERS
    HTML.hh
    QMM.hh
    Regex.hh
    Request.hh
    TensorOps.hh
    Utils.hh
    XHScanner.hh)

set(SLIMT_SOURCES
    Aligned.cc
    Annotation.cc
    Batcher.cc
    Frontend.cc
    HTML.cc
    Input.cc
    Io.cc
    Model.cc
    Modules.cc
    QMM.cc
    Regex.cc
    Request.cc
    Response.cc
    Search.cc
    Shortlist.cc
    Splitter.cc
    Tensor.cc
    TensorOps.cc
    TextProcessor.cc
    Transformer.cc
    Transliterator.cc
    Utils.cc
    Vocabulary.cc
    XHScanner.cc)

if(WITH_GEMMOLOGY)
  list(APPEND SLIMT_SOURCES qmm/Gemmology.cc)
  set_source_files_properties(
    qmm/Gemmology.cc PROPERTIES COMPILE_OPTIONS "${SLIMT_QMM_COMPILE_OPTIONS}")
endif(WITH_GEMMOLOGY)

set(SLIMT_LIBRARIES)

if(BUILD_SHARED)
  list(APPEND SLIMT_LIBRARIES slimt-shared)
  add_library(slimt-shared SHARED ${SLIMT_SOURCES} ${SLIMT_PUBLIC_HEADERS}
                                  ${SLIMT_PRIVATE_HEADERS})
  set_target_properties(
    slimt-shared
    PROPERTIES VERSION ${PROJECT_VERSION_STRING}
               SOVERSION 0
               EXPORT_NAME slimt-shared)

endif(BUILD_SHARED)

if(BUILD_STATIC)
  add_library(slimt-static STATIC ${SLIMT_SOURCES} ${SLIMT_PUBLIC_HEADERS}
                                  ${SLIMT_PRIVATE_HEADERS})
  list(APPEND SLIMT_LIBRARIES slimt-static)
  add_library(slimt ALIAS slimt-static)
endif(BUILD_STATIC)

if(NOT BUILD_STATIC)
  add_library(slimt ALIAS slimt-shared)
endif(NOT BUILD_STATIC)

# Once the library is defined, it's possible to generate_export_header.
include(GenerateExportHeader)
generate_export_header(slimt BASE_NAME slimt EXPORT_FILE_NAME
                       "${CMAKE_CURRENT_BINARY_DIR}/Export.hh")

foreach(SLIMT_LIB IN LISTS SLIMT_LIBRARIES)
  set_target_properties(${SLIMT_LIB} PROPERTIES OUTPUT_NAME "slimt")

  target_link_libraries(
    ${SLIMT_LIB}
    PUBLIC ${SLIMT_PUBLIC_LIBS}
    INTERFACE "$<BUILD_INTERFACE:${SLIMT_INTERFACE_LIBS}>"
    PRIVATE "$<BUILD_INTERFACE:${SLIMT_PRIVATE_LIBS}>")

  target_include_directories(
    ${SLIMT_LIB}
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
      "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>"
  )

  target_link_options(${SLIMT_LIB} PUBLIC ${SLIMT_LINK_OPTIONS})
  target_compile_options(${SLIMT_LIB} PRIVATE ${SLIMT_COMPILE_OPTIONS})
  target_compile_definitions(${SLIMT_LIB} PRIVATE ${SLIMT_COMPILE_DEFINITIONS})
endforeach()

if(UNIX)
  install(FILES ${SLIMT_PUBLIC_HEADERS}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slimt)
  install(FILES ${SLIMT_GENERATED_HEADERS}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slimt/)
  install(TARGETS ${SLIMT_LIBRARIES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(UNIX)

# The following is partial code for enabling `find_package(slimt ...)` Not
# working at the moment, due to the inablility to export sentencepiece
# (accounting for it as an installed target).
if(SLIMT_PACKAGE)
  # Export cmake
  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/slimtConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/slimtConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/slimt")

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/slimtConfigVersion.cmake"
    VERSION "${PROJECT_VERSION_STRING}"
    COMPATIBILITY SameMajorVersion)

  # This will double install, but is-ok.
  install(
    TARGETS ${SLIMT_LIBRARIES} sentencepiece
    EXPORT slimtTargets
    LIBRARY COMPONENT slimt-lib DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER COMPONENT slimt-dev
                  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/slimt")
  install(
    EXPORT slimtTargets
    NAMESPACE slimt::
    COMPONENT slimt-dev
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/slimt")

  export(EXPORT slimtTargets
         FILE "${CMAKE_CURRENT_BINARY_DIR}/slimtTargets.cmake")

  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/slimtConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/slimtConfigVersion.cmake"
    COMPONENT slimt-dev
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/slimt")

  # Export pkgconfig provision. Hardcoded for now.
  set(SLIMT_PKGCONFIG_REQUIRES_PUBLIC "sentencepiece")
  set(SLIMT_PKGCONFIG_REQUIRES_PRIVATE "openblas xsimd")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/slimt.pc.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/slimt.pc" @ONLY)
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/slimt.pc"
    COMPONENT slimt-dev
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
endif(SLIMT_PACKAGE)
