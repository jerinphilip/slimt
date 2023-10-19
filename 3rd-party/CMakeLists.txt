# Create an alias target for SentencePiece
if(USE_BUILTIN_SENTENCEPIECE)
  # SentencePiece
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(SPM_USE_BUILTIN_PROTOBUF ON)
  add_subdirectory(sentencepiece EXCLUDE_FROM_ALL)

  add_library(spiece_iface INTERFACE)
  target_link_libraries(spiece_iface INTERFACE sentencepiece-static)
  get_target_property(SentencePiece_INCLUDES sentencepiece-static
                      INCLUDE_DIRECTORIES)
  target_include_directories(spiece_iface INTERFACE ${SentencePiece_INCLUDES}
                                                    sentencepiece/src/)

  add_library(SentencePiece::SentencePiece ALIAS spiece_iface)
else(USE_BUILTIN_SENTENCEPIECE)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(Sentencepiece REQUIRED sentencepiece)
  add_library(sentencepiece INTERFACE)
  target_link_libraries(sentencepiece INTERFACE ${Sentencepiece_LINK_LIBRARIES})
  target_include_directories(sentencepiece
                             INTERFACE ${Sentencepiece_INCLUDE_DIRS})
  target_link_options(sentencepiece INTERFACE ${SentencePiece_LDFLAGS}
                      ${SentencePiece_LDFLAGS_OTHER})
  target_link_directories(sentencepiece INTERFACE ${Sentencepiece_LIBRARY_DIRS})

  # This is an attempt to use libsentencepiece-dev or sentencepiece-git avaiable
  # via official repositories. However, the existing state of sentencepiece does
  # not ship sentencepiece.pb.h, and since we use it for manipulating
  # SentencepieceText, which is required to look into the original source
  # surface, it becomes a problem. See:
  # https://github.com/google/sentencepiece/issues/922
  target_include_directories(sentencepiece
                             INTERFACE sentencepiece/third_party/protobuf-lite)
  target_include_directories(sentencepiece
                             INTERFACE sentencepiece/src/builtin_pb)

  add_library(SentencePiece::SentencePiece ALIAS sentencepiece)
endif(USE_BUILTIN_SENTENCEPIECE)

# intgemm
if(WITH_INTGEMM)
  set(INTGEMM_DONT_BUILD_TESTS ON)
  add_subdirectory(intgemm EXCLUDE_FROM_ALL)
endif(WITH_INTGEMM)

# ruy
if(WITH_RUY)
  add_subdirectory(ruy EXCLUDE_FROM_ALL)
endif(WITH_RUY)
