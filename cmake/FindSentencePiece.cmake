find_package(PkgConfig REQUIRED)

pkg_check_modules(PC_SentencePiece REQUIRED sentencepiece)

find_path(
  SentencePiece_INCLUDE_DIR
  NAMES sentencepiece_processor.h
  PATHS ${PC_SentencePiece_INCLUDE_DIRS}
  PATH_SUFFIXES sentencepiece)

find_library(
  SentencePiece_TRAIN_LIB
  NAMES sentencepiece_train
  PATHS ${PC_SentencePiece_LIBRARY_DIRS}
  PATH_SUFFIXES sentenecepiece)
find_library(
  SentencePiece_INFERENCE_LIB
  NAMES sentencepiece
  PATHS ${PC_SentencePiece_LIBRARY_DIRS}
  PATH_SUFFIXES sentenecepiece)

set(SentencePiece_VERSION ${PC_SentencePiece_VERSION})

mark_as_advanced(SentencePiece_FOUND SentencePiece_INCLUDE_DIR
                 SentencePiece_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  SentencePiece
  REQUIRED_VARS SentencePiece_INCLUDE_DIR
  VERSION_VAR SentencePiece_VERSION)

if(SentencePiece_FOUND)
  set(SentencePiece_INCLUDE_DIRS ${SentencePiece_INCLUDE_DIR})
endif()

if(SentencePiece_FOUND AND NOT TARGET SentencePiece::SentencePiece)
  add_library(SentencePiece::SentencePiece INTERFACE IMPORTED)
  set_target_properties(
    SentencePiece::SentencePiece
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SentencePiece_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${SentencePiece_INFERENCE_LIB}"
               INTERFACE_LINK_OPTIONS "${PC_SentencePiece_LDFLAGS}")

  add_library(SentencePiece::Train INTERFACE IMPORTED)
  set_target_properties(
    SentencePiece::Train
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SentencePiece_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${SentencePiece_TRAIN_LIB}"
               INTERFACE_LINK_OPTIONS "${PC_SentencePiece_LDFLAGS}")
  message(
    STATUS
      "${SentencePiece_INCLUDE_DIR} ${SentencePiece_TRAIN_LIB} ${SentencePiece_INFERENCE_LIB}"
  )

endif()
