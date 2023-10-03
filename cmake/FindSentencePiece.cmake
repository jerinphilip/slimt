find_package(PkgConfig REQUIRED)

set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(PC_SentencePiece REQUIRED sentencepiece)

find_path(
  SentencePiece_INCLUDE_DIR
  NAMES sentencepiece_processor.h
  PATHS ${PC_SentencePiece_INCLUDE_DIRS}
  PATH_SUFFIXES sentencepiece_processor)

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
    SentencePiece::SentencePiece PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                            "${SentencePiece_INCLUDE_DIR}")
endif()
